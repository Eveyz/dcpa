"""Comprehensive tests for DCPA-Flat v3.3 WS-2: static_grid, SGSCFlat, PMDFlat, and HolographicCopyRing."""

from __future__ import annotations

import ast
import inspect
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.interfaces import DCPAFlatConfig, StaticGridBatch
from dcpa.modules.copy_ring import HolographicCopyRing
from dcpa.modules.pmd import PMDFlat
from dcpa.modules.sgsc import SGSCFlat
from dcpa.ops.rope import apply_concept_rope_2d
from dcpa.ops.static_grid import (
    build_static_segment_map,
    compute_static_seg_depths_and_triggers,
    gather_seg_to_byte,
    static_lif_scan,
    static_segment_mean,
    static_segment_scatter_add,
    static_segment_softmax,
)


def _pack_docs_into_static_batch(
    lane_docs: list[list[tuple[bytes, int, int]]],
    P: int,
    T: int,
    cap_per_lane: int = 64,
    device: torch.device | str = "cpu",
) -> StaticGridBatch:
    """Helper packing lists of `(raw_bytes, lang_id, style_id)` per lane into a StaticGridBatch."""
    assert len(lane_docs) == P
    x_bytes = torch.zeros((P, T), dtype=torch.uint8)
    doc_id_byte = torch.full((P, T), -1, dtype=torch.int32)
    doc_start_byte = torch.zeros((P, T), dtype=torch.bool)
    doc_end_byte = torch.zeros((P, T), dtype=torch.bool)
    loss_weight = torch.zeros((P, T), dtype=torch.float32)
    doc_meta = torch.zeros((P * cap_per_lane, 3), dtype=torch.int32)

    next_doc_id = 0
    for p, docs in enumerate(lane_docs):
        cursor = 0
        for raw, lang_id, style_id in docs:
            n = len(raw)
            assert n > 0 and cursor + n <= T
            x_bytes[p, cursor : cursor + n] = torch.tensor(list(raw), dtype=torch.uint8)
            doc_id_byte[p, cursor : cursor + n] = next_doc_id
            doc_start_byte[p, cursor] = True
            doc_end_byte[p, cursor + n - 1] = True
            loss_weight[p, cursor : cursor + n] = 1.0
            doc_meta[next_doc_id, 0] = lang_id
            doc_meta[next_doc_id, 1] = style_id
            doc_meta[next_doc_id, 2] = 0
            next_doc_id += 1
            cursor += n

    batch = StaticGridBatch(
        x_bytes=x_bytes,
        doc_id_byte=doc_id_byte,
        doc_start_byte=doc_start_byte,
        doc_end_byte=doc_end_byte,
        loss_weight=loss_weight,
        doc_meta=doc_meta,
    ).to(torch.device(device))
    batch.validate()
    return batch


def _sequential_lif_ref(
    info: torch.Tensor,
    delta_info: torch.Tensor,
    raw_bytes: torch.Tensor,
    doc_id_byte: torch.Tensor,
    doc_start_byte: torch.Tensor,
    doc_end_byte: torch.Tensor,
    lif_lambda: float,
    beta1: float,
    beta2: float,
    theta: float,
    max_seg_len: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brute-force sequential reference for `static_lif_scan` parity verification."""
    P, T = info.shape
    spikes = torch.zeros((P, T), dtype=torch.bool)
    v_out = torch.zeros((P, T), dtype=torch.float32)
    for p in range(P):
        v = 0.0
        seg_len = 0
        utf8_rem = 0
        for t in range(T):
            did = int(doc_id_byte[p, t].item())
            if did < 0:
                v = 0.0
                seg_len = 0
                utf8_rem = 0
                continue
            is_start = (
                t == 0
                or bool(doc_start_byte[p, t].item())
                or int(doc_id_byte[p, t - 1].item()) != did
            )
            if is_start:
                v = 0.0
                seg_len = 0
                utf8_rem = 0
            is_end = (
                t == T - 1
                or bool(doc_end_byte[p, t].item())
                or int(doc_id_byte[p, t + 1].item()) != did
                or bool(doc_start_byte[p, t + 1].item())
            )
            bv = int(raw_bytes[p, t].item())
            if (bv & 0xC0) == 0x80:
                utf8_rem = max(utf8_rem - 1, 0)
            elif (bv & 0xE0) == 0xC0:
                utf8_rem = 1
            elif (bv & 0xF0) == 0xE0:
                utf8_rem = 2
            elif (bv & 0xF8) == 0xF0:
                utf8_rem = 3
            else:
                utf8_rem = 0
            drive = beta1 * float(info[p, t].item()) + beta2 * float(delta_info[p, t].item())
            v = lif_lambda * v + drive
            v_out[p, t] = v
            seg_len += 1
            fire = is_end or (seg_len >= max_seg_len) or (utf8_rem == 0 and v >= theta)
            spikes[p, t] = fire
            if fire:
                v = 0.0
                seg_len = 0
    return v_out, spikes


def test_static_grid_multidoc_overflow_sentinels_depths_and_compile() -> None:
    """Test R4 trash-slot overflow isolation, unique sentinels, exact tau in 1..16, depths/triggers & compile."""
    P, T, S = 4, 64, 8
    # Lane 0: 3 docs with deliberate overflow (> 8 segments!) + overflow \x1dE\n (must NOT leak!) + trailing pad
    # Lane 1: 3 docs with nested brackets, \x1dE\n pin trigger, 0x04 unpin trigger (inside slots 0..7) + doc-boundary reset
    # Lane 2: exact S=8 segments (zero overflow!) + cross-doc split \x1dE | \n (must NOT trigger pin!) + trailing pad
    # Lane 3: completely empty (all pad) lane
    lane_docs = [
        [
            (b"a b c d e f g h i j ", 0, 0),  # 20 bytes -> 10 segs (> S=8!)
            (b"ov\x1dE\n\x04", 1, 2),          # in overflow slots -> must NOT leak into pin_t[0]/unpin_t[0]
            (b"third", 3, 4),
        ],
        [
            (b"f(\x1dE\n)\x04", 3, 4),         # 7 bytes: valid slots, hits pin & unpin
            (b"(((u", 0, 1),                   # unclosed depth 3
            (b"ok(1)", 2, 0),                  # fresh doc: start depth MUST reset to 0!
        ],
        [
            (b"0123456789\x1dE", 0, 0),        # 12 bytes -> 6 segs, ends with \x1dE
            (b"\nabc", 0, 0),                  # 4 bytes -> 2 segs, starts with \n (total 8 == S, NOT a pin!)
        ],
        [],
    ]
    batch = _pack_docs_into_static_batch(lane_docs, P=P, T=T)

    # Spike every 2 bytes on valid positions
    spikes = torch.zeros((P, T), dtype=torch.bool)
    spikes[:, 1::2] = True
    # Lane 1: spike every 4 bytes so all 3 docs fit within S=8 segments
    spikes[1, :] = False
    spikes[1, 3::4] = True

    seg_map = build_static_segment_map(
        spikes=spikes,
        doc_id_byte=batch.doc_id_byte,
        doc_start_byte=batch.doc_start_byte,
        doc_end_byte=batch.doc_end_byte,
        grid_segs=S,
        max_seg_len=16,
    )

    # 1. Overflow & trash-slot routing checks
    assert seg_map.seg_counts[0].item() == S
    assert seg_map.byte_overflow[0].any().item(), "Lane 0 must deliberately overflow S=8"
    assert not seg_map.byte_overflow[2].any().item(), "Lane 2 (exact S=8 + trailing pad) must have ZERO overflow!"
    assert seg_map.seg_counts[2].item() == S
    assert seg_map.seg_counts[3].item() == 0
    assert not seg_map.seg_valid[3].any().item()

    # Overflow and pad bytes MUST map strictly to trash slot S
    assert (seg_map.byte_to_seg[seg_map.byte_overflow] == S).all().item()
    assert (seg_map.byte_to_seg[batch.doc_id_byte < 0] == S).all().item()

    # 2. Zero cross-doc contamination in slots 0..S-1
    for p in range(P):
        for s in range(S):
            mask = (seg_map.byte_to_seg[p] == s)
            if seg_map.seg_valid[p, s].item():
                assert mask.any().item()
                d_vals = batch.doc_id_byte[p][mask]
                assert (d_vals == seg_map.seg_doc_id[p, s]).all().item(), (
                    f"Cross-doc contamination at lane {p}, seg {s}"
                )
            else:
                assert not mask.any().item()

    # 3. Unique Negative Sentinel Isolation Law: -(p * S + s + 2)
    invalid_sentinels: list[int] = []
    for p in range(P):
        for s in range(S):
            if not seg_map.seg_valid[p, s].item():
                expected = -(p * S + s + 2)
                got = int(seg_map.seg_doc_id[p, s].item())
                assert got == expected
                assert got <= -2
                invalid_sentinels.append(got)
    assert len(invalid_sentinels) == len(set(invalid_sentinels)), "Invalid sentinels must be globally unique!"

    # seg_reset contract
    assert seg_map.seg_reset[:, 0].all().item()
    assert torch.equal(seg_map.seg_reset[:, 1:], seg_map.seg_doc_id[:, 1:] != seg_map.seg_doc_id[:, :-1])

    # 4. Exact tau in 1..16
    assert (seg_map.intra_seg_pos >= 1).all().item() and (seg_map.intra_seg_pos <= 16).all().item()
    for p in range(P):
        cur_seg = -1
        cur_tau = 0
        for t in range(T):
            if batch.doc_id_byte[p, t].item() < 0:
                continue
            if seg_map.is_seg_start[p, t].item():
                cur_tau = 1
                cur_seg += 1
            else:
                cur_tau += 1
            assert int(seg_map.intra_seg_pos[p, t].item()) == cur_tau

    # 5. Softmax, scatter-add & gather_seg_to_byte drop trash slot S cleanly
    scores = torch.randn((P, T), requires_grad=True)
    vals = torch.randn((P, T, 12), requires_grad=True)
    weights = static_segment_softmax(scores, seg_map.byte_to_seg, S)
    scattered = static_segment_scatter_add(weights.unsqueeze(-1) * vals, seg_map.byte_to_seg, S)
    assert scattered.shape == (P, S, 12)
    # Weights sum to 1.0 in every valid segment 0..S-1
    w_sums = static_segment_scatter_add(weights, seg_map.byte_to_seg, S)
    assert torch.allclose(w_sums[seg_map.seg_valid], torch.ones_like(w_sums[seg_map.seg_valid]), atol=1e-5)

    # Overflow/pad bytes MUST contribute ZERO gradient into slots 0..S-1
    scattered.sum().backward()
    trash_mask = seg_map.byte_to_seg == S
    assert (vals.grad[trash_mask] == 0).all().item()
    assert (scores.grad[trash_mask] == 0).all().item()

    gathered = gather_seg_to_byte(scattered.detach(), seg_map.byte_to_seg)
    assert gathered.shape == (P, T, 12)
    assert (gathered[trash_mask] == 0).all().item()

    # 6. Causal AST bracket depths + pin/unpin triggers
    start_d, end_d, pin_t, unpin_t = compute_static_seg_depths_and_triggers(
        batch.x_bytes, batch.doc_id_byte, batch.doc_start_byte, seg_map, grid_segs=S, max_depth=31
    )
    assert (start_d >= 0).all().item() and (start_d <= 31).all().item()
    assert (end_d >= 0).all().item() and (end_d <= 31).all().item()
    # Lane 1 doc 0 contains \x1dE\n and 0x04 -> triggers must fire in lane 1;
    # Lane 0 has \x1dE\n and 0x04 ONLY in overflow bytes -> must NOT leak;
    # Lane 2 splits \x1dE and \n across doc boundary -> must NOT trigger pin!
    assert pin_t[1].any().item() and not pin_t[0].any().item() and not pin_t[2].any().item()
    assert unpin_t[1].any().item() and not unpin_t[0].any().item()
    # Lane 1 doc 2 ("ok(1)") follows unclosed "(((u" -> its start depth MUST reset to 0!
    doc2_off = len(lane_docs[1][0][0]) + len(lane_docs[1][1][0])
    doc2_id = int(batch.doc_id_byte[1, doc2_off].item())
    doc2_segs = (seg_map.seg_doc_id[1] == doc2_id).nonzero(as_tuple=False).squeeze(-1)
    assert doc2_segs.numel() > 0
    assert int(start_d[1, doc2_segs[0]].item()) == 0

    # 7. fullgraph=True, dynamic=False compilation check on static_grid pipeline
    def _grid_pipeline(x_b, d_id, d_st, d_en, sp, sc, v):
        sm = build_static_segment_map(sp, d_id, d_st, d_en, grid_segs=S, max_seg_len=16)
        w = static_segment_softmax(sc, sm.byte_to_seg, S)
        agg = static_segment_scatter_add(w.unsqueeze(-1) * v, sm.byte_to_seg, S)
        mn = static_segment_mean(sc, sm.byte_to_seg, sm.seg_lens, S)
        back = gather_seg_to_byte(agg, sm.byte_to_seg)
        sd, ed, pt, ut = compute_static_seg_depths_and_triggers(x_b, d_id, d_st, sm, grid_segs=S)
        return agg, mn, back, sd, ed, pt, ut

    compiled_fn = torch.compile(_grid_pipeline, backend="aot_eager", fullgraph=True, dynamic=False)
    sc_c = torch.randn((P, T), requires_grad=True)
    v_c = torch.randn((P, T, 12), requires_grad=True)
    out_c = compiled_fn(batch.x_bytes, batch.doc_id_byte, batch.doc_start_byte, batch.doc_end_byte, spikes, sc_c, v_c)
    out_e = _grid_pipeline(batch.x_bytes, batch.doc_id_byte, batch.doc_start_byte, batch.doc_end_byte, spikes, sc_c.detach(), v_c.detach())
    for tc, te in zip(out_c, out_e):
        if tc.is_floating_point():
            assert torch.allclose(tc, te, atol=1e-5)
        else:
            assert torch.equal(tc, te)
    (out_c[0].sum() + out_c[1].sum() + out_c[2].sum()).backward()
    assert sc_c.grad is not None and torch.isfinite(sc_c.grad).all().item()


def test_static_lif_scan_exact_parity_utf8_and_compile() -> None:
    """Verify pointer-doubling `static_lif_scan` matches sequential reference bit-for-bit on UTF-8 & multi-doc."""
    torch.manual_seed(123)
    P, T = 3, 96
    lane_docs = [
        [("你好世界😀Python函数(a, b) ".encode("utf-8"), 1, 0), (b"0123456789abcdefghijklmnopqrstuvwxyz", 0, 1)],
        [(b"short", 0, 0), ("中文测试字符串长度超过十六字节用来触发最大段长限制".encode("utf-8"), 1, 2)],
        [(b"x" * 40, 3, 4)],
    ]
    batch = _pack_docs_into_static_batch(lane_docs, P=P, T=T)
    info = torch.rand((P, T), dtype=torch.float32) * 4.0
    info.requires_grad_(True)
    delta_info = torch.rand((P, T), dtype=torch.float32) * 2.0
    theta = torch.tensor(2.2, dtype=torch.float32, requires_grad=True)

    v_par, sp_par = static_lif_scan(
        info,
        delta_info,
        batch.x_bytes,
        batch.doc_id_byte,
        batch.doc_start_byte,
        batch.doc_end_byte,
        lif_lambda=0.85,
        beta1=0.25,
        beta2=0.75,
        theta=theta,
        max_seg_len=16,
    )
    v_ref, sp_ref = _sequential_lif_ref(
        info.detach(),
        delta_info.detach(),
        batch.x_bytes,
        batch.doc_id_byte,
        batch.doc_start_byte,
        batch.doc_end_byte,
        lif_lambda=0.85,
        beta1=0.25,
        beta2=0.75,
        theta=float(theta.item()),
        max_seg_len=16,
    )
    assert torch.equal(sp_par, sp_ref), "Pointer-doubling LIF spikes must match sequential reference bit-for-bit!"
    assert torch.allclose(v_par, v_ref, atol=1e-5)

    # Differentiable gradient through v_tilde -> info
    v_par.sum().backward()
    assert info.grad is not None and (info.grad.abs() > 0).any().item()

    # Zero-graph-break compile verification
    def _lif_fn(inf, dinf, xb, did, dst, den, th):
        return static_lif_scan(inf, dinf, xb, did, dst, den, 0.85, 0.25, 0.75, th, 16)

    c_lif = torch.compile(_lif_fn, backend="aot_eager", fullgraph=True, dynamic=False)
    v_c, sp_c = c_lif(
        info.detach().clone().requires_grad_(True),
        delta_info,
        batch.x_bytes,
        batch.doc_id_byte,
        batch.doc_start_byte,
        batch.doc_end_byte,
        theta.detach(),
    )
    assert torch.equal(sp_c, sp_ref)
    assert torch.allclose(v_c, v_ref, atol=1e-5)


def test_sgsc_flat_and_pmd_flat_end_to_end_temps_overflow_and_compile() -> None:
    """Test SGSCFlat + PMDFlat forward/backward, ScatterBridgeOut.validate(), L2 vocab, overflow mask & compile."""
    torch.manual_seed(42)
    cfg = DCPAFlatConfig.flat_micro()
    P, T, S, d_s, D = cfg.grid_lanes, cfg.grid_bytes, cfg.grid_segs, cfg.d_s, cfg.d_concept

    sgsc = SGSCFlat(cfg)
    pmd = PMDFlat(cfg, shared_embed=sgsc.embed, shared_w_out=sgsc.w_out)

    # Single-track contract: ZERO Broca CQ code on PMDFlat
    assert not hasattr(pmd, "w_ord")
    assert not hasattr(pmd, "w_coh")
    assert not hasattr(pmd, "broca_cq_step")

    # Shared weights & independent temperatures initialized to ln(8.0)
    assert pmd.embed is sgsc.embed
    assert pmd.w_out is sgsc.w_out
    assert math.isclose(float(sgsc.vocab_temp.item()), cfg.vocab_temp_init, rel_tol=1e-5)
    assert math.isclose(float(pmd.vocab_temp.item()), cfg.vocab_temp_init, rel_tol=1e-5)

    lane_docs = [
        [(b"def foo(x):\n    return x * 2\n", 3, 4), ("你好世界, hello world! \x1dE\n\x04".encode("utf-8"), 1, 2)],
        [(b"SELECT id, name FROM users WHERE active = 1;\n", 4, 0), (b"const y = [1, 2, 3];", 5, 4)],
        [(b"The quick brown fox jumps over the lazy dog. " * 4, 0, 3)],
        [(b"Theorem 1: for all x in R, (x + 1)^2 >= 0.\n", 2, 0)],
    ]
    batch = _pack_docs_into_static_batch(lane_docs, P=P, T=T, cap_per_lane=cfg.doc_meta_cap_per_lane)

    bridge = sgsc(batch)
    bridge.validate(P, T, S, d_s, D)

    # Feed bridge.f_stem through PMDFlat
    pmd_out = pmd(batch, bridge, bridge.f_stem)
    assert pmd_out.byte_logits.shape == (P, T, 256)
    assert pmd_out.stack_xi.shape == (P, T, 16, 4)
    assert torch.isfinite(pmd_out.ce_loss).item() and torch.isfinite(pmd_out.halt_loss).item()

    # Check row-L2-normalization invariance of shared w_out
    with torch.no_grad():
        prior_before = bridge.prior_logits.clone()
        logits_before = pmd_out.byte_logits.clone()
        sgsc.w_out.weight.mul_(3.7)
        bridge_scaled = sgsc(batch)
        pmd_scaled = pmd(batch, bridge_scaled, bridge_scaled.f_stem)
        assert torch.allclose(bridge_scaled.prior_logits, prior_before, atol=1e-4)
        assert torch.allclose(pmd_scaled.byte_logits, logits_before, atol=1e-4)

        # Mutating log_temp_pmd changes ONLY pmd logits, NOT sgsc prior_logits
        pmd.log_temp_pmd.add_(0.5)
        bridge_t = sgsc(batch)
        pmd_t = pmd(batch, bridge_t, bridge_t.f_stem)
        assert torch.allclose(bridge_t.prior_logits, prior_before, atol=1e-4)
        assert not torch.allclose(pmd_t.byte_logits, logits_before, atol=1e-2)
        pmd.log_temp_pmd.sub_(0.5)

    # End-to-end backward: verify gradients flow to BOTH log_temp_sgsc, log_temp_pmd, theta_raw, w_out, embed
    sgsc.zero_grad()
    pmd.zero_grad()
    bridge2 = sgsc(batch)
    pmd_out2 = pmd(batch, bridge2, bridge2.f_stem)
    sgsc_prior_ce = F.cross_entropy(
        bridge2.prior_logits.reshape(-1, 256), batch.x_bytes.reshape(-1).long()
    )
    loss = pmd_out2.ce_loss + pmd_out2.halt_loss + 0.2 * sgsc_prior_ce + 0.01 * bridge2.x0.pow(2).mean()
    loss.backward()
    for name, param in [
        ("sgsc.log_temp_sgsc", sgsc.log_temp_sgsc),
        ("pmd.log_temp_pmd", pmd.log_temp_pmd),
        ("sgsc.theta_raw", sgsc.theta_raw),
        ("sgsc.w_out.weight", sgsc.w_out.weight),
        ("sgsc.embed.weight", sgsc.embed.weight),
        ("sgsc.lang_embed.weight", sgsc.lang_embed.weight),
        ("sgsc.style_embed.weight", sgsc.style_embed.weight),
        ("pmd.u_demod.weight", pmd.u_demod.weight),
        ("pmd.w_act.weight", pmd.w_act.weight),
    ]:
        assert param.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(param.grad).all().item(), f"Non-finite grad for {name}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero grad for {name}"

    # Overflow loss masking test: force spikes every 2 bytes on 200-byte doc with S=16 -> massive overflow!
    long_docs = [[(b"ab" * 100, 0, 0)] for _ in range(P)]
    ov_batch = _pack_docs_into_static_batch(long_docs, P=P, T=T, cap_per_lane=cfg.doc_meta_cap_per_lane)
    forced_sp = torch.zeros((P, T), dtype=torch.bool)
    forced_sp[:, 1::2] = True
    ov_bridge = sgsc(ov_batch, forced_spikes=forced_sp, grid_segs=16)
    assert ov_bridge.byte_overflow.any().item()
    ov_out_a = pmd(ov_batch, ov_bridge, ov_bridge.f_stem)
    assert (ov_out_a.eff_weight[ov_bridge.byte_overflow] == 0.0).all().item()

    # Corrupt loss_weight on overflow positions -> ce_loss MUST remain strictly identical!
    corrupted_lw = ov_batch.loss_weight.clone()
    corrupted_lw[ov_bridge.byte_overflow] = 999.0
    ov_batch_b = StaticGridBatch(
        x_bytes=ov_batch.x_bytes,
        doc_id_byte=ov_batch.doc_id_byte,
        doc_start_byte=ov_batch.doc_start_byte,
        doc_end_byte=ov_batch.doc_end_byte,
        loss_weight=corrupted_lw,
        doc_meta=ov_batch.doc_meta,
    )
    ov_out_b = pmd(ov_batch_b, ov_bridge, ov_bridge.f_stem)
    assert torch.allclose(ov_out_a.ce_loss, ov_out_b.ce_loss, atol=1e-6)

    # torch.compile(backend="aot_eager", fullgraph=True, dynamic=False) end-to-end forward + backward
    class _WS2BridgeModel(nn.Module):
        def __init__(self, s_mod: SGSCFlat, p_mod: PMDFlat) -> None:
            super().__init__()
            self.sgsc = s_mod
            self.pmd = p_mod

        def forward(self, xb, did, dst, den, lw, meta):
            b = StaticGridBatch(xb, did, dst, den, lw, meta)
            br = self.sgsc(b)
            po = self.pmd(b, br, br.f_stem)
            return po.ce_loss, po.halt_loss, br.x0, po.byte_logits

    wrapper = _WS2BridgeModel(sgsc, pmd)
    compiled_ws2 = torch.compile(wrapper, backend="aot_eager", fullgraph=True, dynamic=False)
    sgsc.zero_grad()
    pmd.zero_grad()
    ce_c, halt_c, x0_c, logits_c = compiled_ws2(
        batch.x_bytes, batch.doc_id_byte, batch.doc_start_byte, batch.doc_end_byte, batch.loss_weight, batch.doc_meta
    )
    assert torch.allclose(ce_c, pmd_out2.ce_loss.detach(), atol=1e-4)
    assert torch.allclose(x0_c, bridge2.x0.detach(), atol=1e-4)
    (ce_c + halt_c + 0.01 * x0_c.pow(2).mean()).backward()
    assert sgsc.log_temp_sgsc.grad is not None and torch.isfinite(sgsc.log_temp_sgsc.grad).all().item()
    assert pmd.log_temp_pmd.grad is not None and torch.isfinite(pmd.log_temp_pmd.grad).all().item()


def test_sgsc_flat_and_pmd_flat_streaming_parity() -> None:
    """Verify single-stream SGSCFlat & PMDFlat step_streaming_byte match batched forward < 1e-4."""
    torch.manual_seed(77)
    cfg = DCPAFlatConfig.flat_micro()
    sgsc = SGSCFlat(cfg).eval()
    pmd = PMDFlat(cfg, shared_embed=sgsc.embed, shared_w_out=sgsc.w_out).eval()

    doc_bytes = "fn calc(x) { return [x + 1]; } // 你好世界 \x1dE\n(ok)\x04".encode("utf-8")
    T_doc = len(doc_bytes)
    lane_docs = [[(doc_bytes, 3, 4)]] + [[] for _ in range(cfg.grid_lanes - 1)]
    batch = _pack_docs_into_static_batch(lane_docs, P=cfg.grid_lanes, T=cfg.grid_bytes)

    with torch.no_grad():
        bridge = sgsc(batch)
        pmd_out = pmd(batch, bridge, bridge.f_stem)

        # 1. SGSCFlat streaming parity
        raw_1d = batch.x_bytes[0, :T_doc]
        st_sgsc = sgsc.encode_streaming_segments(raw_1d, lang_id=3, style_id=4)
        n_segs = int(bridge.seg_counts[0].item())
        assert torch.equal(st_sgsc["spikes"], bridge.spikes[0, :T_doc])
        assert st_sgsc["f_stem"].shape[0] == n_segs
        assert (st_sgsc["prior_logits"] - bridge.prior_logits[0, :T_doc]).abs().max().item() < 1e-5
        assert (st_sgsc["f_stem"] - bridge.f_stem[0, :n_segs]).abs().max().item() < 1e-5
        assert (st_sgsc["x0"] - bridge.x0[0, :n_segs]).abs().max().item() < 1e-5
        assert torch.equal(st_sgsc["seg_start_depth"], bridge.seg_start_depth[0, :n_segs])
        assert torch.equal(st_sgsc["seg_end_depth"], bridge.seg_end_depth[0, :n_segs])
        assert torch.equal(st_sgsc["seg_pin_trig"], bridge.seg_pin_trig[0, :n_segs])
        assert torch.equal(st_sgsc["seg_unpin_trig"], bridge.seg_unpin_trig[0, :n_segs])

        # 2. PMDFlat streaming parity byte-by-byte
        m_cur = pmd.m0.unsqueeze(0).clone()
        xi_cur = torch.zeros((1, 16, 4), dtype=torch.float32)
        charge_cur = torch.zeros((1,), dtype=torch.float32)
        prev_b = torch.zeros((1,), dtype=torch.long)
        l_reg_1 = st_sgsc["seg_l_reg"][0:1]

        logits_stream = []
        charge_stream = []
        for t in range(T_doc):
            s_idx = int(bridge.byte_to_seg[0, t].item())
            tau = int(bridge.intra_seg_pos[0, t].item())
            f_out_s = bridge.f_stem[0, s_idx : s_idx + 1]
            if tau == 1:
                charge_cur = pmd.init_segment_charge(f_out_s, l_reg_1)
            extra_rst = torch.tensor(
                [t >= 3 and doc_bytes[t - 3 : t] == b"\x1dE\n"],
                dtype=torch.bool,
            )
            lg, m_cur, xi_cur, charge_cur, _ = pmd.step_streaming_byte(
                prev_byte=prev_b,
                f_out=f_out_s,
                l_reg=l_reg_1,
                tau_step=tau,
                m_prev=m_cur,
                xi_prev=xi_cur,
                charge_prev=charge_cur,
                extra_stack_reset=extra_rst,
            )
            logits_stream.append(lg[0])
            charge_stream.append(charge_cur[0])
            prev_b = batch.x_bytes[0, t : t + 1].to(torch.long)

        logits_st = torch.stack(logits_stream, dim=0)
        charge_st = torch.stack(charge_stream, dim=0)
        assert (logits_st - pmd_out.byte_logits[0, :T_doc]).abs().max().item() < 1e-5
        assert (charge_st - pmd_out.charge_e[0, :T_doc]).abs().max().item() < 1e-5


def _brute_force_copy_ring_ref(
    ring: HolographicCopyRing,
    q_state: torch.Tensor,
    f_stem: torch.Tensor,
    seg_doc_id: torch.Tensor,
    seg_start_depth: torch.Tensor,
    seg_end_depth: torch.Tensor,
) -> torch.Tensor:
    """Exact O(S^2) brute-force causal reference for `HolographicCopyRing`."""
    P, S, _ = q_state.shape
    W = ring.window_size
    H = ring.n_heads
    d_k = ring.d_k
    device = q_state.device

    pos = torch.arange(S, device=device, dtype=torch.int32).unsqueeze(0).expand(P, S)
    q_raw = ring.w_q(ring.norm_q(q_state)).view(P, S, H, d_k)
    k_raw = ring.w_k(f_stem).view(P, S, H, d_k)
    q_rot = apply_concept_rope_2d(
        q_raw, pos, seg_start_depth, theta_base=ring.cfg.rope_theta_concept, scope_period=ring.cfg.rope_period
    ).permute(0, 2, 1, 3)  # [P, H, S, d_k]
    k_rot = apply_concept_rope_2d(
        k_raw, pos, seg_end_depth, theta_base=ring.cfg.rope_theta_concept, scope_period=ring.cfg.rope_period
    ).permute(0, 2, 1, 3)  # [P, H, S, d_k]

    scores = torch.matmul(q_rot.float(), k_rot.float().transpose(-2, -1)) * ring.scale  # [P, H, S, S]
    idx = torch.arange(S, device=device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # [S, S]
    band = (diff >= 1) & (diff <= W)
    same_doc = (seg_doc_id.unsqueeze(2) == seg_doc_id.unsqueeze(1)) & (seg_doc_id.unsqueeze(2) >= 0)
    valid = (same_doc & band.unsqueeze(0)).unsqueeze(1)  # [P, 1, S, S]
    scores = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
    sink = ring.b_sink.float().view(1, H, 1, 1).expand(P, H, S, 1)
    attn = F.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :S]  # [P, H, S, S]
    copy_w = attn.mean(dim=1)  # [P, S, S]
    return torch.matmul(copy_w, f_stem.float()).to(f_stem.dtype)


def test_holographic_copy_ring_exact_ref_all_first_seg_and_streaming() -> None:
    """Test HolographicCopyRing vs O(S^2) ref (< 1e-5), all-first-seg zero NaN bwd, streaming parity & compile."""
    torch.manual_seed(99)
    cfg = DCPAFlatConfig.flat_micro()  # P=4, S=64, W=32, N_c=2
    P, S, W, D, d_s = cfg.grid_lanes, cfg.grid_segs, cfg.window_size, cfg.d_concept, cfg.d_s
    ring = HolographicCopyRing(cfg)
    with torch.no_grad():
        ring.b_sink.copy_(torch.linspace(-0.3, 0.5, cfg.n_heads))

    q_state = torch.randn((P, S, D))
    f_stem = torch.randn((P, S, d_s))
    seg_start_depth = torch.randint(0, 8, (P, S), dtype=torch.int32)
    seg_end_depth = torch.randint(0, 8, (P, S), dtype=torch.int32)

    # Multi-doc packed lanes crossing chunk boundaries + unique negative invalid sentinels at tail
    seg_doc_id = torch.zeros((P, S), dtype=torch.int32)
    seg_doc_id[0, :45] = 0
    seg_doc_id[0, 45:60] = 1
    seg_doc_id[1, :20] = 2
    seg_doc_id[1, 20:55] = 3
    seg_doc_id[2, :] = 4  # full 64-seg doc spanning both W=32 chunks
    # Lane 3: single-segment docs + invalid tail
    for s in range(10):
        seg_doc_id[3, s] = 5 + s
    # Assign unique negative sentinels to invalid tail slots
    for p, valid_end in [(0, 60), (1, 55), (2, 64), (3, 10)]:
        for s in range(valid_end, S):
            seg_doc_id[p, s] = -(p * S + s + 2)

    # 1. Exact comparison vs brute-force O(S^2) causal lookup (< 1e-5)
    out = ring(q_state, f_stem, seg_doc_id, seg_start_depth=seg_start_depth, seg_end_depth=seg_end_depth)
    ref = _brute_force_copy_ring_ref(ring, q_state, f_stem, seg_doc_id, seg_start_depth, seg_end_depth)
    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-5, f"CopyRing chunked vs brute-force max err {max_err} >= 1e-5"

    # 2. All-first-segment doc inputs: forward + backward ZERO NaN!
    unique_docs = torch.arange(P * S, dtype=torch.int32).view(P, S)
    q_first = torch.randn((P, S, D), requires_grad=True)
    f_first = torch.randn((P, S, d_s), requires_grad=True)
    ring.zero_grad()
    out_first = ring(q_first, f_first, unique_docs, seg_start_depth=seg_start_depth, seg_end_depth=seg_end_depth)
    assert torch.isfinite(out_first).all().item()
    assert (out_first == 0.0).all().item(), "All-first-segment queries must attend 100% to zero-valued sink!"
    (out_first.sum() + 0.0 * ring.b_sink.sum()).backward()
    for t_name, grad in [
        ("q_first.grad", q_first.grad),
        ("f_first.grad", f_first.grad),
        ("w_q.weight.grad", ring.w_q.weight.grad),
        ("w_k.weight.grad", ring.w_k.weight.grad),
        ("b_sink.grad", ring.b_sink.grad),
    ]:
        assert grad is not None, f"Missing {t_name}"
        assert torch.isfinite(grad).all().item(), f"NaN/Inf in {t_name} for all-first-segment doc inputs!"

    # 3. step_streaming parity < 1e-5 over full multi-chunk doc (lane 2, len=64 > W=32)
    r_stem, r_pos, r_dep, r_val = ring.init_ring_state(batch_size=1)
    stream_outs = []
    with torch.no_grad():
        for s in range(S):
            cur_p = torch.tensor([s], dtype=torch.int32)
            cur_sd = seg_start_depth[2, s : s + 1]
            cur_ed = seg_end_depth[2, s : s + 1]
            y_s = ring.step_streaming(
                q_state=q_state[2, s : s + 1],
                ring_stem=r_stem,
                ring_pos=r_pos,
                ring_valid=r_val,
                cur_pos=cur_p,
                ring_depth=r_dep,
                cur_depth=cur_sd,
            )
            stream_outs.append(y_s[0])
            r_stem, r_pos, r_dep, r_val = ring.push_ring(
                r_stem, r_pos, r_dep, r_val, f_stem[2, s : s + 1], cur_p, cur_ed
            )
    stream_cat = torch.stack(stream_outs, dim=0)
    st_err = (stream_cat - out[2]).abs().max().item()
    assert st_err < 1e-5, f"CopyRing streaming vs batched max err {st_err} >= 1e-5"

    # 4. torch.compile(fullgraph=True, dynamic=False) check (forward + backward)
    c_ring = torch.compile(ring, backend="aot_eager", fullgraph=True, dynamic=False)
    q_c = q_state.clone().requires_grad_(True)
    f_c = f_stem.clone().requires_grad_(True)
    ring.zero_grad()
    out_c = c_ring(q_c, f_c, seg_doc_id, seg_start_depth, seg_end_depth)
    assert torch.allclose(out_c, ref, atol=1e-5)
    out_c.pow(2).sum().backward()
    assert q_c.grad is not None and torch.isfinite(q_c.grad).all().item()
    assert f_c.grad is not None and torch.isfinite(f_c.grad).all().item()
    assert ring.b_sink.grad is not None and torch.isfinite(ring.b_sink.grad).all().item()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_ws2_mps_inductor_fullgraph_compile() -> None:
    """Verify WS-2 modules compile cleanly under MPS inductor with fullgraph=True, dynamic=False."""
    device = torch.device("mps")
    torch.manual_seed(7)
    cfg = DCPAFlatConfig.flat_micro()
    P, T = cfg.grid_lanes, cfg.grid_bytes

    sgsc = SGSCFlat(cfg).to(device)
    pmd = PMDFlat(cfg, shared_embed=sgsc.embed, shared_w_out=sgsc.w_out).to(device)
    ring = HolographicCopyRing(cfg).to(device)

    lane_docs = [
        [(b"def f(a): return a + 1\n", 3, 4), (b"hello world \x1dE\n\x04", 0, 1)],
        [(b"SELECT * FROM t;\n", 4, 0)],
        [("中文测试代码(123)".encode("utf-8"), 1, 2)],
        [(b"x = [1, 2, 3]\n", 3, 4)],
    ]
    batch = _pack_docs_into_static_batch(lane_docs, P=P, T=T, device=device)

    class _FullWS2MPS(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sgsc = sgsc
            self.pmd = pmd
            self.ring = ring

        def forward(self, xb, did, dst, den, lw, meta):
            b = StaticGridBatch(xb, did, dst, den, lw, meta)
            br = self.sgsc(b)
            f_cp = self.ring(br.x0, br.f_stem, br.seg_doc_id, br.seg_start_depth, br.seg_end_depth)
            po = self.pmd(b, br, br.f_stem + f_cp)
            return po.ce_loss, po.halt_loss, f_cp

    mod = _FullWS2MPS()
    eager_ce, eager_halt, eager_cp = mod(
        batch.x_bytes, batch.doc_id_byte, batch.doc_start_byte, batch.doc_end_byte, batch.loss_weight, batch.doc_meta
    )
    compiled = torch.compile(mod, backend="inductor", fullgraph=True, dynamic=False)
    comp_ce, comp_halt, comp_cp = compiled(
        batch.x_bytes, batch.doc_id_byte, batch.doc_start_byte, batch.doc_end_byte, batch.loss_weight, batch.doc_meta
    )
    assert torch.allclose(comp_ce, eager_ce, atol=1e-4)
    assert torch.allclose(comp_halt, eager_halt, atol=1e-4)
    assert torch.allclose(comp_cp, eager_cp, atol=1e-4)
    (comp_ce + comp_halt).backward()
    assert sgsc.log_temp_sgsc.grad is not None and torch.isfinite(sgsc.log_temp_sgsc.grad).all().item()
    assert pmd.log_temp_pmd.grad is not None and torch.isfinite(pmd.log_temp_pmd.grad).all().item()
    assert ring.b_sink.grad is not None and torch.isfinite(ring.b_sink.grad).all().item()


def test_ws2_static_red_line_source_audit() -> None:
    """AST-level audit ensuring ZERO forbidden dynamic/sync ops in WS-2 forward hot paths."""
    import textwrap
    import dcpa.ops.static_grid as sg_mod
    import dcpa.modules.copy_ring as cr_mod

    forbidden_attrs = {"nonzero", "repeat_interleave", "item", "cpu", "tolist"}
    targets = [
        ("static_grid.py", inspect.getsource(sg_mod)),
        ("copy_ring.py", inspect.getsource(cr_mod)),
        ("SGSCFlat.forward", textwrap.dedent(inspect.getsource(SGSCFlat.forward))),
        ("PMDFlat.forward", textwrap.dedent(inspect.getsource(PMDFlat.forward))),
    ]
    for label, src in targets:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                raise AssertionError(
                    f"CRITICAL RED LINE VIOLATION: `.{node.attr}` found in {label} at line {node.lineno}!"
                )
            if isinstance(node, ast.Name) and node.id in forbidden_attrs:
                raise AssertionError(
                    f"CRITICAL RED LINE VIOLATION: `{node.id}` found in {label} at line {node.lineno}!"
                )
