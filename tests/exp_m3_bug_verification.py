"""Empirical M3 verification suite for DCPA architectural bugs and design trade-offs.

Runs fast (<10s) on an Apple M3 MacBook Pro (`cpu` or `mps`) and provides:
1. Standalone RED/GREEN diagnostic report:
   `./dcpa-train-py/.venv/bin/python tests/exp_m3_bug_verification.py`
2. Strict assertion mode for post-fix verification:
   `./dcpa-train-py/.venv/bin/python tests/exp_m3_bug_verification.py --assert-green`
3. Pytest integration (`test_exp*`) validating all 6 fixes stay GREEN.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_PKG_ROOT = REPO_ROOT / "dcpa-train-py"
for p in (str(REPO_ROOT), str(PY_PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cli.chat_m3 import DCPAChatSession  # noqa: E402
from dcpa.config import DCPAConfig  # noqa: E402
from dcpa.model import DCPAModel  # noqa: E402
from dcpa.modules.hippocampus import ColumnMajorFastWeights, CompactHouseholderState  # noqa: E402
from dcpa.ops.scan import lif_membrane_scan  # noqa: E402
from train import train_stage5_sft  # noqa: E402
from utils.chat_protocol import detect_lang_id  # noqa: E402


@dataclass
class ExpResult:
    exp_id: str
    name: str
    passed: bool
    metric: str
    detail: str
    is_informational: bool = False


def _pick_device(requested: str = "cpu") -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_sparse_keys(
    batch_size: int,
    d_h: int,
    s_active: int,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_list = []
    for _ in range(batch_size):
        perm = torch.randperm(d_h, generator=gen)[:s_active]
        idx_list.append(perm)
    idx = torch.stack(idx_list, dim=0).to(device=device, dtype=torch.long)
    val = torch.randn((batch_size, s_active), generator=gen).to(device=device, dtype=torch.float32)
    val = val / val.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return idx, val


# ---------------------------------------------------------------------------
# Exp 1: CompactHouseholderState overflow (1A), packed crosstalk (1B), anti-memory (1C)
# ---------------------------------------------------------------------------

def run_exp_1a_overflow(device: torch.device) -> ExpResult:
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(101)
    B, d_h, d_c, d_s, s_act, w_max = 1, 64, 32, 16, 4, 32
    gamma = 0.985
    n_writes = 96  # 3x W_max

    comp = CompactHouseholderState(
        batch_size=B,
        d_h=d_h,
        d_concept=d_c,
        d_s=d_s,
        s_active=s_act,
        w_max=w_max,
        gamma_hip=gamma,
        device=device,
    )
    dense = ColumnMajorFastWeights(
        d_h=d_h,
        d_concept=d_c,
        d_s=d_s,
        gamma_hip=gamma,
        device=device,
    )

    probes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for step in range(n_writes):
        e_idx, e_val = _make_sparse_keys(B, d_h, s_act, gen, device)
        phi_idx, phi_val = _make_sparse_keys(B, d_h, s_act, gen, device)
        s_bar = torch.randn((B, d_c), generator=gen).to(device=device, dtype=torch.float32)
        f_tilde = torch.randn((B, d_s), generator=gen).to(device=device, dtype=torch.float32)
        beta = torch.tensor([0.85], device=device, dtype=torch.float32)
        comp.write_slot(
            beta, e_idx, e_val, s_bar, phi_idx, phi_val, f_tilde, write_threshold=0.05
        )
        dense.write_slot(
            beta, e_idx, e_val, s_bar, phi_idx, phi_val, f_tilde, write_threshold=0.05
        )
        if step in (0, 5, 31, 63, 95):
            probes.append((e_idx, e_val, phi_idx, phi_val))
        comp.advance_step()
        dense.advance_step()

    cos_c_min = 1.0
    max_err_c = 0.0
    cos_l_min = 1.0
    max_err_l = 0.0
    for e_idx, e_val, phi_idx, phi_val in probes:
        c_comp = comp.read_c_from_l(e_idx, e_val)
        c_dens = dense.read_c_from_l(e_idx, e_val)
        l_comp = comp.read_l_from_c(phi_idx, phi_val)
        l_dens = dense.read_l_from_c(phi_idx, phi_val)
        max_err_c = max(max_err_c, float((c_comp - c_dens).abs().max().item()))
        max_err_l = max(max_err_l, float((l_comp - l_dens).abs().max().item()))
        if c_dens.norm().item() > 1e-5:
            if c_comp.norm().item() > 1e-5:
                cos_c_min = min(cos_c_min, float(F.cosine_similarity(c_comp, c_dens, dim=-1).item()))
            else:
                cos_c_min = 0.0
        if l_dens.norm().item() > 1e-5:
            if l_comp.norm().item() > 1e-5:
                cos_l_min = min(cos_l_min, float(F.cosine_similarity(l_comp, l_dens, dim=-1).item()))
            else:
                cos_l_min = 0.0

    passed = (max_err_c < 1e-4) and (max_err_l < 1e-4) and (cos_c_min > 0.9999) and (cos_l_min > 0.9999)
    return ExpResult(
        exp_id="1A",
        name="CompactHouseholder >W_max overflow vs dense",
        passed=passed,
        metric=f"max_err_c={max_err_c:.6f}, max_err_l={max_err_l:.6f}, min_cos_c={cos_c_min:.4f}",
        detail=f"{n_writes} writes with w_max={w_max}; exact folded matrix match requires err<1e-4 & cos>0.9999.",
    )


def run_exp_1b_packed_crosstalk(device: torch.device) -> ExpResult:
    gen = torch.Generator().manual_seed(202)
    B_pack, d_h, d_c, d_s, s_act, w_max = 8, 64, 32, 16, 4, 16
    gamma = 0.99
    thresh = 0.08
    n_steps = 36

    solo = CompactHouseholderState(
        batch_size=1, d_h=d_h, d_concept=d_c, d_s=d_s, s_active=s_act, w_max=w_max, gamma_hip=gamma, device=device
    )
    packed = CompactHouseholderState(
        batch_size=B_pack, d_h=d_h, d_concept=d_c, d_s=d_s, s_active=s_act, w_max=w_max, gamma_hip=gamma, device=device
    )

    first_e_idx, first_e_val = None, None
    for step in range(n_steps):
        e_idx, e_val = _make_sparse_keys(B_pack, d_h, s_act, gen, device)
        phi_idx, phi_val = _make_sparse_keys(B_pack, d_h, s_act, gen, device)
        s_bar = torch.randn((B_pack, d_c), generator=gen).to(device=device, dtype=torch.float32)
        f_tilde = torch.randn((B_pack, d_s), generator=gen).to(device=device, dtype=torch.float32)

        # Doc 0 writes actively in first 12 steps, then goes low-salience (<= 0.08) while Docs 1..7 keep writing.
        beta_pack = torch.full((B_pack,), 0.80, device=device, dtype=torch.float32)
        beta_pack[0] = 0.80 if step < 12 else 0.04

        if step == 0:
            first_e_idx = e_idx[:1].clone()
            first_e_val = e_val[:1].clone()

        solo.write_slot(
            beta_pack[:1],
            e_idx[:1],
            e_val[:1],
            s_bar[:1],
            phi_idx[:1],
            phi_val[:1],
            f_tilde[:1],
            write_threshold=thresh,
        )
        packed.write_slot(
            beta_pack,
            e_idx,
            e_val,
            s_bar,
            phi_idx,
            phi_val,
            f_tilde,
            write_threshold=thresh,
        )
        solo.advance_step()
        packed.advance_step()

    c_solo = solo.read_c_from_l(first_e_idx, first_e_val)
    c_pack_doc0 = packed.read_c_from_l(first_e_idx.expand(B_pack, -1), first_e_val.expand(B_pack, -1))[0:1]
    max_diff = float((c_solo - c_pack_doc0).abs().max().item())
    passed = max_diff < 1e-4
    return ExpResult(
        exp_id="1B",
        name="CompactHouseholder Stage-5 packed-batch crosstalk",
        passed=passed,
        metric=f"max|c_solo - c_packed_doc0|={max_diff:.6f}",
        detail=f"Doc 0 with beta=0.04 <= thresh={thresh} must remain isolated from Docs 1..7.",
    )


def run_exp_1c_ghost_antimemory(device: torch.device) -> ExpResult:
    d_h, d_c, d_s, s_act, w_max = 32, 16, 8, 2, 4
    gamma = 0.99
    comp = CompactHouseholderState(
        batch_size=1, d_h=d_h, d_concept=d_c, d_s=d_s, s_active=s_act, w_max=w_max, gamma_hip=gamma, device=device
    )
    dense = ColumnMajorFastWeights(
        d_h=d_h, d_concept=d_c, d_s=d_s, gamma_hip=gamma, device=device
    )

    k0_idx = torch.tensor([[0, 1]], device=device, dtype=torch.long)
    k0_val = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float32)
    p0_idx = torch.tensor([[0, 1]], device=device, dtype=torch.long)
    p0_val = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float32)

    v_old = torch.zeros((1, d_c), device=device, dtype=torch.float32)
    v_old[0, 0] = 1.0
    v_new = torch.zeros((1, d_c), device=device, dtype=torch.float32)
    v_new[0, 1] = 1.0
    f_dummy = torch.ones((1, d_s), device=device, dtype=torch.float32) * 0.1
    beta = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Step 0: write k0 -> v_old
    comp.write_slot(beta, k0_idx, k0_val, v_old, p0_idx, p0_val, f_dummy, write_threshold=0.0)
    dense.write_slot(beta, k0_idx, k0_val, v_old, p0_idx, p0_val, f_dummy, write_threshold=0.0)
    comp.advance_step()
    dense.advance_step()

    # Step 1: overwrite k0 -> v_new (orthogonal)
    comp.write_slot(beta, k0_idx, k0_val, v_new, p0_idx, p0_val, f_dummy, write_threshold=0.0)
    dense.write_slot(beta, k0_idx, k0_val, v_new, p0_idx, p0_val, f_dummy, write_threshold=0.0)
    comp.advance_step()
    dense.advance_step()

    # Steps 2..4: write 3 orthogonal filler keys so only Step 0 (v_old) is evicted (total 5 > w_max=4)
    gen = torch.Generator().manual_seed(303)
    for i in range(3):
        ki = torch.tensor([[2 + i * 2, 3 + i * 2]], device=device, dtype=torch.long)
        kv = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float32)
        vi = torch.randn((1, d_c), generator=gen).to(device=device, dtype=torch.float32)
        comp.write_slot(beta, ki, kv, vi, ki, kv, f_dummy, write_threshold=0.0)
        dense.write_slot(beta, ki, kv, vi, ki, kv, f_dummy, write_threshold=0.0)
        comp.advance_step()
        dense.advance_step()

    c_comp = comp.read_c_from_l(k0_idx, k0_val)
    c_dens = dense.read_c_from_l(k0_idx, k0_val)
    cos_new_comp = float(F.cosine_similarity(c_comp, v_new, dim=-1).item())
    cos_old_comp = float(F.cosine_similarity(c_comp, v_old, dim=-1).item())
    cos_new_dens = float(F.cosine_similarity(c_dens, v_new, dim=-1).item())
    cos_old_dens = float(F.cosine_similarity(c_dens, v_old, dim=-1).item())
    err = float((c_comp - c_dens).abs().max().item())

    passed = (cos_new_comp > 0.99) and (abs(cos_old_comp) < 0.05) and (err < 1e-4)
    return ExpResult(
        exp_id="1C",
        name="CompactHouseholder overwrite+evict ghost anti-memory",
        passed=passed,
        metric=(
            f"Comp(cos_new={cos_new_comp:+.4f}, cos_old={cos_old_comp:+.4f}) vs "
            f"Dense(cos_new={cos_new_dens:+.4f}, cos_old={cos_old_dens:+.4f}), err={err:.6f}"
        ),
        detail="Overwriting k0 then evicting original delta via pop(0) creates -v_old ghost unless folded.",
    )


# ---------------------------------------------------------------------------
# Exp 2: Multi-turn Slot-0 Thesis Pinning Lockup
# ---------------------------------------------------------------------------

def run_exp_2_multiturn_pinning(device: torch.device) -> ExpResult:
    torch.manual_seed(7)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device).eval()

    turn1 = b"\x1dE\nAlpha evidence: the quantum key is cobalt-42.\n\x04\x1e"
    turn2 = b"\x1dE\nBravo evidence: the updated target is amber-99.\n"
    langs = torch.zeros((1,), device=device, dtype=torch.long)

    with torch.no_grad():
        b1 = torch.tensor(list(turn1), device=device, dtype=torch.uint8).unsqueeze(0)
        b2 = torch.tensor(list(turn2), device=device, dtype=torch.uint8).unsqueeze(0)

        out1, carry1 = model.forward_streaming_chunk(b1, langs, stage="C")
        slot0_t1 = carry1.graph_state.slots[:, 0, :].clone()
        pinned_after_t1 = float(carry1.graph_state.slot0_pinned.item())

        out2, carry2 = model.forward_streaming_chunk(b2, langs, carry_state=carry1, stage="C")
        slot0_t2 = carry2.graph_state.slots[:, 0, :].clone()
        stream_diff = float((slot0_t2 - slot0_t1).abs().max().item())

        # Packed multi-turn forward path check: compare Turn 1+EOT/RS+Turn 2 vs same bytes without EOT/RS
        b_packed = torch.cat([b1, b2], dim=1)
        out_packed_both = model(b_packed, langs, stage="C")
        t1_len = b1.shape[1]
        logits_t2_packed = out_packed_both.pmd.byte_logits[:, t1_len:, :]

        turn1_no_eot = b"\x1dE\nAlpha evidence: the quantum key is cobalt-42.\n  "
        b_packed_no_eot = torch.cat(
            [
                torch.tensor(list(turn1_no_eot), device=device, dtype=torch.uint8).unsqueeze(0),
                b2,
            ],
            dim=1,
        )
        out_no_eot = model(b_packed_no_eot, langs, stage="C")
        logits_t2_no_eot = out_no_eot.pmd.byte_logits[:, t1_len:, :]
        packed_unpin_delta = float((logits_t2_packed - logits_t2_no_eot).abs().max().item())

        # Also check Slot-0 trace in packed forward changed from Turn-1 Slot-0 vs Turn-2 Slot-0
        pinned_ratio_t2 = float(out2.slot0_pinned_trace[-1].item()) if out2.slot0_pinned_trace.numel() > 0 else 0.0

    passed = (stream_diff > 1e-3) and (packed_unpin_delta > 1e-4)
    return ExpResult(
        exp_id="2",
        name="Multi-turn Slot-0 turn-boundary unpinning",
        passed=passed,
        metric=(
            f"stream max|Slot0_t2 - Slot0_t1|={stream_diff:.6f} "
            f"(pinned_after_t1={pinned_after_t1:.1f}, t2_end_pin={pinned_ratio_t2:.1f}), "
            f"packed_unpin_delta={packed_unpin_delta:.6f}"
        ),
        detail="Turn boundary 0x04/0x1E must clear stale slot0_pinned so Turn 2 \\x1dE\\n can re-pin.",
    )


# ---------------------------------------------------------------------------
# Exp 3: Stage-5 SFT W_gen anchor (3A) & YAML config wiring (3B)
# ---------------------------------------------------------------------------

def run_exp_3a_wgen_anchor(device: torch.device) -> ExpResult:
    torch.manual_seed(123)
    cfg = DCPAConfig.micro()
    cfg.tau_skip = 0.32
    model = DCPAModel(cfg).to(device)
    model.train()

    train_stage5_sft.apply_stage5_surgical_freeze(model)
    bcg_mod = getattr(model.bcg_pc, "_orig_mod", model.bcg_pc)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.05)

    B, T = 2, 48
    langs = torch.zeros((B,), device=device, dtype=torch.long)
    probe_bytes = torch.randint(32, 122, (B, T), device=device, dtype=torch.uint8)
    probe_bytes[:, 0] = 0x1D
    probe_bytes[:, 1] = ord("H")
    probe_bytes[:, 2] = ord("\n")
    probe_bytes[:, 16] = 0x1D
    probe_bytes[:, 17] = ord("E")
    probe_bytes[:, 18] = ord("\n")
    probe_bytes[:, 32] = 0x1F
    probe_bytes[:, 33] = ord("A")
    probe_bytes[:, 34] = ord("\n")

    def _measure_wgen_mse() -> tuple[float, float]:
        was_train = model.training
        model.eval()
        with torch.no_grad():
            # Stage C runs exact z_star without mutating fast memory, returning MSE(c_slow, W_gen(z_star)) * 0.05
            out_c = model(probe_bytes, langs, stage="C")
            mse_c = float(out_c.metrics["loss_gen_align"]) / 0.05
            w_norm = float(bcg_mod.w_gen.weight.norm().item())
        if was_train:
            model.train()
        return mse_c, w_norm

    mse0, wnorm0 = _measure_wgen_mse()

    loss_gen_seen = 0.0
    for step in range(35):
        batch = probe_bytes.clone()
        payload_noise = torch.randint(97, 122, (B, T), device=device, dtype=torch.uint8)
        mask_data = (batch >= 32) & (batch != ord("\n"))
        batch = torch.where(mask_data, payload_noise, batch)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            batch,
            langs,
            stage="D",
            write_threshold_override=0.08,
        )
        loss = out.loss_total
        loss_gen_seen = float(out.metrics.get("loss_gen_align", 0.0))
        loss.backward()
        optimizer.step()

    mse1, wnorm1 = _measure_wgen_mse()

    passed = (loss_gen_seen > 1e-6) and (mse1 < mse0 * 1.10)
    return ExpResult(
        exp_id="3A",
        name="Stage-5 SFT W_gen alignment anchor",
        passed=passed,
        metric=(
            f"loss_gen_align={loss_gen_seen:.6f}, "
            f"MSE(c,W_gen(z*)) {mse0:.4f}->{mse1:.4f}, "
            f"||W_gen|| {wnorm0:.4f}->{wnorm1:.4f}"
        ),
        detail="Stage-5 training (stage='D', wd=0.05) must keep loss_gen_align active so W_gen does not drift.",
    )


def run_exp_3b_yaml_wiring() -> ExpResult:
    yaml_path = REPO_ROOT / "configs" / "v2" / "dcpa_210m_v2_cognitive.yaml"
    train_sh_text = (REPO_ROOT / "scripts" / "train.sh").read_text(encoding="utf-8")

    has_train_sh_flag = "--train-config" in train_sh_text
    has_resolver = hasattr(train_stage5_sft, "resolve_stage5_yaml_defaults")
    sig = inspect.signature(train_stage5_sft.run_stage5_sft)
    has_arg = "train_config_path" in sig.parameters or "write_threshold_override" in sig.parameters

    resolved_thresh = None
    if has_resolver:
        cfg_dict = train_stage5_sft.resolve_stage5_yaml_defaults(yaml_path)
        resolved_thresh = cfg_dict.get("write_threshold_override")

    passed = (
        has_train_sh_flag
        and has_resolver
        and has_arg
        and (resolved_thresh is not None)
        and abs(float(resolved_thresh) - 0.08) < 1e-6
    )
    return ExpResult(
        exp_id="3B",
        name="Stage-5 SFT YAML config & train.sh wiring",
        passed=passed,
        metric=(
            f"train_sh_flag={has_train_sh_flag}, resolver={has_resolver}, "
            f"run_sig_arg={has_arg}, write_thresh={resolved_thresh}"
        ),
        detail=f"scripts/train.sh & train_stage5_sft.py must wire stage5_sft block from {yaml_path.name}.",
    )


# ---------------------------------------------------------------------------
# Exp 4: Stage C/D Broca CQ Causality Leak + Gradient Check
# ---------------------------------------------------------------------------

def run_exp_4_broca_causality(device: torch.device) -> ExpResult:
    torch.manual_seed(99)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device)

    B = 1
    langs = torch.zeros((B,), device=device, dtype=torch.long)
    prefix = torch.randint(48, 90, (B, 64), device=device, dtype=torch.uint8)
    prefix[:, 0] = 0x1D
    prefix[:, 1] = ord("H")
    prefix[:, 2] = ord("\n")
    prefix[:, 15] = ord(" ")
    prefix[:, 31] = ord(".")
    prefix[:, 47] = ord(" ")

    # Use a 3-byte suffix (< k_slots=8 segments even at high initial spike rate) so tail CQ overlaps prefix[:64]
    suffix_a = torch.tensor(list(b"aaa"), device=device, dtype=torch.uint8).unsqueeze(0)
    suffix_b = torch.tensor(list(b"zzz"), device=device, dtype=torch.uint8).unsqueeze(0)

    x_a = torch.cat([prefix, suffix_a], dim=1)
    x_b = torch.cat([prefix, suffix_b], dim=1)

    model.eval()
    with torch.no_grad():
        out_a = model(x_a, langs, stage="C", use_broca_cq=True)
        out_b = model(x_b, langs, stage="C", use_broca_cq=True)
        leak = float((out_a.pmd.byte_logits[:, :64, :] - out_b.pmd.byte_logits[:, :64, :]).abs().max().item())

    model.train()
    model.zero_grad(set_to_none=True)
    out_grad = model(x_a, langs, stage="C", use_broca_cq=True)
    out_grad.loss_total.backward()
    w_ord_grad = float(model.pmd.w_ord.weight.grad.abs().max().item()) if model.pmd.w_ord.weight.grad is not None else 0.0
    w_coh_grad = float(model.pmd.w_coh.weight.grad.abs().max().item()) if model.pmd.w_coh.weight.grad is not None else 0.0

    passed = (leak < 1e-5) and (w_ord_grad > 1e-7) and (w_coh_grad > 1e-7)
    return ExpResult(
        exp_id="4",
        name="Stage-C Broca CQ strict causality & gradients",
        passed=passed,
        metric=f"future_leak={leak:.6f}, w_ord_grad={w_ord_grad:.2e}, w_coh_grad={w_coh_grad:.2e}",
        detail="Rolling causal Broca CQ must leak 0 future information onto bytes [:64] while keeping grads.",
    )


# ---------------------------------------------------------------------------
# Exp 5: LIF Membrane Scan Multi-Byte UTF-8 Span Truncation
# ---------------------------------------------------------------------------

def run_exp_5_lif_utf8_span(device: torch.device) -> ExpResult:
    # Build T=4120 bytes with a 19B segment caused by 4B UTF-8 emoji continuation bytes past max_seg_len=16
    chunk = b"abcdefghijklmnop"  # 16 ASCII bytes -> max_seg_len=16
    emoji_chunk = b"abcdefg" + ("🦀" * 3).encode("utf-8")  # 7 + 12 = 19 bytes; last continuation ends at 18
    # Wait: make byte 15 the start of a 4-byte emoji so cont mask spans 16, 17, 18!
    emoji_chunk = b"a" * 15 + "🦀".encode("utf-8")  # bytes 0..14 ASCII, 15 = 0xF0, 16..18 = 0x80..0xBF conts!
    unit = emoji_chunk + b"\n" + (chunk + b"\n") * 8
    raw = (unit * 30)[:4120]
    raw_bytes = torch.tensor(list(raw), device=device, dtype=torch.uint8).unsqueeze(0)

    T = raw_bytes.shape[1]
    info = torch.full((1, T), 0.15, device=device, dtype=torch.float32)
    delta_info = torch.full((1, T), 0.05, device=device, dtype=torch.float32)
    theta = torch.tensor(1.0, device=device, dtype=torch.float32)

    v_par, sp_par = lif_membrane_scan(
        info=info,
        delta_info=delta_info,
        raw_bytes=raw_bytes,
        lif_lambda=0.90,
        beta1=0.25,
        beta2=0.75,
        theta=theta,
        max_seg_len=16,
    )
    # Exact sequential scalar recurrence reference over t = 0..24
    drive_0 = 0.25 * info[0, :25] + 0.75 * delta_info[0, :25]
    exact_vals: list[float] = []
    v_acc = 0.0
    for t in range(25):
        v_acc = (0.0 if (t == 0 or bool(sp_par[0, t - 1].item())) else 0.90 * v_acc) + float(drive_0[t].item())
        exact_vals.append(v_acc)
    v_exact_25 = torch.tensor(exact_vals, device=device, dtype=torch.float32)

    max_err = float((v_par[0, :25] - v_exact_25).abs().max().item())
    first_spikes = torch.nonzero(sp_par[0, :25] > 0.5, as_tuple=False).squeeze(-1).tolist()
    passed = max_err < 1e-5
    return ExpResult(
        exp_id="5",
        name="lif_membrane_scan UTF-8 19B span parity (T=4120)",
        passed=passed,
        metric=f"max|v_tilde_par[:25] - v_exact[:25]|={max_err:.6f}, first_spikes={first_spikes}",
        detail="Parallel LIF scan max_span must cover 16+3=19B UTF-8 continuation extensions.",
    )


# ---------------------------------------------------------------------------
# Exp 6: detect_lang_id SQL/Formal (6A) & SWR Sleep Consolidation (6B)
# ---------------------------------------------------------------------------

def run_exp_6a_lang_detect() -> ExpResult:
    cases = [
        ("EN", "The quick brown fox jumps over the lazy dog in cognitive architecture.", 0),
        ("ZH", "今天天气真好，我们去公园散步吧。", 1),
        ("JA", "おはようございます。今日の会議について確認します。", 2),
        ("FOL", "forall x in S, exists y: P(x, y) -> Q(x) |- R(y)", 2),
        ("PY", "def solve(items):\n    return [x * 2 for x in items if x > 0]\n", 3),
        ("RS", "pub fn compute(mut val: usize) -> usize { let mut acc = 0; acc + val }", 4),
        ("SQL", "SELECT u.id, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id;", 4),
        ("JS", "const handler = async (req) => { console.log(req.url); return true; };", 5),
    ]
    results = [(tag, detect_lang_id(sample), expected) for tag, sample, expected in cases]
    failures = [f"{tag}:{got}!=exp({exp})" for tag, got, exp in results if got != exp]
    passed = len(failures) == 0
    return ExpResult(
        exp_id="6A",
        name="detect_lang_id v2 SQL(4) & Formal(2) + legacy parity",
        passed=passed,
        metric="PASS all 8 domains" if passed else f"FAILED: {', '.join(failures)}",
        detail="v2 6x5 curriculum expects SQL->4 and Formal Logic->2 alongside EN/ZH/JA/PY/RS/JS.",
    )


def run_exp_6b_swr_consolidation(device: torch.device) -> ExpResult:
    torch.manual_seed(55)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device).eval()
    session = DCPAChatSession(model, device, preload_builtin_tools=False)

    prompt_bytes = b"\x1dE\nNovel protocol zyxwvu means obsidian_vault.\n"
    session.ingest_bytes(prompt_bytes, lang_id=0)

    ring = list(session._swr_ring)
    assert len(ring) > 0, "Expected non-empty _swr_ring after ingest_bytes"
    # Check ring captures per-slot h_end (4-tuple) AND after SWR consolidation the true SGSC forward path
    # norm_c(w_res(h_end) + w_sem(f_stem)) aligns closely to all target s_bar entries in the ring.
    last_entry = ring[-1]
    has_h_end = len(last_entry) >= 4
    f_all = torch.stack([e[1].to(device=device, dtype=torch.float32) for e in ring], dim=0)
    s_bar_all = torch.stack([e[2].to(device=device, dtype=torch.float32) for e in ring], dim=0)

    # Also capture true SGSC h_end for the last segment independently if len < 4 (RED path fallback)
    raw_t = torch.tensor(list(prompt_bytes), device=device, dtype=torch.uint8).unsqueeze(0)
    with torch.no_grad():
        sg_ref = model.sgsc(raw_t)
        end_pos = int(sg_ref.cu_seqlens[-1].item()) - 1
        true_h_end = sg_ref.h_seq[:, end_pos, :].float()
    h_all = torch.stack(
        [
            e[3].to(device=device, dtype=torch.float32) if len(e) >= 4 else true_h_end.squeeze(0)
            for e in ring
        ],
        dim=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "consolidated.safetensors"
        report = session.run_sleep_consolidation(
            steps=80,
            lr=1e-2,
            out_path=out_path,
            familiarity_threshold=0.0,
        )

    with torch.no_grad():
        c_true_after = model.sgsc.norm_c(
            model.sgsc.w_res(h_all) + model.sgsc.w_sem(f_all)
        ).float()
        c_wsem_only = model.sgsc.w_sem(f_all).float()
        cos_true = float(F.cosine_similarity(c_true_after, s_bar_all, dim=-1).mean().item())
        cos_wsem = float(F.cosine_similarity(c_wsem_only, s_bar_all, dim=-1).mean().item())

    passed = has_h_end and (cos_true >= 0.98)
    return ExpResult(
        exp_id="6B",
        name="chat_m3 SWR true norm_c(w_res + w_sem(f)) target",
        passed=passed,
        metric=(
            f"has_h_end={has_h_end} (tuple_len={len(last_entry)}), "
            f"cos(true_c_slow, s_bar)={cos_true:.4f}, cos(raw_wsem, s_bar)={cos_wsem:.4f}, "
            f"loss {report.get('loss_before', 0.0):.3f}->{report.get('loss_after', 0.0):.3f}"
        ),
        detail="SWR consolidation must retain h_end and train norm_c(w_res(h_end) + w_sem(f)) -> s_bar.",
    )


# ---------------------------------------------------------------------------
# Informational Trade-off Observations (Roadmap v3, non-gating)
# ---------------------------------------------------------------------------

def run_obs_a_carry_damping(device: torch.device) -> ExpResult:
    torch.manual_seed(11)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device).train()

    text = ("ab " * 16).encode("utf-8")
    raw = torch.tensor(list(text), device=device, dtype=torch.uint8).unsqueeze(0)
    langs = torch.zeros((1,), device=device, dtype=torch.long)
    out = model(raw, langs, stage="C", use_broca_cq=False)
    out.loss_total.backward()
    m_segs = int(out.s_trace.shape[0]) if out.s_trace is not None else int(out.sgsc.seg_counts.sum().item())
    u_src_grad = (
        float(model.bcg_pc.u_src.weight.grad.abs().mean().item())
        if model.bcg_pc.u_src.weight.grad is not None
        else 0.0
    )
    return ExpResult(
        exp_id="OBS-A",
        name="[Trade-off v1/v2] 0.5x damped carry & s_tilde.detach()",
        passed=True,
        metric=f"m_segs={m_segs}, u_src_grad={u_src_grad:.3e}, 0.5^m_segs={0.5**m_segs:.2e}",
        detail="Intentional v1/v2 stability trade-off; log-space diagonal SSM deferred to v3.",
        is_informational=True,
    )


def run_obs_b_pmd_stack(device: torch.device) -> ExpResult:
    torch.manual_seed(12)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg).to(device).eval()

    langs = torch.zeros((1,), device=device, dtype=torch.long)
    s_lit = torch.tensor(list(b's = "(" '), device=device, dtype=torch.uint8).unsqueeze(0)
    s_call = torch.tensor(list(b"s = f() "), device=device, dtype=torch.uint8).unsqueeze(0)
    with torch.no_grad():
        out_lit = model(s_lit, langs, stage="A")
        out_call = model(s_call, langs, stage="A")
        p_close_lit = float(F.softmax(out_lit.pmd.byte_logits[0, 5].float(), dim=-1)[ord(")")].item())
        p_close_call = float(F.softmax(out_call.pmd.byte_logits[0, 5].float(), dim=-1)[ord(")")].item())

    return ExpResult(
        exp_id="OBS-B",
        name="[Trade-off v1/v2] PMD single-byte stack lexical sensitivity",
        passed=True,
        metric=f"P(')' | '\"(')={p_close_lit:.4f}, P(')' | 'f(')={p_close_call:.4f}",
        detail="Intentional v1/v2 1-byte gate trade-off; Conv1d local context gate deferred to v3.",
        is_informational=True,
    )


# ---------------------------------------------------------------------------
# Suite runner & Pytest entrypoints
# ---------------------------------------------------------------------------

def run_all_experiments(device: torch.device) -> list[ExpResult]:
    return [
        run_exp_1a_overflow(device),
        run_exp_1b_packed_crosstalk(device),
        run_exp_1c_ghost_antimemory(device),
        run_exp_2_multiturn_pinning(device),
        run_exp_3a_wgen_anchor(device),
        run_exp_3b_yaml_wiring(),
        run_exp_4_broca_causality(device),
        run_exp_5_lif_utf8_span(device),
        run_exp_6a_lang_detect(),
        run_exp_6b_swr_consolidation(device),
        run_obs_a_carry_damping(device),
        run_obs_b_pmd_stack(device),
    ]


def test_exp_1_compact_householder_fixes() -> None:
    dev = _pick_device("cpu")
    for fn in (run_exp_1a_overflow, run_exp_1b_packed_crosstalk, run_exp_1c_ghost_antimemory):
        res = fn(dev)
        assert res.passed, f"{res.exp_id} FAILED: {res.metric}"


def test_exp_2_multiturn_slot0_unpinning() -> None:
    res = run_exp_2_multiturn_pinning(_pick_device("cpu"))
    assert res.passed, f"{res.exp_id} FAILED: {res.metric}"


def test_exp_3_stage5_wgen_and_yaml_wiring() -> None:
    res_a = run_exp_3a_wgen_anchor(_pick_device("cpu"))
    res_b = run_exp_3b_yaml_wiring()
    assert res_a.passed, f"{res_a.exp_id} FAILED: {res_a.metric}"
    assert res_b.passed, f"{res_b.exp_id} FAILED: {res_b.metric}"


def test_exp_4_broca_cq_causality() -> None:
    res = run_exp_4_broca_causality(_pick_device("cpu"))
    assert res.passed, f"{res.exp_id} FAILED: {res.metric}"


def test_exp_5_lif_utf8_span_parity() -> None:
    res = run_exp_5_lif_utf8_span(_pick_device("cpu"))
    assert res.passed, f"{res.exp_id} FAILED: {res.metric}"


def test_exp_6_lang_detect_and_swr_alignment() -> None:
    res_a = run_exp_6a_lang_detect()
    res_b = run_exp_6b_swr_consolidation(_pick_device("cpu"))
    assert res_a.passed, f"{res_a.exp_id} FAILED: {res_a.metric}"
    assert res_b.passed, f"{res_b.exp_id} FAILED: {res_b.metric}"


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 bug verification suite for DCPA.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="Compute device.")
    parser.add_argument("--assert-green", action="store_true", help="Exit non-zero if any bug exp is RED.")
    args = parser.parse_args()

    dev = _pick_device(args.device)
    t0 = time.perf_counter()
    results = run_all_experiments(dev)
    elapsed = time.perf_counter() - t0

    print(f"=== DCPA M3 Verification Suite (device={dev}, elapsed={elapsed:.2f}s) ===")
    gating_failures = 0
    for r in results:
        status = "INFO" if r.is_informational else ("GREEN" if r.passed else "RED")
        if not r.is_informational and not r.passed:
            gating_failures += 1
        print(f"[{status:5s}] Exp {r.exp_id:5s} | {r.name}")
        print(f"        metric: {r.metric}")
        print(f"        note  : {r.detail}")

    total_gating = len([r for r in results if not r.is_informational])
    print(f"Summary: {total_gating - gating_failures}/{total_gating} bug checks GREEN ({gating_failures} RED)")

    if args.assert_green and gating_failures > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
