"""Verification suite for P0 sync/allocation fixes and GPU-utilization optimizations.

Covers:
  1. Single-pass `parallel_stack_scan(..., reset_to_zero=True)` & custom_op `dcpa::tridiag_stack_scan_fwd/bwd`
     forward + backward parity vs sequential and double-scan references (eager & `torch.compile`).
  2. PMD `_motor_scan_rows_impl` single stack scan parity vs legacy double-scan formula.
  3. Cortical Encoder FlexAttention (`ring=None` and `ring!=None` LSE merge, including non-power-of-2 `d_head=96`).
  4. Stage 3 256-entry LUT `build_byte_loss_weights` bit-exact parity across all 256 bytes x 6 languages.
  5. Zero-sync `seg_doc_start` edge cases and deduplicated Mode III forward (Stages A/B/C + stream carry).
  6. `_AsyncStreamCarryPrefetcher` bit-exact batch & cursor determinism + reset rollback.
  7. Single-`torch.save` deduplication (`atomic_copy_or_link` + `staged_path`) and `AsyncCheckpointWriter`
     snapshot isolation against concurrent in-place `optimizer.step()` mutation.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.stream_carry_packer import StreamCarryPacker
from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from dcpa.modules.cortical import CorticalSandwich, RingMemory
from dcpa.modules.pmd import PMD
from dcpa.ops.scan import parallel_stack_scan, sequential_stack_scan_ref
from dcpa.stream_carry import StreamCarryBank
from train.colab_drive_sync import ABDriveSyncManager
from train.curriculum_state_machine import (
    CODE_DELIMITER_BYTES,
    CurriculumController,
    CurriculumStepState,
    build_byte_loss_weights,
)
from train.train_dcpa import _AsyncStreamCarryPrefetcher, _DynamicRaggedCursorStream
from train.watchdog_and_ckpt import AsyncCheckpointWriter, CheckpointManager


def _make_stack_inputs(B: int, T: int, d_tag: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((B, T, 3), generator=g)
    probs = torch.softmax(logits, dim=-1)
    p_push = probs[..., 0].detach().clone().requires_grad_(True)
    p_pop = probs[..., 1].detach().clone().requires_grad_(True)
    p_hold = probs[..., 2].detach().clone().requires_grad_(True)
    v_tag = torch.tanh(torch.randn((B, T, d_tag), generator=g)).detach().clone().requires_grad_(True)
    xi0 = torch.randn((B, 16, d_tag), generator=g).detach().clone().requires_grad_(True)
    reset_mask = torch.rand((B, T), generator=g) < 0.08
    # Explicitly test both reset_mask[:, 0] == True and False across batch items
    if B >= 2 and T >= 1:
        reset_mask[0, 0] = False
        reset_mask[1, 0] = True
    return p_push, p_pop, p_hold, v_tag, xi0, reset_mask


@pytest.mark.parametrize("T,chunk_size", [(1, 16), (37, 16), (310, 64)])
def test_parallel_stack_scan_reset_to_zero_fwd_and_bwd_parity(T: int, chunk_size: int) -> None:
    B, d_tag = 3, 4
    pp, po, ph, vt, xi0, rm = _make_stack_inputs(B, T, d_tag=d_tag, seed=100 + T)
    grad_w = torch.randn((B, T, 16, d_tag), generator=torch.Generator().manual_seed(777 + T))

    # 1. Exact sequential ground truth
    out_ref = sequential_stack_scan_ref(pp, po, ph, vt, xi_init=xi0, reset_mask=rm, reset_to_zero=True)
    (out_ref * grad_w).sum().backward()
    ref_grads = [t.grad.detach().clone() for t in (pp, po, ph, vt, xi0)]

    # 2. Legacy double-scan linearity formula (forward)
    with torch.no_grad():
        xi_main = parallel_stack_scan(pp, po, ph, vt, xi_init=None, reset_mask=rm, chunk_size=chunk_size)
        xi_hom = parallel_stack_scan(
            pp, po, ph, torch.zeros_like(vt), xi_init=xi0, reset_mask=None, chunk_size=chunk_size
        )
        before_first = torch.cumsum(rm.to(torch.int32), dim=1) == 0
        out_double = xi_main + before_first.float()[..., None, None] * xi_hom
    assert torch.allclose(out_ref, out_double, atol=1e-5, rtol=1e-5)

    # 3. Single-pass parallel_stack_scan(..., reset_to_zero=True)
    for t in (pp, po, ph, vt, xi0):
        t.grad = None
    out_par = parallel_stack_scan(
        pp, po, ph, vt, xi_init=xi0, reset_mask=rm, chunk_size=chunk_size, reset_to_zero=True
    )
    (out_par * grad_w).sum().backward()
    assert torch.allclose(out_par, out_ref, atol=1e-5, rtol=1e-5)
    for g_par, g_ref in zip([t.grad for t in (pp, po, ph, vt, xi0)], ref_grads):
        assert g_par is not None
        assert torch.allclose(g_par, g_ref, atol=1e-5, rtol=1e-5)

    # 4. Custom op `dcpa::tridiag_stack_scan_fwd(..., reset_to_zero=True)` (eager + aot_eager compile)
    for use_compile in (False, True):
        for t in (pp, po, ph, vt, xi0):
            t.grad = None

        def _op_fn(a, b, c, v, m, x0):
            xi_o, _ = torch.ops.dcpa.tridiag_stack_scan_fwd(a, b, c, v, m, x0, chunk_size, True)
            return xi_o

        fn = torch.compile(_op_fn, backend="aot_eager", dynamic=False) if use_compile else _op_fn
        out_op = fn(pp, po, ph, vt, rm, xi0)
        (out_op * grad_w).sum().backward()
        assert torch.allclose(out_op, out_ref, atol=1e-5, rtol=1e-5)
        for g_op, g_ref in zip([t.grad for t in (pp, po, ph, vt, xi0)], ref_grads):
            assert g_op is not None
            assert torch.allclose(g_op, g_ref, atol=1e-5, rtol=1e-5)


def test_pmd_motor_scan_rows_single_stack_scan_exact_parity() -> None:
    torch.manual_seed(42)
    cfg = DCPAConfig.micro()
    pmd = DCPAModel(cfg).pmd
    R, L = 4, 48
    u_prev = torch.randn((R, L, cfg.d_s), requires_grad=True)
    d_drive = torch.randn((R, L, cfg.d_s), requires_grad=True)
    l_reg = torch.randn((R, L, cfg.d_l))
    m_start = torch.randn((R, cfg.d_s), requires_grad=True)
    xi0 = torch.randn((R, cfg.stack_depth, cfg.stack_tag_dim), requires_grad=True)
    srm = torch.zeros((R, L), dtype=torch.bool)
    srm[0, 17] = True
    srm[1, 5] = True
    srm[1, 31] = True
    srm[2, 0] = True  # edge case: reset at row start overrides xi0

    # Single-scan implementation
    logits_new, xi_new, m_new = pmd._motor_scan_rows_impl(u_prev, d_drive, l_reg, m_start, xi0, srm)
    loss_new = logits_new.sum() + 0.5 * xi_new.sum() + 0.25 * m_new.sum()
    loss_new.backward()
    grads_new = (u_prev.grad.clone(), d_drive.grad.clone(), m_start.grad.clone(), xi0.grad.clone())

    # Legacy double-scan formula reference
    for t in (u_prev, d_drive, m_start, xi0):
        t.grad = None
    act_probs = torch.softmax(pmd.w_act(torch.cat([u_prev, l_reg], dim=-1)).float(), dim=-1)
    pp, po, ph = act_probs[..., 0], act_probs[..., 1], act_probs[..., 2]
    vt = torch.tanh(pmd.w_tag(u_prev).float())
    xi_main = parallel_stack_scan(pp, po, ph, vt, xi_init=None, reset_mask=srm)
    xi_hom = parallel_stack_scan(pp, po, ph, torch.zeros_like(vt), xi_init=xi0.float(), reset_mask=None)
    before_first = torch.cumsum(srm.to(torch.int32), dim=1) == 0
    xi_ref = xi_main + before_first.to(xi_hom.dtype)[..., None, None] * xi_hom
    xi_top2 = xi_ref[:, :, :2, :].reshape(R, L, 2 * cfg.stack_tag_dim).to(u_prev.dtype)
    x_in = u_prev + d_drive + pmd.w_st(xi_top2)
    alpha_mot = torch.sigmoid(pmd.w_alpha_mot(x_in))
    b_mot = (1.0 - alpha_mot) * torch.nn.functional.silu(pmd.w_m(x_in))
    from dcpa.ops.scan import parallel_diag_scan

    m_ref = parallel_diag_scan(alpha_mot, b_mot, h0=m_start, reset_mask=None)
    logits_ref = torch.nn.functional.linear(m_ref.float(), pmd.w_out.weight.float())
    (logits_ref.sum() + 0.5 * xi_ref.sum() + 0.25 * m_ref.sum()).backward()
    grads_ref = (u_prev.grad.clone(), d_drive.grad.clone(), m_start.grad.clone(), xi0.grad.clone())

    assert torch.allclose(logits_new, logits_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(xi_new, xi_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(m_new, m_ref, atol=1e-5, rtol=1e-5)
    for gn, gr in zip(grads_new, grads_ref):
        assert torch.allclose(gn, gr, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("d_concept,n_heads,window_size", [(128, 4, 64), (192, 2, 64)])
def test_cortical_encoder_flex_attention_fwd_bwd_parity(
    d_concept: int, n_heads: int, window_size: int
) -> None:
    """Verify FlexAttention encoder parity on both pow2 (`d_head=32`) and non-pow2 (`d_head=96`, moe_210m)."""
    torch.manual_seed(17)
    cfg = DCPAConfig.micro()
    cfg.d_concept = d_concept
    cfg.n_heads = n_heads
    cfg.window_size = window_size
    cfg.n_enc_layers = 2
    cort = CorticalSandwich(cfg)
    cort.eval()

    M = 135
    c_slow = torch.randn((M, d_concept), requires_grad=True)
    f_stem = torch.randn((M, cfg.d_s), requires_grad=True)
    doc_ids = torch.cat([torch.zeros(60, dtype=torch.int64), torch.ones(75, dtype=torch.int64)])
    pos = torch.cat([torch.arange(60, dtype=torch.int64), torch.arange(75, dtype=torch.int64)])
    dep = torch.randint(0, 4, (M,), dtype=torch.int32)

    V, W = 2, window_size
    ring = RingMemory(
        ring_kv=torch.randn(V, W, cfg.d_s),
        ring_pos=torch.arange(-W, 0, dtype=torch.int64).unsqueeze(0).expand(V, W).contiguous(),
        ring_depth=torch.zeros(V, W, dtype=torch.int32),
        ring_valid=torch.ones(V, W, dtype=torch.bool),
        seg_pad_pos=torch.cat([torch.arange(60), 80 + torch.arange(75)]),
        q_pad_len=80,
    )

    for active_ring in (None, ring):
        c_slow.grad = None
        f_stem.grad = None
        cort._force_flex_attention = False
        h_ref, ck_ref, b_ref = cort.encode_parallel(c_slow, f_stem, doc_ids, pos, dep, ring=active_ring)
        (h_ref.sum() + 0.3 * ck_ref.sum()).backward()
        gc_ref, gf_ref = c_slow.grad.clone(), f_stem.grad.clone()

        c_slow.grad = None
        f_stem.grad = None
        cort._force_flex_attention = True
        h_flx, ck_flx, b_flx = cort.encode_parallel(c_slow, f_stem, doc_ids, pos, dep, ring=active_ring)
        (h_flx.sum() + 0.3 * ck_flx.sum()).backward()
        gc_flx, gf_flx = c_slow.grad.clone(), f_stem.grad.clone()

        assert torch.allclose(h_flx, h_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(ck_flx, ck_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(b_flx, b_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(gc_flx, gc_ref, atol=1e-4, rtol=1e-4)
        assert torch.allclose(gf_flx, gf_ref, atol=1e-4, rtol=1e-4)

    # Also verify compiled path with varying doc splits (aot_eager)
    cort.enable_compiled_kernels(backend="aot_eager", dynamic=False)
    cort._force_flex_attention = True
    for cut in (40, 77, 101):
        d_var = torch.cat([torch.zeros(cut, dtype=torch.int64), torch.ones(M - cut, dtype=torch.int64)])
        p_var = torch.cat([torch.arange(cut, dtype=torch.int64), torch.arange(M - cut, dtype=torch.int64)])
        c_t = torch.randn((M, d_concept), requires_grad=True)
        f_t = torch.randn((M, cfg.d_s), requires_grad=True)
        h_c, _, _ = cort.encode_parallel(c_t, f_t, d_var, p_var, dep, ring=None)
        h_c.sum().backward()
        assert torch.isfinite(h_c).all() and c_t.grad is not None and torch.isfinite(c_t.grad).all()


def test_stage3_byte_loss_weights_lut_bit_exact() -> None:
    """Verify the 256-entry GPU LUT matches the reference loop bit-for-bit across all bytes & languages."""
    all_bytes = torch.arange(256, dtype=torch.uint8).repeat(6 * 2)
    lang_ids = torch.arange(6, dtype=torch.int64).repeat_interleave(256).repeat(2)
    loss_mask = torch.cat(
        [torch.ones(256 * 6, dtype=torch.bool), torch.zeros(256 * 6, dtype=torch.bool)],
        dim=0,
    )
    wug_mask = torch.zeros_like(loss_mask)

    base_s3 = CurriculumController(stage_boundaries=(0, 2, 8, 14, 20)).get(8)
    assert base_s3.stage_id == 3
    for focal_w in (1.0, 1.5, 2.5):
        st = dataclasses.replace(base_s3, code_delimiter_focal_weight=focal_w)
        got = build_byte_loss_weights(all_bytes, lang_ids, loss_mask, wug_mask, st)
        assert got is not None

        # Reference loop formula
        b = all_bytes.to(torch.int64)
        m_bool = loss_mask.bool()
        w_ref = m_bool.to(torch.float32)
        if focal_w > 1.0:
            is_code = (lang_ids == 3) | (lang_ids == 4) | (lang_ids == 5)
            is_delim = torch.zeros_like(is_code)
            for cp in CODE_DELIMITER_BYTES:
                is_delim = is_delim | (b == cp)
            w_ref = torch.where(is_code & is_delim & m_bool, torch.full_like(w_ref, focal_w), w_ref)
        w_ref = torch.where((b == 0x04) & m_bool, torch.maximum(w_ref, torch.full_like(w_ref, 1.5)), w_ref)
        w_ref = torch.where(
            ((b == 0x11) | (b == 0x13)) & m_bool, torch.maximum(w_ref, torch.full_like(w_ref, 2.0)), w_ref
        )
        assert torch.equal(got, w_ref.unsqueeze(0))


def test_zero_sync_seg_doc_start_and_mode3_dedup_forward() -> None:
    # 1. Edge cases of zero-sync seg_doc_start
    for counts_list in ([0, 0, 0], [0, 0, 1], [1], [2, 0, 3, 0, 1], [1, 1, 1], [4, 0, 0]):
        seg_counts = torch.tensor(counts_list, dtype=torch.int32)
        V = seg_counts.numel()
        m_total = int(seg_counts.sum().item())
        seg_offsets = torch.zeros_like(seg_counts)
        if V > 1:
            seg_offsets[1:] = torch.cumsum(seg_counts[:-1], dim=0)
        seg_doc_ids = torch.repeat_interleave(
            torch.arange(V, dtype=torch.int64), seg_counts.to(torch.int64), output_size=m_total
        )

        # Reference with valid_docs.any()
        ref = torch.zeros((m_total,), dtype=torch.bool)
        valid_docs = seg_counts > 0
        if valid_docs.any():
            ref[seg_offsets[valid_docs].to(torch.int64)] = True

        # New zero-sync version
        got = torch.zeros((m_total,), dtype=torch.bool)
        if m_total > 0:
            got[0] = True
            if m_total > 1:
                got[1:] = seg_doc_ids[1:] != seg_doc_ids[:-1]
        assert torch.equal(got, ref)

    # 2. Full DCPAModel forward/backward across Stages A, B, C (including parallel CQ & stream carry)
    torch.manual_seed(9)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg)
    raw = torch.randint(32, 126, (1, 192), dtype=torch.uint8)
    doc_cu = torch.tensor([0, 48, 112, 192], dtype=torch.int32)
    langs = torch.tensor([0, 3, 4], dtype=torch.int64)
    for stage, use_cq in (("A", False), ("B", False), ("C", True)):
        model.zero_grad(set_to_none=True)
        out = model(raw, lang_ids=langs, stage=stage, use_broca_cq=use_cq, doc_cu_seqlens=doc_cu)
        assert torch.isfinite(out.loss_total)
        out.loss_total.backward()

    # Stream-carry mode (Stage C with Broca CQ)
    bank = StreamCarryBank.from_config(cfg, n_streams=3, device=torch.device("cpu"))
    doc_carry = bank.gather(torch.tensor([-1, -1, -1]), max_entry_bytes=96)
    model.zero_grad(set_to_none=True)
    out_sc = model(
        raw, lang_ids=langs, stage="C", use_broca_cq=True, doc_cu_seqlens=doc_cu, doc_carry=doc_carry
    )
    assert torch.isfinite(out_sc.loss_total)
    assert out_sc.doc_carry_out is not None
    out_sc.loss_total.backward()


def test_async_stream_carry_prefetcher_exact_parity_and_cursor_determinism() -> None:
    controller = CurriculumController(stage_boundaries=(0, 2, 8, 14, 20))
    dev = torch.device("cpu")
    stream_sync = _DynamicRaggedCursorStream(None, None, seed=4242, device=dev, prefetch_depth=0)
    stream_async = _DynamicRaggedCursorStream(None, None, seed=4242, device=dev, prefetch_depth=0)
    packer_sync = StreamCarryPacker(n_streams=4, chunk_bytes=64, min_piece_bytes=16, max_entries=6)
    packer_async = StreamCarryPacker(n_streams=4, chunk_bytes=64, min_piece_bytes=16, max_entries=6)
    prefetcher = _AsyncStreamCarryPrefetcher(packer_async, stream_async, dev, prefetch_depth=2)

    try:
        grad_accum = 2
        for step_idx in range(2, 8):
            sc = controller.get(step_idx)
            for micro_idx in range(grad_accum):
                b_sync = packer_sync.next_batch(
                    lambda: stream_sync.next_batch(target_batch_bytes=256, state=sc, pin_output=False)
                )
                cur_sync = dict(stream_sync.cursor)

                b_async = prefetcher.next_batch(
                    step_idx=step_idx,
                    micro_idx=micro_idx,
                    target_batch_bytes=256,
                    state=sc,
                    grad_accum_steps=grad_accum,
                    total_steps=20,
                    controller=controller,
                )
                cur_async = prefetcher.committed_cursor

                assert cur_async == cur_sync
                assert set(b_async.keys()) == set(b_sync.keys())
                for k in b_sync:
                    assert torch.equal(b_async[k], b_sync[k]), f"Mismatch on key={k} at step={step_idx}, micro={micro_idx}"

        # Test mid-stage reset rewinds stream_async.cursor back to last committed cursor
        sc_s3 = controller.get(8)
        prefetcher.reset(restore_stream_cursor=True)
        packer_sync.reset()
        b_sync_0 = packer_sync.next_batch(
            lambda: stream_sync.next_batch(target_batch_bytes=256, state=sc_s3, pin_output=False)
        )
        b_async_0 = prefetcher.next_batch(
            step_idx=8,
            micro_idx=0,
            target_batch_bytes=256,
            state=sc_s3,
            grad_accum_steps=grad_accum,
            total_steps=20,
            controller=controller,
        )
        assert torch.equal(b_async_0["x_bytes"], b_sync_0["x_bytes"])
        expected_cur = dict(stream_sync.cursor)
        prefetcher.reset(restore_stream_cursor=True)
        assert dict(stream_async.cursor) == expected_cur
    finally:
        prefetcher.close()
        stream_sync.close()
        stream_async.close()


def test_single_torch_save_and_async_checkpoint_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(7)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Run one step so AdamW state tensors exist
    raw = torch.randint(32, 126, (1, 32), dtype=torch.uint8)
    out = model(raw, lang_ids=torch.tensor([0]), stage="A")
    out.loss_total.backward()
    opt.step()

    ckpt_mgr = CheckpointManager(ckpt_dir=tmp_path / "ckpts")
    sync_mgr = ABDriveSyncManager(
        local_cache_dir=tmp_path / "nvme",
        drive_sync_dir=tmp_path / "drive",
        milestone_steps=(10,),
    )

    save_calls: list[Any] = []
    orig_torch_save = torch.save

    def _counting_save(*args, **kwargs):
        save_calls.append(args)
        return orig_torch_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", _counting_save)

    writer = AsyncCheckpointWriter(torch.device("cpu"), enabled=True)
    try:
        ref_weight_before = model.sgsc.embed.weight.detach().clone()
        payload = CheckpointManager.build_payload(
            step=10,
            stage_id=2,
            model=model,
            optimizer=opt,
            scheduler_state={"last_step": 10},
            dynamic_loss_multipliers={},
            dataloader_cursor={"batch_idx": 10, "seed": 42},
            best_val_metric=1.23,
            config_dict=cfg.to_dict(),
            hardware_tier="cpu",
        )
        post_committed = []
        writer.submit_checkpoint(
            raw_payload=payload,
            step=10,
            model_stage="B",
            ckpt_mgr=ckpt_mgr,
            save_healthy=True,
            save_milestone=True,
            save_best=True,
            sync_mgr=sync_mgr,
            drive_reason="milestone",
            post_commit_fn=lambda: post_committed.append(True),
        )
        # Immediately mutate model weights in-place (simulating concurrent next optimizer.step())
        with torch.no_grad():
            model.sgsc.embed.weight.add_(99.0)

        writer.wait()

        # Verify torch.save executed EXACTLY ONCE across all 4 targets (healthy + milestone + best + Drive)!
        assert len(save_calls) == 1
        assert post_committed == [True]
        assert ckpt_mgr.last_healthy_step == 10
        assert ckpt_mgr.last_healthy_path.exists()
        assert ckpt_mgr.milestone_path(10).exists()
        assert ckpt_mgr.best_model_path.exists()
        assert (sync_mgr.drive_sync_dir / "slot_A.pt").exists()
        assert (sync_mgr.drive_sync_dir / "manifest.json").exists()

        # Verify snapshot isolation: saved weights MUST equal pre-mutation weights, NOT +99.0!
        loaded_healthy = ckpt_mgr.load_checkpoint(ckpt_mgr.last_healthy_path, map_location="cpu")
        assert torch.equal(loaded_healthy["model_state_dict"]["sgsc.embed.weight"], ref_weight_before)

        resumed = sync_mgr.load_latest_auto(map_location="cpu")
        assert resumed is not None
        assert resumed.step == 10
        assert torch.equal(resumed.checkpoint["model_state_dict"]["sgsc.embed.weight"], ref_weight_before)
    finally:
        writer.close()
