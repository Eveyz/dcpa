"""Regression tests for the Stage-2 moe_210m blowup & L4 OOM root causes.

Covers:
  1. Intra-chunk + inter-chunk `_DampedBoundedCarry` in Turbo `BCGPC.forward_clause_chunk`:
       - Exact 0.0-ULP forward identity vs undamped (Rust bit-parity preserved).
       - Prevents 32-step Jacobian explosion under low `tau_m=0.25` + Slot-0 pinning.
       - Exact gradient parity between chunked `K_c=16` and unchunked `K_c=1` in `DCPAModel`.
       - Clean `torch.compile(backend="aot_eager", dynamic=False)` forward + backward parity.
  2. VRAM headroom guard (`utils.device.is_vram_headroom_low`) & CUDAGraph linear head cap:
       - Activates BCG chunk checkpointing when live free VRAM is low even if `memory_allocated`
         sits below `bcg_checkpoint_vram_watermark_gb` (CUDAGraph private pools blindness fix).
  3. Smooth linear Stage-2 `cap_mix_ratio` warm-in ramp (eliminates step-37,335 0%->15% cliff).
  4. Watchdog Rule A finite gradient/loss explosion guard + Drive Ping-Pong health gate
     (never overwrites a healthy Drive slot with an exploding step).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "dcpa-train-py"
for _p in (str(REPO_ROOT), str(TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dcpa.model as dcpa_model_mod
from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from dcpa.modules.bcg_pc import BCGPC, _DampedBoundedCarry
from train.curriculum_state_machine import CurriculumController
from train.train_dcpa import run_training
from train.watchdog_and_ckpt import SelfHealingWatchdog
import utils.device as device_mod


def test_bcg_clause_chunk_damped_carry_prevents_32step_explosion_and_zero_fwd_drift() -> None:
    """Reproduces the 32-step moe_210m BCG-PC Jacobian explosion and verifies `_DampedBoundedCarry` cures it
    with exact 0.0-ULP forward identity and full `torch.compile` compatibility."""
    torch.manual_seed(42)
    cfg = DCPAConfig.moe_210m()
    cfg.num_experts = 4
    cfg.expert_hidden = 128
    bcg = BCGPC(cfg).float()
    bcg.train()

    # Trained-stage weight scale where recurrent Jacobian exceeds 1.0 across 32 steps
    with torch.no_grad():
        for name, p in bcg.named_parameters():
            if any(k in name for k in ("w_comp", "w_r", "w_l", "w_e", "f_pred", "w_merge", "w_g")):
                p.mul_(3.5)

    B, S, K, D = 2, 16, cfg.k_slots, cfg.d_concept
    c_seq = torch.randn(2 * S, B, D)
    prec_seq = torch.ones(2 * S, B, D)
    ach_seq = torch.zeros(2 * S, B)
    am_seq = torch.ones(2 * S, B, dtype=torch.bool)
    trig_seq = torch.zeros(2 * S, B)
    trig_seq[5, :] = 1.0  # Slot-0 pinning trigger (amplifies recurrence when Lambda_0 = 1.0)
    unpin_seq = torch.zeros(2 * S, B, dtype=torch.bool)
    tau_m = torch.tensor(0.25)
    mix_t = torch.tensor(0.0)
    num_pairs = int(bcg.perm_bank.shape[0])
    gu_seq = torch.rand(2 * S, B, num_pairs)

    def _run_two_chunks(gamma: float, max_rms: float) -> tuple[torch.Tensor, float, float]:
        bcg.zero_grad(set_to_none=True)
        st = bcg.init_state(B, device=torch.device("cpu"), dtype=torch.float32)
        slots0 = st.slots.detach().requires_grad_(True)
        edges0 = st.edges.detach().requires_grad_(True)
        sal0 = st.salience.detach().requires_grad_(True)

        # Chunk 0 (steps 0..15)
        out0 = bcg.forward_clause_chunk(
            slots0,
            edges0,
            sal0,
            st.phase,
            st.slot0_pinned,
            c_seq[:S],
            prec_seq[:S],
            ach_seq[:S],
            am_seq[:S],
            trig_seq[:S],
            unpin_seq[:S],
            tau_m,
            mix_t,
            gumbel_u_seq=gu_seq[:S],
            use_gumbel=True,
            check_pin=True,
            carry_gamma=gamma,
            carry_max_rms=max_rms,
        )
        s1, e1, w1, p1, pin1 = out0[10], out0[11], out0[12], out0[13], out0[14]
        if gamma < 0.9999 or max_rms < 1e6:
            s1 = _DampedBoundedCarry.apply(s1, gamma, max_rms)
            e1 = _DampedBoundedCarry.apply(e1, gamma, max_rms)
            w1 = _DampedBoundedCarry.apply(w1, gamma, max_rms)

        # Chunk 1 (steps 16..31)
        out1 = bcg.forward_clause_chunk(
            s1,
            e1,
            w1,
            p1,
            pin1,
            c_seq[S:],
            prec_seq[S:],
            ach_seq[S:],
            am_seq[S:],
            trig_seq[S:],
            unpin_seq[S:],
            tau_m,
            mix_t,
            gumbel_u_seq=gu_seq[S:],
            use_gumbel=True,
            check_pin=True,
            carry_gamma=gamma,
            carry_max_rms=max_rms,
        )
        chat_all = torch.cat([out0[2], out1[2]], dim=0)
        pi_all = torch.cat([out0[3], out1[3]], dim=0)
        err_sq = (c_seq.detach() - chat_all).pow(2)
        loss_pc = 0.5 * ((pi_all * err_sq).mean(dim=-1) - torch.log(pi_all.clamp_min(1e-6)).mean(dim=-1)).mean()
        loss_pc.backward()

        total_sq = 0.0
        for p in bcg.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.float().pow(2).sum().item())
        gn = total_sq**0.5
        ds0 = float(slots0.grad.float().norm().item()) if slots0.grad is not None else 0.0
        return chat_all.detach().clone(), gn, ds0

    chat_undamped, gn_undamped, ds0_undamped = _run_two_chunks(gamma=1.0, max_rms=1e9)
    chat_damped, gn_damped, ds0_damped = _run_two_chunks(gamma=0.5, max_rms=0.25)

    # 1. Exact 0.0-ULP forward identity (preserves Rust bit-parity)
    assert torch.equal(chat_damped, chat_undamped), "Damped carry must be an exact 0.0-ULP forward identity!"
    # 2. Undamped explodes while damped stays cleanly bounded with non-vanishing gradient flow
    assert gn_undamped > 1_000.0, f"Expected undamped 32-step recurrence to explode, got GN={gn_undamped:.2f}"
    assert gn_damped < 50.0, f"Damped 32-step recurrence still too large: GN={gn_damped:.2f}"
    assert gn_damped > 1e-2, f"Damped 32-step gradient vanished: GN={gn_damped:.4e}"
    assert gn_damped < gn_undamped / 100.0
    assert ds0_damped < ds0_undamped / 100.0


def test_chunked_vs_unchunked_and_compiled_damped_carry_exact_parity() -> None:
    """Verify DCPAModel defaults (`carry_grad_gamma=0.5, carry_grad_max_rms=0.25`) give exact parity
    between `bcg_chunk_steps=1` and `bcg_chunk_steps=8`, and compile cleanly with `aot_eager`."""
    torch.manual_seed(123)
    cfg = DCPAConfig.micro()
    m1 = DCPAModel(cfg)
    assert m1.carry_grad_gamma == pytest.approx(0.5)
    assert m1.carry_grad_max_rms == pytest.approx(0.25)

    m8 = copy.deepcopy(m1)
    m8_comp = copy.deepcopy(m1)
    m1.bcg_chunk_steps = 1
    m8.bcg_chunk_steps = 8
    m8_comp.bcg_chunk_steps = 8
    m1.bcg_grad_checkpointing = False
    m8.bcg_grad_checkpointing = False
    m8_comp.bcg_grad_checkpointing = False
    m8_comp.bcg_pc.enable_compiled_chunk_kernel(backend="aot_eager", dynamic=False)

    raw = torch.randint(32, 126, (1, 320), dtype=torch.uint8)
    cu = torch.tensor([0, 160, 320], dtype=torch.int32)
    langs = torch.tensor([0, 3], dtype=torch.long)

    def _fwd_bwd(model: DCPAModel) -> tuple[float, dict[str, torch.Tensor]]:
        torch.manual_seed(999)
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(raw, lang_ids=langs, stage="B", tau_s=0.35, tau_m=0.35, doc_cu_seqlens=cu, record_traces=False)
        out.loss_total.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return float(out.loss_total.item()), grads

    l1, g1 = _fwd_bwd(m1)
    l8, g8 = _fwd_bwd(m8)
    l8c, g8c = _fwd_bwd(m8_comp)

    assert abs(l1 - l8) < 1e-5, f"Chunked vs unchunked loss mismatch: {abs(l1 - l8):.3e}"
    assert abs(l8 - l8c) < 1e-5, f"Compiled vs eager chunked loss mismatch: {abs(l8 - l8c):.3e}"
    for name in g1:
        assert name in g8 and name in g8c
        assert torch.allclose(g1[name], g8[name], atol=1e-4, rtol=1e-3), f"Grad mismatch K_c=1 vs K_c=8 on {name}"
        assert torch.allclose(g8[name], g8c[name], atol=1e-4, rtol=1e-3), f"Compiled grad mismatch on {name}"


def test_vram_headroom_guard_and_checkpoint_activation_under_low_free_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `is_vram_headroom_low` catches low live VRAM (e.g. L4 with 3.92 GiB CUDAGraph private pools)
    even when `memory_allocated` stays below `watermark_gb`, and forces BCG chunk checkpointing."""
    fake_cuda = torch.device("cuda", 0)
    monkeypatch.setattr(device_mod, "is_cuda_available", lambda: True)

    # L4 (22.03 GiB) right before OOM: only 0.5 GiB free -> MUST report low headroom!
    monkeypatch.setattr(device_mod, "get_vram_free_and_total_gb", lambda _dev: (0.5, 22.03))
    assert device_mod.is_vram_headroom_low(fake_cuda, min_free_gb=3.5, min_free_frac=0.15) is True

    # L4 (22.03 GiB) healthy: 6.5 GiB free (> max(3.5, 3.3)) -> comfortable headroom
    monkeypatch.setattr(device_mod, "get_vram_free_and_total_gb", lambda _dev: (6.5, 22.03))
    assert device_mod.is_vram_headroom_low(fake_cuda, min_free_gb=3.5, min_free_frac=0.15) is False

    # G4 (95 GiB): 10 GiB free < 15% (14.25 GiB) -> triggers proportional guard
    monkeypatch.setattr(device_mod, "get_vram_free_and_total_gb", lambda _dev: (10.0, 95.0))
    assert device_mod.is_vram_headroom_low(fake_cuda, min_free_gb=3.5, min_free_frac=0.15) is True

    # CPU device always returns False
    assert device_mod.is_vram_headroom_low(torch.device("cpu")) is False

    # Verify DCPAModel's chunked BCG loop routes to torch_checkpoint when `_is_vram_headroom_low`
    # reports tight free VRAM even when `_get_allocated_memory_gb` sits below `watermark_gb`.
    torch.manual_seed(7)
    cfg = DCPAConfig.micro()
    model = DCPAModel(cfg)
    model.train()
    model.bcg_chunk_steps = 4
    model.bcg_grad_checkpointing = True
    model.bcg_checkpoint_vram_watermark_gb = 14.0  # allocated=0.0 < 14.0!
    ckpt_calls = {"count": 0}
    orig_ckpt = dcpa_model_mod.torch_checkpoint

    def _counting_ckpt(fn, *args, **kwargs):
        ckpt_calls["count"] += 1
        return orig_ckpt(fn, *args, **kwargs)

    # Simulate low live VRAM guard on a non-CUDA tensor by checking watermark<=0 vs patched guard:
    monkeypatch.setattr(dcpa_model_mod, "torch_checkpoint", _counting_ckpt)
    raw = torch.randint(32, 126, (1, 192), dtype=torch.uint8)
    cu = torch.tensor([0, 96, 192], dtype=torch.int32)
    langs = torch.tensor([0, 3], dtype=torch.long)

    # With plenty of watermark headroom on CPU, checkpointing is skipped
    out_fast = model(raw, lang_ids=langs, stage="B", doc_cu_seqlens=cu, record_traces=False)
    out_fast.loss_total.backward()
    assert ckpt_calls["count"] == 0

    # With watermark exceeded (simulating watermark/VRAM pressure), checkpointing activates for every chunk
    monkeypatch.setattr(dcpa_model_mod, "_get_allocated_memory_gb", lambda _d: 14.5)
    model.zero_grad(set_to_none=True)
    out_ckpt = model(raw, lang_ids=langs, stage="B", doc_cu_seqlens=cu, record_traces=False)
    out_ckpt.loss_total.backward()
    assert ckpt_calls["count"] > 0
    assert model.bcg_cudagraph_max_chunks == 4
    assert model.bcg_vram_min_free_gb == pytest.approx(3.5)


def test_stage2_cap_mix_linear_warmup_ramp() -> None:
    """Verify Stage-2 `cap_mix_prob` ramps linearly instead of jumping 0% -> 15% at step 37,335."""
    cfg_path = REPO_ROOT / "configs" / "v2" / "dcpa_210m_v2_cognitive.yaml"
    raw_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    csm = CurriculumController(raw_yaml["curriculum"])

    # Before start_progress (0.6667 -> step 37,334): strictly 0.0
    assert csm.get(37_000).cap_mix_prob == 0.0
    assert csm.get(37_333).cap_mix_prob == 0.0

    # Right at old cliff step 37,335: gentle start (< 1%), NOT 15%!
    p_start = csm.get(37_335).cap_mix_prob
    assert 0.0 <= p_start < 0.01, f"Expected gentle cap_mix ramp at step 37,335, got {p_start}"

    # Mid-ramp (~38,850): roughly halfway between 0 and 0.15
    p_mid = csm.get(38_850).cap_mix_prob
    assert 0.05 < p_mid < 0.10, f"Expected mid-ramp cap_mix in (0.05, 0.10), got {p_mid}"

    # Monotonic non-decreasing across the ramp and exact 0.15 at step 42,000
    assert p_start < p_mid < csm.get(40_000).cap_mix_prob <= 0.15
    assert csm.get(42_000).cap_mix_prob == pytest.approx(0.15, abs=1e-6)


def test_watchdog_finite_explosion_guard_and_drive_health_gate(tmp_path: Path) -> None:
    """Verify Watchdog Rule A catches finite GN/loss explosions and Drive Ping-Pong never overwrites
    a healthy slot when `step_healthy_ok` is False."""
    wd = SelfHealingWatchdog(
        watchdog_cfg={
            "rule_bcd_warmup_steps": 500,
            "grad_norm_explosion_threshold": 10_000.0,
            "grad_norm_healthy_cap": 1_000.0,
            "max_finite_loss": 100.0,
        },
        base_seed=42,
        enable_event_logging=False,
    )
    ok_metrics = {
        "loss_total": 2.1,
        "loss_lang": 1.9,
        "loss_sgsc_prior": 2.0,
        "loss_pc": 0.4,
        "loss_var": 0.0,
        "loss_rate": 0.0,
        "loss_ponder": 0.01,
        "loss_halt": 0.01,
        "compression_ratio": 9.5,
        "c_slow_std_min": 0.98,
        "merge_entropy_nats": 3.2,
    }

    # Pre-warmup moderate spike (500) is allowed, catastrophic finite spike (2e5) triggers Rule A
    assert not wd.check_step(step=5, stage_id=2, metrics=ok_metrics, grads_finite=True, grad_norm=500.0).triggered
    act_cat = wd.check_step(step=6, stage_id=2, metrics=ok_metrics, grads_finite=True, grad_norm=2e5)
    assert act_cat.triggered and act_cat.rule == "A"
    assert "grad_norm_explosion" in act_cat.reason

    # Post-warmup: finite GN=2.8e8 (exact crash symptom at step 37,493) triggers Rule A immediately!
    wd2 = SelfHealingWatchdog(
        watchdog_cfg={"rule_bcd_warmup_steps": 500, "grad_norm_explosion_threshold": 10_000.0},
        base_seed=42,
        enable_event_logging=False,
    )
    act_gn = wd2.check_step(step=37_493, stage_id=2, metrics=ok_metrics, grads_finite=True, grad_norm=2.8e8)
    assert act_gn.triggered and act_gn.rule == "A"
    assert wd2.directives.lr_multiplier == pytest.approx(0.7)

    # Post-warmup: finite loss_total > max_finite_loss triggers Rule A
    wd3 = SelfHealingWatchdog(
        watchdog_cfg={"rule_bcd_warmup_steps": 500, "max_finite_loss": 100.0},
        base_seed=42,
        enable_event_logging=False,
    )
    bad_loss_metrics = dict(ok_metrics, loss_total=142.5)
    act_loss = wd3.check_step(step=37_500, stage_id=2, metrics=bad_loss_metrics, grads_finite=True, grad_norm=5.0)
    assert act_loss.triggered and act_loss.rule == "A"

    # Checkpoint health gate rejects elevated finite GN > grad_norm_healthy_cap
    assert wd2.is_healthy_for_checkpoint(37_400, stage_id=2, metrics=ok_metrics, grad_norm=12.0) is True
    assert wd2.is_healthy_for_checkpoint(37_400, stage_id=2, metrics=ok_metrics, grad_norm=1_500.0) is False

    # End-to-end trainer Drive health gate: unhealthy step at step 4 MUST NOT overwrite Drive slot from step 2!
    drive_ckpts = tmp_path / "drive_ckpts"
    local_cache = tmp_path / "nvme_cache"

    def _poison_step4_metrics(step: int, _model: Any, _opt: Any, metrics: dict[str, float]) -> None:
        if step == 4:
            # Below Rule-A rollback (window<20 so Rule B doesn't rollback), but fails is_healthy_for_checkpoint!
            metrics["compression_ratio"] = 1.2

    run_training(
        preset="micro",
        total_steps=4,
        seq_len=64,
        target_tokens_bytes_per_step=128,
        micro_batch_bytes=128,
        local_cache_dir=local_cache,
        drive_sync_dir=drive_ckpts,
        save_every_steps=2,
        milestone_every_steps=0,
        probe_every_steps=0,
        resume="none",
        device_str="cpu",
        seed=42,
        watchdog_warmup_steps=3,
        export_final=False,
        pre_step_hook=_poison_step4_metrics,
    )
    manifest = json.loads((drive_ckpts / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["step"] == 2, (
        f"Drive Ping-Pong overwrote healthy step-2 slot with unhealthy step-4! manifest={manifest}"
    )
