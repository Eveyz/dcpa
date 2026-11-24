"""Full trainer lifecycle, 4-stage freeze zero-corrosion audit, Watchdog self-healing,
preemption resume determinism, and Colab A/B Ping-Pong Drive sync tests.

Covers Deliverable 5 (Tests 1-3) and Global Override Iron Laws #1 & #3.
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from dcpa.config import DCPAConfig
from dcpa.model import DCPAModel
from train.colab_drive_sync import (
    ABDriveSyncManager,
    CheckpointTrigger,
    MAX_MILESTONE_FILES,
    SLOT_FILES,
    assert_not_drive_path,
)
from train.curriculum_state_machine import (
    HIPPOCAMPUS_FAST_BUFFER_NAMES,
    HIPPOCAMPUS_SCALAR_GATE_NAMES,
    expected_stage_param_sets,
    hash_named_parameters,
)
from train.train_dcpa import MeltdownAbortError, run_training
from utils.device import capture_rng_state, get_device, restore_rng_state


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATA_DIR = REPO_ROOT / "artifacts" / "data"
PROD_CONFIG_YAML = REPO_ROOT / "configs" / "v1" / "dcpa_48m_train.yaml"


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# =============================================================================
# Deliverable 5 — Test 1: 80-Step Compressed 4-Stage Lifecycle & Freeze Audit
# =============================================================================


def test_four_stage_lifecycle_freeze_invariance_and_safetensors(tmp_path: Path):
    """Test 1: Compressed 4-stage [0~20, 20~45, 45~60, 60~80] lifecycle verification.

    Verifies:
      1. Exact `requires_grad` parameter sets match curriculum contract at every stage boundary.
      2. Frozen parameters maintain bit-identical `blake2b` raw-byte hashes across frozen intervals
         (proving AdamW decoupled weight_decay causes ZERO weight corrosion!).
      3. Stage 4 (steps 60~80) genuinely updates Hippocampus gating scalars
         (`alpha_1, theta_i, alpha_q, theta_q, alpha_out`), keeps fast buffers `M/u` clean (zeros),
         and never puts fast weights in the optimizer.
      4. Final `artifacts/export/dcpa_final.safetensors` exports atomically and reloads cleanly.
    """
    data_dir = SMOKE_DATA_DIR if (SMOKE_DATA_DIR / "train.bin").exists() else None
    export_dir = REPO_ROOT / "artifacts" / "export"
    ckpt_dir = tmp_path / "ckpts"
    logs_dir = tmp_path / "logs"

    stage_boundaries = (0, 20, 45, 60, 80)
    seen_stages: list[tuple[int, int]] = []
    stage1_frozen_hash_start: dict[str, str] = {}
    stage1_frozen_hash_end: dict[str, str] = {}
    stage4_frozen_hash_start: dict[str, str] = {}
    stage4_frozen_hash_end: dict[str, str] = {}
    stage4_gates_before: dict[str, float] = {}
    stage4_gates_after: dict[str, float] = {}
    stage4_fast_clean_checks: list[bool] = []
    optimizer_fast_overlap: list[int] = []

    def _on_stage_change(completed_step: int, stage_id: int, model: DCPAModel) -> None:
        seen_stages.append((completed_step, stage_id))
        exp_train, exp_frozen = expected_stage_param_sets(model, stage_id)
        act_train = {n for n, p in model.named_parameters() if p.requires_grad}
        act_frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
        assert act_train == exp_train, f"Stage {stage_id} trainable set mismatch!"
        assert act_frozen == exp_frozen, f"Stage {stage_id} frozen set mismatch!"
        # Red Line #2: every frozen parameter must have grad is None
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None, f"Stage {stage_id}: frozen param {n} retained non-None grad!"

        if stage_id == 1:
            stage1_frozen_hash_start.update(hash_named_parameters(model, exp_frozen))
        elif stage_id == 2:
            # Verify Stage 1's frozen parameters did NOT change by a single bit over steps 1..20!
            stage1_frozen_hash_end.update(hash_named_parameters(model, set(stage1_frozen_hash_start.keys())))
        elif stage_id == 4:
            # Verify >= 95% of slow parameters are hard-frozen in Stage 4!
            total_elems = sum(p.numel() for p in model.parameters())
            frozen_elems = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            assert frozen_elems / float(total_elems) >= 0.95, (
                f"Stage 4 must freeze >=95% slow params, got {100.0 * frozen_elems / total_elems:.2f}%"
            )
            stage4_frozen_hash_start.update(hash_named_parameters(model, exp_frozen))
            for gname in HIPPOCAMPUS_SCALAR_GATE_NAMES:
                attr = gname.split(".", 1)[1]
                stage4_gates_before[gname] = float(getattr(model.hippocampus, attr).detach().item())

    def _on_pre_step(completed_step: int, model: DCPAModel, optimizer: torch.optim.Optimizer, _m: dict[str, float]) -> None:
        if completed_step == 61:
            opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
            fast_ids = {
                id(buf)
                for bname, buf in model.named_buffers()
                if bname in HIPPOCAMPUS_FAST_BUFFER_NAMES
                or bname.endswith(("m_c_from_l", "m_l_from_c", "u_lex", "u_sem"))
            }
            optimizer_fast_overlap.append(len(opt_ids.intersection(fast_ids)))

    def _on_step(completed_step: int, model: DCPAModel, _metrics: dict[str, float]) -> None:
        if completed_step >= 61:
            # Red Line #3: fast weight buffers on module must remain strictly clean (all zero) after every step
            clean = (
                float(model.hippocampus.m_c_from_l.abs().max().item()) == 0.0
                and float(model.hippocampus.m_l_from_c.abs().max().item()) == 0.0
                and float(model.hippocampus.u_lex.abs().max().item()) == 0.0
                and float(model.hippocampus.u_sem.abs().max().item()) == 0.0
            )
            stage4_fast_clean_checks.append(clean)
        if completed_step == 80:
            stage4_frozen_hash_end.update(hash_named_parameters(model, set(stage4_frozen_hash_start.keys())))
            for gname in HIPPOCAMPUS_SCALAR_GATE_NAMES:
                attr = gname.split(".", 1)[1]
                stage4_gates_after[gname] = float(getattr(model.hippocampus, attr).detach().item())

    summary = run_training(
        config_path=PROD_CONFIG_YAML,
        preset="micro",
        total_steps=80,
        stage_boundaries=stage_boundaries,
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=512,
        data_dir=data_dir,
        ckpt_dir=ckpt_dir,
        logs_dir=logs_dir,
        export_dir=export_dir,
        local_cache_dir=tmp_path / "nvme_cache",
        save_every_steps=20,
        milestone_every_steps=40,
        probe_every_steps=40,
        console_every_steps=20,
        resume="none",
        device_str="cpu",
        deterministic=True,
        seed=42,
        watchdog_warmup_steps=500,
        export_final=True,
        stage_change_hook=_on_stage_change,
        pre_step_hook=_on_pre_step,
        step_hook=_on_step,
    )

    assert summary.completed_steps == 80
    assert [st for _, st in seen_stages] == [1, 2, 3, 4], f"Unexpected stage transitions: {seen_stages}"
    assert [step for step, _ in seen_stages] == [1, 21, 46, 61], f"Stage boundary steps mismatch: {seen_stages}"

    # Bit-exact blake2b zero-corrosion verification for Stage 1 and Stage 4 frozen parameters
    s1_mismatches = [k for k in stage1_frozen_hash_start if stage1_frozen_hash_start[k] != stage1_frozen_hash_end[k]]
    s4_mismatches = [k for k in stage4_frozen_hash_start if stage4_frozen_hash_start[k] != stage4_frozen_hash_end[k]]
    assert not s1_mismatches, f"Stage 1 frozen weights mutated (weight_decay corrosion!): {s1_mismatches[:5]}"
    assert not s4_mismatches, f"Stage 4 frozen weights mutated (weight_decay corrosion!): {s4_mismatches[:5]}"

    # Stage 4 Hippocampus gating scalar updates & clean fast weights
    required_gates = (
        "hippocampus.alpha_1",
        "hippocampus.theta_i",
        "hippocampus.alpha_q",
        "hippocampus.theta_q",
        "hippocampus.alpha_out",
    )
    gate_deltas = {
        g: abs(stage4_gates_after[g] - stage4_gates_before[g])
        for g in required_gates
    }
    print(f"\n[Test 1] Stage 1 frozen hashes identical: {len(stage1_frozen_hash_start)}/{len(stage1_frozen_hash_start)}")
    print(f"[Test 1] Stage 4 frozen hashes identical: {len(stage4_frozen_hash_start)}/{len(stage4_frozen_hash_start)}")
    print(f"[Test 1] Stage 4 gate updates: {gate_deltas}")
    for g, delta in gate_deltas.items():
        assert delta > 0.0, f"Stage 4 Hippocampus gate {g} received ZERO update (delta={delta})!"

    assert optimizer_fast_overlap == [0], f"Fast buffers leaked into optimizer: {optimizer_fast_overlap}"
    assert len(stage4_fast_clean_checks) == 20 and all(stage4_fast_clean_checks), "Fast buffers were not clean after Stage 4 steps!"

    # Verify milestone checkpoints at stage boundaries
    for b_step in (20, 45, 60, 80):
        assert (ckpt_dir / f"milestone_step_{b_step:06d}.pt").exists(), f"Missing milestone at step {b_step}"

    # Verify artifacts/export/dcpa_final.safetensors reloads cleanly without error
    sf_path = export_dir / "dcpa_final.safetensors"
    cfg_path = export_dir / "config.json"
    assert sf_path.exists() and cfg_path.exists()
    sf_size_mb = sf_path.stat().st_size / (1024.0 * 1024.0)
    loaded_tensors = load_file(str(sf_path))
    assert len(loaded_tensors) > 20
    assert all(not any(b in k for b in ("m_c_from_l", "m_l_from_c", "u_lex", "u_sem")) for k in loaded_tensors)

    # Reconstruct model from config.json + safetensors weights (re-binding tied weights & buffers)
    reloaded_cfg = DCPAConfig.load_json(cfg_path)
    reloaded_model = DCPAModel(reloaded_cfg)
    state_in = dict(loaded_tensors)
    if "sgsc.theta_raw" in state_in:
        state_in["sgsc.theta_raw"] = state_in["sgsc.theta_raw"] / reloaded_cfg.lif_theta_scale
    missing, unexpected = reloaded_model.load_state_dict(state_in, strict=False)
    allowed_missing = {
        "pmd.embed.weight",
        "pmd.w_out.weight",
        "bcg_pc.perm_bank",
        "bcg_pc.pair_i",
        "bcg_pc.pair_j",
        "bcg_pc.pair_u",
        "bcg_pc.pair_v",
        "bcg_pc.pair_u_onehot",
    }
    assert set(missing).issubset(allowed_missing), f"Unexpected missing keys on safetensors reload: {missing}"
    assert not unexpected, f"Unexpected extra keys on safetensors reload: {unexpected}"
    print(f"[Test 1] Exported & reloaded {sf_path} cleanly ({sf_size_mb:.2f} MB, {len(loaded_tensors)} tensors).")


# =============================================================================
# Deliverable 5 — Test 2: Artificial NaN Injection & Watchdog Auto Self-Healing
# =============================================================================


def test_nan_injection_watchdog_rollback_self_healing(tmp_path: Path):
    """Test 2: Inject NaN grad at step 18 -> Watchdog catches Rule A, rolls back to step 10,
    scales LR by 0.7x, skips poisoned cursor, completes to step 30, and logs watchdog_events.jsonl.
    """
    data_dir = SMOKE_DATA_DIR if (SMOKE_DATA_DIR / "train.bin").exists() else None
    ckpt_dir = tmp_path / "ckpts"
    logs_dir = tmp_path / "logs"

    injected = {"done": False}
    lr_before_nan: list[float] = []
    lr_after_rollback: list[float] = []

    def _inject_nan_once(
        completed_step: int,
        model: DCPAModel,
        optimizer: torch.optim.Optimizer,
        _metrics: dict[str, float],
    ) -> None:
        if completed_step == 18 and not injected["done"]:
            injected["done"] = True
            active_lrs = [g["lr"] for g in optimizer.param_groups if g["lr"] > 0.0]
            lr_before_nan.append(max(active_lrs))
            # Corrupt an active parameter's gradient with NaN
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.fill_(float("nan"))
                    break
        elif completed_step == 11 and injected["done"] and not lr_after_rollback:
            active_lrs = [g["lr"] for g in optimizer.param_groups if g["lr"] > 0.0]
            lr_after_rollback.append(max(active_lrs))

    summary = run_training(
        config_path=PROD_CONFIG_YAML,
        preset="micro",
        total_steps=30,
        stage_boundaries=(0, 30, 31, 32, 33),
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=512,
        data_dir=data_dir,
        ckpt_dir=ckpt_dir,
        logs_dir=logs_dir,
        export_dir=tmp_path / "export",
        local_cache_dir=tmp_path / "nvme_cache",
        save_every_steps=10,
        milestone_every_steps=0,
        probe_every_steps=0,
        console_every_steps=10,
        resume="none",
        device_str="cpu",
        deterministic=True,
        seed=42,
        watchdog_warmup_steps=500,
        export_final=False,
        pre_step_hook=_inject_nan_once,
    )

    assert injected["done"] is True
    assert summary.completed_steps == 30
    assert summary.rollbacks_performed == 1
    assert summary.final_lr_multiplier == pytest.approx(0.7, rel=1e-6)
    assert math.isfinite(summary.last_metrics["loss_total"])

    events_file = logs_dir / "watchdog_events.jsonl"
    assert events_file.exists(), "watchdog_events.jsonl was not created!"
    events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1
    ev = events[0]
    assert ev["rule"] == "A"
    assert ev["triggered_step"] == 18
    assert ev["rollback_target_step"] == 10
    assert ev["skip_batches"] == 10
    assert ev["directives_before"]["lr_multiplier"] == pytest.approx(1.0)
    assert ev["directives_after"]["lr_multiplier"] == pytest.approx(0.7)

    print(
        f"\n[Test 2] NaN injected at step=18 -> rolled back to step={ev['rollback_target_step']} | "
        f"LR mult {ev['directives_before']['lr_multiplier']:.2f} -> {ev['directives_after']['lr_multiplier']:.2f} | "
        f"LR before={lr_before_nan[0]:.2e}, after_rb={lr_after_rollback[0]:.2e} | "
        f"final step 30 loss={summary.last_metrics['loss_total']:.4f}"
    )


# =============================================================================
# Deliverable 5 — Test 3: Preemption Checkpoint Resume Determinism (< 1e-5)
# =============================================================================


def test_preemption_checkpoint_resume_determinism(tmp_path: Path):
    """Test 3: Uninterrupted 20-step run vs save at step 10 + full state destruction + resume to 20.

    Asserts Step 11 L_total absolute difference < 1e-5 with aligned optimizer momentum,
    RNG state, DataLoader cursor, and stage state.
    """
    data_dir = SMOKE_DATA_DIR if (SMOKE_DATA_DIR / "train.bin").exists() else None
    bounds = (0, 15, 25, 30, 35)

    # Run A: Uninterrupted 20 steps
    run_a = run_training(
        config_path=PROD_CONFIG_YAML,
        preset="micro",
        total_steps=20,
        stage_boundaries=bounds,
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=512,
        data_dir=data_dir,
        ckpt_dir=tmp_path / "run_a_ckpts",
        logs_dir=tmp_path / "run_a_logs",
        export_dir=tmp_path / "run_a_export",
        local_cache_dir=tmp_path / "run_a_cache",
        save_every_steps=10,
        milestone_every_steps=0,
        probe_every_steps=0,
        console_every_steps=10,
        resume="none",
        device_str="cpu",
        deterministic=True,
        seed=2026,
        export_final=False,
    )

    # Run B Part 1: Run 10 steps and save last_healthy.pt
    ckpt_b_dir = tmp_path / "run_b_ckpts"
    run_b1 = run_training(
        config_path=PROD_CONFIG_YAML,
        preset="micro",
        total_steps=10,
        stage_boundaries=bounds,
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=512,
        data_dir=data_dir,
        ckpt_dir=ckpt_b_dir,
        logs_dir=tmp_path / "run_b_logs",
        export_dir=tmp_path / "run_b_export",
        local_cache_dir=tmp_path / "run_b_cache",
        save_every_steps=10,
        milestone_every_steps=0,
        probe_every_steps=0,
        console_every_steps=10,
        resume="none",
        device_str="cpu",
        deterministic=True,
        seed=2026,
        export_final=False,
    )
    assert run_b1.completed_steps == 10
    step10_ckpt = ckpt_b_dir / "last_healthy.pt"
    assert step10_ckpt.exists()

    # Destroy all in-memory objects & scramble global RNG before resuming
    del run_b1
    gc.collect()
    torch.manual_seed(999999)

    # Run B Part 2: Reconstruct fresh trainer from step-10 checkpoint & finish steps 11..20
    run_b2 = run_training(
        config_path=PROD_CONFIG_YAML,
        preset="micro",
        total_steps=20,
        stage_boundaries=bounds,
        target_tokens_bytes_per_step=512,
        micro_batch_bytes=512,
        data_dir=data_dir,
        ckpt_dir=ckpt_b_dir,
        logs_dir=tmp_path / "run_b_logs",
        export_dir=tmp_path / "run_b_export",
        local_cache_dir=tmp_path / "run_b_cache",
        save_every_steps=10,
        milestone_every_steps=0,
        probe_every_steps=0,
        console_every_steps=10,
        resume=str(step10_ckpt),
        device_str="cpu",
        deterministic=True,
        seed=999,  # intentionally different CLI seed — checkpoint RNG + cursor MUST override!
        export_final=False,
    )

    assert run_b2.resumed_from_step == 10
    assert run_b2.completed_steps == 20

    diff_step11 = abs(run_a.loss_history[11] - run_b2.loss_history[11])
    max_diff_11_20 = max(abs(run_a.loss_history[s] - run_b2.loss_history[s]) for s in range(11, 21))
    print(
        f"\n[Test 3] Step 11 L_total diff = {diff_step11:.3e} (< 1e-5) | "
        f"Max diff over steps 11..20 = {max_diff_11_20:.3e} | "
        f"RunA[11]={run_a.loss_history[11]:.6f}, RunB[11]={run_b2.loss_history[11]:.6f}"
    )
    assert diff_step11 < 1e-5, f"Step 11 resume loss drift {diff_step11:.3e} >= 1e-5!"
    assert max_diff_11_20 < 1e-5, f"Steps 11..20 resume loss drift {max_diff_11_20:.3e} >= 1e-5!"

    # Also verify final step-20 checkpoints have identical cursor, stage, and optimizer state
    ck_a = torch.load(str(tmp_path / "run_a_ckpts" / "last_healthy.pt"), map_location="cpu", weights_only=False)
    ck_b = torch.load(str(ckpt_b_dir / "last_healthy.pt"), map_location="cpu", weights_only=False)
    assert ck_a["dataloader_cursor"] == ck_b["dataloader_cursor"]
    assert ck_a["stage_id"] == ck_b["stage_id"]


# =============================================================================
# Iron Laws #1 & #3 — Colab Anti-Disconnect A/B Ping-Pong & Circuit Breaker Tests
# =============================================================================


def test_mps_fallback_and_drive_isolation_guard():
    dev = get_device("auto", deterministic=False)
    if dev.type == "mps":
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    with pytest.raises(ValueError, match="/content/drive"):
        assert_not_drive_path("/content/drive/MyDrive/DCPA_Project/nvme_cache")

    with pytest.raises(ValueError, match="/content/drive"):
        ABDriveSyncManager(local_cache_dir="/content/drive/MyDrive/illegal_cache")


def test_dual_trigger_step_and_ten_minute_wall_clock():
    clk = _FakeClock(start=0.0)
    trig = CheckpointTrigger(save_every_steps=1000, save_every_minutes=10.0, clock=clk)

    assert trig.should_save(1) == (False, "")
    assert trig.should_save(999) == (False, "")
    assert trig.should_save(1000) == (True, "step_interval")
    trig.mark_saved(1000)

    clk.advance(599.0)
    assert trig.should_save(1005) == (False, "")

    clk.advance(1.0)
    assert trig.should_save(1006) == (True, "wall_clock")
    trig.mark_saved(1006)
    assert trig.should_save(1007) == (False, "")


def test_ab_ping_pong_overwrite_and_self_healing_auto_resume(tmp_path: Path):
    """Colab lifecycle: A->B ping-pong, corrupt active slot, auto-fallback to backup slot."""
    local_cache = tmp_path / "nvme_cache"
    drive_ckpts = tmp_path / "drive_checkpoints"
    drive_logs = tmp_path / "drive_logs"
    drive_export = tmp_path / "drive_export"

    summary1 = run_training(
        preset="micro",
        total_steps=4,
        seq_len=64,
        target_tokens_bytes_per_step=128,
        micro_batch_bytes=128,
        local_cache_dir=local_cache,
        drive_sync_dir=drive_ckpts,
        drive_logs_dir=drive_logs,
        drive_export_dir=drive_export,
        save_every_steps=2,
        milestone_every_steps=0,
        save_every_minutes=10.0,
        probe_every_steps=2,
        heartbeat_seconds=1.0,
        resume="none",
        device_str="cpu",
        seed=42,
        milestone_steps=(2, 4),
        export_final=False,
    )
    assert summary1.completed_steps == 4
    assert (drive_ckpts / SLOT_FILES["A"]).exists()
    assert (drive_ckpts / SLOT_FILES["B"]).exists()
    assert (drive_logs / "latest_probe.md").exists()
    assert (drive_logs / "metrics.jsonl").exists()

    manifest = json.loads((drive_ckpts / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_slot"] == "B"
    assert manifest["backup_slot"] == "A"
    assert manifest["step"] == 4
    assert len(manifest["blake2b"]) == 128

    slot_files = sorted(p.name for p in drive_ckpts.glob("slot_*.pt"))
    assert slot_files == ["slot_A.pt", "slot_B.pt"]

    # Corrupt active slot_B.pt to simulate abrupt mid-sync disconnect
    (drive_ckpts / SLOT_FILES["B"]).write_bytes(b"TRUNCATED_CORRUPT_FUSE_PAYLOAD")

    summary2 = run_training(
        preset="micro",
        total_steps=6,
        seq_len=64,
        target_tokens_bytes_per_step=128,
        micro_batch_bytes=128,
        local_cache_dir=local_cache,
        drive_sync_dir=drive_ckpts,
        drive_logs_dir=drive_logs,
        drive_export_dir=drive_export,
        save_every_steps=2,
        milestone_every_steps=0,
        save_every_minutes=10.0,
        probe_every_steps=2,
        heartbeat_seconds=1.0,
        resume="auto",
        device_str="cpu",
        seed=42,
        milestone_steps=(2, 4, 6),
        export_final=True,
    )
    assert summary2.fallback_resume_used is True
    assert summary2.resumed_from_step == 2
    assert summary2.completed_steps == 6
    assert (drive_export / "dcpa_final.safetensors").exists()
    assert (drive_export / "config.json").exists()
    assert (drive_export / "dcpa_final.safetensors").stat().st_size > 1024

    mgr = ABDriveSyncManager(local_cache_dir=local_cache, drive_sync_dir=drive_ckpts)
    assert len(mgr.list_milestone_files()) <= MAX_MILESTONE_FILES


def test_cross_platform_rng_payload_and_wall_clock_triggered_save(tmp_path: Path):
    """Verify 10-min wall-clock save trigger & cross-platform RNG loading (Mac <-> Colab)."""
    clk = _FakeClock(start=500.0)
    local_cache = tmp_path / "nvme_cache"
    drive_ckpts = tmp_path / "drive_ckpts"

    def _hook(completed_step: int, _model: Any, _metrics: Any) -> None:
        if completed_step == 1:
            clk.advance(605.0)

    summary = run_training(
        preset="micro",
        total_steps=2,
        seq_len=64,
        target_tokens_bytes_per_step=128,
        micro_batch_bytes=128,
        local_cache_dir=local_cache,
        drive_sync_dir=drive_ckpts,
        save_every_steps=1000,
        milestone_every_steps=0,
        save_every_minutes=10.0,
        probe_every_steps=0,
        heartbeat_seconds=60.0,
        resume="none",
        device_str="cpu",
        seed=7,
        milestone_steps=(),
        export_final=False,
        clock=clk,
        step_hook=_hook,
    )
    assert summary.saves_performed >= 2

    rng = capture_rng_state()
    rng["cuda_rng_state_all"] = [torch.zeros(16, dtype=torch.uint8)]
    restore_rng_state(rng)


def test_circuit_breaker_aborts_after_three_consecutive_meltdowns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify 3 consecutive unrecoverable NaN steps trip Watchdog circuit breaker cleanly."""
    orig_forward = DCPAModel.forward

    def _nan_forward(self, *args, **kwargs):
        out = orig_forward(self, *args, **kwargs)
        out.loss_total = out.loss_total * float("nan")
        return out

    monkeypatch.setattr(DCPAModel, "forward", _nan_forward)

    with pytest.raises(MeltdownAbortError, match="3 consecutive"):
        run_training(
            preset="micro",
            total_steps=5,
            seq_len=64,
            target_tokens_bytes_per_step=128,
            micro_batch_bytes=128,
            local_cache_dir=tmp_path / "nvme_cache",
            drive_sync_dir=tmp_path / "drive_ckpts",
            save_every_steps=10,
            milestone_every_steps=0,
            probe_every_steps=0,
            resume="none",
            device_str="cpu",
            max_consecutive_meltdowns=3,
            export_final=False,
        )
