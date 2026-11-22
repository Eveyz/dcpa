"""Trainer lifecycle, A/B Ping-Pong Drive sync, self-healing resume, and circuit-breaker tests.

Covers Iron Laws #1.2, #1.3, #1.4, #3, and Colab Cell 4 cloud verification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from train.colab_drive_sync import (
    ABDriveSyncManager,
    CheckpointTrigger,
    MAX_MILESTONE_FILES,
    SLOT_FILES,
    assert_not_drive_path,
)
from train.train_dcpa import MeltdownAbortError, run_training
from utils.device import capture_rng_state, get_device, restore_rng_state


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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

    # Advance 9m59s -> not due yet
    clk.advance(599.0)
    assert trig.should_save(1005) == (False, "")

    # Advance 1 more second (total 600s = 10 min) -> wall-clock trigger fires!
    clk.advance(1.0)
    assert trig.should_save(1006) == (True, "wall_clock")
    trig.mark_saved(1006)
    assert trig.should_save(1007) == (False, "")


def test_ab_ping_pong_overwrite_and_self_healing_auto_resume(tmp_path: Path):
    """End-to-end trainer lifecycle: A->B ping-pong, corrupt active slot, auto-fallback to backup slot."""
    local_cache = tmp_path / "nvme_cache"
    drive_ckpts = tmp_path / "drive_checkpoints"
    drive_logs = tmp_path / "drive_logs"
    drive_export = tmp_path / "drive_export"

    # Phase 1: Train 4 steps saving every 2 steps (step 2 -> slot_A, step 4 -> slot_B)
    summary1 = run_training(
        preset="micro",
        total_steps=4,
        seq_len=64,
        target_tokens_bytes_per_step=128,
        local_cache_dir=local_cache,
        drive_sync_dir=drive_ckpts,
        drive_logs_dir=drive_logs,
        drive_export_dir=drive_export,
        save_every_steps=2,
        save_every_minutes=10.0,
        probe_every_steps=2,
        heartbeat_seconds=1.0,
        resume="none",
        device_str="auto",
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

    # Verify only slot_A.pt and slot_B.pt exist as rolling files (no Drive trash churn)
    slot_files = sorted(p.name for p in drive_ckpts.glob("slot_*.pt"))
    assert slot_files == ["slot_A.pt", "slot_B.pt"]

    # Phase 2: Simulate abrupt Colab mid-sync disconnect corrupting active slot_B.pt!
    corrupted_active = drive_ckpts / SLOT_FILES["B"]
    corrupted_active.write_bytes(b"TRUNCATED_CORRUPT_FUSE_PAYLOAD")

    # Phase 3: Resume with `--resume auto` up to total_steps=6 -> MUST self-heal to backup slot_A (step 2)!
    summary2 = run_training(
        preset="micro",
        total_steps=6,
        seq_len=64,
        target_tokens_bytes_per_step=128,
        local_cache_dir=local_cache,
        drive_sync_dir=drive_ckpts,
        drive_logs_dir=drive_logs,
        drive_export_dir=drive_export,
        save_every_steps=2,
        save_every_minutes=10.0,
        probe_every_steps=2,
        heartbeat_seconds=1.0,
        resume="auto",
        device_str="auto",
        seed=42,
        milestone_steps=(2, 4, 6),
        export_final=True,
    )
    assert summary2.fallback_resume_used is True, "Expected self-healing fallback to backup slot_A"
    assert summary2.resumed_from_step == 2, f"Expected fallback resume from step 2, got {summary2.resumed_from_step}"
    assert summary2.completed_steps == 6
    assert (drive_export / "dcpa_final.safetensors").exists()
    assert (drive_export / "config.json").exists()
    assert (drive_export / "dcpa_final.safetensors").stat().st_size > 1024

    # Verify milestone file count never exceeds 5 (4 boundaries + best_model.pt)
    mgr = ABDriveSyncManager(local_cache_dir=local_cache, drive_sync_dir=drive_ckpts)
    assert len(mgr.list_milestone_files()) <= MAX_MILESTONE_FILES


def test_cross_platform_rng_payload_and_wall_clock_triggered_save(tmp_path: Path):
    """Verify wall-clock save trigger & cross-platform checkpoint loading (Mac <-> Colab RNG)."""
    clk = _FakeClock(start=500.0)
    local_cache = tmp_path / "nvme_cache"
    drive_ckpts = tmp_path / "drive_ckpts"

    def _hook(completed_step: int, _model, _metrics):
        # Advance virtual clock by 605 seconds on step 1 so wall-clock trigger fires!
        if completed_step == 1:
            clk.advance(605.0)

    summary = run_training(
        preset="micro",
        total_steps=2,
        seq_len=64,
        target_tokens_bytes_per_step=128,
        local_cache_dir=local_cache,
        drive_sync_dir=drive_ckpts,
        save_every_steps=1000,  # step interval will NOT fire; only 10-min wall clock fires!
        save_every_minutes=10.0,
        probe_every_steps=0,
        heartbeat_seconds=60.0,
        resume="none",
        device_str="auto",
        seed=7,
        milestone_steps=(),
        export_final=False,
        clock=clk,
        step_hook=_hook,
    )
    assert summary.saves_performed >= 2  # step 1 (wall_clock) + step 2 (final)

    # Simulate a checkpoint transferred from Colab with CUDA RNG bytes on CPU/MPS (or vice versa)
    rng = capture_rng_state()
    rng["cuda_rng_state_all"] = [torch.zeros(16, dtype=torch.uint8)]
    restore_rng_state(rng)  # Must never raise on non-CUDA Mac!


def test_circuit_breaker_aborts_after_three_consecutive_meltdowns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify 3 consecutive NaN/Inf steps trigger MeltdownAbortError cleanly."""
    from dcpa.model import DCPAModel

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
            local_cache_dir=tmp_path / "nvme_cache",
            drive_sync_dir=tmp_path / "drive_ckpts",
            save_every_steps=10,
            probe_every_steps=0,
            resume="none",
            device_str="cpu",
            max_consecutive_meltdowns=3,
            export_final=False,
        )
