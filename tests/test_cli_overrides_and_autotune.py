"""Unit & smoke tests for CLI `--set` YAML overrides, perf flags, and `scripts/autotune_gpu.py`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "dcpa-train-py"
for _p in (str(REPO_ROOT), str(TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.autotune_gpu import build_candidate_grid, parse_trial_log, stage_boundaries_for_probe
from train.train_dcpa import apply_config_overrides, load_yaml_config, run_training


def test_apply_config_overrides_types_strictness_and_plus_create() -> None:
    base = load_yaml_config(REPO_ROOT / "configs" / "v2" / "dcpa_210m_v2_cognitive.yaml")
    orig_chunk = base["hardware"]["bcg_chunk_steps"]

    updated, applied = apply_config_overrides(
        base,
        [
            "hardware.bcg_chunk_steps=8",
            "hardware.bcg_cudagraphs=false",
            "optimizer.groups.group_matrix_decay.lr=3e-4",
            "curriculum.stage_boundaries=[0, 1000, 2000, 3000, 4000]",
            "+hardware.custom_probe_flag=true",
        ],
    )
    # Original dict must remain untouched
    assert base["hardware"]["bcg_chunk_steps"] == orig_chunk
    assert "custom_probe_flag" not in base["hardware"]

    assert updated["hardware"]["bcg_chunk_steps"] == 8
    assert updated["hardware"]["bcg_cudagraphs"] is False
    assert isinstance(updated["optimizer"]["groups"]["group_matrix_decay"]["lr"], float)
    assert abs(updated["optimizer"]["groups"]["group_matrix_decay"]["lr"] - 3.0e-4) < 1e-12
    assert updated["curriculum"]["stage_boundaries"] == [0, 1000, 2000, 3000, 4000]
    assert updated["hardware"]["custom_probe_flag"] is True
    assert len(applied) == 5

    # Unknown key without '+' must fail fast to catch typos
    with pytest.raises(KeyError, match="Unknown config key"):
        apply_config_overrides(base, ["hardware.bcg_chonk_steps=8"])

    with pytest.raises(ValueError, match="expected 'key.path=value'"):
        apply_config_overrides(base, ["hardware.bcg_chunk_steps"])


def test_run_training_perf_flags_and_set_overrides_cpu(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def _hook(step: int, model, optimizer, metrics) -> None:
        if step == 1:
            seen["bcg_chunk_steps"] = model.bcg_chunk_steps
            seen["bcg_tbptt_window"] = model.bcg_tbptt_window
            seen["base_lrs"] = {g["name"]: float(g["base_lr"]) for g in optimizer.param_groups}

    summary = run_training(
        config_path=REPO_ROOT / "configs" / "v2" / "dcpa_210m_v2_cognitive.yaml",
        preset="micro",
        total_steps=2,
        stage_boundaries=(0, 1, 2, 3, 4),
        target_tokens_bytes_per_step=256,
        micro_batch_bytes=256,
        local_cache_dir=tmp_path / "nvme",
        ckpt_dir=tmp_path / "ckpts",
        logs_dir=tmp_path / "logs",
        export_dir=tmp_path / "export",
        resume="none",
        device_str="cpu",
        compile_mode="none",
        bcg_chunk_steps=6,
        bcg_tbptt_window=12,
        lr_scale=0.5,
        config_overrides=["optimizer.groups.group_matrix_decay.lr=2e-4"],
        export_final=False,
        pre_step_hook=_hook,
    )
    assert summary.completed_steps == 2
    assert seen["bcg_chunk_steps"] == 6
    assert seen["bcg_tbptt_window"] == 12
    base_lrs = seen["base_lrs"]
    assert isinstance(base_lrs, dict)
    # 2e-4 from --set * 0.5 from --lr-scale == 1e-4
    assert abs(base_lrs["group_matrix_decay:cortical_io"] - 1.0e-4) < 1e-12


def test_autotune_candidate_grid_and_log_parser() -> None:
    # L4 (22 GB): watermark=0.0 must never be paired with bcg_cudagraphs=True
    l4_grid = build_candidate_grid(vram_gb=22.0, target_step_bytes=131_072, max_trials=20)
    assert len(l4_grid) > 0
    for c in l4_grid:
        assert c.micro_batch_bytes <= 131_072
        assert c.bcg_ckpt_watermark_gb <= round(0.88 * 22.0, 1)
        if c.bcg_ckpt_watermark_gb == 0.0:
            assert c.bcg_cudagraphs is False

    # G4 (95 GB): starts with 128KB micro-batch and high watermark
    g4_grid = build_candidate_grid(vram_gb=95.0, target_step_bytes=131_072, max_trials=12)
    assert g4_grid[0].micro_batch_bytes == 131_072
    assert g4_grid[0].bcg_ckpt_watermark_gb > 50.0

    # Stage probe boundaries
    assert stage_boundaries_for_probe(2, 14) == (0, 1, 15, 16, 17)
    assert stage_boundaries_for_probe(3, 14) == (0, 1, 2, 15, 16)

    sample_log = """
    Step | Stg |    Loss |    BPC |     PC |   rho |   Std(m/mn) |   Ent |  Skip% |  L_m |     GN |       LR | MemGB |    KB/s
       1 | S1  | 10.0000 | 12.000 |  0.000 |  9.50 |  1.00/ 0.90 |  4.20 |   0.0% | 1.00 |  10.00 | 1.00e-03 |  4.10 |     1.0
       2 | S2  | 19.1418 | 21.034 |  0.508 |  9.61 |  1.12/ 1.02 |  4.28 |   0.0% | 3.79 | 443.78 | 1.00e-03 |  5.25 |     0.1
       3 | S2  | 17.5036 | 19.599 |  0.508 |  9.02 |  1.12/ 1.03 |  4.28 |   0.0% | 3.79 | 386.31 | 1.00e-03 |  6.92 |     3.5
       4 | S2  | 14.7250 | 17.323 |  0.508 |  9.11 |  1.12/ 1.02 |  4.28 |   0.0% | 3.77 | 257.07 | 1.00e-03 |  7.40 |    62.0
       5 | S2  | 12.7893 | 15.497 |  0.508 |  9.04 |  1.12/ 1.01 |  4.28 |   0.0% | 3.78 | 118.34 | 1.00e-03 |  7.40 |    64.0
       6 | S2  | 11.7188 | 14.211 |  0.508 |  8.98 |  1.13/ 1.02 |  4.28 |   0.0% | 3.77 |  34.26 | 1.00e-03 |  7.40 |    66.0
    """
    med_kb, mean_kb, peak_gb, n_st = parse_trial_log(sample_log, target_stage=2, warmup_steps=2)
    assert n_st == 3
    assert abs(med_kb - 64.0) < 1e-6
    assert abs(mean_kb - 64.0) < 1e-6
    assert abs(peak_gb - 7.40) < 1e-6


def test_autotune_cli_dry_run_and_micro_cpu_trial(tmp_path: Path) -> None:
    dry = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "autotune_gpu.py"),
            "--config",
            "configs/v2/dcpa_210m_v2_cognitive.yaml",
            "--preset",
            "moe_210m",
            "--vram-gb-override",
            "22.0",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "mb=64K_" in dry.stdout
    assert "--bcg-ckpt-watermark-gb" in dry.stdout

    out_json = tmp_path / "autotune_res.json"
    live = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "autotune_gpu.py"),
            "--config",
            "configs/v2/dcpa_210m_v2_cognitive.yaml",
            "--preset",
            "micro",
            "--device",
            "cpu",
            "--compile-mode",
            "none",
            "--stage",
            "2",
            "--steps",
            "4",
            "--warmup-steps",
            "1",
            "--target-step-bytes",
            "512",
            "--micro-bytes",
            "256",
            "--bcg-chunks",
            "4",
            "--stream-chunks",
            "128",
            "--cudagraphs",
            "false",
            "--watermarks",
            "0.0",
            "--local-cache-dir",
            str(tmp_path / "nvme"),
            "--out-json",
            str(out_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[Autotune][WINNER]" in live.stdout
    assert out_json.is_file()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["best"]["status"] == "ok"
    assert report["best"]["steady_steps_measured"] == 2
    assert "--micro-batch-bytes 256" in report["recommended_extra_train_args"]
