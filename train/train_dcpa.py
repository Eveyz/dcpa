"""Production cross-platform 100,000-step DCPA developmental training orchestrator.

Integrates Deliverables 1-5 and Global Override Iron Laws #1-#5:
  - YAML hyperparameter & curriculum contract (`configs/dcpa_48m_train.yaml`).
  - Four-stage developmental state machine (`train/curriculum_state_machine.py`) with hard
    `requires_grad_(False)` + `grad = None` freezing (NEVER grad-zeroing) and fixed Step-0
    three-way differentiated AdamW parameter groups.
  - Pure `bf16` autocast on CUDA ONLY (NO `GradScaler`!), with all 5 numerically sensitive ops
    isolated in pure `float32` precision islands; automatic pure `float32` fallback on MPS/CPU.
  - Constant-Byte-Budget zero-padding Ragged UTF-8 `RaggedByteDataset` with deterministic
    checkpointable `dataloader_cursor` and dynamic hardware-tier gradient accumulation.
  - 20-step sliding-window Self-Healing Watchdog (`train/watchdog_and_ckpt.py`) with atomic
    `*.pt.tmp` + `fsync` + `os.replace` dual-track checkpoints, `SIGINT`/`SIGTERM` preemption
    protection, optional Colab A/B Ping-Pong Drive sync, and 3-rollback circuit breaker (exit 42).
  - Concise 50-step single-line console table and 500-step zero-grad visual diagnostic probe
    (`train/visual_probe.py`) + automatic final `dcpa_final.safetensors` export.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.ragged_dataloader import RaggedByteDataset
from data.wug_generator import sample_wug_episode_fast
from dcpa.config import DCPAConfig
from dcpa.data.synthetic import generate_dyck_sequence, generate_grid_physics_episode
from dcpa.export import extract_flat_state_dict
from dcpa.model import DCPAModel
from train.colab_drive_sync import (
    ABDriveSyncManager,
    CheckpointTrigger,
    MILESTONE_STEPS,
    assert_not_drive_path,
    sync_file_to_drive,
)
from train.curriculum_state_machine import (
    CurriculumController,
    CurriculumStepState,
    apply_stage_freeze,
    build_byte_loss_weights,
    create_dcpa_optimizer,
)
from train.visual_probe import format_console_header, format_console_row, run_visual_probe
from train.watchdog_and_ckpt import (
    CheckpointManager,
    MeltdownAbortError,
    PreemptionSignalGuard,
    SelfHealingWatchdog,
    reseed_all_rngs,
)
from utils.device import (
    amp_autocast,
    capture_rng_state,
    configure_cuda_tf32,
    describe_hardware_profile,
    detect_hardware_profile,
    get_peak_memory_gb,
    reset_peak_memory_stats,
    restore_rng_state,
    seed_everything,
    synchronize,
)


DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "dcpa_48m_train.yaml"


@dataclass
class TrainRunSummary:
    completed_steps: int
    total_steps: int
    resumed_from_step: int
    fallback_resume_used: bool
    last_stage: str
    last_stage_id: int
    last_metrics: dict[str, float]
    best_loss: float
    saves_performed: int
    rollbacks_performed: int
    rollback_events: list[dict[str, Any]] = field(default_factory=list)
    loss_history: dict[int, float] = field(default_factory=dict)
    final_export_path: Optional[Path] = None
    hardware_tier: str = ""
    interrupted_by_signal: Optional[str] = None
    final_lr_multiplier: float = 1.0


def load_yaml_config(config_path: Optional[str | Path]) -> dict[str, Any]:
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Training config YAML not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return dict(data or {})


def build_dcpa_config(preset: str, yaml_cfg: Mapping[str, Any]) -> DCPAConfig:
    """Resolve `DCPAConfig` from preset name and optional YAML overrides."""
    preset_key = preset.strip().lower()
    factory_map: dict[str, Callable[[], DCPAConfig]] = {
        "proto_48m": DCPAConfig.proto_48m,
        "48m": DCPAConfig.proto_48m,
        "standard_152m": DCPAConfig.standard_152m,
        "152m": DCPAConfig.standard_152m,
        "base": DCPAConfig.base,
        "tiny": DCPAConfig.tiny,
        "micro": DCPAConfig.micro,
    }
    if preset_key not in factory_map:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {sorted(factory_map)}")
    base_cfg = factory_map[preset_key]()
    merged = base_cfg.to_dict()

    presets_block = yaml_cfg.get("presets", {}) if isinstance(yaml_cfg, Mapping) else {}
    yaml_model = yaml_cfg.get("model", {}) if isinstance(yaml_cfg, Mapping) else {}
    # Apply YAML model defaults when using the YAML's own default preset
    if yaml_model and str(yaml_model.get("preset", "")).lower() == preset_key:
        for k, v in yaml_model.items():
            if k in merged:
                merged[k] = v
    if isinstance(presets_block, Mapping) and preset_key in presets_block:
        for k, v in dict(presets_block[preset_key]).items():
            if k in merged:
                merged[k] = v

    return DCPAConfig(**merged)


def _parse_int_tuple(raw: Optional[str], default: tuple[int, ...]) -> tuple[int, ...]:
    if not raw:
        return default
    vals = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    return tuple(vals)


def _sample_synthetic_ragged_batch(
    batch_idx: int,
    seed: int,
    target_bytes: int,
    max_doc_bytes: int,
    wug_ratio: float,
) -> dict[str, torch.Tensor]:
    """Deterministic zero-padding Ragged synthetic fallback when no `.bin/.idx` shard is passed."""
    rng = random.Random(int(seed) + int(batch_idx) * 1_000_003 + 17)
    budget = max(64, int(target_bytes))
    max_doc = max(32, min(int(max_doc_bytes), budget))

    chunks: list[bytes] = []
    cu_list: list[int] = [0]
    doc_langs: list[int] = []
    byte_langs: list[int] = []
    loss_masks: list[bool] = []
    wug_masks: list[bool] = []
    cursor = 0

    while cursor < budget:
        rem = budget - cursor
        if rem < 32 and chunks:
            break
        lid = rng.randrange(5)
        if wug_ratio > 0.0 and rng.random() < wug_ratio:
            raw_b, t1_len, tgt_s, tgt_e = sample_wug_episode_fast(rng, lid)
            if len(raw_b) <= rem or not chunks:
                if len(raw_b) > rem:
                    # Keep full episode if first item so Wug masks stay 100% aligned
                    pass
                ep_len = len(raw_b)
                chunks.append(raw_b)
                doc_langs.append(lid)
                byte_langs.extend([lid] * ep_len)
                loss_masks.extend([False] * t1_len + [True] * (ep_len - t1_len))
                wug_masks.extend(
                    [False] * tgt_s + [True] * (tgt_e - tgt_s) + [False] * (ep_len - tgt_e)
                )
                cursor += ep_len
                cu_list.append(cursor)
                continue

        doc_cap = min(max_doc, rem)
        if lid in (3, 4):
            txt, _ = generate_dyck_sequence(rng, max_depth=6, min_bytes=min(64, doc_cap), rust_flavored=(lid == 4))
            raw_b = txt.encode("utf-8")[:doc_cap]
        else:
            txt = generate_grid_physics_episode(rng, num_steps=8)
            raw_b = txt.encode("utf-8")[:doc_cap]
        if not raw_b:
            raw_b = b"fn main() { let x = 1; }\n"[:doc_cap]
        seg_len = len(raw_b)
        chunks.append(raw_b)
        doc_langs.append(lid)
        byte_langs.extend([lid] * seg_len)
        loss_masks.extend([True] * seg_len)
        wug_masks.extend([False] * seg_len)
        cursor += seg_len
        cu_list.append(cursor)

    joined = b"".join(chunks)
    total_t = len(joined)
    cu_arr = np.asarray(cu_list, dtype=np.int32)
    is_doc_start = np.zeros(total_t, dtype=np.bool_)
    if len(cu_arr) > 1:
        is_doc_start[cu_arr[:-1]] = True

    return {
        "x_bytes": torch.from_numpy(np.frombuffer(joined, dtype=np.uint8).copy()),
        "cu_seqlens": torch.from_numpy(cu_arr),
        "is_doc_start": torch.from_numpy(is_doc_start),
        "doc_lang_ids": torch.tensor(doc_langs, dtype=torch.long),
        "byte_lang_ids": torch.tensor(byte_langs, dtype=torch.long),
        "loss_mask": torch.tensor(loss_masks, dtype=torch.bool),
        "wug_target_mask": torch.tensor(wug_masks, dtype=torch.bool),
    }


def _compute_familiar_doc_mask(
    cu_seqlens: torch.Tensor,
    loss_mask: torch.Tensor,
    wug_target_mask: torch.Tensor,
) -> torch.Tensor:
    """Identify regular replay (familiar) documents vs online Wug episodes in a packed Stage 4 batch."""
    n_docs = int(cu_seqlens.numel()) - 1
    fam = torch.ones((n_docs,), dtype=torch.bool, device=loss_mask.device)
    cu_cpu = cu_seqlens.detach().cpu().tolist()
    for d in range(n_docs):
        s, e = cu_cpu[d], cu_cpu[d + 1]
        if wug_target_mask[s:e].any() or (~loss_mask[s:e]).any():
            fam[d] = False
    return fam


def _check_active_grads_finite(active_params: Sequence[torch.nn.Parameter]) -> bool:
    for p in active_params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def export_final_safetensors(
    model: DCPAModel,
    export_dir: str | Path,
    filename: str = "dcpa_final.safetensors",
) -> dict[str, Path]:
    """Atomically export slow weights & frozen orthogonal projections to `.safetensors` + `config.json`."""
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dst = out_dir / filename
    weights_tmp = out_dir / f"{filename}.tmp"
    cfg_dst = out_dir / "config.json"

    flat_weights = extract_flat_state_dict(model, dtype=torch.float32)
    # Double-check Red Line #3: NEVER export fast hippocampal state matrices
    for k in flat_weights:
        if any(bad in k for bad in ("m_c_from_l", "m_l_from_c", "u_lex", "u_sem")):
            raise RuntimeError(f"Red Line #3 violation: fast weight {k!r} leaked into safetensors export!")

    save_file(flat_weights, str(weights_tmp))
    with weights_tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(weights_tmp, weights_dst)

    cfg_tmp = out_dir / "config.json.tmp"
    with cfg_tmp.open("w", encoding="utf-8") as cf:
        cf.write(json.dumps(model.cfg.to_dict(), indent=2))
        cf.flush()
        os.fsync(cf.fileno())
    os.replace(cfg_tmp, cfg_dst)

    return {"weights": weights_dst, "config": cfg_dst}


class _DynamicRaggedCursorStream:
    """Deterministic cursor-addressable Ragged byte stream adapting automatically to curriculum stage."""

    def __init__(
        self,
        bin_path: Optional[Path],
        idx_path: Optional[Path],
        seed: int,
    ) -> None:
        self.bin_path = bin_path if (bin_path is not None and bin_path.exists()) else None
        self.idx_path = idx_path if (idx_path is not None and idx_path.exists()) else None
        self.has_shards = self.bin_path is not None and self.idx_path is not None
        self._ds: Optional[RaggedByteDataset] = None
        self._ds_key: Optional[tuple[Any, ...]] = None
        self.cursor: dict[str, Any] = {
            "batch_idx": 0,
            "epoch": 0,
            "seed": int(seed),
            "consumed_bytes": 0,
        }
        self._corpus_bytes: int = int(self.bin_path.stat().st_size) if self.bin_path is not None else 1

    def _ensure_dataset(
        self,
        *,
        target_batch_bytes: int,
        max_doc_bytes: int,
        domain_weights: Mapping[int, float],
        wug_ratio: float,
    ) -> Optional[RaggedByteDataset]:
        if not self.has_shards or self.bin_path is None or self.idx_path is None:
            return None
        dw_key = tuple(sorted((int(k), round(float(v), 6)) for k, v in domain_weights.items()))
        key = (int(target_batch_bytes), int(max_doc_bytes), dw_key, round(float(wug_ratio), 6), int(self.cursor["seed"]))
        if self._ds is not None and self._ds_key == key:
            return self._ds
        if self._ds is not None:
            self._ds.close()
        self._ds = RaggedByteDataset(
            bin_path=self.bin_path,
            idx_path=self.idx_path,
            target_batch_bytes=int(target_batch_bytes),
            max_doc_bytes=int(max_doc_bytes),
            domain_weights=dict(domain_weights),
            wug_ratio=float(wug_ratio),
            num_batches=10_000_000,
            seed=int(self.cursor["seed"]),
        )
        self._ds_key = key
        return self._ds

    def next_batch(
        self,
        *,
        target_batch_bytes: int,
        state: CurriculumStepState,
    ) -> dict[str, torch.Tensor]:
        b_idx = int(self.cursor["batch_idx"])
        ds = self._ensure_dataset(
            target_batch_bytes=target_batch_bytes,
            max_doc_bytes=state.max_doc_bytes,
            domain_weights=state.domain_weights,
            wug_ratio=state.wug_ratio,
        )
        if ds is not None:
            batch = ds[b_idx]
        else:
            batch = _sample_synthetic_ragged_batch(
                batch_idx=b_idx,
                seed=int(self.cursor["seed"]),
                target_bytes=target_batch_bytes,
                max_doc_bytes=state.max_doc_bytes,
                wug_ratio=state.wug_ratio,
            )
        self.cursor["batch_idx"] = b_idx + 1
        self.cursor["consumed_bytes"] = int(self.cursor["consumed_bytes"]) + int(batch["x_bytes"].numel())
        self.cursor["epoch"] = int(self.cursor["consumed_bytes"]) // max(self._corpus_bytes, 1)
        return batch

    def restore_cursor(self, cursor: Optional[Mapping[str, Any]]) -> None:
        if not cursor:
            return
        self.cursor["batch_idx"] = int(cursor.get("batch_idx", 0))
        self.cursor["epoch"] = int(cursor.get("epoch", 0))
        self.cursor["seed"] = int(cursor.get("seed", self.cursor["seed"]))
        self.cursor["consumed_bytes"] = int(cursor.get("consumed_bytes", 0))

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None
            self._ds_key = None


def run_training(
    *,
    config_path: Optional[str | Path] = None,
    preset: Optional[str] = None,
    total_steps: Optional[int] = None,
    stage_boundaries: Optional[Sequence[int]] = None,
    seq_len: int = 2048,
    target_tokens_bytes_per_step: Optional[int] = None,
    micro_batch_bytes: Optional[int] = None,
    data_dir: Optional[str | Path] = None,
    ckpt_dir: Optional[str | Path] = None,
    logs_dir: Optional[str | Path] = None,
    export_dir: Optional[str | Path] = None,
    local_cache_dir: str | Path = "artifacts/nvme_cache",
    drive_sync_dir: Optional[str | Path] = None,
    drive_logs_dir: Optional[str | Path] = None,
    drive_export_dir: Optional[str | Path] = None,
    save_every_steps: Optional[int] = None,
    milestone_every_steps: Optional[int] = None,
    save_every_minutes: float = 10.0,
    probe_every_steps: Optional[int] = None,
    console_every_steps: Optional[int] = None,
    heartbeat_seconds: float = 60.0,
    resume: str = "auto",
    device_str: str = "auto",
    deterministic: bool = False,
    seed: int = 1337,
    max_consecutive_meltdowns: int = 3,
    watchdog_warmup_steps: Optional[int] = None,
    milestone_steps: Optional[tuple[int, ...]] = None,
    export_final: bool = True,
    clock: Callable[[], float] = time.monotonic,
    pre_step_hook: Optional[Callable[[int, DCPAModel, torch.optim.Optimizer, dict[str, float]], None]] = None,
    step_hook: Optional[Callable[[int, DCPAModel, dict[str, float]], None]] = None,
    stage_change_hook: Optional[Callable[[int, int, DCPAModel], None]] = None,
) -> TrainRunSummary:
    """Execute the full DCPA 4-stage self-healing developmental training lifecycle."""
    yaml_cfg = load_yaml_config(config_path) if config_path is not None else {}
    hw_cfg = dict(yaml_cfg.get("hardware", {}))
    opt_cfg = dict(yaml_cfg.get("optimizer", {}))
    curr_cfg = dict(yaml_cfg.get("curriculum", {}))
    wd_cfg = dict(yaml_cfg.get("watchdog", {}))
    ck_cfg = dict(yaml_cfg.get("checkpointing", {}))
    vp_cfg = dict(yaml_cfg.get("visual_probe", {}))

    eff_preset = preset or str(yaml_cfg.get("model", {}).get("preset", "proto_48m"))

    # Resolve boundaries & total_steps cleanly
    if stage_boundaries is not None:
        eff_boundaries = tuple(int(x) for x in stage_boundaries)
        eff_total_steps = int(total_steps) if total_steps is not None else eff_boundaries[-1]
    elif total_steps is not None:
        eff_total_steps = int(total_steps)
        if eff_total_steps == 100_000 and "stage_boundaries" in curr_cfg:
            eff_boundaries = tuple(int(x) for x in curr_cfg["stage_boundaries"])
        elif eff_total_steps == 100_000:
            eff_boundaries = CurriculumController.NOMINAL_BOUNDARIES
        else:
            # Proportional boundaries when a custom short total_steps is requested without explicit boundaries
            b1 = max(1, int(round(eff_total_steps * 0.15)))
            b2 = max(b1 + 1, int(round(eff_total_steps * 0.65)))
            b3 = max(b2 + 1, int(round(eff_total_steps * 0.85)))
            b4 = max(b3 + 1, eff_total_steps)
            if b4 > eff_total_steps:
                # Tiny test runs (e.g. total_steps <= 6): keep Stage 1..4 cleanly bounded
                eff_boundaries = (0, eff_total_steps, eff_total_steps + 1, eff_total_steps + 2, eff_total_steps + 3)
            else:
                eff_boundaries = (0, b1, b2, b3, eff_total_steps)
    else:
        eff_boundaries = tuple(int(x) for x in curr_cfg.get("stage_boundaries", CurriculumController.NOMINAL_BOUNDARIES))
        eff_total_steps = eff_boundaries[-1]

    # Enforce Iron Law #3.1 NVMe isolation on hot cache & data paths
    local_cache = assert_not_drive_path(local_cache_dir, role="--local-cache-dir")
    if data_dir is not None:
        assert_not_drive_path(data_dir, role="--data-dir")

    eff_ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else Path(ck_cfg.get("ckpt_dir", local_cache / "checkpoints"))
    eff_logs_dir = Path(logs_dir) if logs_dir is not None else Path(vp_cfg.get("logs_dir", local_cache / "logs"))
    eff_export_dir = Path(export_dir) if export_dir is not None else Path(ck_cfg.get("export_dir", local_cache / "export"))
    eff_ckpt_dir.mkdir(parents=True, exist_ok=True)
    eff_logs_dir.mkdir(parents=True, exist_ok=True)
    eff_export_dir.mkdir(parents=True, exist_ok=True)

    # Hardware detection & auto grad-accum sharding (Iron Law #1 + Deliverable 1)
    yaml_target_bytes = hw_cfg.get("target_tokens_bytes_per_step")
    profile = detect_hardware_profile(
        preferred=device_str,
        deterministic=deterministic,
        target_tokens_bytes_per_step=(
            target_tokens_bytes_per_step
            if target_tokens_bytes_per_step is not None
            else (int(yaml_target_bytes) if (yaml_target_bytes is not None and device_str == "cuda") else None)
        ),
    )
    device = profile.device
    allow_tf32 = False if deterministic else bool(hw_cfg.get("allow_tf32", True))
    configure_cuda_tf32(device, allow=allow_tf32)
    reset_peak_memory_stats(device)

    use_bf16 = bool(profile.use_bf16 and hw_cfg.get("bf16_autocast", True) and not deterministic)
    if hw_cfg.get("use_grad_scaler", False):
        raise ValueError("Red Line #1 violation: GradScaler is strictly forbidden with bfloat16!")

    eff_target_bytes = (
        int(target_tokens_bytes_per_step)
        if target_tokens_bytes_per_step is not None
        else int(profile.target_tokens_bytes_per_step)
    )
    eff_micro_bytes = (
        int(micro_batch_bytes)
        if micro_batch_bytes is not None
        else min(int(profile.micro_batch_bytes), eff_target_bytes)
    )
    eff_micro_bytes = max(64, eff_micro_bytes)
    grad_accum_steps = max(1, math.ceil(eff_target_bytes / eff_micro_bytes))
    actual_step_bytes = eff_micro_bytes * grad_accum_steps

    print(describe_hardware_profile(profile), flush=True)
    print(
        f"[Plan] preset={eff_preset} | total_steps={eff_total_steps:,} | "
        f"boundaries={list(eff_boundaries)} | micro_batch_bytes={eff_micro_bytes:,} | "
        f"grad_accum_steps={grad_accum_steps} | effective_bytes_per_step={actual_step_bytes:,} | "
        f"bf16={use_bf16} | grad_scaler=False",
        flush=True,
    )

    seed_everything(seed, device=device)
    cfg = build_dcpa_config(eff_preset, yaml_cfg)
    model = DCPAModel(cfg).to(device)
    optimizer = create_dcpa_optimizer(model, optimizer_cfg=opt_cfg)
    grad_clip_norm = float(opt_cfg.get("grad_clip_norm", cfg.grad_clip_norm))

    controller = CurriculumController(curriculum_cfg=curr_cfg, stage_boundaries=eff_boundaries)

    if watchdog_warmup_steps is not None:
        wd_cfg["rule_bcd_warmup_steps"] = int(watchdog_warmup_steps)
    wd_cfg["max_consecutive_rollbacks"] = int(max_consecutive_meltdowns)
    watchdog = SelfHealingWatchdog(
        watchdog_cfg=wd_cfg,
        base_seed=seed,
        events_log_path=eff_logs_dir / "watchdog_events.jsonl",
    )

    eff_healthy_every = (
        int(save_every_steps)
        if save_every_steps is not None
        else int(ck_cfg.get("healthy_save_every_steps", 1000))
    )
    eff_milestone_every = (
        int(milestone_every_steps)
        if milestone_every_steps is not None
        else int(ck_cfg.get("milestone_every_steps", 5000))
    )
    eff_probe_every = (
        int(probe_every_steps)
        if probe_every_steps is not None
        else int(vp_cfg.get("probe_every_steps", 500))
    )
    eff_console_every = (
        int(console_every_steps)
        if console_every_steps is not None
        else int(vp_cfg.get("console_every_steps", 50))
    )
    eff_stage_milestones = (
        milestone_steps if milestone_steps is not None else controller.stage_boundary_milestones()
    )

    ckpt_mgr = CheckpointManager(
        ckpt_dir=eff_ckpt_dir,
        healthy_save_every_steps=eff_healthy_every,
        milestone_every_steps=eff_milestone_every,
        stage_boundaries=eff_stage_milestones,
    )

    # Optional Colab A/B Ping-Pong Drive sync + 10-minute wall-clock trigger (Iron Law #3)
    sync_mgr: Optional[ABDriveSyncManager] = None
    if drive_sync_dir is not None:
        sync_mgr = ABDriveSyncManager(
            local_cache_dir=local_cache,
            drive_sync_dir=drive_sync_dir,
            milestone_steps=eff_stage_milestones if eff_stage_milestones else MILESTONE_STEPS,
        )
    wall_trigger = CheckpointTrigger(
        save_every_steps=eff_healthy_every,
        save_every_minutes=save_every_minutes,
        clock=clock,
    )

    # Hot NVMe Ragged byte stream
    train_bin = (Path(data_dir) / "train.bin") if data_dir is not None else None
    train_idx = (Path(data_dir) / "train.idx") if data_dir is not None else None
    val_bin = (Path(data_dir) / "val.bin") if data_dir is not None else None
    val_idx = (Path(data_dir) / "val.idx") if data_dir is not None else None
    stream = _DynamicRaggedCursorStream(train_bin, train_idx, seed=seed)
    if stream.has_shards:
        print(f"[Data] Opened NVMe Ragged UTF-8 stream: {train_bin}", flush=True)

    # Resume handling (auto: Colab A/B Ping-Pong first if configured, else local CheckpointManager)
    start_step = 0
    resumed_from_step = 0
    fallback_resume_used = False
    best_val_metric = float("inf")

    if resume and resume.lower() != "none":
        if resume.lower() == "auto":
            loaded_ckpt: Optional[dict[str, Any]] = None
            if sync_mgr is not None:
                res = sync_mgr.load_latest_auto(map_location=device)
                if res is not None:
                    loaded_ckpt = res.checkpoint
                    fallback_resume_used = res.fallback_triggered
            if loaded_ckpt is None:
                latest_local = ckpt_mgr.find_latest_checkpoint()
                if latest_local is not None:
                    loaded_ckpt = ckpt_mgr.load_checkpoint(latest_local, map_location=device)
                    print(f"[Resume] Loaded local checkpoint: {latest_local}", flush=True)
            if loaded_ckpt is not None:
                start_step, restored_cursor = CheckpointManager.restore_into(
                    loaded_ckpt,
                    model=model,
                    optimizer=optimizer,
                    controller=controller,
                    watchdog=watchdog,
                    preserve_watchdog_repairs=False,
                    restore_rng=True,
                )
                stream.restore_cursor(restored_cursor)
                resumed_from_step = start_step
                best_val_metric = float(loaded_ckpt.get("best_val_metric", loaded_ckpt.get("best_loss", float("inf"))))
                ckpt_mgr.last_healthy_step = start_step
                wall_trigger.mark_saved(start_step)
        else:
            explicit = Path(resume)
            loaded_ckpt = ckpt_mgr.load_checkpoint(explicit, map_location=device)
            start_step, restored_cursor = CheckpointManager.restore_into(
                loaded_ckpt,
                model=model,
                optimizer=optimizer,
                controller=controller,
                watchdog=watchdog,
                preserve_watchdog_repairs=False,
                restore_rng=True,
            )
            stream.restore_cursor(restored_cursor)
            resumed_from_step = start_step
            best_val_metric = float(loaded_ckpt.get("best_val_metric", loaded_ckpt.get("best_loss", float("inf"))))
            ckpt_mgr.last_healthy_step = start_step
            wall_trigger.mark_saved(start_step)
            print(f"[Resume] Restored explicit checkpoint {explicit} at step={start_step}", flush=True)

    current_stage_id: int = -1
    last_stage_str: str = "A"
    last_metrics: dict[str, float] = {}
    loss_history: dict[int, float] = {}
    rollback_events: list[dict[str, Any]] = []
    saves_performed = 0
    steps_since_start = 0
    t_run_start = clock()
    last_heartbeat_t = t_run_start
    header_printed = False
    interrupted_sig: Optional[str] = None

    def _build_ckpt_payload(completed_step: int, st_id: int) -> dict[str, Any]:
        return CheckpointManager.build_payload(
            step=completed_step,
            stage_id=st_id,
            model=model,
            optimizer=optimizer,
            scheduler_state=controller.state_dict(),
            dynamic_loss_multipliers=watchdog.state_dict(),
            dataloader_cursor=dict(stream.cursor),
            best_val_metric=best_val_metric,
            config_dict=cfg.to_dict(),
            hardware_tier=profile.tier,
        )

    try:
        with PreemptionSignalGuard() as sig_guard:
            step_idx = start_step
            while step_idx < eff_total_steps:
                completed_step = step_idx + 1
                sc = controller.get(step_idx)
                last_stage_str = sc.model_stage

                if sc.stage_id != current_stage_id:
                    freeze_stats = apply_stage_freeze(model, sc.stage_id)
                    current_stage_id = sc.stage_id
                    print(
                        f"[Stage] Entering Stage {sc.stage_id} ({sc.model_stage}: {sc.stage_name}) "
                        f"at step={completed_step:,}/{eff_total_steps:,} | "
                        f"active_params={freeze_stats['active_params']:,} | "
                        f"frozen_params={freeze_stats['frozen_params']:,} | "
                        f"max_doc={sc.max_doc_bytes}B | wug_ratio={sc.wug_ratio:.2f}",
                        flush=True,
                    )
                    if stage_change_hook is not None:
                        stage_change_hook(completed_step, sc.stage_id, model)

                applied_lrs = controller.apply_to_optimizer(
                    optimizer,
                    sc,
                    watchdog_lr_multiplier=watchdog.directives.lr_multiplier,
                )
                eff_tau_s = watchdog.effective_tau_s(sc.tau_s)
                eff_tau_m = watchdog.effective_tau_m(sc.tau_m)
                eff_lam_rate = watchdog.effective_lambda_rate(sc.lambda_rate)
                eff_lam_var = watchdog.effective_lambda_var(sc.lambda_var)
                eff_uniform_mix = watchdog.active_merge_uniform_mix()
                eff_merge_kl = watchdog.active_merge_kl_weight()

                optimizer.zero_grad(set_to_none=True)
                accum_metrics: dict[str, float] = {}
                step_loss_finite = True

                for _micro_idx in range(grad_accum_steps):
                    batch = stream.next_batch(target_batch_bytes=eff_micro_bytes, state=sc)
                    x_bytes = batch["x_bytes"].to(device)
                    cu_seqlens = batch["cu_seqlens"].to(device)
                    doc_lang_ids = batch["doc_lang_ids"].to(device)
                    byte_lang_ids = batch["byte_lang_ids"].to(device)
                    loss_mask = batch["loss_mask"].to(device)
                    wug_target_mask = batch["wug_target_mask"].to(device)

                    loss_w = build_byte_loss_weights(
                        x_bytes=x_bytes,
                        byte_lang_ids=byte_lang_ids,
                        loss_mask=loss_mask,
                        wug_target_mask=wug_target_mask,
                        state=sc,
                    )
                    fam_mask = (
                        _compute_familiar_doc_mask(cu_seqlens, loss_mask, wug_target_mask)
                        if sc.stage_id == 4
                        else None
                    )

                    with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                        out = model(
                            raw_bytes=x_bytes.unsqueeze(0),
                            lang_ids=doc_lang_ids,
                            stage=sc.model_stage,
                            tau_s=eff_tau_s,
                            tau_m=eff_tau_m,
                            use_broca_cq=sc.use_broca_cq,
                            loss_weights=loss_w,
                            lambda_rate_override=eff_lam_rate,
                            lambda_var_override=eff_lam_var,
                            lambda_pc_override=sc.lambda_pc,
                            lambda_act_override=sc.lambda_act,
                            lambda_halt_override=sc.lambda_halt,
                            target_spike_rate_override=sc.target_spike_rate,
                            merge_kl_weight=eff_merge_kl,
                            merge_uniform_mix=eff_uniform_mix,
                            familiar_ach_reg_weight=sc.lambda_ach_reg if sc.stage_id == 4 else 0.0,
                            familiar_doc_mask=fam_mask,
                            doc_cu_seqlens=cu_seqlens,
                        )

                    for k, v in out.metrics.items():
                        accum_metrics[k] = accum_metrics.get(k, 0.0) + float(v) / float(grad_accum_steps)

                    if not torch.isfinite(out.loss_total):
                        step_loss_finite = False
                        break

                    scaled_loss = out.loss_total / float(grad_accum_steps)
                    scaled_loss.backward()

                # Test injection / instrumentation hook before Watchdog check & optimizer.step()
                if pre_step_hook is not None:
                    pre_step_hook(completed_step, model, optimizer, accum_metrics)

                active_params = [p for p in model.parameters() if p.requires_grad]
                grads_finite = step_loss_finite and _check_active_grads_finite(active_params)
                grad_norm: Optional[float] = None
                if grads_finite and active_params:
                    gn_t = torch.nn.utils.clip_grad_norm_(active_params, max_norm=grad_clip_norm)
                    grad_norm = float(gn_t.detach().item())
                    if not math.isfinite(grad_norm):
                        grads_finite = False

                action = watchdog.check_step(
                    step=completed_step,
                    stage_id=sc.stage_id,
                    metrics=accum_metrics,
                    grads_finite=grads_finite,
                    grad_norm=grad_norm,
                    rollback_target_step=ckpt_mgr.last_healthy_step,
                )
                if action.triggered:
                    optimizer.zero_grad(set_to_none=True)
                    rollback_events.append(action.event_record)
                    print(
                        f"[Watchdog Rule {action.rule}] {action.reason} | "
                        f"rollback_count={watchdog.rollback_count} | "
                        f"lr_mult={watchdog.directives.lr_multiplier:.4f} | "
                        f"skip_batches={action.skip_batches} | new_seed={action.new_seed}",
                        flush=True,
                    )
                    if ckpt_mgr.last_healthy_path.exists():
                        healthy_ckpt = ckpt_mgr.load_checkpoint(ckpt_mgr.last_healthy_path, map_location=device)
                        restored_step, restored_cursor = CheckpointManager.restore_into(
                            healthy_ckpt,
                            model=model,
                            optimizer=optimizer,
                            controller=controller,
                            watchdog=watchdog,
                            preserve_watchdog_repairs=True,
                            restore_rng=False,
                        )
                        stream.restore_cursor(restored_cursor)
                        stream.cursor["batch_idx"] = int(stream.cursor["batch_idx"]) + int(action.skip_batches)
                        step_idx = restored_step
                        current_stage_id = -1  # force exact freeze re-verification on next iteration
                    else:
                        stream.cursor["batch_idx"] = int(stream.cursor["batch_idx"]) + int(action.skip_batches)

                    if action.new_seed is not None:
                        stream.cursor["seed"] = int(action.new_seed)
                        reseed_all_rngs(action.new_seed, device=device)
                    continue

                # Healthy optimization step
                optimizer.step()
                watchdog.tick_post_healthy_step(completed_step)
                steps_since_start += 1
                last_metrics = accum_metrics
                loss_history[completed_step] = float(accum_metrics["loss_total"])

                if step_hook is not None:
                    step_hook(completed_step, model, last_metrics)

                now_t = clock()
                elapsed = max(now_t - t_run_start, 1e-6)
                kb_per_sec = float(steps_since_start * actual_step_bytes) / (elapsed * 1024.0)
                active_lrs = [v for v in applied_lrs.values() if v > 0.0]
                disp_lr = max(active_lrs) if active_lrs else 0.0

                is_final_step = completed_step == eff_total_steps

                # Concise single-line console row every `console_every_steps` (default 50)
                if (
                    completed_step == start_step + 1
                    or is_final_step
                    or (eff_console_every > 0 and completed_step % eff_console_every == 0)
                    or (heartbeat_seconds > 0 and (now_t - last_heartbeat_t) >= heartbeat_seconds)
                ):
                    if not header_printed:
                        print(format_console_header(), flush=True)
                        header_printed = True
                    print(
                        format_console_row(
                            step=completed_step,
                            stage_id=sc.stage_id,
                            metrics=last_metrics,
                            lr=disp_lr,
                            mem_gb=get_peak_memory_gb(device),
                            kb_per_sec=kb_per_sec,
                        ),
                        flush=True,
                    )
                    last_heartbeat_t = now_t

                # 500-step zero-grad visual diagnostic probe (isolated RNG so training trajectory is 100% invariant)
                is_best = False
                if eff_probe_every > 0 and (completed_step % eff_probe_every == 0 or is_final_step):
                    rng_snap = capture_rng_state()
                    probe_res = run_visual_probe(
                        model,
                        step=completed_step,
                        total_steps=eff_total_steps,
                        stage_id=sc.stage_id,
                        model_stage=sc.model_stage,
                        tau_s=eff_tau_s,
                        tau_m=eff_tau_m,
                        lr=disp_lr,
                        train_metrics=last_metrics,
                        device=device,
                        use_bf16=use_bf16,
                        kb_per_sec=kb_per_sec,
                        max_doc_bytes=sc.max_doc_bytes,
                        val_bin_path=val_bin,
                        val_idx_path=val_idx,
                        logs_dir=eff_logs_dir,
                        val_budget_bytes=min(2048, max(256, eff_micro_bytes)),
                        val_batches_per_domain=int(vp_cfg.get("val_batches_per_domain", 2)),
                    )
                    restore_rng_state(rng_snap)

                    eval_metric = (
                        probe_res.wug_target_bpc
                        if sc.stage_id == 4
                        else (
                            probe_res.val_bpc_overall
                            if probe_res.val_bpc_overall is not None
                            else probe_res.train_bpc
                        )
                    )
                    if math.isfinite(eval_metric) and eval_metric < best_val_metric:
                        best_val_metric = eval_metric
                        is_best = True

                    if drive_logs_dir is not None:
                        sync_file_to_drive(eff_logs_dir / "latest_probe.md", drive_logs_dir, dest_name="latest_probe.md")
                        sync_file_to_drive(eff_logs_dir / "metrics.jsonl", drive_logs_dir, dest_name="metrics.jsonl")

                # Dual-track atomic checkpointing + optional Colab Drive A/B Ping-Pong sync
                healthy_due = ckpt_mgr.is_healthy_save_due(completed_step)
                milestone_due = ckpt_mgr.is_milestone_step(completed_step)
                wall_due, wall_reason = wall_trigger.should_save(completed_step)
                need_payload = (
                    healthy_due
                    or milestone_due
                    or is_best
                    or wall_due
                    or is_final_step
                    or sig_guard.interrupted
                )
                if need_payload:
                    synchronize(device)
                    payload = _build_ckpt_payload(completed_step, sc.stage_id)

                    if sig_guard.interrupted:
                        ckpt_mgr.save_emergency(payload)
                        saves_performed += 1
                        interrupted_sig = sig_guard.signal_name
                        print(
                            f"[PreemptionGuard] Saved atomic emergency checkpoint at step={completed_step} "
                            f"-> {ckpt_mgr.emergency_interrupt_path}",
                            flush=True,
                        )
                        step_idx += 1
                        break

                    if (healthy_due or is_final_step) and watchdog.is_healthy_for_checkpoint(
                        completed_step, sc.stage_id, last_metrics
                    ):
                        ckpt_mgr.save_healthy(payload)
                        saves_performed += 1

                    if milestone_due:
                        ckpt_mgr.save_milestone(payload)
                        saves_performed += 1

                    if is_best:
                        ckpt_mgr.save_best(payload)
                        saves_performed += 1

                    if sync_mgr is not None and (wall_due or milestone_due or is_final_step or is_best):
                        reason_s = wall_reason or ("milestone" if milestone_due else ("final" if is_final_step else "best"))
                        sync_mgr.save_ping_pong(
                            payload=payload,
                            step=completed_step,
                            stage_id=sc.model_stage,
                            reason=reason_s,
                            is_best=is_best,
                        )
                        wall_trigger.mark_saved(completed_step)
                        saves_performed += 1
                    elif wall_due and not healthy_due:
                        # Non-Drive local wall-clock safety save
                        if watchdog.is_healthy_for_checkpoint(completed_step, sc.stage_id, last_metrics):
                            ckpt_mgr.save_healthy(payload)
                            saves_performed += 1
                        wall_trigger.mark_saved(completed_step)

                step_idx += 1

        # Final `.safetensors` export on clean completion
        final_export_path: Optional[Path] = None
        if export_final and interrupted_sig is None:
            synchronize(device)
            sf_name = str(ck_cfg.get("final_safetensors_name", "dcpa_final.safetensors"))
            exported = export_final_safetensors(model, eff_export_dir, filename=sf_name)
            final_export_path = exported["weights"]
            print(
                f"[Export] Atomically exported {final_export_path} "
                f"({final_export_path.stat().st_size / (1024 * 1024):.2f} MB)",
                flush=True,
            )
            if drive_export_dir is not None:
                final_export_path = sync_file_to_drive(
                    exported["weights"], drive_export_dir, dest_name=sf_name
                )
                sync_file_to_drive(exported["config"], drive_export_dir, dest_name="config.json")
                print(f"[Export] Mirrored to Drive -> {final_export_path}", flush=True)
            if hasattr(os, "sync"):
                os.sync()

        return TrainRunSummary(
            completed_steps=step_idx,
            total_steps=eff_total_steps,
            resumed_from_step=resumed_from_step,
            fallback_resume_used=fallback_resume_used,
            last_stage=last_stage_str,
            last_stage_id=max(1, current_stage_id),
            last_metrics=last_metrics,
            best_loss=best_val_metric,
            saves_performed=saves_performed,
            rollbacks_performed=watchdog.rollback_count,
            rollback_events=rollback_events,
            loss_history=loss_history,
            final_export_path=final_export_path,
            hardware_tier=profile.tier,
            interrupted_by_signal=interrupted_sig,
            final_lr_multiplier=watchdog.directives.lr_multiplier,
        )
    finally:
        stream.close()
        if hasattr(os, "sync"):
            os.sync()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DCPA 100k-Step Four-Stage Self-Healing Developmental Trainer")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)
    ap.add_argument(
        "--preset",
        choices=["micro", "tiny", "proto_48m", "48m", "base", "standard_152m", "152m"],
        default=None,
    )
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--stage-boundaries", type=str, default=None, help="Comma-separated 5 boundaries e.g. 0,20,45,60,80")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--target-tokens-bytes-per-step", type=int, default=None)
    ap.add_argument("--micro-batch-bytes", type=int, default=None)
    ap.add_argument("--data-dir", type=str, default="artifacts/data")
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--logs-dir", type=str, default=None)
    ap.add_argument("--export-dir", type=str, default=None)
    ap.add_argument("--local-cache-dir", type=str, default="artifacts/nvme_cache")
    ap.add_argument("--drive-sync-dir", type=str, default=None)
    ap.add_argument("--drive-logs-dir", type=str, default=None)
    ap.add_argument("--drive-export-dir", type=str, default=None)
    ap.add_argument("--save-every-steps", type=int, default=None)
    ap.add_argument("--milestone-every-steps", type=int, default=None)
    ap.add_argument("--save-every-minutes", type=float, default=10.0)
    ap.add_argument("--probe-every-steps", type=int, default=None)
    ap.add_argument("--console-every-steps", type=int, default=None)
    ap.add_argument("--heartbeat-seconds", type=float, default=60.0)
    ap.add_argument("--resume", type=str, default="auto")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-consecutive-meltdowns", type=int, default=3)
    ap.add_argument("--milestone-steps", type=str, default=None)
    ap.add_argument("--no-export", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    bounds = _parse_int_tuple(args.stage_boundaries, ()) if args.stage_boundaries else None
    m_steps = _parse_int_tuple(args.milestone_steps, MILESTONE_STEPS) if args.milestone_steps else None
    try:
        summary = run_training(
            config_path=args.config,
            preset=args.preset,
            total_steps=args.total_steps,
            stage_boundaries=bounds,
            seq_len=args.seq_len,
            target_tokens_bytes_per_step=args.target_tokens_bytes_per_step,
            micro_batch_bytes=args.micro_batch_bytes,
            data_dir=args.data_dir,
            ckpt_dir=args.ckpt_dir,
            logs_dir=args.logs_dir,
            export_dir=args.export_dir,
            local_cache_dir=args.local_cache_dir,
            drive_sync_dir=args.drive_sync_dir,
            drive_logs_dir=args.drive_logs_dir,
            drive_export_dir=args.drive_export_dir,
            save_every_steps=args.save_every_steps,
            milestone_every_steps=args.milestone_every_steps,
            save_every_minutes=args.save_every_minutes,
            probe_every_steps=args.probe_every_steps,
            console_every_steps=args.console_every_steps,
            heartbeat_seconds=args.heartbeat_seconds,
            resume=args.resume,
            device_str=args.device,
            deterministic=args.deterministic,
            seed=args.seed,
            max_consecutive_meltdowns=args.max_consecutive_meltdowns,
            milestone_steps=m_steps,
            export_final=not args.no_export,
        )
        print(
            f"[DONE] Completed {summary.completed_steps:,}/{summary.total_steps:,} steps "
            f"(saves={summary.saves_performed}, rollbacks={summary.rollbacks_performed}, "
            f"best_metric={summary.best_loss:.4f}, export={summary.final_export_path})",
            flush=True,
        )
    except MeltdownAbortError as exc:
        print(f"[FATAL MELTDOWN] {exc}", file=sys.stderr, flush=True)
        sys.exit(42)


if __name__ == "__main__":
    main()
