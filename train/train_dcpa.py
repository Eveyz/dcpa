"""Production cross-platform 100,000-step DCPA developmental training orchestrator.

Integrates Deliverables 1-5 and Global Override Iron Laws #1-#5:
  - YAML hyperparameter & curriculum contract (`configs/v1/dcpa_48m_train.yaml` or `configs/v2/*.yaml`).
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
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import contextlib
import copy
import json
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

# Prevent CUDA caching allocator fragmentation (e.g. fb=94GB on 96GB Blackwell / 32GB RTX 5090)
# and persist TorchInductor FX + AOTAutograd graphs on disk so restarts skip cold compilation!
# Must be set BEFORE `import torch` so it also applies automatically under older Colab notebooks!
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")

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
from data.stream_carry_packer import StreamCarryPacker
from data.wug_generator import sample_wug_episode_fast
from dcpa.compile_utils import configure_model_compile, unwrap_compiled_module
from dcpa.config import DCPAConfig
from dcpa.data.synthetic import generate_dyck_sequence, generate_grid_physics_episode
from dcpa.export import extract_flat_state_dict
from dcpa.model import DCPAModel, materialize_metric_vec
from dcpa.stream_carry import StreamCarryBank
from train.colab_drive_sync import (
    ABDriveSyncManager,
    CheckpointTrigger,
    MILESTONE_STEPS,
    RunLogTee,
    assert_not_drive_path,
    restore_drive_log_file,
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
    AsyncCheckpointWriter,
    CheckpointManager,
    MeltdownAbortError,
    PreemptionSignalGuard,
    SelfHealingWatchdog,
    reseed_all_rngs,
)
from utils.device import (
    DEFAULT_PROD_STEP_BYTES,
    HardwareProfile,
    amp_autocast,
    capture_rng_state,
    configure_cuda_tf32,
    describe_hardware_profile,
    detect_hardware_profile,
    get_peak_memory_gb,
    pin_batch_if_cuda,
    reset_peak_memory_stats,
    resolve_gpu_profile,
    restore_rng_state,
    seed_everything,
    synchronize,
    to_device_async,
)
from utils.distributed import (
    DistContext,
    all_reduce_gradients_coalesced,
    all_reduce_metrics_dict,
    broadcast_model_state,
    cleanup_distributed_context,
    dist_barrier,
    init_distributed_context,
    maybe_relaunch_distributed,
    sync_bool_any,
    sync_tensor_avg_,
)
from utils.wandb_tracker import WandbTracker, resolve_wandb_settings


DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "v1" / "dcpa_48m_train.yaml"


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
    wandb_run_id: Optional[str] = None


def load_yaml_config(config_path: Optional[str | Path]) -> dict[str, Any]:
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Training config YAML not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return dict(data or {})


_SCI_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _parse_override_value(raw_val: str) -> Any:
    """Parse a `--set key=val` value using YAML semantics plus YAML-1.1 scientific notation (e.g. `3e-4`)."""
    txt = raw_val.strip()
    if not txt:
        return ""
    val = yaml.safe_load(txt)
    if isinstance(val, str) and _SCI_FLOAT_RE.match(val):
        return float(val)
    return val


def apply_config_overrides(
    cfg_dict: Mapping[str, Any],
    overrides: Optional[Sequence[str]],
) -> tuple[dict[str, Any], list[tuple[str, Any, Any]]]:
    """Apply repeatable `--set key.path=val` overrides onto a deep copy of `cfg_dict`.

    By default, every path segment and leaf key must already exist in `cfg_dict` to catch typos
    immediately. Prefix a path with `+` (e.g. `+hardware.custom_flag=true`) to allow inserting
    a brand-new key.
    """
    out = copy.deepcopy(dict(cfg_dict))
    applied: list[tuple[str, Any, Any]] = []
    if not overrides:
        return out, applied

    for raw_expr in overrides:
        expr = str(raw_expr).strip()
        if not expr:
            continue
        if "=" not in expr:
            raise ValueError(f"Invalid --set override {raw_expr!r}: expected 'key.path=value'")
        raw_path, raw_val = expr.split("=", 1)
        path_str = raw_path.strip()
        allow_create = path_str.startswith("+")
        if allow_create:
            path_str = path_str[1:].strip()
        parts = [p.strip() for p in path_str.split(".") if p.strip()]
        if not parts:
            raise ValueError(f"Invalid --set override {raw_expr!r}: empty key path")

        cur: Any = out
        traversed: list[str] = []
        for seg in parts[:-1]:
            if not isinstance(cur, dict):
                raise ValueError(
                    f"Cannot traverse --set {path_str!r}: {'.'.join(traversed)!r} is not a mapping"
                )
            if seg not in cur:
                if allow_create:
                    cur[seg] = {}
                else:
                    avail = ", ".join(sorted(str(k) for k in cur.keys()))
                    raise KeyError(
                        f"Unknown config section {'.'.join(traversed + [seg])!r} in --set {raw_expr!r} "
                        f"(available keys: {avail}; prefix with '+' to create a new key)"
                    )
            cur = cur[seg]
            traversed.append(seg)

        leaf = parts[-1]
        if not isinstance(cur, dict):
            raise ValueError(
                f"Cannot set {path_str!r}: parent {'.'.join(traversed)!r} is not a mapping"
            )
        if leaf not in cur and not allow_create and bool(out):
            avail = ", ".join(sorted(str(k) for k in cur.keys()))
            raise KeyError(
                f"Unknown config key {path_str!r} in --set {raw_expr!r} "
                f"(available keys: {avail}; prefix with '+' to create a new key)"
            )
        old_val = cur.get(leaf, None)
        new_val = _parse_override_value(raw_val)
        cur[leaf] = new_val
        applied.append((path_str, old_val, new_val))
    return out, applied


def build_dcpa_config(preset: str, yaml_cfg: Mapping[str, Any]) -> DCPAConfig:
    """Resolve `DCPAConfig` from preset name and optional YAML overrides."""
    preset_key = preset.strip().lower()
    alias_map = {
        "48m": "proto_48m",
        "152m": "standard_152m",
        "210m": "moe_210m",
        "650m": "moe_650m",
        "1.8b": "moe_1_8b",
        "1_8b": "moe_1_8b",
        "2.6b": "moe_2_6b",
        "2_6b": "moe_2_6b",
        "3.5b": "moe_3_5b",
        "3_5b": "moe_3_5b",
    }
    canon_key = alias_map.get(preset_key, preset_key)
    factory_map: dict[str, Callable[[], DCPAConfig]] = {
        "proto_48m": DCPAConfig.proto_48m,
        "standard_152m": DCPAConfig.standard_152m,
        "moe_210m": DCPAConfig.moe_210m,
        "moe_650m": DCPAConfig.moe_650m,
        "moe_1_8b": DCPAConfig.moe_1_8b,
        "moe_2_6b": DCPAConfig.moe_2_6b,
        "moe_3_5b": DCPAConfig.moe_3_5b,
        "base": DCPAConfig.base,
        "tiny": DCPAConfig.tiny,
        "micro": DCPAConfig.micro,
    }
    if canon_key not in factory_map:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {sorted(set(factory_map) | set(alias_map))}")
    base_cfg = factory_map[canon_key]()
    merged = base_cfg.to_dict()

    presets_block = yaml_cfg.get("presets", {}) if isinstance(yaml_cfg, Mapping) else {}
    yaml_model = yaml_cfg.get("model", {}) if isinstance(yaml_cfg, Mapping) else {}
    # Apply YAML model defaults when using the YAML's own default preset
    if yaml_model and alias_map.get(str(yaml_model.get("preset", "")).lower(), str(yaml_model.get("preset", "")).lower()) == canon_key:
        for k, v in yaml_model.items():
            if k in merged:
                merged[k] = v
    for key_cand in (canon_key, preset_key):
        if isinstance(presets_block, Mapping) and key_cand in presets_block:
            for k, v in dict(presets_block[key_cand]).items():
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
        lid = rng.choice((0, 1, 3, 5))
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
        if lid in (3, 5):
            txt, _ = generate_dyck_sequence(rng, max_depth=6, min_bytes=min(64, doc_cap), rust_flavored=(lid == 5))
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
    if n_docs <= 0:
        return torch.ones((0,), dtype=torch.bool, device=loss_mask.device)
    non_fam_byte = (wug_target_mask.bool() | (~loss_mask.bool())).to(torch.int64)
    cs = torch.zeros((non_fam_byte.numel() + 1,), dtype=torch.int64, device=loss_mask.device)
    if non_fam_byte.numel() > 0:
        cs[1:] = torch.cumsum(non_fam_byte, dim=0)
    starts = cu_seqlens[:-1].to(torch.int64)
    ends = cu_seqlens[1:].to(torch.int64)
    return (cs[ends] - cs[starts]) == 0


def _check_active_grads_finite(active_params: Sequence[torch.nn.Parameter]) -> bool:
    grads = [p.grad for p in active_params if p.grad is not None]
    if not grads:
        return True
    norms = torch._foreach_norm(grads)
    return bool(torch.isfinite(torch.stack(norms)).all().item())


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
    """Deterministic cursor-addressable Ragged byte stream with multi-rank sharding & async prefetch."""

    def __init__(
        self,
        bin_path: Optional[Path],
        idx_path: Optional[Path],
        seed: int,
        cap_data_dir: Optional[Path] = None,
        stage3_cap_mix_ratio: float = 0.35,
        device: Optional[torch.device] = None,
        prefetch_depth: int = 0,
        rank: int = 0,
        world_size: int = 1,
        length_bucketing: bool = True,
        prefetch_workers: int = 2,
        cap_shard_weights: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.bin_path = bin_path if (bin_path is not None and bin_path.exists()) else None
        self.idx_path = idx_path if (idx_path is not None and idx_path.exists()) else None
        self.has_shards = self.bin_path is not None and self.idx_path is not None
        self.rank = max(0, int(rank))
        self.world_size = max(1, int(world_size))
        self.length_bucketing = bool(length_bucketing)
        self._ds: Optional[RaggedByteDataset] = None
        self._ds_key: Optional[tuple[Any, ...]] = None
        self._cap_shards: list[tuple[Path, Path]] = []
        self._cap_names: list[str] = []
        self._cap_datasets: list[Optional[RaggedByteDataset]] = []
        self._cap_keys: list[Optional[tuple[Any, ...]]] = []
        self.stage3_cap_mix_ratio = float(stage3_cap_mix_ratio)
        self._cap_shard_weights: dict[str, int] = (
            {str(k): max(1, int(v)) for k, v in cap_shard_weights.items()}
            if cap_shard_weights
            else {}
        )
        if cap_data_dir is not None and cap_data_dir.exists():
            for pref in ("cap_reasoning", "cap_instruct", "cap_code", "cap_toolcall", "cap_writing"):
                bp = cap_data_dir / f"{pref}_train.bin"
                ip = cap_data_dir / f"{pref}_train.idx"
                if bp.exists() and ip.exists():
                    self._cap_shards.append((bp, ip))
                    self._cap_names.append(pref)
                    self._cap_datasets.append(None)
                    self._cap_keys.append(None)
        self._base_b_idx: int = 0
        self._phys_base: int = 0
        self.cursor: dict[str, Any] = {
            "batch_idx": 0,
            "phys_offset": 0,
            "epoch": 0,
            "seed": int(seed),
            "consumed_bytes": 0,
        }
        self._corpus_bytes: int = int(self.bin_path.stat().st_size) if self.bin_path is not None else 1
        self._device: torch.device = device if device is not None else torch.device("cpu")
        self._ds_lock = threading.Lock()
        self._prefetch_depth = int(prefetch_depth) if (self.has_shards and self._device.type == "cuda") else 0
        self._prefetch_futures: dict[tuple[Any, ...], Future[dict[str, torch.Tensor]]] = {}
        n_workers = max(1, min(int(prefetch_workers), max(1, self._prefetch_depth)))
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix=f"dcpa_prefetch_r{self.rank}")
            if self._prefetch_depth > 0
            else None
        )

    def _ensure_cap_dataset(
        self,
        cap_idx: int,
        *,
        target_batch_bytes: int,
        max_doc_bytes: int,
        seed_val: int,
    ) -> RaggedByteDataset:
        key = (int(target_batch_bytes), int(max_doc_bytes), int(seed_val), self.length_bucketing, self.world_size)
        if self._cap_datasets[cap_idx] is not None and self._cap_keys[cap_idx] == key:
            return self._cap_datasets[cap_idx]  # type: ignore[return-value]
        if self._cap_datasets[cap_idx] is not None:
            self._cap_datasets[cap_idx].close()  # type: ignore[union-attr]
        bp, ip = self._cap_shards[cap_idx]
        ds = RaggedByteDataset(
            bin_path=bp,
            idx_path=ip,
            target_batch_bytes=int(target_batch_bytes),
            max_doc_bytes=int(max_doc_bytes),
            domain_weights=None,
            wug_ratio=0.0,
            num_batches=10_000_000,
            seed=int(seed_val) + (cap_idx + 1) * 4099,
            length_bucketing=self.length_bucketing,
            bucket_sync_world_size=self.world_size,
        )
        self._cap_datasets[cap_idx] = ds
        self._cap_keys[cap_idx] = key
        return ds

    def _ensure_dataset(
        self,
        *,
        target_batch_bytes: int,
        max_doc_bytes: int,
        domain_weights: Mapping[int, float],
        wug_ratio: float,
        seed_val: int,
        max_stage_tier: int = 0,
        anchor_doc_start: bool = False,
        wug_mode: str = "classic",
    ) -> Optional[RaggedByteDataset]:
        if not self.has_shards or self.bin_path is None or self.idx_path is None:
            return None
        dw_key = tuple(sorted((int(k), round(float(v), 6)) for k, v in domain_weights.items()))
        key = (
            int(target_batch_bytes),
            int(max_doc_bytes),
            dw_key,
            round(float(wug_ratio), 6),
            int(seed_val),
            self.length_bucketing,
            self.world_size,
            int(max_stage_tier),
            bool(anchor_doc_start),
            str(wug_mode),
        )
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
            seed=int(seed_val),
            length_bucketing=self.length_bucketing,
            bucket_sync_world_size=self.world_size,
            max_stage_tier=int(max_stage_tier),
            anchor_doc_start=bool(anchor_doc_start),
            wug_mode=str(wug_mode),
        )
        self._ds_key = key
        return self._ds

    @staticmethod
    def _make_req_key(
        b_idx: int,
        target_batch_bytes: int,
        stage_id: int,
        stage_half: int,
        max_doc_bytes: int,
        domain_weights: Mapping[int, float],
        wug_ratio: float,
        seed_val: int,
        max_stage_tier: int = 0,
        anchor_doc_start: bool = False,
        cap_mix_prob: float = -1.0,
        wug_mode: str = "classic",
        pin_output: bool = True,
    ) -> tuple[Any, ...]:
        dw_key = tuple(sorted((int(k), round(float(v), 6)) for k, v in domain_weights.items()))
        return (
            int(b_idx),
            int(target_batch_bytes),
            int(stage_id),
            int(stage_half),
            int(max_doc_bytes),
            dw_key,
            round(float(wug_ratio), 6),
            int(seed_val),
            int(max_stage_tier),
            bool(anchor_doc_start),
            round(float(cap_mix_prob), 4),
            str(wug_mode),
            bool(pin_output),
        )

    def _select_cap_shard_idx(self, logical_b: int, *, stage_id: int) -> int:
        n_cap = len(self._cap_shards)
        if n_cap <= 1:
            return 0
        if stage_id == 2:
            # Stage-2 late warm-in focuses on reasoning, writing & code skeletons before tool protocols
            early_idxs = [
                i for i, nm in enumerate(self._cap_names) if nm in ("cap_reasoning", "cap_writing", "cap_code", "cap_instruct")
            ]
            if early_idxs:
                return early_idxs[logical_b % len(early_idxs)]
        default_weights = (
            {
                "cap_reasoning": 35,
                "cap_writing": 35,
                "cap_code": 20,
                "cap_instruct": 5,
                "cap_toolcall": 5,
            }
            if "cap_code" in self._cap_names
            else {
                "cap_reasoning": 35,
                "cap_instruct": 30,
                "cap_toolcall": 20,
                "cap_writing": 15,
            }
        )
        pref_weights = self._cap_shard_weights if self._cap_shard_weights else default_weights
        slots: list[int] = []
        for idx_c, nm in enumerate(self._cap_names):
            slots.extend([idx_c] * pref_weights.get(nm, 20))
        return slots[(logical_b * 13) % len(slots)]

    def _fetch_one(
        self,
        b_idx: int,
        target_batch_bytes: int,
        stage_id: int,
        stage_half: int,
        max_doc_bytes: int,
        domain_weights: dict[int, float],
        wug_ratio: float,
        seed_val: int,
        max_stage_tier: int = 0,
        anchor_doc_start: bool = False,
        cap_mix_prob: float = -1.0,
        wug_mode: str = "classic",
        pin_output: bool = True,
    ) -> dict[str, torch.Tensor]:
        # Key shard choice AND length-bucket tier on logical `b_idx` so ALL ranks draw homogeneous
        # document lengths on the same step, while `phys_b_idx` gives each rank disjoint documents
        # AND advances monotonically across elastic world_size changes!
        logical_b = int(b_idx)
        phys_b_idx = self._phys_base + (logical_b - self._base_b_idx) * self.world_size + self.rank
        cap_prob = 0.0
        if len(self._cap_shards) > 0:
            if cap_mix_prob >= 0.0:
                cap_prob = float(cap_mix_prob)
            elif self.stage3_cap_mix_ratio > 0.0:
                if stage_id == 3:
                    cap_prob = self.stage3_cap_mix_ratio
                elif stage_id == 2 and stage_half >= 1:
                    cap_prob = min(0.15, 0.35 * self.stage3_cap_mix_ratio)
        use_cap = cap_prob > 0.0 and ((logical_b * 37) % 100) < int(round(cap_prob * 100.0))
        if use_cap:
            c_idx = self._select_cap_shard_idx(logical_b, stage_id=stage_id)
            with self._ds_lock:
                cap_ds = self._ensure_cap_dataset(
                    c_idx,
                    target_batch_bytes=target_batch_bytes,
                    max_doc_bytes=max_doc_bytes,
                    seed_val=seed_val,
                )
            batch = cap_ds.get_batch(phys_b_idx, sync_group=logical_b)
        else:
            with self._ds_lock:
                ds = self._ensure_dataset(
                    target_batch_bytes=target_batch_bytes,
                    max_doc_bytes=max_doc_bytes,
                    domain_weights=domain_weights,
                    wug_ratio=wug_ratio,
                    seed_val=seed_val,
                    max_stage_tier=max_stage_tier,
                    anchor_doc_start=anchor_doc_start,
                    wug_mode=wug_mode,
                )
            if ds is not None:
                batch = ds.get_batch(phys_b_idx, sync_group=logical_b)
            else:
                batch = _sample_synthetic_ragged_batch(
                    batch_idx=phys_b_idx,
                    seed=seed_val,
                    target_bytes=target_batch_bytes,
                    max_doc_bytes=max_doc_bytes,
                    wug_ratio=wug_ratio,
                )
        return pin_batch_if_cuda(batch, self._device) if pin_output else batch

    def _clear_prefetch(self) -> None:
        for fut in list(self._prefetch_futures.values()):
            fut.cancel()
        self._prefetch_futures.clear()

    def next_batch(
        self,
        *,
        target_batch_bytes: int,
        state: CurriculumStepState,
        pin_output: bool = True,
    ) -> dict[str, torch.Tensor]:
        b_idx = int(self.cursor["batch_idx"])
        seed_val = int(self.cursor["seed"])
        dw_copy = dict(state.domain_weights)
        stage_half = 1 if float(state.stage_progress) >= 0.50 else 0
        max_tier = int(state.max_stage_tier)
        anc_start = bool(state.anchor_doc_start)
        cap_p = float(state.cap_mix_prob)
        w_mode = str(state.wug_mode)

        if self._executor is not None and self._prefetch_depth > 0:
            req_key = self._make_req_key(
                b_idx,
                target_batch_bytes,
                state.stage_id,
                stage_half,
                state.max_doc_bytes,
                dw_copy,
                state.wug_ratio,
                seed_val,
                max_tier,
                anc_start,
                cap_p,
                w_mode,
                pin_output,
            )
            fut = self._prefetch_futures.pop(req_key, None)
            if fut is not None:
                batch = fut.result()
            else:
                self._clear_prefetch()
                batch = self._fetch_one(
                    b_idx,
                    target_batch_bytes,
                    state.stage_id,
                    stage_half,
                    state.max_doc_bytes,
                    dw_copy,
                    state.wug_ratio,
                    seed_val,
                    max_tier,
                    anc_start,
                    cap_p,
                    w_mode,
                    pin_output,
                )
            for off in range(1, self._prefetch_depth + 1):
                look_b = b_idx + off
                look_key = self._make_req_key(
                    look_b,
                    target_batch_bytes,
                    state.stage_id,
                    stage_half,
                    state.max_doc_bytes,
                    dw_copy,
                    state.wug_ratio,
                    seed_val,
                    max_tier,
                    anc_start,
                    cap_p,
                    w_mode,
                    pin_output,
                )
                if look_key not in self._prefetch_futures:
                    self._prefetch_futures[look_key] = self._executor.submit(
                        self._fetch_one,
                        look_b,
                        int(target_batch_bytes),
                        int(state.stage_id),
                        stage_half,
                        int(state.max_doc_bytes),
                        dict(dw_copy),
                        float(state.wug_ratio),
                        seed_val,
                        max_tier,
                        anc_start,
                        cap_p,
                        w_mode,
                        pin_output,
                    )
        else:
            batch = self._fetch_one(
                b_idx,
                target_batch_bytes,
                state.stage_id,
                stage_half,
                state.max_doc_bytes,
                dw_copy,
                state.wug_ratio,
                seed_val,
                max_tier,
                anc_start,
                cap_p,
                w_mode,
                pin_output,
            )

        next_b = b_idx + 1
        self.cursor["batch_idx"] = next_b
        self.cursor["phys_offset"] = self._phys_base + (next_b - self._base_b_idx) * self.world_size
        self.cursor["consumed_bytes"] = (
            int(self.cursor["consumed_bytes"]) + int(batch["x_bytes"].numel()) * self.world_size
        )
        self.cursor["epoch"] = int(self.cursor["consumed_bytes"]) // max(self._corpus_bytes, 1)
        return batch

    def restore_cursor(self, cursor: Optional[Mapping[str, Any]]) -> None:
        self._clear_prefetch()
        if not cursor:
            return
        restored_b = int(cursor.get("batch_idx", 0))
        restored_phys = int(cursor.get("phys_offset", restored_b * max(self.world_size, 1)))
        self._base_b_idx = restored_b
        self._phys_base = restored_phys
        self.cursor["batch_idx"] = restored_b
        self.cursor["phys_offset"] = restored_phys
        self.cursor["epoch"] = int(cursor.get("epoch", 0))
        self.cursor["seed"] = int(cursor.get("seed", self.cursor["seed"]))
        self.cursor["consumed_bytes"] = int(cursor.get("consumed_bytes", 0))

    def close(self) -> None:
        self._clear_prefetch()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        with self._ds_lock:
            if self._ds is not None:
                self._ds.close()
                self._ds = None
                self._ds_key = None
            for i, cds in enumerate(self._cap_datasets):
                if cds is not None:
                    cds.close()
                    self._cap_datasets[i] = None
                    self._cap_keys[i] = None


class _AsyncStreamCarryPrefetcher:
    """Single-worker sequential background prefetcher for stateful `StreamCarryPacker`.

    Overlaps CPU stream packing + page-locked `pin_memory()` with GPU forward/backward while
    snapshotting `stream.cursor` per packed batch so checkpoints, stage transitions, and Watchdog
    rollbacks remain 100% deterministic.
    """

    def __init__(
        self,
        packer: StreamCarryPacker,
        stream: _DynamicRaggedCursorStream,
        device: torch.device,
        *,
        prefetch_depth: int = 0,
    ) -> None:
        self.packer = packer
        self.stream = stream
        self.device = device
        self.prefetch_depth = max(0, int(prefetch_depth))
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"dcpa_carry_r{stream.rank}")
            if self.prefetch_depth > 0
            else None
        )
        self._pending: deque[tuple[int, int, int, Future[tuple[dict[str, torch.Tensor], dict[str, Any]]]]] = deque()
        self._committed_cursor: Optional[dict[str, Any]] = None
        self._cursor_ran_ahead: bool = False

    @property
    def committed_cursor(self) -> Optional[dict[str, Any]]:
        return dict(self._committed_cursor) if self._committed_cursor is not None else None

    def _produce_one(
        self,
        target_batch_bytes: int,
        state: CurriculumStepState,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        batch = self.packer.next_batch(
            lambda: self.stream.next_batch(
                target_batch_bytes=target_batch_bytes,
                state=state,
                pin_output=False,
            )
        )
        cursor_snap = dict(self.stream.cursor)
        batch = pin_batch_if_cuda(batch, self.device)
        return batch, cursor_snap

    def _schedule_ahead(
        self,
        *,
        next_step_idx: int,
        next_micro_idx: int,
        grad_accum_steps: int,
        total_steps: int,
        active_stage_id: int,
        target_batch_bytes: int,
        controller: CurriculumController,
    ) -> None:
        if self._executor is None or self.prefetch_depth <= 0:
            return
        if self._pending:
            last_step, last_micro, _, _ = self._pending[-1]
            s_cur, m_cur = last_step, last_micro + 1
            if m_cur >= grad_accum_steps:
                s_cur += 1
                m_cur = 0
        else:
            s_cur, m_cur = next_step_idx, next_micro_idx

        while len(self._pending) < self.prefetch_depth and s_cur < total_steps:
            sc_next = controller.get(s_cur)
            if sc_next.stage_id != active_stage_id or sc_next.model_stage not in ("B", "C"):
                break
            fut = self._executor.submit(self._produce_one, int(target_batch_bytes), sc_next)
            self._pending.append((s_cur, m_cur, int(sc_next.stage_id), fut))
            self._cursor_ran_ahead = True
            m_cur += 1
            if m_cur >= grad_accum_steps:
                s_cur += 1
                m_cur = 0

    def next_batch(
        self,
        *,
        step_idx: int,
        micro_idx: int,
        target_batch_bytes: int,
        state: CurriculumStepState,
        grad_accum_steps: int,
        total_steps: int,
        controller: CurriculumController,
    ) -> dict[str, torch.Tensor]:
        if self._executor is not None and self.prefetch_depth > 0:
            if self._pending:
                p_step, p_micro, p_stage, fut = self._pending.popleft()
                if p_step == step_idx and p_micro == micro_idx and p_stage == state.stage_id:
                    batch, cursor_snap = fut.result()
                else:
                    fut.result()
                    self._drain_and_realign()
                    batch, cursor_snap = self._produce_one(int(target_batch_bytes), state)
            else:
                batch, cursor_snap = self._produce_one(int(target_batch_bytes), state)
            self._committed_cursor = cursor_snap
            if not self._pending:
                self._cursor_ran_ahead = False

            next_s, next_m = step_idx, micro_idx + 1
            if next_m >= grad_accum_steps:
                next_s += 1
                next_m = 0
            self._schedule_ahead(
                next_step_idx=next_s,
                next_micro_idx=next_m,
                grad_accum_steps=grad_accum_steps,
                total_steps=total_steps,
                active_stage_id=int(state.stage_id),
                target_batch_bytes=int(target_batch_bytes),
                controller=controller,
            )
            return batch

        batch, cursor_snap = self._produce_one(int(target_batch_bytes), state)
        self._committed_cursor = cursor_snap
        self._cursor_ran_ahead = False
        return batch

    def _drain_pending(self) -> None:
        while self._pending:
            _, _, _, fut = self._pending.popleft()
            try:
                fut.result()
            except Exception:
                pass

    def _drain_and_realign(self) -> None:
        self._drain_pending()
        if self._cursor_ran_ahead and self._committed_cursor is not None:
            self.stream.restore_cursor(self._committed_cursor)
        self._cursor_ran_ahead = False

    def reset(self, *, restore_stream_cursor: bool = True) -> None:
        """Drain any in-flight worker future, reset packer, and restore stream cursor if ran ahead."""
        self._drain_pending()
        if restore_stream_cursor and self._cursor_ran_ahead and self._committed_cursor is not None:
            self.stream.restore_cursor(self._committed_cursor)
        self._cursor_ran_ahead = False
        self._committed_cursor = None
        self.packer.reset()

    def close(self) -> None:
        self._drain_pending()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


def _stage_microbatch_to_device(
    batch: dict[str, torch.Tensor],
    stage_id: int,
    device: torch.device,
    h2d_stream: Optional[Any],
) -> tuple[dict[str, Optional[torch.Tensor]], Optional[bool], Optional[Any]]:
    """Transfer a micro-batch to `device`, using dedicated CUDA `h2d_stream` when available."""
    # Check Stage-2 loss_mask homogeneity on CPU before async H2D DMA (zero CUDA sync!)
    loss_mask_all_ones: Optional[bool] = bool(batch["loss_mask"].all()) if stage_id == 2 else None
    keys = (
        "x_bytes",
        "cu_seqlens",
        "doc_lang_ids",
        "doc_style_ids",
        "doc_ponder_tax_scale",
        "byte_lang_ids",
        "loss_mask",
        "wug_target_mask",
    )
    if h2d_stream is not None:
        with torch.cuda.stream(h2d_stream):
            staged = {k: to_device_async(batch.get(k), device) for k in keys}
            ev = torch.cuda.Event()
            ev.record(h2d_stream)
        return staged, loss_mask_all_ones, ev
    staged = {k: to_device_async(batch.get(k), device) for k in keys}
    return staged, loss_mask_all_ones, None


def _await_staged_microbatch(
    staged: dict[str, Optional[torch.Tensor]],
    h2d_event: Optional[Any],
    device: torch.device,
) -> None:
    """Synchronize compute stream on `h2d_event` and protect async-copied tensors via `record_stream`."""
    if h2d_event is not None and device.type == "cuda":
        cur_stream = torch.cuda.current_stream(device)
        cur_stream.wait_event(h2d_event)
        for t in staged.values():
            if t is not None:
                t.record_stream(cur_stream)


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
    cap_data_dir: Optional[str | Path] = None,
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
    gpu_profile: Optional[str] = None,
    compile_mode: Optional[str] = None,
    allow_cpu_compile: bool = False,
    deterministic: bool = False,
    seed: int = 1337,
    max_consecutive_meltdowns: int = 3,
    watchdog_warmup_steps: Optional[int] = None,
    milestone_steps: Optional[tuple[int, ...]] = None,
    export_final: bool = True,
    fixed_global_batch: bool = False,
    auto_lr_scale: bool = True,
    length_bucketing: bool = True,
    bf16_grad_comm: bool = True,
    stream_carry: Optional[bool] = None,
    stream_chunk_bytes: Optional[int] = None,
    bcg_cudagraphs: Optional[bool] = None,
    bcg_ckpt_watermark_gb: Optional[float] = None,
    bcg_chunk_steps: Optional[int] = None,
    bcg_tbptt_window: Optional[int] = None,
    lr_scale: Optional[float] = None,
    config_overrides: Optional[Sequence[str]] = None,
    use_wandb: Optional[bool] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    wandb_tags: Optional[str] = None,
    wandb_log_every: Optional[int] = None,
    clock: Callable[[], float] = time.monotonic,
    pre_step_hook: Optional[Callable[[int, DCPAModel, torch.optim.Optimizer, dict[str, float]], None]] = None,
    step_hook: Optional[Callable[[int, DCPAModel, dict[str, float]], None]] = None,
    stage_change_hook: Optional[Callable[[int, int, DCPAModel], None]] = None,
) -> TrainRunSummary:
    """Execute the full DCPA 4-stage self-healing developmental training lifecycle (1..N GPUs)."""
    dist_ctx, dist_owned = init_distributed_context(preferred_device=device_str)
    is_primary = dist_ctx.is_primary

    def _r0_print(*args: Any, **kwargs: Any) -> None:
        if is_primary:
            print(*args, **kwargs)

    yaml_cfg_raw = load_yaml_config(config_path) if config_path is not None else {}
    yaml_cfg, applied_overrides = apply_config_overrides(yaml_cfg_raw, config_overrides)
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
        cfg_bounds = curr_cfg.get("stage_boundaries")
        if cfg_bounds is not None and len(cfg_bounds) == 5 and eff_total_steps == int(cfg_bounds[-1]):
            eff_boundaries = tuple(int(x) for x in cfg_bounds)
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
    if cap_data_dir is not None:
        assert_not_drive_path(cap_data_dir, role="--cap-data-dir")

    eff_ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else Path(ck_cfg.get("ckpt_dir", local_cache / "checkpoints"))
    eff_logs_dir = Path(logs_dir) if logs_dir is not None else Path(vp_cfg.get("logs_dir", local_cache / "logs"))
    eff_export_dir = Path(export_dir) if export_dir is not None else Path(ck_cfg.get("export_dir", local_cache / "export"))
    if is_primary:
        eff_ckpt_dir.mkdir(parents=True, exist_ok=True)
        eff_logs_dir.mkdir(parents=True, exist_ok=True)
        eff_export_dir.mkdir(parents=True, exist_ok=True)
        # Restore append-only Drive logs onto fresh NVMe before appending so Colab reconnects
        # preserve full continuous training telemetry across sessions.
        restore_drive_log_file(drive_logs_dir, eff_logs_dir, "metrics.jsonl")
        restore_drive_log_file(drive_logs_dir, eff_logs_dir, "watchdog_events.jsonl")
        restore_drive_log_file(drive_logs_dir, eff_logs_dir, "wandb_run.json")
    dist_barrier(dist_ctx)

    run_log: Any = (
        RunLogTee(
            local_logs_dir=eff_logs_dir,
            drive_logs_dir=drive_logs_dir,
            sync_interval_seconds=max(30.0, float(heartbeat_seconds)),
            clock=clock,
        )
        if is_primary
        else contextlib.nullcontext()
    )
    run_log.__enter__()
    _exit_exc: tuple[Any, Any, Any] = (None, None, None)
    stream: Optional[_DynamicRaggedCursorStream] = None
    wb_tracker: Optional[WandbTracker] = None
    try:
        if is_primary and hasattr(run_log, "local_log_path"):
            _r0_print(
                f"[Logs] Local console log: {run_log.local_log_path}"
                + (
                    f" | Drive mirror: {Path(drive_logs_dir) / run_log.log_filename} & train_latest.log"
                    if drive_logs_dir is not None
                    else ""
                ),
                flush=True,
            )
        for ov_path, ov_old, ov_new in applied_overrides:
            _r0_print(f"[Override] --set {ov_path}: {ov_old!r} -> {ov_new!r}", flush=True)
        # Hardware detection & GPU profile resolution (auto-detects T4/L4/A100/H100/G4 or honors `--gpu-profile`)
        eff_stream_carry_pref = bool(hw_cfg.get("stream_carry", True)) if stream_carry is None else bool(stream_carry)
        yaml_per_gpu_bytes = hw_cfg.get("per_gpu_tokens_bytes_per_step")
        yaml_target_bytes = hw_cfg.get("target_tokens_bytes_per_step")
        cuda_target_active = (
            device_str == "cuda"
            or device_str.startswith("cuda:")
            or (device_str == "auto" and torch.cuda.is_available())
        )
        eff_fixed_global = bool(fixed_global_batch or hw_cfg.get("fixed_global_batch", False))
        explicit_target_bytes: Optional[int]
        if target_tokens_bytes_per_step is not None:
            explicit_target_bytes = int(target_tokens_bytes_per_step)
        elif yaml_per_gpu_bytes is not None and cuda_target_active and not eff_fixed_global:
            explicit_target_bytes = int(yaml_per_gpu_bytes)
        elif (
            yaml_target_bytes is not None
            and int(yaml_target_bytes) != DEFAULT_PROD_STEP_BYTES
            and cuda_target_active
        ):
            explicit_target_bytes = int(yaml_target_bytes)
        else:
            explicit_target_bytes = None

        raw_profile = detect_hardware_profile(
            preferred=device_str,
            deterministic=deterministic,
            target_tokens_bytes_per_step=explicit_target_bytes,
        )
        gpu_spec, gpu_profile_explicit = resolve_gpu_profile(
            gpu_profile,
            raw_profile,
            stream_carry=eff_stream_carry_pref,
        )
        device = raw_profile.device
        if (
            gpu_profile_explicit
            and device.type == "cuda"
            and raw_profile.vram_gb > 0.0
            and raw_profile.vram_gb + 2.0 < gpu_spec.min_recommended_vram_gb
        ):
            _r0_print(
                f"[Hardware][WARN] --gpu-profile={gpu_spec.name} expects >={gpu_spec.min_recommended_vram_gb:.0f} GB VRAM, "
                f"but detected {raw_profile.device_name} ({raw_profile.vram_gb:.1f} GB). "
                "Clamping BCG checkpoint watermark safely to avoid OOM.",
                flush=True,
            )

        # Effective target bytes: CLI > explicit --gpu-profile > YAML per_gpu/target > profile default
        if target_tokens_bytes_per_step is not None:
            eff_target_bytes = int(target_tokens_bytes_per_step)
        elif gpu_profile_explicit and device.type == "cuda":
            eff_target_bytes = int(gpu_spec.target_tokens_bytes_per_step)
        elif explicit_target_bytes is not None:
            eff_target_bytes = int(explicit_target_bytes)
        elif device.type == "cuda":
            eff_target_bytes = int(gpu_spec.target_tokens_bytes_per_step)
        else:
            eff_target_bytes = int(raw_profile.target_tokens_bytes_per_step)

        profile_micro_bytes = int(gpu_spec.micro_batch_bytes) if device.type == "cuda" else int(raw_profile.micro_batch_bytes)
        profile_tier = gpu_spec.tier if (device.type == "cuda" or gpu_profile_explicit) else raw_profile.tier
        allow_tf32 = False if deterministic else bool(hw_cfg.get("allow_tf32", True))
        configure_cuda_tf32(device, allow=allow_tf32)
        reset_peak_memory_stats(device)

        use_bf16 = bool(raw_profile.use_bf16 and hw_cfg.get("bf16_autocast", True) and not deterministic)
        if hw_cfg.get("use_grad_scaler", False):
            raise ValueError("Red Line #1 violation: GradScaler is strictly forbidden with bfloat16!")

        if dist_ctx.is_distributed:
            if (
                eff_fixed_global
                or (
                    target_tokens_bytes_per_step is not None
                    and eff_target_bytes >= profile_micro_bytes * dist_ctx.world_size
                )
                or (
                    yaml_per_gpu_bytes is None
                    and eff_target_bytes >= profile_micro_bytes * dist_ctx.world_size
                )
            ):
                # Partition target global byte budget evenly across ranks (zero GPU starvation when >= W * micro!)
                per_rank_target = max(64, math.ceil(eff_target_bytes / dist_ctx.world_size))
            else:
                # Per-GPU throughput scaling: each GPU runs its full per-GPU step budget (131 KB x W GPUs!)
                per_rank_target = max(profile_micro_bytes, eff_target_bytes)
        else:
            per_rank_target = eff_target_bytes

        eff_micro_bytes = (
            int(micro_batch_bytes)
            if micro_batch_bytes is not None
            else min(profile_micro_bytes, per_rank_target)
        )
        eff_micro_bytes = max(64, eff_micro_bytes)
        grad_accum_steps = max(1, math.ceil(per_rank_target / eff_micro_bytes))
        per_rank_step_bytes = eff_micro_bytes * grad_accum_steps
        actual_step_bytes = per_rank_step_bytes * dist_ctx.world_size
        profile = HardwareProfile(
            device_type=raw_profile.device_type,
            device_name=raw_profile.device_name,
            vram_gb=raw_profile.vram_gb,
            tier=profile_tier,
            micro_batch_bytes=eff_micro_bytes,
            grad_accum_steps=grad_accum_steps,
            target_tokens_bytes_per_step=per_rank_target,
            use_bf16=use_bf16,
            device_index=raw_profile.device_index,
        )

        # Bidirectional sqrt-batch LR scaling across 1..N GPUs relative to `lr_reference_bytes`
        eff_auto_lr = bool(auto_lr_scale and hw_cfg.get("auto_lr_scale_multi_gpu", True) and not eff_fixed_global)
        lr_ref_cfg = opt_cfg.get("lr_reference_bytes")
        lr_ref_bytes = (
            int(lr_ref_cfg)
            if (lr_ref_cfg is not None and device.type == "cuda" and target_tokens_bytes_per_step is None)
            else int(eff_target_bytes)
        )
        lr_batch_scale = 1.0
        if eff_auto_lr and lr_ref_bytes > 0 and (
            (dist_ctx.is_distributed and actual_step_bytes != lr_ref_bytes)
            or (lr_ref_cfg is not None and device.type == "cuda" and actual_step_bytes != lr_ref_bytes)
        ):
            lr_batch_scale = min(
                2.5,
                max(0.45, math.sqrt(float(actual_step_bytes) / float(max(lr_ref_bytes, 1)))),
            )
            scaled_groups: dict[str, dict[str, Any]] = {}
            base_groups = dict(opt_cfg.get("groups", {}))
            for g_name, g_vals in base_groups.items():
                gd = dict(g_vals)
                if "lr" in gd:
                    gd["lr"] = float(gd["lr"]) * lr_batch_scale
                scaled_groups[g_name] = gd
            if scaled_groups:
                opt_cfg = dict(opt_cfg)
                opt_cfg["groups"] = scaled_groups

        seed_everything(seed, device=device)
        cfg = build_dcpa_config(eff_preset, yaml_cfg)
        # Scale matrix/norm LR smoothly by sqrt(yaml_d_concept / cfg.d_concept) when CLI --preset changes width
        yaml_d_concept = int(yaml_cfg.get("model", {}).get("d_concept", cfg.d_concept) or cfg.d_concept)
        lr_width_scale = 1.0
        if yaml_d_concept > 0 and cfg.d_concept > 0 and yaml_d_concept != cfg.d_concept and cfg.d_concept >= 512:
            lr_width_scale = min(1.45, max(0.75, math.sqrt(float(yaml_d_concept) / float(cfg.d_concept))))
            w_groups: dict[str, dict[str, Any]] = {}
            for g_name, g_vals in dict(opt_cfg.get("groups", {})).items():
                gd = dict(g_vals)
                if "lr" in gd and g_name != "group_gates_thresholds":
                    gd["lr"] = float(gd["lr"]) * lr_width_scale
                w_groups[g_name] = gd
            if w_groups:
                opt_cfg = dict(opt_cfg)
                opt_cfg["groups"] = w_groups
        if lr_batch_scale != 1.0:
            cfg.lr_micro *= lr_batch_scale
            cfg.lr_concept *= lr_batch_scale
            cfg.lr_scalar_gate *= lr_batch_scale
        eff_user_lr_scale = float(lr_scale) if lr_scale is not None else 1.0
        if eff_user_lr_scale <= 0.0:
            raise ValueError(f"--lr-scale must be > 0, got {eff_user_lr_scale}")
        if eff_user_lr_scale != 1.0:
            u_groups: dict[str, dict[str, Any]] = {}
            for g_name, g_vals in dict(opt_cfg.get("groups", {})).items():
                gd = dict(g_vals)
                if "lr" in gd:
                    gd["lr"] = float(gd["lr"]) * eff_user_lr_scale
                u_groups[g_name] = gd
            if u_groups:
                opt_cfg = dict(opt_cfg)
                opt_cfg["groups"] = u_groups
            cfg.lr_micro *= eff_user_lr_scale
            cfg.lr_concept *= eff_user_lr_scale
            cfg.lr_scalar_gate *= eff_user_lr_scale
        model = DCPAModel(cfg).to(device)
        broadcast_model_state(model, dist_ctx)
        if dist_ctx.is_distributed:
            # Decorrelate per-rank Gumbel exploration noise while keeping model weights bit-identical!
            reseed_all_rngs(seed + dist_ctx.rank * 104729, device=device)

        if device.type == "cuda":
            # GPU-profile-aware VRAM watermark (enables zero-recompute + A2 CUDA Graphs on L4/4090/5090/A100/H100/G4,
            # while always clamping safely against actual physical VRAM so a mismatched --gpu-profile cannot OOM):
            if profile.vram_gb >= 88.0:
                safe_cap_wm = round(0.78 * profile.vram_gb, 1)
            elif profile.vram_gb >= 75.0:
                safe_cap_wm = round(0.76 * profile.vram_gb, 1)
            elif profile.vram_gb >= 20.0:
                safe_cap_wm = round(0.65 * profile.vram_gb, 1)
            else:
                safe_cap_wm = 0.0
            if bcg_ckpt_watermark_gb is not None:
                req_wm = max(0.0, float(bcg_ckpt_watermark_gb))
                if profile.vram_gb > 0.0 and req_wm > 0.85 * profile.vram_gb:
                    _r0_print(
                        f"[Hardware][WARN] --bcg-ckpt-watermark-gb={req_wm:.1f} exceeds 85% of detected "
                        f"{profile.vram_gb:.1f} GB VRAM; OOM risk is elevated.",
                        flush=True,
                    )
                model.bcg_checkpoint_vram_watermark_gb = req_wm
            else:
                spec_wm = float(gpu_spec.bcg_ckpt_watermark_gb)
                yaml_wm = float(hw_cfg.get("bcg_checkpoint_vram_watermark_gb", spec_wm))
                if spec_wm <= 0.0 or safe_cap_wm <= 0.0:
                    model.bcg_checkpoint_vram_watermark_gb = 0.0
                elif gpu_profile_explicit:
                    model.bcg_checkpoint_vram_watermark_gb = min(spec_wm, safe_cap_wm)
                else:
                    eff_wm = min(yaml_wm, spec_wm) if yaml_wm > 0.0 else spec_wm
                    model.bcg_checkpoint_vram_watermark_gb = min(eff_wm, safe_cap_wm)

        yaml_bcg_chunk = int(
            bcg_chunk_steps if bcg_chunk_steps is not None else hw_cfg.get("bcg_chunk_steps", cfg.clause_chunk_size)
        )
        model.bcg_chunk_steps = max(
            1,
            yaml_bcg_chunk if (device.type == "cuda" or bcg_chunk_steps is not None) else min(yaml_bcg_chunk, 2),
        )
        model.bcg_tbptt_window = max(
            0,
            int(bcg_tbptt_window if bcg_tbptt_window is not None else hw_cfg.get("bcg_tbptt_window", 0)),
        )
        raw_tbptt_replay = hw_cfg.get("bcg_tbptt_replay", False)
        if device.type == "cuda" and (
            gpu_profile_explicit or (isinstance(raw_tbptt_replay, str) and raw_tbptt_replay.strip().lower() == "auto")
        ):
            model.bcg_tbptt_replay = bool(gpu_spec.bcg_tbptt_replay and model.bcg_tbptt_window > 0)
        else:
            model.bcg_tbptt_replay = bool(raw_tbptt_replay) and model.bcg_tbptt_window > 0
        optimizer = create_dcpa_optimizer(model, optimizer_cfg=opt_cfg)
        grad_clip_norm = float(opt_cfg.get("grad_clip_norm", cfg.grad_clip_norm))

        _r0_print(describe_hardware_profile(profile), flush=True)
        if dist_ctx.is_distributed:
            _r0_print(
                f"[Dist] world_size={dist_ctx.world_size} | backend={dist_ctx.backend} | "
                f"per_gpu_bytes={per_rank_step_bytes:,} | global_bytes_per_step={actual_step_bytes:,} | "
                f"lr_batch_scale={lr_batch_scale:.3f}x | bf16_grad_comm={bool(use_bf16 and bf16_grad_comm)}",
                flush=True,
            )
        _r0_print(
            f"[Plan] preset={eff_preset} | gpu_profile={gpu_spec.name} ({'explicit' if gpu_profile_explicit else 'auto'}) | "
            f"total_steps={eff_total_steps:,} | "
            f"boundaries={list(eff_boundaries)} | micro_batch_bytes={eff_micro_bytes:,} | "
            f"grad_accum_steps={grad_accum_steps} | effective_bytes_per_step={actual_step_bytes:,} | "
            f"bf16={use_bf16} | grad_scaler=False | "
            f"bcg_ckpt_watermark_gb={model.bcg_checkpoint_vram_watermark_gb:.1f} | "
            f"bcg_chunk={model.bcg_chunk_steps} | "
            f"tbptt_w={model.bcg_tbptt_window} (replay={model.bcg_tbptt_replay})",
            flush=True,
        )

        # Selective static-shape torch.compile on Hotspot A (BCG step) & Hotspot B (SGSC/PMD scans) ONLY
        req_compile_mode = (
            compile_mode
            if compile_mode is not None
            else str(hw_cfg.get("compile_mode", "none"))
        )
        if bcg_cudagraphs is not None:
            eff_bcg_cudagraphs = bool(bcg_cudagraphs)
        elif device.type == "cuda":
            eff_bcg_cudagraphs = bool(gpu_spec.bcg_cudagraphs and hw_cfg.get("bcg_cudagraphs", True))
        else:
            eff_bcg_cudagraphs = bool(hw_cfg.get("bcg_cudagraphs", True))
        compile_report = configure_model_compile(
            model,
            compile_mode=req_compile_mode,
            device=device,
            doc_bucket_size=int(hw_cfg.get("compile_doc_bucket_size", 16)),
            cache_size_limit=int(hw_cfg.get("compile_cache_size_limit", 128)),
            allow_cpu_compile=allow_cpu_compile,
            cache_dir=local_cache / "compile_cache",
            bcg_cudagraphs=eff_bcg_cudagraphs,
        )
        _r0_print(
            f"[Compile] requested={compile_report.requested_mode} -> effective={compile_report.effective_mode} "
            f"| backend={compile_report.backend} | hotspots={list(compile_report.compiled_hotspots)} "
            f"| doc_bucket={compile_report.doc_bucket_size} | cache_limit={compile_report.cache_size_limit}"
            + (f" ({compile_report.fallback_reason})" if compile_report.fallback_reason else ""),
            flush=True,
        )

        controller = CurriculumController(curriculum_cfg=curr_cfg, stage_boundaries=eff_boundaries)

        if watchdog_warmup_steps is not None:
            wd_cfg["rule_bcd_warmup_steps"] = int(watchdog_warmup_steps)
        wd_cfg["max_consecutive_rollbacks"] = int(max_consecutive_meltdowns)
        watchdog = SelfHealingWatchdog(
            watchdog_cfg=wd_cfg,
            base_seed=seed,
            events_log_path=eff_logs_dir / "watchdog_events.jsonl",
            enable_event_logging=is_primary,
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
        if drive_sync_dir is not None and is_primary:
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

        # Hot NVMe Ragged byte stream (+ optional Stage 3 cap_reasoning & cap_writing 35% mix)
        train_bin = (Path(data_dir) / "train.bin") if data_dir is not None else None
        train_idx = (Path(data_dir) / "train.idx") if data_dir is not None else None
        val_bin = (Path(data_dir) / "val.bin") if data_dir is not None else None
        val_idx = (Path(data_dir) / "val.idx") if data_dir is not None else None
        s3_cap_ratio = float(curr_cfg.get("stage3", {}).get("cap_mix_ratio", 0.35))
        cap_shard_weights_cfg = curr_cfg.get("cap_shard_weights")
        cap_dir_path = Path(cap_data_dir) if cap_data_dir is not None else None
        eff_prefetch_depth = int(hw_cfg.get("prefetch_depth", 2)) if device.type == "cuda" else 0
        eff_prefetch_workers = int(hw_cfg.get("prefetch_workers", 2))
        eff_len_bucketing = bool(length_bucketing and hw_cfg.get("length_bucketing", True))
        stream = _DynamicRaggedCursorStream(
            train_bin,
            train_idx,
            seed=seed,
            cap_data_dir=cap_dir_path,
            stage3_cap_mix_ratio=s3_cap_ratio,
            device=device,
            prefetch_depth=eff_prefetch_depth,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            length_bucketing=eff_len_bucketing,
            prefetch_workers=eff_prefetch_workers,
            cap_shard_weights=cap_shard_weights_cfg if isinstance(cap_shard_weights_cfg, Mapping) else None,
        )
        if stream.has_shards:
            _r0_print(
                f"[Data] Opened NVMe Ragged UTF-8 stream: {train_bin} "
                f"(length_bucketing={eff_len_bucketing}, prefetch_depth={eff_prefetch_depth})",
                flush=True,
            )
        elif eff_total_steps >= 1000:
            raise FileNotFoundError(
                f"Production training (total_steps={eff_total_steps:,}) requires real corpus shards at "
                f"{train_bin} and {train_idx}! Restore corpus onto local NVMe first — "
                "refusing to fall back to synthetic smoke data and overwrite checkpoints."
            )
        if stream._cap_shards:
            _r0_print(
                f"[Data] Stage 3 capability mix active ({s3_cap_ratio*100:.0f}%): "
                f"{[p[0].name for p in stream._cap_shards]}",
                flush=True,
            )

        # A1: cross-step stateful stream carry for Stages B/C (serial BCG depth ~chunk/rho instead of ~doc/rho)
        eff_stream_carry = bool(hw_cfg.get("stream_carry", True)) if stream_carry is None else bool(stream_carry)
        eff_stream_chunk = int(
            stream_chunk_bytes if stream_chunk_bytes is not None else hw_cfg.get("stream_chunk_bytes", 512)
        )
        carry_packer: Optional[StreamCarryPacker] = None
        carry_prefetcher: Optional[_AsyncStreamCarryPrefetcher] = None
        carry_bank: Optional[StreamCarryBank] = None
        carry_active_stage: Optional[int] = None
        if eff_stream_carry:
            eff_stream_chunk = max(64, min(eff_stream_chunk, eff_micro_bytes))
            raw_rows = max(1, eff_micro_bytes // eff_stream_chunk)
            entry_cap = 1 << (raw_rows.bit_length() - 1)  # static power-of-two BCG batch width
            n_carry_streams = max(1, entry_cap - entry_cap // 8)  # leave 1/8 headroom for same-step refills
            carry_packer = StreamCarryPacker(
                n_carry_streams,
                eff_stream_chunk,
                min_piece_bytes=int(hw_cfg.get("stream_min_piece_bytes", 64)),
                max_entries=entry_cap,
            )
            eff_carry_prefetch = max(2, eff_prefetch_depth) if device.type == "cuda" else 0
            carry_prefetcher = _AsyncStreamCarryPrefetcher(
                carry_packer,
                stream,
                device,
                prefetch_depth=eff_carry_prefetch,
            )
            carry_bank = StreamCarryBank.from_config(cfg, n_carry_streams, device)
            _r0_print(
                f"[StreamCarry] enabled for Stages B/C | streams={n_carry_streams} | chunk={eff_stream_chunk}B | "
                f"entry_cap={entry_cap} | prefetch={eff_carry_prefetch} | bcg_cudagraphs={eff_bcg_cudagraphs}",
                flush=True,
            )

        def _reset_stream_carry() -> None:
            nonlocal carry_active_stage
            if carry_prefetcher is not None:
                carry_prefetcher.reset(restore_stream_cursor=True)
            elif carry_packer is not None:
                carry_packer.reset()
            if carry_bank is not None:
                carry_bank.reset()
            carry_active_stage = None

        h2d_stream: Optional[Any] = (
            torch.cuda.Stream(device=device) if (device.type == "cuda" and torch.cuda.is_available()) else None
        )
        async_ckpt: Optional[AsyncCheckpointWriter] = (
            AsyncCheckpointWriter(
                device,
                max_pinned_mirror_gb=float(ck_cfg.get("async_pinned_mirror_gb", 8.0)),
                enabled=bool(ck_cfg.get("async_checkpoint", True)),
            )
            if is_primary
            else None
        )

        # Resume handling (auto: Colab A/B Ping-Pong first if configured, else local CheckpointManager)
        start_step = 0
        resumed_from_step = 0
        resumed_stage_id = -1
        fallback_resume_used = False
        best_val_metric = float("inf")
        resumed_wandb_id: Optional[str] = None

        if resume and resume.lower() != "none":
            if resume.lower() == "auto":
                loaded_ckpt: Optional[dict[str, Any]] = None
                if sync_mgr is not None:
                    res = sync_mgr.load_latest_auto(map_location="cpu")
                    if res is not None:
                        loaded_ckpt = res.checkpoint
                        fallback_resume_used = res.fallback_triggered
                        if dist_ctx.is_distributed:
                            ckpt_mgr.save_healthy(loaded_ckpt)
                dist_barrier(dist_ctx)
                if loaded_ckpt is None:
                    latest_local = ckpt_mgr.find_latest_checkpoint()
                    if latest_local is not None:
                        loaded_ckpt = ckpt_mgr.load_checkpoint(latest_local, map_location="cpu")
                        _r0_print(f"[Resume] Loaded local checkpoint: {latest_local}", flush=True)
                if loaded_ckpt is not None:
                    start_step, restored_cursor = CheckpointManager.restore_into(
                        loaded_ckpt,
                        model=model,
                        optimizer=optimizer,
                        controller=controller,
                        watchdog=watchdog,
                        preserve_watchdog_repairs=False,
                        restore_rng=not dist_ctx.is_distributed,
                    )
                    resumed_stage_id = int(loaded_ckpt.get("stage_id", -1))
                    best_val_metric = float(loaded_ckpt.get("best_val_metric", loaded_ckpt.get("best_loss", float("inf"))))
                    if loaded_ckpt.get("wandb_run_id"):
                        resumed_wandb_id = str(loaded_ckpt["wandb_run_id"])
                    del loaded_ckpt
                    broadcast_model_state(model, dist_ctx)
                    stream.restore_cursor(restored_cursor)
                    if dist_ctx.is_distributed:
                        reseed_all_rngs(int(stream.cursor["seed"]) + dist_ctx.rank * 104729, device=device)
                    resumed_from_step = start_step
                    ckpt_mgr.last_healthy_step = start_step
                    wall_trigger.mark_saved(start_step)
            else:
                explicit = Path(resume)
                loaded_ckpt = ckpt_mgr.load_checkpoint(explicit, map_location="cpu")
                start_step, restored_cursor = CheckpointManager.restore_into(
                    loaded_ckpt,
                    model=model,
                    optimizer=optimizer,
                    controller=controller,
                    watchdog=watchdog,
                    preserve_watchdog_repairs=False,
                    restore_rng=not dist_ctx.is_distributed,
                )
                resumed_stage_id = int(loaded_ckpt.get("stage_id", -1))
                best_val_metric = float(loaded_ckpt.get("best_val_metric", loaded_ckpt.get("best_loss", float("inf"))))
                if loaded_ckpt.get("wandb_run_id"):
                    resumed_wandb_id = str(loaded_ckpt["wandb_run_id"])
                del loaded_ckpt
                broadcast_model_state(model, dist_ctx)
                stream.restore_cursor(restored_cursor)
                if dist_ctx.is_distributed:
                    reseed_all_rngs(int(stream.cursor["seed"]) + dist_ctx.rank * 104729, device=device)
                resumed_from_step = start_step
                ckpt_mgr.last_healthy_step = start_step
                wall_trigger.mark_saved(start_step)
                _r0_print(f"[Resume] Restored explicit checkpoint {explicit} at step={start_step}", flush=True)

        wb_settings = resolve_wandb_settings(
            yaml_wandb_cfg=yaml_cfg.get("wandb") if isinstance(yaml_cfg.get("wandb"), Mapping) else None,
            cli_enabled=use_wandb,
            cli_project=wandb_project,
            cli_entity=wandb_entity,
            cli_run_name=wandb_run_name,
            cli_mode=wandb_mode,
            cli_tags=wandb_tags,
            cli_log_every=wandb_log_every,
        )
        wb_tracker = WandbTracker(
            enabled=bool(wb_settings["enabled"]),
            is_primary=is_primary,
            project=str(wb_settings["project"]),
            entity=wb_settings["entity"],
            run_name=wb_settings["run_name"] or f"dcpa-turbo-{eff_preset}",
            mode=str(wb_settings["mode"]),
            tags=list(wb_settings["tags"]),
            log_every_steps=int(wb_settings["log_every_steps"]),
            logs_dir=eff_logs_dir,
            drive_logs_dir=drive_logs_dir,
            config_payload={
                "preset": eff_preset,
                "total_steps": eff_total_steps,
                "stage_boundaries": list(eff_boundaries),
                "hardware_tier": profile.tier,
                "world_size": dist_ctx.world_size,
                "effective_bytes_per_step": actual_step_bytes,
                "model": cfg.to_dict(),
            },
        )
        wb_tracker.start(resumed_run_id=resumed_wandb_id, resume_step=start_step)

        def _eval_metric_regime(st_id: int) -> str:
            if st_id == 1:
                return "autoencoder"
            if st_id in (2, 3):
                return "causal_lm"
            return "wug_fastmap"

        current_stage_id: int = -1
        last_stage_str: str = "A"
        last_metrics: dict[str, float] = {}
        loss_history: dict[int, float] = {}
        rollback_events: list[dict[str, Any]] = []
        saves_performed = 0
        steps_since_start = 0
        t_run_start = clock()
        last_heartbeat_t = t_run_start
        last_log_t = t_run_start
        last_log_steps = 0
        header_printed = False
        interrupted_sig: Optional[str] = None

        def _current_dataloader_cursor() -> dict[str, Any]:
            if carry_prefetcher is not None and carry_active_stage is not None:
                committed = carry_prefetcher.committed_cursor
                if committed is not None:
                    return committed
            return dict(stream.cursor)

        def _build_ckpt_payload(completed_step: int, st_id: int) -> dict[str, Any]:
            return CheckpointManager.build_payload(
                step=completed_step,
                stage_id=st_id,
                model=model,
                optimizer=optimizer,
                scheduler_state=controller.state_dict(),
                dynamic_loss_multipliers=watchdog.state_dict(),
                dataloader_cursor=_current_dataloader_cursor(),
                best_val_metric=best_val_metric,
                config_dict=cfg.to_dict(),
                hardware_tier=profile.tier,
                wandb_run_id=wb_tracker.run_id if wb_tracker is not None else None,
            )

        def _sync_log(force: bool = False) -> None:
            if is_primary and hasattr(run_log, "sync_to_drive"):
                run_log.sync_to_drive(force=force)

        eff_bf16_comm = bool(use_bf16 and bf16_grad_comm and hw_cfg.get("bf16_grad_comm", True))
        _sync_log(force=True)
        with PreemptionSignalGuard() as sig_guard:
            step_idx = start_step
            while step_idx < eff_total_steps:
                completed_step = step_idx + 1
                sc = controller.get(step_idx)
                last_stage_str = sc.model_stage

                if sc.stage_id != current_stage_id:
                    prev_stage = current_stage_id if current_stage_id > 0 else resumed_stage_id
                    if prev_stage > 0 and _eval_metric_regime(prev_stage) != _eval_metric_regime(sc.stage_id):
                        # Reset best_val_metric across incompatible evaluation regimes
                        # (e.g., Stage 1 autoencoder ~0.000 BPC -> Stage 2 causal predictive LM BPC,
                        # or Stage 3 causal LM -> Stage 4 Wug target BPC) so best_model.pt stays live!
                        best_val_metric = float("inf")
                        watchdog.clear_window()
                    freeze_stats = apply_stage_freeze(model, sc.stage_id)
                    current_stage_id = sc.stage_id
                    resumed_stage_id = sc.stage_id
                    _r0_print(
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
                accum_metric_vec: Optional[torch.Tensor] = None
                step_loss_finite = True

                use_carry = carry_packer is not None and carry_bank is not None and sc.model_stage in ("B", "C")
                if use_carry:
                    if carry_active_stage != sc.stage_id:
                        _reset_stream_carry()
                        carry_active_stage = sc.stage_id
                elif carry_active_stage is not None:
                    _reset_stream_carry()

                def _pull_microbatch(_m_i: int) -> dict[str, torch.Tensor]:
                    if use_carry:
                        assert carry_prefetcher is not None
                        return carry_prefetcher.next_batch(
                            step_idx=step_idx,
                            micro_idx=_m_i,
                            target_batch_bytes=eff_micro_bytes,
                            state=sc,
                            grad_accum_steps=grad_accum_steps,
                            total_steps=eff_total_steps,
                            controller=controller,
                        )
                    return stream.next_batch(target_batch_bytes=eff_micro_bytes, state=sc)

                batch_cur = _pull_microbatch(0)
                staged_cur, lmao_cur, ev_cur = _stage_microbatch_to_device(batch_cur, sc.stage_id, device, h2d_stream)

                for _micro_idx in range(grad_accum_steps):
                    _await_staged_microbatch(staged_cur, ev_cur, device)
                    batch = batch_cur
                    loss_mask_all_ones = lmao_cur
                    x_bytes = staged_cur["x_bytes"]
                    cu_seqlens = staged_cur["cu_seqlens"]
                    doc_lang_ids = staged_cur["doc_lang_ids"]
                    doc_style_ids_t = staged_cur["doc_style_ids"]
                    doc_pts_t = staged_cur["doc_ponder_tax_scale"]
                    byte_lang_ids = staged_cur["byte_lang_ids"]
                    loss_mask = staged_cur["loss_mask"]
                    wug_target_mask = staged_cur["wug_target_mask"]
                    assert (
                        x_bytes is not None
                        and cu_seqlens is not None
                        and doc_lang_ids is not None
                        and byte_lang_ids is not None
                        and loss_mask is not None
                        and wug_target_mask is not None
                    )

                    doc_carry = None
                    if use_carry:
                        assert carry_bank is not None
                        doc_carry = carry_bank.gather(batch["carry_from"], max_entry_bytes=eff_stream_chunk)

                    # Double-buffer next micro-batch H2D DMA on `h2d_stream` overlapped with forward + backward
                    if _micro_idx + 1 < grad_accum_steps and h2d_stream is not None:
                        batch_next = _pull_microbatch(_micro_idx + 1)
                        staged_next, lmao_next, ev_next = _stage_microbatch_to_device(
                            batch_next, sc.stage_id, device, h2d_stream
                        )
                    else:
                        batch_next = None
                        staged_next, lmao_next, ev_next = None, None, None

                    loss_w = build_byte_loss_weights(
                        x_bytes=x_bytes,
                        byte_lang_ids=byte_lang_ids,
                        loss_mask=loss_mask,
                        wug_target_mask=wug_target_mask,
                        state=sc,
                        loss_mask_all_ones=loss_mask_all_ones,
                    )
                    fam_mask = (
                        _compute_familiar_doc_mask(cu_seqlens, loss_mask, wug_target_mask)
                        if sc.stage_id == 4
                        else None
                    )

                    use_tbptt_replay_step = bool(
                        model.bcg_tbptt_replay
                        and model.bcg_tbptt_window > 0
                        and sc.model_stage in ("B", "C")
                    )
                    with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
                        if use_tbptt_replay_step:
                            out = model.forward_and_backward_tbptt(
                                raw_bytes=x_bytes.unsqueeze(0),
                                lang_ids=doc_lang_ids,
                                style_ids=doc_style_ids_t,
                                ponder_tax_scale=doc_pts_t,
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
                                familiar_ach_reg_weight=0.0,
                                familiar_doc_mask=None,
                                doc_cu_seqlens=cu_seqlens,
                                record_traces=False,
                                sync_metrics=False,
                                loss_scale=1.0 / float(grad_accum_steps),
                                doc_carry=doc_carry,
                            )
                        else:
                            out = model(
                                raw_bytes=x_bytes.unsqueeze(0),
                                lang_ids=doc_lang_ids,
                                style_ids=doc_style_ids_t,
                                ponder_tax_scale=doc_pts_t,
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
                                record_traces=False,
                                sync_metrics=False,
                                doc_carry=doc_carry,
                            )

                    # Launch backward IMMEDIATELY with zero pre-backward GPU->CPU sync bubbles!
                    if not use_tbptt_replay_step:
                        scaled_loss = out.loss_total / float(grad_accum_steps)
                        scaled_loss.backward()
                    if use_carry and carry_bank is not None:
                        carry_bank.store(out.doc_carry_out, batch["carry_save"])

                    if out.metrics_tensor is not None:
                        m_vec = torch.cat(
                            [out.loss_total.detach().float().reshape(1), out.metrics_tensor[1:].detach().float()],
                            dim=0,
                        )
                        accum_metric_vec = m_vec if accum_metric_vec is None else (accum_metric_vec + m_vec)
                    else:
                        for k, v in out.metrics.items():
                            accum_metrics[k] = accum_metrics.get(k, 0.0) + float(v) / float(grad_accum_steps)

                    if _micro_idx + 1 < grad_accum_steps:
                        if batch_next is not None and staged_next is not None:
                            batch_cur = batch_next
                            staged_cur, lmao_cur, ev_cur = staged_next, lmao_next, ev_next
                        else:
                            batch_cur = _pull_microbatch(_micro_idx + 1)
                            staged_cur, lmao_cur, ev_cur = _stage_microbatch_to_device(
                                batch_cur, sc.stage_id, device, h2d_stream
                            )

                if accum_metric_vec is not None:
                    accum_metrics = materialize_metric_vec(accum_metric_vec / float(grad_accum_steps))
                local_bad_loss = not math.isfinite(float(accum_metrics.get("loss_total", float("nan"))))
                if sync_bool_any(local_bad_loss, dist_ctx, device):
                    step_loss_finite = False

                # Test injection / instrumentation hook before Watchdog check & optimizer.step()
                if pre_step_hook is not None:
                    pre_step_hook(completed_step, model, optimizer, accum_metrics)

                active_params = [p for p in model.parameters() if p.requires_grad]
                grads_finite = step_loss_finite
                grad_norm: Optional[float] = None
                if grads_finite and active_params:
                    if dist_ctx.is_distributed:
                        all_reduce_gradients_coalesced(
                            active_params,
                            dist_ctx,
                            use_bf16_comm=eff_bf16_comm,
                        )
                    if any(p.grad is not None for p in active_params):
                        gn_t = torch.nn.utils.clip_grad_norm_(active_params, max_norm=grad_clip_norm)
                        gn_val = float(gn_t.detach().item())
                        if math.isfinite(gn_val):
                            grad_norm = gn_val
                        else:
                            grads_finite = False
                    else:
                        grad_norm = 0.0

                if dist_ctx.is_distributed:
                    if sync_bool_any(not grads_finite, dist_ctx, device):
                        grads_finite = False
                    accum_metrics = all_reduce_metrics_dict(accum_metrics, dist_ctx, device)

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
                    _reset_stream_carry()
                    rollback_events.append(action.event_record)
                    if wb_tracker is not None:
                        wb_tracker.log_watchdog_rollback(
                            step=completed_step,
                            rule=str(action.rule),
                            reason=str(action.reason),
                            rollback_count=watchdog.rollback_count,
                            lr_multiplier=watchdog.directives.lr_multiplier,
                        )
                    _r0_print(
                        f"[Watchdog Rule {action.rule}] {action.reason} | "
                        f"rollback_count={watchdog.rollback_count} | "
                        f"lr_mult={watchdog.directives.lr_multiplier:.4f} | "
                        f"skip_batches={action.skip_batches} | new_seed={action.new_seed}",
                        flush=True,
                    )
                    _sync_log(force=True)
                    if async_ckpt is not None:
                        async_ckpt.wait()
                    dist_barrier(dist_ctx)
                    if ckpt_mgr.last_healthy_path.exists():
                        healthy_ckpt = ckpt_mgr.load_checkpoint(ckpt_mgr.last_healthy_path, map_location="cpu")
                        restored_step, restored_cursor = CheckpointManager.restore_into(
                            healthy_ckpt,
                            model=model,
                            optimizer=optimizer,
                            controller=controller,
                            watchdog=watchdog,
                            preserve_watchdog_repairs=True,
                            restore_rng=False,
                        )
                        del healthy_ckpt
                        if device.type == "cuda" and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        broadcast_model_state(model, dist_ctx)
                        stream.restore_cursor(restored_cursor)
                        skip_n = int(action.skip_batches)
                        stream.cursor["batch_idx"] = int(stream.cursor["batch_idx"]) + skip_n
                        stream.cursor["phys_offset"] = int(stream.cursor.get("phys_offset", 0)) + skip_n * stream.world_size
                        stream._base_b_idx = int(stream.cursor["batch_idx"])
                        stream._phys_base = int(stream.cursor["phys_offset"])
                        step_idx = restored_step
                        current_stage_id = -1  # force exact freeze re-verification on next iteration
                    else:
                        skip_n = int(action.skip_batches)
                        stream.cursor["batch_idx"] = int(stream.cursor["batch_idx"]) + skip_n
                        stream.cursor["phys_offset"] = int(stream.cursor.get("phys_offset", 0)) + skip_n * stream.world_size
                        stream._base_b_idx = int(stream.cursor["batch_idx"])
                        stream._phys_base = int(stream.cursor["phys_offset"])

                    if action.new_seed is not None:
                        stream.cursor["seed"] = int(action.new_seed)
                        eff_r_seed = (
                            int(action.new_seed) + dist_ctx.rank * 104729
                            if dist_ctx.is_distributed
                            else int(action.new_seed)
                        )
                        reseed_all_rngs(eff_r_seed, device=device)
                    continue

                # Healthy optimization step (zero-CPU-sync wait if previous async checkpoint D2H still in flight)
                if async_ckpt is not None:
                    async_ckpt.wait_for_d2h_on_stream()
                optimizer.step()
                watchdog.tick_post_healthy_step(completed_step)
                steps_since_start += 1
                if grad_norm is not None:
                    accum_metrics["grad_norm"] = float(grad_norm)
                last_metrics = accum_metrics
                loss_history[completed_step] = float(accum_metrics["loss_total"])

                is_final_step = completed_step == eff_total_steps
                if dist_ctx.is_distributed and sc.stage_id in (2, 3, 4) and (completed_step % 50 == 0 or is_final_step):
                    sync_tensor_avg_(unwrap_compiled_module(model.bcg_pc).c0, dist_ctx)

                if step_hook is not None:
                    step_hook(completed_step, model, last_metrics)

                now_t = clock()
                interval_steps = max(steps_since_start - last_log_steps, 1)
                interval_dt = max(now_t - last_log_t, 1e-6)
                kb_per_sec = float(interval_steps * actual_step_bytes) / (interval_dt * 1024.0)
                active_lrs = [v for v in applied_lrs.values() if v > 0.0]
                disp_lr = max(active_lrs) if active_lrs else 0.0
                if wb_tracker is not None and wb_tracker.should_log_step(completed_step, total_steps=eff_total_steps):
                    wb_tracker.log_train_step(
                        step=completed_step,
                        stage_id=sc.stage_id,
                        metrics=last_metrics,
                        applied_lrs=applied_lrs,
                        kb_per_sec=kb_per_sec,
                        peak_mem_gb=get_peak_memory_gb(device),
                        watchdog_lr_mult=watchdog.directives.lr_multiplier,
                        total_steps=eff_total_steps,
                    )

                # Concise single-line console row every `console_every_steps` (default 50)
                # plus first 10 steps after launch/resume so compile warmup -> steady-state KB/s is visible!
                if is_primary and (
                    steps_since_start <= 10
                    or is_final_step
                    or (eff_console_every > 0 and completed_step % eff_console_every == 0)
                    or (heartbeat_seconds > 0 and (now_t - last_heartbeat_t) >= heartbeat_seconds)
                ):
                    if not header_printed:
                        _r0_print(format_console_header(), flush=True)
                        header_printed = True
                    _r0_print(
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
                    last_log_t = now_t
                    last_log_steps = steps_since_start
                    _sync_log(force=False)

                # 500-step zero-grad visual diagnostic probe (Rank 0 only; isolated RNG)
                is_best = False
                if is_primary and eff_probe_every > 0 and (completed_step % eff_probe_every == 0 or is_final_step):
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
                    if wb_tracker is not None:
                        wb_tracker.log_visual_probe(
                            step=completed_step,
                            probe_result=probe_res,
                            best_val_metric=best_val_metric,
                        )

                    if drive_logs_dir is not None:
                        sync_file_to_drive(eff_logs_dir / "latest_probe.md", drive_logs_dir, dest_name="latest_probe.md")
                        sync_file_to_drive(eff_logs_dir / "metrics.jsonl", drive_logs_dir, dest_name="metrics.jsonl")
                        _sync_log(force=True)

                # Dual-track atomic checkpointing + optional Colab Drive A/B Ping-Pong sync (Rank 0 writes)
                any_interrupted = sync_bool_any(sig_guard.interrupted, dist_ctx, device)
                healthy_due = ckpt_mgr.is_healthy_save_due(completed_step)
                milestone_due = ckpt_mgr.is_milestone_step(completed_step)
                wall_due_local, wall_reason = wall_trigger.should_save(completed_step) if is_primary else (False, None)
                wall_due = sync_bool_any(wall_due_local, dist_ctx, device)
                step_healthy_ok = watchdog.is_healthy_for_checkpoint(
                    completed_step, sc.stage_id, last_metrics, grad_norm=grad_norm
                )
                need_sync_ckpt = (
                    healthy_due
                    or milestone_due
                    or wall_due
                    or is_final_step
                    or any_interrupted
                )
                if need_sync_ckpt or (is_primary and is_best):
                    if dist_ctx.is_distributed and need_sync_ckpt:
                        sync_tensor_avg_(unwrap_compiled_module(model.bcg_pc).c0, dist_ctx)
                    if is_primary:
                        if any_interrupted:
                            if async_ckpt is not None:
                                async_ckpt.wait()
                            synchronize(device)
                            payload = _build_ckpt_payload(completed_step, sc.stage_id)
                            ckpt_mgr.save_emergency(payload)
                            saves_performed += 1
                            interrupted_sig = sig_guard.signal_name or "SIGTERM"
                            _r0_print(
                                f"[PreemptionGuard] Saved atomic emergency checkpoint at step={completed_step} "
                                f"-> {ckpt_mgr.emergency_interrupt_path}",
                                flush=True,
                            )
                        else:
                            if sync_mgr is not None and (wall_due or milestone_due) and not step_healthy_ok:
                                _r0_print(
                                    f"[DriveSync][SKIP] Skipped Drive slot overwrite at step={completed_step} "
                                    f"(unhealthy step: grad_norm={grad_norm}, loss={last_metrics.get('loss_total')}); "
                                    "preserving last healthy Drive slot.",
                                    flush=True,
                                )
                            want_drive = bool(
                                sync_mgr is not None
                                and (
                                    (wall_due and step_healthy_ok)
                                    or (milestone_due and step_healthy_ok)
                                    or (is_final_step and step_healthy_ok)
                                    or is_best
                                )
                            )
                            want_healthy = bool(
                                ((healthy_due or is_final_step) and step_healthy_ok)
                                or (wall_due and not want_drive and not healthy_due and step_healthy_ok)
                            )
                            want_milestone = bool(milestone_due and step_healthy_ok)
                            want_best = bool(is_best)
                            drive_reason: Optional[str] = (
                                (wall_reason or ("milestone" if milestone_due else ("final" if is_final_step else "best")))
                                if want_drive
                                else None
                            )
                            if want_healthy or want_milestone or want_best or want_drive:
                                payload = _build_ckpt_payload(completed_step, sc.stage_id)
                                assert async_ckpt is not None
                                async_ckpt.submit_checkpoint(
                                    raw_payload=payload,
                                    step=completed_step,
                                    model_stage=sc.model_stage,
                                    ckpt_mgr=ckpt_mgr,
                                    save_healthy=want_healthy,
                                    save_milestone=want_milestone,
                                    save_best=want_best,
                                    sync_mgr=sync_mgr if want_drive else None,
                                    drive_reason=drive_reason,
                                    post_commit_fn=(lambda: _sync_log(force=True)) if want_drive else None,
                                )
                                if want_healthy:
                                    saves_performed += 1
                                if want_milestone:
                                    saves_performed += 1
                                if want_best:
                                    saves_performed += 1
                                if want_drive:
                                    saves_performed += 1
                            if want_drive or (want_healthy and wall_due):
                                wall_trigger.mark_saved(completed_step)
                            if is_final_step and async_ckpt is not None:
                                async_ckpt.wait()
                    else:
                        if not any_interrupted and step_healthy_ok and (healthy_due or is_final_step or (wall_due and drive_sync_dir is None)):
                            ckpt_mgr.last_healthy_step = completed_step

                    # Only barrier across ranks on emergency preemption or final step;
                    # regular checkpoints overlap Rank-0 disk/Drive I/O without stalling other ranks.
                    if any_interrupted or is_final_step:
                        dist_barrier(dist_ctx)
                    if any_interrupted:
                        interrupted_sig = sig_guard.signal_name or "SIGTERM"
                        step_idx += 1
                        break

                step_idx += 1

        # Drain any in-flight async checkpoint before final export or returning summary
        if async_ckpt is not None:
            async_ckpt.wait()

        # Final `.safetensors` export on clean completion (Rank 0 exclusive)
        final_export_path: Optional[Path] = None
        if export_final and interrupted_sig is None:
            if dist_ctx.is_distributed:
                sync_tensor_avg_(unwrap_compiled_module(model.bcg_pc).c0, dist_ctx)
            if is_primary:
                synchronize(device)
                sf_name = str(ck_cfg.get("final_safetensors_name", "dcpa_final.safetensors"))
                exported = export_final_safetensors(model, eff_export_dir, filename=sf_name)
                final_export_path = exported["weights"]
                _r0_print(
                    f"[Export] Atomically exported {final_export_path} "
                    f"({final_export_path.stat().st_size / (1024 * 1024):.2f} MB)",
                    flush=True,
                )
                if drive_export_dir is not None:
                    final_export_path = sync_file_to_drive(
                        exported["weights"], drive_export_dir, dest_name=sf_name
                    )
                    sync_file_to_drive(exported["config"], drive_export_dir, dest_name="config.json")
                    _r0_print(f"[Export] Mirrored to Drive -> {final_export_path}", flush=True)
                if hasattr(os, "sync"):
                    os.sync()
            dist_barrier(dist_ctx)
            if not is_primary:
                sf_name = str(ck_cfg.get("final_safetensors_name", "dcpa_final.safetensors"))
                final_export_path = eff_export_dir / sf_name

        _r0_print(
            f"[Train] Summary step={step_idx:,}/{eff_total_steps:,} | "
            f"saves={saves_performed} | rollbacks={watchdog.rollback_count} | "
            f"best_metric={best_val_metric:.4f}",
            flush=True,
        )
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
            wandb_run_id=wb_tracker.run_id if wb_tracker is not None else None,
        )
    except BaseException:
        _exit_exc = sys.exc_info()
        raise
    finally:
        if "async_ckpt" in locals() and async_ckpt is not None:
            async_ckpt.close()
        if wb_tracker is not None:
            wb_tracker.finish()
        if "carry_prefetcher" in locals() and carry_prefetcher is not None:
            carry_prefetcher.close()
        if stream is not None:
            stream.close()
        run_log.__exit__(*_exit_exc)
        if hasattr(os, "sync") and is_primary:
            os.sync()
        cleanup_distributed_context(dist_owned)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DCPA 100k-Step Four-Stage Self-Healing Developmental Trainer")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)
    ap.add_argument(
        "--preset",
        choices=[
            "micro",
            "tiny",
            "proto_48m",
            "48m",
            "base",
            "standard_152m",
            "152m",
            "moe_210m",
            "210m",
            "moe_650m",
            "650m",
            "moe_1_8b",
            "1.8b",
            "moe_2_6b",
            "2.6b",
            "moe_3_5b",
            "3.5b",
        ],
        default=None,
    )
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--stage-boundaries", type=str, default=None, help="Comma-separated 5 boundaries e.g. 0,20,45,60,80")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--target-tokens-bytes-per-step", type=int, default=None)
    ap.add_argument("--micro-batch-bytes", type=int, default=None)
    ap.add_argument("--fixed-global-batch", action="store_true", help="Keep exact global step bytes across ranks instead of saturating per-GPU micro-batch.")
    ap.add_argument("--no-auto-lr-scale", action="store_true", help="Disable automatic sqrt(batch) LR scaling when multi-GPU expands global batch.")
    ap.add_argument("--no-length-bucketing", action="store_true", help="Disable length-homogeneous Ragged batch packing.")
    ap.add_argument(
        "--stream-carry",
        dest="stream_carry",
        action="store_true",
        default=None,
        help="Stages B/C: stream-TBPTT across steps (parallel byte streams + carried detached state).",
    )
    ap.add_argument("--no-stream-carry", dest="stream_carry", action="store_false", help="Disable stream carry.")
    ap.add_argument("--stream-chunk-bytes", type=int, default=None, help="Bytes per stream per micro-step (default 512).")
    ap.add_argument(
        "--bcg-cudagraphs",
        dest="bcg_cudagraphs",
        action="store_true",
        default=None,
        help="CUDA: capture the static-shape BCG-PC clause-chunk kernel with CUDA Graphs (reduce-overhead).",
    )
    ap.add_argument("--no-bcg-cudagraphs", dest="bcg_cudagraphs", action="store_false")
    ap.add_argument(
        "--bcg-ckpt-watermark-gb",
        type=float,
        default=None,
        help="Override VRAM watermark (GB) above which BCG chunk checkpointing engages (0.0 = always checkpoint).",
    )
    ap.add_argument(
        "--bcg-chunk-steps",
        type=int,
        default=None,
        help="Override BCG-PC static clause-chunk unroll length (e.g. 4, 8, 16, 32).",
    )
    ap.add_argument(
        "--bcg-tbptt-window",
        type=int,
        default=None,
        help="Override BCG-PC truncated BPTT window length (0 = disabled).",
    )
    ap.add_argument(
        "--lr-scale",
        type=float,
        default=None,
        help="Global multiplicative LR scale applied across all optimizer parameter groups (e.g. 0.5, 0.8, 1.2).",
    )
    ap.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VAL",
        help="Repeatable dot-path YAML override (e.g. --set hardware.prefetch_workers=4 --set curriculum.stage2.lambda_act_target=0.02). Prefix with '+' to insert a new key.",
    )
    ap.add_argument("--data-dir", type=str, default="artifacts/data")
    ap.add_argument("--cap-data-dir", type=str, default=None, help="Optional Stage 3 capability shards dir (cap_reasoning/writing)")
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
    ap.add_argument(
        "--gpu-profile",
        type=str,
        default="auto",
        help="GPU execution profile: 'auto' (default, auto-detects live GPU), or explicit name: "
        "t4, l4, 4090, 5090, a100_40g, a100, h100, g4, g4_256k.",
    )
    ap.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="GPU count: 'auto' (all visible GPUs; 1 process if <=1 GPU), '1' (force single-GPU), or N (e.g. 2, 4, 8).",
    )
    ap.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Optional comma-separated physical GPU IDs (e.g. '2' for single GPU #2, or '0,2' for 2 GPUs).",
    )
    ap.add_argument(
        "--compile-mode",
        choices=["none", "auto", "reduce-overhead"],
        default=None,
        help="Selective torch.compile on static-shape hotspots (auto-falls back to none on cpu/mps).",
    )
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-consecutive-meltdowns", type=int, default=3)
    ap.add_argument("--milestone-steps", type=str, default=None)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--wandb", dest="use_wandb", action="store_true", default=None, help="Enable Weights & Biases live telemetry.")
    ap.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases telemetry.")
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    ap.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags.")
    ap.add_argument("--wandb-log-every", type=int, default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    spawn_rc = maybe_relaunch_distributed(
        __file__,
        sys.argv[1:],
        gpus=args.gpus,
        gpu_ids=args.gpu_ids,
        device=args.device,
    )
    if spawn_rc is not None:
        sys.exit(spawn_rc)

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
            cap_data_dir=args.cap_data_dir,
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
            gpu_profile=args.gpu_profile,
            compile_mode=args.compile_mode,
            deterministic=args.deterministic,
            seed=args.seed,
            max_consecutive_meltdowns=args.max_consecutive_meltdowns,
            milestone_steps=m_steps,
            export_final=not args.no_export,
            fixed_global_batch=args.fixed_global_batch,
            auto_lr_scale=not args.no_auto_lr_scale,
            length_bucketing=not args.no_length_bucketing,
            stream_carry=args.stream_carry,
            stream_chunk_bytes=args.stream_chunk_bytes,
            bcg_cudagraphs=args.bcg_cudagraphs,
            bcg_ckpt_watermark_gb=args.bcg_ckpt_watermark_gb,
            bcg_chunk_steps=args.bcg_chunk_steps,
            bcg_tbptt_window=args.bcg_tbptt_window,
            lr_scale=args.lr_scale,
            config_overrides=args.config_overrides,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_mode=args.wandb_mode,
            wandb_tags=args.wandb_tags,
            wandb_log_every=args.wandb_log_every,
        )
        if int(os.environ.get("RANK", "0") or "0") == 0:
            print(
                f"[DONE] Completed {summary.completed_steps:,}/{summary.total_steps:,} steps "
                f"(saves={summary.saves_performed}, rollbacks={summary.rollbacks_performed}, "
                f"best_metric={summary.best_loss:.4f}, export={summary.final_export_path})",
                flush=True,
            )
    except MeltdownAbortError as exc:
        if int(os.environ.get("RANK", "0") or "0") == 0:
            print(f"[FATAL MELTDOWN] {exc}", file=sys.stderr, flush=True)
        sys.exit(42)


if __name__ == "__main__":
    main()
