"""Production 5-Stage Developmental Trainer for DCPA-Flat v3.3 (`DCPAModelFlat`).

Features:
  - Reads `configs/v3/dcpa_flat_*.yaml` (48M / 206M / 650M / micro smoke).
  - Drives `StaticGridByteDataset` (zero mid-doc truncation, fixed-point x10 `.loss_mask.bin`)
    with automatic capability shard (`cap_*`) and Stage 5 SFT mixing.
  - 5-stage developmental progression (Stage 1 Sensory `[64, 2048, S=512]` -> Stages 2-3 Cortex
    `[32, 4096, S=640]` -> Stage 4 Hippocampus `HIPR <-> G2` -> Stage 5 Polish + Frozen DG Whitening).
  - Colab/Cloud anti-disconnect A/B Ping-Pong Google Drive sync (`ABDriveSyncManager`) with
    strict health gating (never overwrites healthy Drive slots with NaN/diverged gradients).
  - Weights & Biases (`WandbTracker`) run-ID continuity across Colab reconnects.
  - Strictly adheres to Iron Law #1 (`utils/device.py` abstraction, zero bare `torch.cuda.*`).
  - Automatic Rust-ready `model.safetensors` + `config.json` + `golden.safetensors` export on finish.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.ragged_dataloader import StaticGridByteDataset, build_static_grid_dataloader
from dcpa.compile_utils import get_clean_state_dict
from dcpa.export import export_flat_v3_and_golden
from dcpa.interfaces import DCPAFlatConfig, StaticGridBatch
from dcpa.model_flat import DCPAModelFlat, FlatStageMode
from gates.run_flat_killgate_5k import _make_synthetic_grid_batch
from train.colab_drive_sync import (
    ABDriveSyncManager,
    CheckpointTrigger,
    RunLogTee,
    sync_file_to_drive,
)
from utils.device import (
    amp_autocast,
    capture_rng_state,
    configure_cuda_tf32,
    detect_hardware_profile,
    get_device,
    is_bf16_enabled,
    restore_rng_state,
    seed_everything,
    synchronize,
)
from utils.wandb_tracker import WandbTracker, resolve_wandb_settings


@dataclass(frozen=True)
class StageWindow:
    name: str
    mode: FlatStageMode
    start_step: int
    end_step: int
    grid_lanes: int
    grid_bytes: int
    grid_segs: int
    hip_enabled: bool
    cap_mix_prob: float


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def build_flat_config_from_yaml(raw: dict[str, Any], preset_override: Optional[str] = None) -> DCPAFlatConfig:
    preset_name = (preset_override or str(raw.get("preset", "flat_48m"))).strip()
    if preset_name == "flat_micro":
        base = DCPAFlatConfig.flat_micro()
    elif preset_name == "flat_48m":
        base = DCPAFlatConfig.flat_48m()
    elif preset_name == "flat_206m":
        base = DCPAFlatConfig.flat_206m()
    elif preset_name == "flat_650m":
        base = DCPAFlatConfig.flat_650m()
    else:
        raise ValueError(f"Unknown DCPA-Flat preset: {preset_name!r}")

    if preset_override is None and isinstance(raw.get("model"), dict):
        d = base.to_dict()
        for k, v in raw["model"].items():
            if k in d:
                d[k] = tuple(int(x) for x in v) if k == "local_window_layers" else v
        if isinstance(d.get("local_window_layers"), list):
            d["local_window_layers"] = tuple(int(x) for x in d["local_window_layers"])
        return DCPAFlatConfig(**d)
    return base


def build_stage_schedule(raw: dict[str, Any], total_steps: int, cfg: DCPAFlatConfig) -> list[StageWindow]:
    t_cfg = raw.get("training", {})
    stages_cfg = t_cfg.get("stages", {})
    if not stages_cfg:
        return [
            StageWindow(
                name="stage1_cortex",
                mode="cortex",
                start_step=0,
                end_step=total_steps,
                grid_lanes=cfg.grid_lanes,
                grid_bytes=cfg.grid_bytes,
                grid_segs=cfg.grid_segs,
                hip_enabled=False,
                cap_mix_prob=0.0,
            )
        ]

    ordered = list(stages_cfg.items())
    frac_sum = sum(float(v.get("fraction", 0.0)) for _, v in ordered) or 1.0
    windows: list[StageWindow] = []
    cur = 0
    for idx, (name, sc) in enumerate(ordered):
        frac = float(sc.get("fraction", 0.0)) / frac_sum
        end = total_steps if idx == len(ordered) - 1 else min(total_steps, cur + max(1, int(round(total_steps * frac))))
        if end <= cur:
            end = min(total_steps, cur + 1)
        g = sc.get("grid", {})
        mode_raw = str(sc.get("mode", "cortex"))
        mode: FlatStageMode = mode_raw if mode_raw in ("sensory", "cortex", "hippocampus", "full") else "cortex"  # type: ignore[assignment]
        windows.append(
            StageWindow(
                name=str(name),
                mode=mode,
                start_step=cur,
                end_step=end,
                grid_lanes=int(g.get("P", cfg.grid_lanes)),
                grid_bytes=int(g.get("T", cfg.grid_bytes)),
                grid_segs=int(g.get("S", cfg.grid_segs)),
                hip_enabled=bool(sc.get("hip_enabled", mode in ("hippocampus", "full"))),
                cap_mix_prob=float(sc.get("cap_mix_prob", 0.0)),
            )
        )
        cur = end
    return windows


def resolve_stage_for_step(schedule: list[StageWindow], step: int) -> StageWindow:
    for w in schedule:
        if w.start_step <= step < w.end_step:
            return w
    return schedule[-1]


def cosine_lr_multiplier(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denom = max(total_steps - warmup_steps, 1)
    prog = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    return float(min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog)))


class _BatchStreamPool:
    """Manages StaticGridByteDataset iterators for base corpus, capability shards, and Stage 5 SFT."""

    def __init__(
        self,
        *,
        data_dir: Optional[Path],
        cap_data_dir: Optional[Path],
        sft_data_dir: Optional[Path],
        cfg: DCPAFlatConfig,
        domain_weights: Optional[dict[int, float]],
        seed: int,
        smoke: bool,
    ) -> None:
        self.data_dir = data_dir
        self.cap_data_dir = cap_data_dir
        self.sft_data_dir = sft_data_dir
        self.cfg = cfg
        self.domain_weights = domain_weights
        self.seed = int(seed)
        self.smoke = bool(smoke)
        self._rng = random.Random(seed + 1337)
        self._iter_cache: dict[tuple[str, int, int], Iterator[StaticGridBatch]] = {}

        self.cap_pairs: list[tuple[Path, Path]] = []
        if cap_data_dir and cap_data_dir.is_dir():
            for name in (
                "cap_reasoning_train",
                "cap_writing_train",
                "cap_code_train",
                "cap_toolcall_train",
                "cap_instruct_train",
            ):
                bp = cap_data_dir / f"{name}.bin"
                ip = cap_data_dir / f"{name}.idx"
                if bp.is_file() and ip.is_file() and bp.stat().st_size > 0 and ip.stat().st_size >= 16:
                    self.cap_pairs.append((bp, ip))

        self.sft_pair: Optional[tuple[Path, Path]] = None
        if sft_data_dir and sft_data_dir.is_dir():
            bp = sft_data_dir / "sft_train.bin"
            ip = sft_data_dir / "sft_train.idx"
            if bp.is_file() and ip.is_file() and bp.stat().st_size > 0 and ip.stat().st_size >= 16:
                self.sft_pair = (bp, ip)

    def _make_iter(
        self,
        bin_p: Path,
        idx_p: Path,
        P: int,
        T: int,
        dw: Optional[dict[int, float]],
        salt: int,
        *,
        use_cuda_workers: bool = False,
    ) -> Iterator[StaticGridBatch]:
        nw = 0 if (self.smoke or not use_cuda_workers) else min(4, max(2, (os.cpu_count() or 2) // 2))
        loader = build_static_grid_dataloader(
            bin_path=bin_p,
            idx_path=idx_p,
            grid_lanes=P,
            grid_bytes=T,
            doc_meta_cap_per_lane=self.cfg.doc_meta_cap_per_lane,
            domain_weights=dw,
            num_batches=None,
            num_workers=nw,
            pin_memory=use_cuda_workers,
            seed=self.seed + salt,
        )
        return iter(loader)

    def next_batch(self, stage: StageWindow, step: int, device: torch.device, eff_cap_mix: Optional[float] = None) -> StaticGridBatch:
        P, T = stage.grid_lanes, stage.grid_bytes
        if self.smoke or self.data_dir is None or not (self.data_dir / "train.bin").is_file():
            sub_cfg = DCPAFlatConfig(**{**self.cfg.to_dict(), "grid_lanes": P, "grid_bytes": T, "grid_segs": stage.grid_segs})
            return _make_synthetic_grid_batch(sub_cfg, lang_id=(step % 6), seed=self.seed + step, device=device)

        use_cuda_w = device.type == "cuda"
        cap_p = stage.cap_mix_prob if eff_cap_mix is None else float(eff_cap_mix)
        use_sft = stage.mode == "full" and self.sft_pair is not None and self._rng.random() < cap_p * 0.5
        use_cap = (not use_sft) and bool(self.cap_pairs) and self._rng.random() < cap_p

        if use_sft and self.sft_pair is not None:
            key = ("sft", P, T)
            if key not in self._iter_cache:
                self._iter_cache[key] = self._make_iter(self.sft_pair[0], self.sft_pair[1], P, T, None, 501, use_cuda_workers=use_cuda_w)
            batch = next(self._iter_cache[key])
        elif use_cap:
            c_idx = self._rng.randrange(len(self.cap_pairs))
            bp, ip = self.cap_pairs[c_idx]
            key = (f"cap_{c_idx}", P, T)
            if key not in self._iter_cache:
                self._iter_cache[key] = self._make_iter(bp, ip, P, T, None, 100 + c_idx, use_cuda_workers=use_cuda_w)
            batch = next(self._iter_cache[key])
        else:
            key = ("base", P, T)
            if key not in self._iter_cache:
                self._iter_cache[key] = self._make_iter(
                    self.data_dir / "train.bin",
                    self.data_dir / "train.idx",
                    P,
                    T,
                    self.domain_weights,
                    1,
                    use_cuda_workers=use_cuda_w,
                )
            batch = next(self._iter_cache[key])
        return batch.to(device=device, non_blocking=use_cuda_w)

    def eval_batches(self, P: int, T: int, S: int, device: torch.device, n_batches: int = 4) -> list[StaticGridBatch]:
        if self.smoke or self.data_dir is None or not (self.data_dir / "val.bin").is_file():
            sub_cfg = DCPAFlatConfig(**{**self.cfg.to_dict(), "grid_lanes": P, "grid_bytes": T, "grid_segs": S})
            return [_make_synthetic_grid_batch(sub_cfg, lang_id=(i % 6), seed=9000 + i, device=device) for i in range(n_batches)]
        ds = StaticGridByteDataset(
            bin_path=self.data_dir / "val.bin",
            idx_path=self.data_dir / "val.idx",
            grid_lanes=P,
            grid_bytes=T,
            doc_meta_cap_per_lane=self.cfg.doc_meta_cap_per_lane,
            num_batches=n_batches,
            seed=9999,
        )
        return [b.to(device=device) for b in ds]


@torch.no_grad()
def evaluate_val_bpc(
    model: DCPAModelFlat,
    batches: list[StaticGridBatch],
    stage: StageWindow,
    use_bf16: bool,
    device: torch.device,
) -> float:
    was_training = model.training
    old_segs = model.cfg.grid_segs
    model.cfg.grid_segs = stage.grid_segs
    model.eval()
    total_ce = 0.0
    try:
        for b in batches:
            with amp_autocast(device, enabled=use_bf16):
                out = model(b, stage=stage.mode, hip_enabled=stage.hip_enabled)
            total_ce += float(out.loss_lang.item())
    finally:
        model.cfg.grid_segs = old_segs
        model.train(was_training)
    return (total_ce / max(len(batches), 1)) / math.log(2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="DCPA-Flat v3.3 Production 5-Stage Trainer")
    ap.add_argument("--config", default="configs/v3/dcpa_flat_206m_v3.yaml")
    ap.add_argument("--preset", default=None, help="Optional override: flat_micro | flat_48m | flat_206m | flat_650m")
    ap.add_argument("--data-dir", default=None, help="Local NVMe dir containing train.bin/train.idx/val.bin/val.idx")
    ap.add_argument("--cap-data-dir", default=None, help="Optional NVMe dir containing cap_* shards")
    ap.add_argument("--sft-data-dir", default=None, help="Optional NVMe dir containing sft_train.bin/idx")
    ap.add_argument("--local-cache-dir", default="artifacts/flat_run")
    ap.add_argument("--drive-sync-dir", default=None, help="Google Drive A/B Ping-Pong checkpoint dir")
    ap.add_argument("--drive-logs-dir", default=None, help="Google Drive persistent logs dir")
    ap.add_argument("--export-dir", default=None, help="Local Rust export dir on finish")
    ap.add_argument("--drive-export-dir", default=None, help="Optional Drive mirror for Rust export")
    ap.add_argument("--resume", default="auto", help="auto | none | /path/to/ckpt.pt")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--grad-accum-steps", type=int, default=None)
    ap.add_argument("--grad-checkpoint", choices=("auto", "on", "off"), default=None)
    ap.add_argument("--compile-chunks", action="store_true", default=None)
    ap.add_argument("--no-compile-chunks", dest="compile_chunks", action="store_false")
    ap.add_argument("--use-local-window-layers", choices=("true", "false"), default=None)
    ap.add_argument("--carrier-grad-guard", choices=("none", "router_sg", "soft_clip"), default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Fast 4-step end-to-end smoke + checkpoint + export check")
    ap.add_argument("--wandb", action="store_true", default=None)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    raw_yaml = load_yaml_config(args.config)
    preset = "flat_micro" if args.smoke else args.preset
    cfg = build_flat_config_from_yaml(raw_yaml, preset_override=preset)
    if args.use_local_window_layers is not None:
        cfg.use_local_window_layers = args.use_local_window_layers == "true"
    if args.carrier_grad_guard is not None:
        cfg.carrier_grad_guard = args.carrier_grad_guard  # type: ignore[assignment]

    device = torch.device("cpu") if args.smoke else get_device(preferred=args.device)
    hw = detect_hardware_profile("cpu" if args.smoke else args.device)
    configure_cuda_tf32(device, allow=bool(raw_yaml.get("hardware", {}).get("allow_tf32", True)))
    use_bf16 = is_bf16_enabled(device, enabled=bool(raw_yaml.get("hardware", {}).get("bf16", True)))
    seed_everything(args.seed, device=device)

    t_cfg = raw_yaml.get("training", {})
    hw_cfg = raw_yaml.get("hardware", {})
    total_steps = 4 if args.smoke else int(args.total_steps or t_cfg.get("total_steps", 100_000))
    warmup_steps = 1 if args.smoke else int(t_cfg.get("warmup_steps", 2000))
    min_lr_ratio = float(t_cfg.get("min_lr_ratio", 0.10))
    grad_accum = 1 if args.smoke else int(args.grad_accum_steps or hw_cfg.get("grad_accum_steps", 1))
    eval_every = 2 if args.smoke else int(t_cfg.get("eval_every_steps", 1000))
    save_every_steps = 2 if args.smoke else int(t_cfg.get("save_every_steps", 1000))
    save_every_min = float(t_cfg.get("save_every_minutes", 10.0))
    max_healthy_gn = float(t_cfg.get("max_healthy_grad_norm", 1000.0))

    gc_raw = args.grad_checkpoint if args.grad_checkpoint is not None else hw_cfg.get("grad_checkpoint", "auto")
    if isinstance(gc_raw, bool):
        use_ckpt = gc_raw
    else:
        gc_mode = str(gc_raw).strip().lower()
        if gc_mode in ("on", "true", "1", "yes"):
            use_ckpt = True
        elif gc_mode in ("off", "false", "0", "no"):
            use_ckpt = False
        else:
            use_ckpt = bool(device.type == "cuda" and hw.vram_gb < 45.0)

    schedule = build_stage_schedule(raw_yaml, total_steps=total_steps, cfg=cfg)
    if args.smoke:
        # Exercise sensory -> cortex -> hippocampus -> full within 4 smoke steps!
        schedule = [
            StageWindow("stage1_sensory", "sensory", 0, 1, cfg.grid_lanes, cfg.grid_bytes, cfg.grid_segs, False, 0.0),
            StageWindow("stage2_cortex", "cortex", 1, 2, cfg.grid_lanes, cfg.grid_bytes, cfg.grid_segs, False, 0.0),
            StageWindow("stage4_hip", "hippocampus", 2, 3, cfg.grid_lanes, cfg.grid_bytes, cfg.grid_segs, True, 0.0),
            StageWindow("stage5_full", "full", 3, 4, cfg.grid_lanes, cfg.grid_bytes, cfg.grid_segs, True, 0.0),
        ]

    local_cache = Path(args.local_cache_dir)
    logs_dir = local_cache / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    drive_logs_dir = Path(args.drive_logs_dir) if args.drive_logs_dir else None

    milestones = tuple(sorted({w.end_step for w in schedule if w.end_step < total_steps} | {total_steps}))
    sync_mgr = ABDriveSyncManager(
        local_cache_dir=local_cache,
        drive_sync_dir=args.drive_sync_dir,
        milestone_steps=milestones,
    )
    trigger = CheckpointTrigger(save_every_steps=save_every_steps, save_every_minutes=save_every_min)

    model = DCPAModelFlat(cfg).to(device=device)
    param_info = model.count_parameters()
    opt_kw: dict[str, Any] = {"betas": (0.9, 0.95)}
    if device.type == "cuda":
        opt_kw["fused"] = True
    optimizer = torch.optim.AdamW(model.parameter_groups(), **opt_kw)

    start_step = 0
    best_val_bpc = float("inf")
    resumed_wandb_id: Optional[str] = None

    if args.resume != "none":
        res = None
        if args.resume == "auto":
            res = sync_mgr.load_latest_auto(map_location="cpu")
        else:
            p = Path(args.resume)
            if p.is_file():
                ck = torch.load(str(p), map_location="cpu", weights_only=False)
                res = type("R", (), {"checkpoint": ck, "step": int(ck.get("step", 0)), "reason": str(p)})()
        if res is not None:
            ck = res.checkpoint
            # Strip any legacy `_orig_mod.` prefix defensively before loading
            clean_sd = {k.replace("._orig_mod.", ".").removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
            model.load_state_dict(clean_sd, strict=True)
            if "optimizer" in ck:
                try:
                    optimizer.load_state_dict(ck["optimizer"])
                except Exception as opt_exc:
                    print(f"[Resume][WARN] Optimizer state skipped ({opt_exc}); starting fresh AdamW moments.", flush=True)
            start_step = int(ck.get("step", res.step))
            best_val_bpc = float(ck.get("best_val_bpc", float("inf")))
            resumed_wandb_id = ck.get("wandb_run_id")
            if "rng_state" in ck:
                restore_rng_state(ck["rng_state"])
            print(f"[Resume] Restored DCPA-Flat checkpoint at step={start_step} ({res.reason})", flush=True)

    want_compile = (
        False
        if args.smoke
        else (args.compile_chunks if args.compile_chunks is not None else bool(hw_cfg.get("compile_chunks", True)))
    )
    compiled_mods = model.enable_compiled_submodules(device) if want_compile else []

    wb_settings = resolve_wandb_settings(
        yaml_wandb_cfg=raw_yaml.get("wandb"),
        cli_enabled=False if args.smoke else args.wandb,
        cli_project=args.wandb_project,
        cli_run_name=args.wandb_run_name or str(raw_yaml.get("experiment_name", "dcpa-flat-v3")),
        default_project="dcpa-flat-v3",
    )
    tracker = WandbTracker(
        enabled=wb_settings["enabled"],
        is_primary=True,
        project=wb_settings["project"],
        entity=wb_settings["entity"],
        run_name=wb_settings["run_name"],
        mode=wb_settings["mode"],
        tags=["dcpa-flat-v3.3", str(raw_yaml.get("preset", "custom"))],
        log_every_steps=wb_settings["log_every_steps"],
        logs_dir=logs_dir,
        drive_logs_dir=drive_logs_dir,
        config_payload={"model": cfg.to_dict(), "params": param_info, "total_steps": total_steps},
    )
    tracker.start(resumed_run_id=resumed_wandb_id, resume_step=start_step)

    raw_dw = raw_yaml.get("data", {}).get("domain_weights")
    domain_weights = {int(k): float(v) for k, v in raw_dw.items()} if isinstance(raw_dw, dict) else None
    stream_pool = _BatchStreamPool(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        cap_data_dir=Path(args.cap_data_dir) if args.cap_data_dir else None,
        sft_data_dir=Path(args.sft_data_dir) if args.sft_data_dir else None,
        cfg=cfg,
        domain_weights=domain_weights,
        seed=args.seed,
        smoke=args.smoke,
    )

    metrics_jsonl = logs_dir / "metrics_flat.jsonl"
    active_stage_mode: Optional[FlatStageMode] = None
    stage_enter_step: int = start_step
    win_bytes = 0
    win_t0 = time.perf_counter()

    with RunLogTee(logs_dir, drive_logs_dir=drive_logs_dir) as log_tee:
        print(
            f"[DCPA-Flat v3.3] device={device} ({hw.device_name}, {hw.vram_gb:.1f}GB) | "
            f"bf16={use_bf16} | compile={compiled_mods or 'eager'} | grad_ckpt={use_ckpt} | grad_accum={grad_accum} | "
            f"trainable={param_info['trainable_total']/1e6:.2f}M | active={param_info['active_avg_per_seg']/1e6:.2f}M | "
            f"steps={start_step}..{total_steps}",
            flush=True,
        )

        for step in range(start_step, total_steps):
            st = resolve_stage_for_step(schedule, step)
            if st.mode != active_stage_mode:
                if active_stage_mode is not None:
                    stage_enter_step = step
                model.set_stage_mode(st.mode)
                active_stage_mode = st.mode
                print(
                    f"[Stage Transition] step={step} -> {st.name} (mode={st.mode}, grid=[{st.grid_lanes}, {st.grid_bytes}], "
                    f"S={st.grid_segs}, hip={st.hip_enabled}, cap_mix={st.cap_mix_prob:.2f})",
                    flush=True,
                )

            cfg.grid_lanes = st.grid_lanes
            cfg.grid_bytes = st.grid_bytes
            cfg.grid_segs = st.grid_segs

            # Smooth stage-transition micro-warmup (prevents cold-layer gradient shock on Stage 1->2 & 3->4)
            lr_mult = cosine_lr_multiplier(step, warmup_steps=warmup_steps, total_steps=total_steps, min_ratio=min_lr_ratio)
            if not args.smoke and stage_enter_step > 0 and (step - stage_enter_step) < 500:
                stage_ramp = float(step - stage_enter_step + 1) / 500.0
                lr_mult *= 0.1 + 0.9 * stage_ramp
            for pg in optimizer.param_groups:
                pg["lr"] = float(pg.get("base_lr", cfg.lr_concept)) * lr_mult

            # Linear cap_mix_prob ramp across first 20% of stage window (prevents sudden control-byte cliff)
            st_span = max(st.end_step - st.start_step, 1)
            ramp_len = max(1, int(0.20 * st_span))
            cap_ramp = min(1.0, float(step - st.start_step + 1) / float(ramp_len))
            eff_cap_mix = st.cap_mix_prob * cap_ramp

            optimizer.zero_grad(set_to_none=True)

            step_bytes = 0
            last_out = None
            acc_expert_counts: Optional[list[torch.Tensor]] = None
            model.train()
            for acc_i in range(grad_accum):
                batch = stream_pool.next_batch(st, step * grad_accum + acc_i, device=device, eff_cap_mix=eff_cap_mix)
                step_bytes += int(batch.x_bytes.numel())
                with amp_autocast(device, enabled=use_bf16):
                    out = model(batch, stage=st.mode, hip_enabled=st.hip_enabled, use_checkpoint=use_ckpt)
                    loss_scaled = out.loss_total / float(grad_accum)
                loss_scaled.backward()
                last_out = out
                if out.expert_counts:
                    if acc_expert_counts is None:
                        acc_expert_counts = [c.detach().clone() for c in out.expert_counts]
                    else:
                        for li, c in enumerate(out.expert_counts):
                            acc_expert_counts[li] = acc_expert_counts[li] + c.detach()

            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            completed_step = step + 1
            win_bytes += step_bytes

            due, reason = trigger.should_save(completed_step)
            is_final = completed_step == total_steps
            is_log_step = (completed_step == 1) or (completed_step % 10 == 0) or (completed_step % eval_every == 0) or is_final or due

            # Single D2H sync per optimizer step with full Watchdog health & `max_healthy_gn` coverage
            loss_t = last_out.loss_total.detach() if last_out is not None else torch.tensor(float("nan"), device=device)
            if is_log_step or device.type != "cuda":
                stats_cpu = torch.stack([loss_t.float(), grad_norm_t.float()]).to("cpu")
                loss_val = float(stats_cpu[0].item())
                grad_norm = float(stats_cpu[1].item())
                step_healthy = bool(math.isfinite(loss_val) and math.isfinite(grad_norm) and grad_norm <= max_healthy_gn)
            else:
                healthy_t = torch.isfinite(loss_t) & torch.isfinite(grad_norm_t) & (grad_norm_t <= max_healthy_gn)
                step_healthy = bool(healthy_t.item())
                grad_norm = 0.0
                loss_val = 0.0

            if step_healthy:
                optimizer.step()
                if last_out is not None:
                    model.post_optimizer_step(last_out, expert_counts_override=acc_expert_counts)
            else:
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[Watchdog][SKIP] step={completed_step} unhealthy loss={loss_val:.4f} grad_norm={grad_norm:.2f} "
                    "(skipping optimizer step & Drive write)",
                    flush=True,
                )

            val_bpc: Optional[float] = None
            is_best = False
            if completed_step % eval_every == 0 or is_final:
                v_batches = stream_pool.eval_batches(st.grid_lanes, st.grid_bytes, st.grid_segs, device=device, n_batches=2 if args.smoke else 4)
                val_bpc = evaluate_val_bpc(model, v_batches, st, use_bf16=use_bf16, device=device)
                if math.isfinite(val_bpc) and val_bpc < best_val_bpc:
                    best_val_bpc = val_bpc
                    is_best = True

            if is_log_step:
                synchronize(device)
                dt = max(time.perf_counter() - win_t0, 1e-6)
                kb_s = (win_bytes / 1024.0) / dt
                win_bytes = 0
                win_t0 = time.perf_counter()

                m = last_out.materialize_metrics() if last_out is not None else {}
                train_bpc = float(m.get("loss_lang", 0.0)) / math.log(2.0)
                v_str = f" | val_bpc={val_bpc:.4f}" if val_bpc is not None else ""
                print(
                    f"[Step {completed_step:>6d}/{total_steps} | {st.name}] "
                    f"bpc={train_bpc:.4f}{v_str} | jepa={m.get('loss_jepa', 0.0):.4f} | unc={m.get('loss_unc', 0.0):.4f} | "
                    f"ovf={m.get('overflow_byte_ratio', 0.0):.4f} | gn={grad_norm:.2f} | {kb_s:.1f} KB/s",
                    flush=True,
                )

                rec = {
                    "step": completed_step,
                    "stage": st.name,
                    "train_bpc": train_bpc,
                    "val_bpc": val_bpc,
                    "grad_norm": grad_norm,
                    "kb_s": kb_s,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{k: float(v) for k, v in m.items()},
                }
                with metrics_jsonl.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(rec) + "\n")

                if tracker.active and (tracker.should_log_step(completed_step, total_steps=total_steps) or val_bpc is not None):
                    wb_payload = {
                        "train/bpc": train_bpc,
                        "train/loss_total": loss_val,
                        "train/loss_jepa": float(m.get("loss_jepa", 0.0)),
                        "train/loss_unc": float(m.get("loss_unc", 0.0)),
                        "train/overflow_ratio": float(m.get("overflow_byte_ratio", 0.0)),
                        "train/grad_norm": grad_norm,
                        "perf/kb_per_sec": kb_s,
                        "perf/tau_think": float(m.get("tau_think", 0.0)),
                    }
                    if val_bpc is not None:
                        wb_payload["val/bpc"] = val_bpc
                    tracker._run.log(wb_payload, step=completed_step)  # type: ignore[union-attr]

            if (due or is_best or is_final) and step_healthy:
                payload = {
                    "schema_version": 3,
                    "architecture": "dcpa_flat_v3_3",
                    "step": completed_step,
                    "stage_name": st.name,
                    "cfg": cfg.to_dict(),
                    "model": get_clean_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "best_val_bpc": best_val_bpc,
                    "wandb_run_id": tracker.run_id,
                    "rng_state": capture_rng_state(),
                }
                sync_mgr.save_ping_pong(
                    payload=payload,
                    step=completed_step,
                    stage_id=st.name,
                    reason="final" if is_final else ("best" if is_best else reason),
                    is_best=is_best,
                )
                trigger.mark_saved(completed_step)
                if drive_logs_dir is not None:
                    sync_file_to_drive(metrics_jsonl, drive_logs_dir, dest_name="metrics_flat.jsonl")
                log_tee.sync_to_drive(force=is_final)

        # Auto-export Rust-ready Flat v3.3 weights on completion (unwrap any torch.compile wrappers first)
        model.disable_compiled_submodules()
        export_dir = Path(args.export_dir) if args.export_dir else (local_cache / "export")
        export_paths = export_flat_v3_and_golden(
            model,
            export_dir,
            export_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )
        print(f"[Export] Saved Rust-ready Flat v3.3 bundle -> {export_dir}", flush=True)
        if args.drive_export_dir:
            d_exp = Path(args.drive_export_dir)
            for k in ("config", "weights", "meta"):
                if k in export_paths:
                    sync_file_to_drive(export_paths[k], d_exp, dest_name=export_paths[k].name)
            print(f"[Export] Mirrored Flat v3.3 export to Drive -> {d_exp}", flush=True)

    tracker.finish(summary_extra={"best_val_bpc": best_val_bpc, "completed_steps": total_steps})


if __name__ == "__main__":
    main()
