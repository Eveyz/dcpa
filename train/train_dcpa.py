"""Production cross-platform DCPA developmental trainer (Apple Silicon M3 Pro + Google Colab H100/A100).

Enforces Iron Laws #1, #3, and #4:
  - Routes all device selection, AMP autocast, hardware-tier micro-batch sharding, and RNG
    checkpointing exclusively through `utils/device.py`.
  - Enforces two-tier storage isolation: hot `.bin/.idx` mmap reads and staging checkpoints
    live strictly on local NVMe (`--local-cache-dir`, `--data-dir`), never `/content/drive`.
  - Dual-trigger checkpointing: saves every `--save-every-steps` (default 1,000) OR every
    `--save-every-minutes` (default 10 min wall-clock) via A/B Ping-Pong in-place overwrite
    (`slot_A.pt` <-> `slot_B.pt` + `manifest.json`) with `flush()` + `os.fsync()`.
  - Self-healing `--resume auto`: verifies `active_slot` size + `blake2b` hash and seamlessly
    falls back to `backup_slot` if Colab disconnected mid-sync.
  - Emits one concise heartbeat line every 60s (keeping Colab WebSocket active without flooding
    browser memory) and syncs `latest_probe.md` + `metrics.jsonl` to Drive every 500 steps.
  - Circuit breaker aborts cleanly after 3 consecutive unrecoverable NaN/Inf meltdowns.
  - On reaching `total_steps`, exports `dcpa_final.safetensors` + `config.json` into Drive
    `export/` and flushes OS buffers (`os.sync()`).
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
from typing import Any, Callable, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dcpa.config import DCPAConfig
from dcpa.data.shard_loader import BinaryShardDataset
from dcpa.data.synthetic import (
    generate_dyck_sequence,
    generate_fast_mapping_episode,
    generate_grid_physics_episode,
)
from dcpa.export import export_model_and_golden
from dcpa.model import DCPAModel
from dcpa.train.curriculum import CurriculumSchedule
from dcpa.train.monitor import AntiCollapseMonitor, project_spectral_norm
from train.colab_drive_sync import (
    ABDriveSyncManager,
    CheckpointTrigger,
    MILESTONE_STEPS,
    assert_not_drive_path,
    sync_file_to_drive,
    write_text_fsync,
)
from utils.device import (
    amp_autocast,
    capture_rng_state,
    describe_hardware_profile,
    detect_hardware_profile,
    resolve_micro_batch_plan,
    restore_rng_state,
    synchronize,
)


class MeltdownAbortError(RuntimeError):
    """Raised when training hits max consecutive unrecoverable NaN/Inf meltdowns."""


@dataclass
class TrainRunSummary:
    completed_steps: int
    total_steps: int
    resumed_from_step: int
    fallback_resume_used: bool
    last_stage: str
    last_metrics: dict[str, float]
    best_loss: float
    saves_performed: int
    final_export_path: Optional[Path]
    hardware_tier: str


def _parse_milestone_steps(raw: Optional[str]) -> tuple[int, ...]:
    if not raw:
        return MILESTONE_STEPS
    vals = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    return tuple(sorted(set(vals)))


def _sample_synthetic_batch(
    rng: random.Random,
    batch_size: int,
    seq_len: int,
    stage: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    raw_rows = []
    lang_ids = []
    weight_rows = []
    has_weights = stage == "D"

    for _ in range(batch_size):
        if stage == "D":
            ep = generate_fast_mapping_episode(rng, lang="en")
            b = ep.text.encode("utf-8")[:seq_len]
            w = torch.ones(seq_len, dtype=torch.float32)
            s_idx = min(ep.target_byte_start, seq_len)
            e_idx = min(ep.target_byte_end, seq_len)
            w[s_idx:e_idx] = 4.0
            lid = ep.lang_id
        elif rng.random() < 0.5:
            txt = generate_grid_physics_episode(rng, num_steps=16)
            b = txt.encode("utf-8")[:seq_len]
            w = torch.ones(seq_len, dtype=torch.float32)
            lid = 5
        else:
            txt, _ = generate_dyck_sequence(
                rng, max_depth=8, min_bytes=seq_len, rust_flavored=(stage == "C")
            )
            b = txt.encode("utf-8")[:seq_len]
            w = torch.ones(seq_len, dtype=torch.float32)
            lid = 4 if stage == "C" else 5

        if len(b) < seq_len:
            b = b + b"\n" * (seq_len - len(b))
        raw_rows.append(torch.tensor(list(b), dtype=torch.uint8))
        lang_ids.append(lid)
        weight_rows.append(w)

    raw_t = torch.stack(raw_rows, dim=0).to(device)
    lang_t = torch.tensor(lang_ids, dtype=torch.long, device=device)
    w_t = torch.stack(weight_rows, dim=0).to(device) if has_weights else None
    return raw_t, lang_t, w_t


def _sample_micro_batch(
    rng: random.Random,
    shard_ds: Optional[BinaryShardDataset],
    batch_size: int,
    seq_len: int,
    stage: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Stage D mixes online fast-mapping episodes (100% synthetic category 4 episodes)
    # Stages A/B/C sample predominantly from NVMe shards (with 15% synthetic structural drills)
    use_shards = (
        shard_ds is not None
        and len(shard_ds) > 0
        and stage != "D"
        and rng.random() >= 0.15
    )
    if use_shards and shard_ds is not None:
        items = [shard_ds[rng.randrange(len(shard_ds))] for _ in range(batch_size)]
        raw_bytes = torch.stack([x[0] for x in items], dim=0).to(device)
        lang_ids = torch.stack([x[1] for x in items], dim=0).to(device)
        return raw_bytes, lang_ids, None
    return _sample_synthetic_batch(rng, batch_size, seq_len, stage, device)


@torch.no_grad()
def _run_probe_diagnostics(
    model: DCPAModel,
    val_ds: Optional[BinaryShardDataset],
    rng: random.Random,
    device: torch.device,
    use_bf16: bool,
    step: int,
    total_steps: int,
    stage: str,
    tau_s: float,
    tau_m: float,
    lr: float,
    train_metrics: dict[str, float],
    alerts: list[str],
    elapsed_sec: float,
    bytes_per_sec: float,
    seq_len: int,
) -> tuple[Optional[float], str]:
    """Compute lightweight validation probe + SGSC segmentation preview markdown."""
    was_training = model.training
    model.eval()

    val_loss: Optional[float] = None
    if val_ds is not None and len(val_ds) > 0:
        n_eval = min(2, len(val_ds))
        items = [val_ds[rng.randrange(len(val_ds))] for _ in range(n_eval)]
        raw_v = torch.stack([x[0] for x in items], dim=0).to(device)
        lang_v = torch.stack([x[1] for x in items], dim=0).to(device)
        with amp_autocast(device, dtype=torch.bfloat16, enabled=use_bf16):
            v_out = model(
                raw_bytes=raw_v,
                lang_ids=lang_v,
                stage=stage,
                tau_s=tau_s,
                tau_m=tau_m,
                use_broca_cq=False,
            )
        val_loss = float(v_out.loss_total.detach().item())

    # SGSC live segmentation sample across ZH + EN + Rust + Emoji
    sample_text = "A zibber is a tiny frog! 你好世界🦀 fn main() { let x = 42; }"
    sample_raw = sample_text.encode("utf-8")
    x_probe = torch.tensor(list(sample_raw), dtype=torch.uint8, device=device).unsqueeze(0)
    sgsc_out = model.sgsc(x_probe, tau_s=tau_s)
    cu = sgsc_out.cu_seqlens.detach().cpu().tolist()
    segs = [sample_raw[cu[i] : cu[i + 1]].decode("utf-8", errors="replace") for i in range(len(cu) - 1)]
    segmented_preview = " | ".join(segs)

    if was_training:
        model.train()

    val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
    alert_str = ", ".join(alerts) if alerts else "None"
    md = "\n".join(
        [
            f"# DCPA Live Diagnostic Probe — Step {step:,} / {total_steps:,}",
            "",
            f"- **UTC Timestamp**: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
            f"- **Developmental Stage**: `{stage}` (`tau_s={tau_s:.3f}`, `tau_m={tau_m:.3f}`, `lr={lr:.2e}`)",
            f"- **Throughput**: `{bytes_per_sec:,.0f} bytes/sec` | **Elapsed**: `{elapsed_sec / 60.0:.1f} min` | **SeqLen**: `{seq_len}`",
            f"- **Validation Loss**: `{val_str}`",
            f"- **Active Rescue Alerts**: `{alert_str}`",
            "",
            "## Loss & Anti-Collapse Metrics",
            "",
            "| Metric | Value |",
            "| :--- | ---: |",
            f"| `loss_total` | {train_metrics.get('loss_total', 0.0):.4f} |",
            f"| `loss_lang` | {train_metrics.get('loss_lang', 0.0):.4f} |",
            f"| `loss_sgsc_prior` | {train_metrics.get('loss_sgsc_prior', 0.0):.4f} |",
            f"| `loss_pc` | {train_metrics.get('loss_pc', 0.0):.4f} |",
            f"| `loss_var` | {train_metrics.get('loss_var', 0.0):.4f} |",
            f"| `loss_rate` | {train_metrics.get('loss_rate', 0.0):.4f} |",
            f"| `loss_halt` | {train_metrics.get('loss_halt', 0.0):.4f} |",
            f"| `spike_rate` | {train_metrics.get('spike_rate', 0.0):.4f} |",
            f"| `compression_ratio` | {train_metrics.get('compression_ratio', 0.0):.2f}x |",
            f"| `theta` | {train_metrics.get('theta', 0.0):.4f} |",
            f"| `c_slow_std_mean` | {train_metrics.get('c_slow_std_mean', 0.0):.4f} |",
            f"| `merge_entropy_nats` | {train_metrics.get('merge_entropy_nats', 0.0):.4f} |",
            f"| `avg_ponder_steps` | {train_metrics.get('avg_ponder_steps', 0.0):.3f} |",
            f"| `mean_ach` | {train_metrics.get('mean_ach', 0.0):.4f} |",
            f"| `mean_beta` | {train_metrics.get('mean_beta', 0.0):.4f} |",
            "",
            "## Live SGSC Byte Segmentation Preview",
            "",
            f"```text\n{segmented_preview}\n```",
            "",
        ]
    )
    return val_loss, md


def _serialize_monitor(monitor: AntiCollapseMonitor) -> dict[str, Any]:
    return {
        "target_spike_rate": monitor.target_spike_rate,
        "bad_spike_streak": monitor.bad_spike_streak,
        "dual_lagrange": monitor.dual_lagrange,
        "tau_m_reheat_steps": monitor.tau_m_reheat_steps,
        "step_count": monitor.step_count,
    }


def _restore_monitor(monitor: AntiCollapseMonitor, state: Optional[dict[str, Any]]) -> None:
    if not state:
        return
    monitor.bad_spike_streak = int(state.get("bad_spike_streak", 0))
    monitor.dual_lagrange = float(state.get("dual_lagrange", 0.0))
    monitor.tau_m_reheat_steps = int(state.get("tau_m_reheat_steps", 0))
    monitor.step_count = int(state.get("step_count", 0))


def _check_grads_finite(model: DCPAModel) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def run_training(
    preset: str = "tiny",
    total_steps: int = 100_000,
    seq_len: int = 2048,
    target_tokens_bytes_per_step: Optional[int] = None,
    micro_batch_bytes: Optional[int] = None,
    data_dir: Optional[str | Path] = None,
    local_cache_dir: str | Path = "artifacts/nvme_cache",
    drive_sync_dir: Optional[str | Path] = None,
    drive_logs_dir: Optional[str | Path] = None,
    drive_export_dir: Optional[str | Path] = None,
    save_every_steps: int = 1000,
    save_every_minutes: float = 10.0,
    probe_every_steps: int = 500,
    heartbeat_seconds: float = 60.0,
    resume: str = "auto",
    device_str: str = "auto",
    seed: int = 1337,
    max_consecutive_meltdowns: int = 3,
    milestone_steps: tuple[int, ...] = MILESTONE_STEPS,
    export_final: bool = True,
    clock: Callable[[], float] = time.monotonic,
    step_hook: Optional[Callable[[int, DCPAModel, dict[str, float]], None]] = None,
) -> TrainRunSummary:
    """Execute resilient DCPA developmental training with NVMe cache + Drive A/B sync."""
    local_cache = assert_not_drive_path(local_cache_dir, role="--local-cache-dir")
    local_logs_dir = local_cache / "logs"
    local_export_dir = local_cache / "export"
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    local_export_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
        assert_not_drive_path(data_dir, role="--data-dir")

    profile = detect_hardware_profile(
        preferred=device_str,
        deterministic=False,
        target_tokens_bytes_per_step=target_tokens_bytes_per_step,
    )
    device = profile.device
    micro_batch_size, grad_accum_steps, actual_step_bytes = resolve_micro_batch_plan(
        profile=profile,
        seq_len=seq_len,
        target_tokens_bytes_per_step=target_tokens_bytes_per_step,
        micro_batch_bytes_override=micro_batch_bytes,
    )
    print(describe_hardware_profile(profile), flush=True)
    print(
        f"[Plan] preset={preset} | total_steps={total_steps:,} | seq_len={seq_len} | "
        f"micro_batch_size={micro_batch_size} | grad_accum_steps={grad_accum_steps} | "
        f"effective_bytes_per_step={actual_step_bytes:,}",
        flush=True,
    )

    torch.manual_seed(seed)
    rng = random.Random(seed)

    cfg = {"base": DCPAConfig.base, "tiny": DCPAConfig.tiny, "micro": DCPAConfig.micro}[preset]()
    model = DCPAModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups())
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    schedule = CurriculumSchedule(total_steps=total_steps)
    monitor = AntiCollapseMonitor(target_spike_rate=cfg.target_spike_rate)

    sync_mgr = ABDriveSyncManager(
        local_cache_dir=local_cache,
        drive_sync_dir=drive_sync_dir,
        milestone_steps=milestone_steps,
    )
    trigger = CheckpointTrigger(
        save_every_steps=save_every_steps,
        save_every_minutes=save_every_minutes,
        clock=clock,
    )

    start_step = 0
    resumed_from_step = 0
    fallback_resume_used = False
    best_loss = float("inf")

    # Self-healing resume logic
    if resume and resume.lower() != "none":
        if resume.lower() == "auto":
            res = sync_mgr.load_latest_auto(map_location=device)
            if res is not None:
                ckpt = res.checkpoint
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                _restore_monitor(monitor, ckpt.get("monitor_state"))
                restore_rng_state(ckpt.get("rng_state"))
                start_step = int(ckpt.get("step", res.step))
                resumed_from_step = start_step
                fallback_resume_used = res.fallback_triggered
                best_loss = float(ckpt.get("best_loss", float("inf")))
                trigger.mark_saved(start_step)
        else:
            explicit_path = Path(resume)
            ckpt = torch.load(str(explicit_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            _restore_monitor(monitor, ckpt.get("monitor_state"))
            restore_rng_state(ckpt.get("rng_state"))
            start_step = int(ckpt.get("step", 0))
            resumed_from_step = start_step
            best_loss = float(ckpt.get("best_loss", float("inf")))
            trigger.mark_saved(start_step)

    # Open hot NVMe mmap datasets (strictly non-Drive)
    train_ds: Optional[BinaryShardDataset] = None
    val_ds: Optional[BinaryShardDataset] = None
    if data_dir is not None:
        dd = Path(data_dir)
        t_bin, t_idx = dd / "train.bin", dd / "train.idx"
        v_bin, v_idx = dd / "val.bin", dd / "val.idx"
        if t_bin.exists() and t_idx.exists():
            train_ds = BinaryShardDataset(t_bin, t_idx, seq_len=seq_len)
            print(f"[Data] Opened local NVMe train shard: {t_bin} ({len(train_ds):,} docs)", flush=True)
        if v_bin.exists() and v_idx.exists():
            val_ds = BinaryShardDataset(v_bin, v_idx, seq_len=seq_len)
            print(f"[Data] Opened local NVMe val shard: {v_bin} ({len(val_ds):,} docs)", flush=True)

    local_metrics_path = local_logs_dir / "metrics.jsonl"
    local_probe_path = local_logs_dir / "latest_probe.md"

    last_stage: Optional[str] = None
    last_metrics: dict[str, float] = {}
    directives = monitor.observe({}, stage=schedule.get(min(start_step, total_steps - 1)).stage, step_in_stage=0)
    saves_performed = 0
    consecutive_meltdowns = 0
    lr_backoff_factor = 1.0
    t_run_start = clock()
    last_heartbeat_t = t_run_start
    steps_since_start = 0

    def _build_payload(completed_step: int, stage_id: str) -> dict[str, Any]:
        return {
            "step": int(completed_step),
            "stage_id": str(stage_id),
            "preset": preset,
            "config": cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "monitor_state": _serialize_monitor(monitor),
            "best_loss": float(best_loss),
            "rng_state": capture_rng_state(),
            "hardware_tier": profile.tier,
            "total_steps": int(total_steps),
            "seq_len": int(seq_len),
            "seed": int(seed),
        }

    try:
        for step_idx in range(start_step, total_steps):
            completed_step = step_idx + 1
            sc = schedule.get(step_idx)
            if sc.stage != last_stage:
                model.set_stage_freeze(sc.stage)
                print(
                    f"[Stage] Entering Stage {sc.stage} at step={completed_step:,}/{total_steps:,}",
                    flush=True,
                )
                last_stage = sc.stage

            for g, blr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = blr * sc.lr_scale * lr_backoff_factor

            optimizer.zero_grad(set_to_none=True)
            tau_m = sc.tau_m if directives.tau_m_override is None else directives.tau_m_override

            accum_metrics: dict[str, float] = {}
            step_finite = True

            for _micro_idx in range(grad_accum_steps):
                raw_bytes, lang_ids, loss_w = _sample_micro_batch(
                    rng=rng,
                    shard_ds=train_ds,
                    batch_size=micro_batch_size,
                    seq_len=seq_len,
                    stage=sc.stage,
                    device=device,
                )
                with amp_autocast(device, dtype=torch.bfloat16, enabled=profile.use_bf16):
                    out = model(
                        raw_bytes=raw_bytes,
                        lang_ids=lang_ids,
                        stage=sc.stage,
                        tau_s=sc.tau_s,
                        tau_m=tau_m,
                        use_broca_cq=sc.use_broca_cq,
                        loss_weights=loss_w,
                        lambda_rate_override=directives.lambda_rate_override,
                        dual_rate_lagrange=directives.dual_rate_lagrange,
                        lambda_var_override=directives.lambda_var_override,
                        merge_kl_weight=directives.merge_kl_weight,
                        ach_hint_weight=directives.ach_hint_weight,
                    )

                if not torch.isfinite(out.loss_total):
                    step_finite = False
                    break

                scaled_loss = out.loss_total / float(grad_accum_steps)
                scaled_loss.backward()

                for k, v in out.metrics.items():
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + float(v) / float(grad_accum_steps)

            if step_finite and not _check_grads_finite(model):
                step_finite = False

            if not step_finite:
                consecutive_meltdowns += 1
                optimizer.zero_grad(set_to_none=True)
                lr_backoff_factor *= 0.5
                print(
                    f"[CIRCUIT_BREAKER] Non-finite loss/grad at step={completed_step} "
                    f"(meltdown {consecutive_meltdowns}/{max_consecutive_meltdowns}); "
                    f"lr_backoff={lr_backoff_factor:.4f}",
                    flush=True,
                )
                if consecutive_meltdowns >= max_consecutive_meltdowns:
                    raise MeltdownAbortError(
                        f"Aborting after {consecutive_meltdowns} consecutive unrecoverable NaN/Inf meltdowns "
                        f"at step {completed_step}."
                    )
                continue

            consecutive_meltdowns = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            steps_since_start += 1

            directives = monitor.observe(accum_metrics, stage=sc.stage, step_in_stage=sc.step_in_stage)
            if directives.apply_wsem_spectral_norm:
                project_spectral_norm(model.sgsc.w_sem.weight, max_sigma=1.0)

            last_metrics = accum_metrics
            if step_hook is not None:
                step_hook(completed_step, model, last_metrics)

            now_t = clock()
            elapsed = max(now_t - t_run_start, 1e-6)
            bytes_per_sec = float(steps_since_start * actual_step_bytes) / elapsed

            # 60-second concise heartbeat (plus first and final steps)
            is_final_step = completed_step == total_steps
            if (
                completed_step == start_step + 1
                or is_final_step
                or (heartbeat_seconds > 0 and (now_t - last_heartbeat_t) >= heartbeat_seconds)
            ):
                alert_s = f" ALERTS={directives.alerts}" if directives.alerts else ""
                print(
                    f"[HB step={completed_step:06d}/{total_steps:06d} | Stage {sc.stage}] "
                    f"loss={last_metrics['loss_total']:.4f} lang={last_metrics['loss_lang']:.4f} "
                    f"pc={last_metrics['loss_pc']:.4f} rate={last_metrics['spike_rate']:.3f} "
                    f"std={last_metrics['c_slow_std_mean']:.2f} "
                    f"speed={bytes_per_sec / 1024.0:.1f}KB/s{alert_s}",
                    flush=True,
                )
                last_heartbeat_t = now_t

            # Diagnostic probe + Drive logs sync every `probe_every_steps` (default 500)
            is_best = False
            if probe_every_steps > 0 and (completed_step % probe_every_steps == 0 or is_final_step):
                val_loss, probe_md = _run_probe_diagnostics(
                    model=model,
                    val_ds=val_ds,
                    rng=rng,
                    device=device,
                    use_bf16=profile.use_bf16,
                    step=completed_step,
                    total_steps=total_steps,
                    stage=sc.stage,
                    tau_s=sc.tau_s,
                    tau_m=tau_m,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    train_metrics=last_metrics,
                    alerts=directives.alerts,
                    elapsed_sec=elapsed,
                    bytes_per_sec=bytes_per_sec,
                    seq_len=seq_len,
                )
                eval_metric = val_loss if val_loss is not None else float(last_metrics["loss_total"])
                if math.isfinite(eval_metric) and eval_metric < best_loss:
                    best_loss = eval_metric
                    is_best = True

                record = {
                    "step": completed_step,
                    "stage": sc.stage,
                    "wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "bytes_per_sec": round(bytes_per_sec, 1),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "val_loss": val_loss,
                    "alerts": directives.alerts,
                    **last_metrics,
                }
                with local_metrics_path.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    mf.flush()
                write_text_fsync(local_probe_path, probe_md)

                if drive_logs_dir is not None:
                    sync_file_to_drive(local_probe_path, drive_logs_dir, dest_name="latest_probe.md")
                    sync_file_to_drive(local_metrics_path, drive_logs_dir, dest_name="metrics.jsonl")

            # Dual-trigger (step OR wall-clock timer) OR milestone OR final step checkpoint
            due, reason = trigger.should_save(completed_step)
            is_milestone = completed_step in milestone_steps
            if due or is_milestone or is_final_step or is_best:
                save_reason = reason or ("milestone" if is_milestone else ("final" if is_final_step else "best"))
                synchronize(device)
                payload = _build_payload(completed_step, sc.stage)
                sync_mgr.save_ping_pong(
                    payload=payload,
                    step=completed_step,
                    stage_id=sc.stage,
                    reason=save_reason,
                    is_best=is_best,
                )
                trigger.mark_saved(completed_step)
                saves_performed += 1

        # Final safetensors export on completion
        final_export_path: Optional[Path] = None
        if export_final:
            synchronize(device)
            exported = export_model_and_golden(model, local_export_dir)
            # Move model back to original device if caller inspects it later
            model.to(device)
            final_export_path = exported["weights"]
            if drive_export_dir is not None:
                final_export_path = sync_file_to_drive(
                    exported["weights"], drive_export_dir, dest_name="dcpa_final.safetensors"
                )
                sync_file_to_drive(exported["config"], drive_export_dir, dest_name="config.json")
                sync_file_to_drive(exported["golden"], drive_export_dir, dest_name="golden.safetensors")
                sync_file_to_drive(exported["meta"], drive_export_dir, dest_name="golden_meta.json")
                print(f"[Export] Flushed dcpa_final.safetensors -> {final_export_path}", flush=True)
            if hasattr(os, "sync"):
                os.sync()

        return TrainRunSummary(
            completed_steps=total_steps,
            total_steps=total_steps,
            resumed_from_step=resumed_from_step,
            fallback_resume_used=fallback_resume_used,
            last_stage=last_stage or schedule.get(max(0, total_steps - 1)).stage,
            last_metrics=last_metrics,
            best_loss=best_loss,
            saves_performed=saves_performed,
            final_export_path=final_export_path,
            hardware_tier=profile.tier,
        )
    finally:
        if train_ds is not None:
            train_ds.close()
        if val_ds is not None:
            val_ds.close()
        if hasattr(os, "sync"):
            os.sync()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DCPA Cross-Platform & Colab Anti-Disconnect Trainer")
    ap.add_argument("--preset", choices=["micro", "tiny", "base"], default="tiny")
    ap.add_argument("--total-steps", type=int, default=100_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--target-tokens-bytes-per-step", type=int, default=None)
    ap.add_argument("--micro-batch-bytes", type=int, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--local-cache-dir", type=str, default="artifacts/nvme_cache")
    ap.add_argument("--drive-sync-dir", type=str, default=None)
    ap.add_argument("--drive-logs-dir", type=str, default=None)
    ap.add_argument("--drive-export-dir", type=str, default=None)
    ap.add_argument("--save-every-steps", type=int, default=1000)
    ap.add_argument("--save-every-minutes", type=float, default=10.0)
    ap.add_argument("--probe-every-steps", type=int, default=500)
    ap.add_argument("--heartbeat-seconds", type=float, default=60.0)
    ap.add_argument("--resume", type=str, default="auto")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-consecutive-meltdowns", type=int, default=3)
    ap.add_argument("--milestone-steps", type=str, default="15000,65000,85000,100000")
    ap.add_argument("--no-export", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        summary = run_training(
            preset=args.preset,
            total_steps=args.total_steps,
            seq_len=args.seq_len,
            target_tokens_bytes_per_step=args.target_tokens_bytes_per_step,
            micro_batch_bytes=args.micro_batch_bytes,
            data_dir=args.data_dir,
            local_cache_dir=args.local_cache_dir,
            drive_sync_dir=args.drive_sync_dir,
            drive_logs_dir=args.drive_logs_dir,
            drive_export_dir=args.drive_export_dir,
            save_every_steps=args.save_every_steps,
            save_every_minutes=args.save_every_minutes,
            probe_every_steps=args.probe_every_steps,
            heartbeat_seconds=args.heartbeat_seconds,
            resume=args.resume,
            device_str=args.device,
            seed=args.seed,
            max_consecutive_meltdowns=args.max_consecutive_meltdowns,
            milestone_steps=_parse_milestone_steps(args.milestone_steps),
            export_final=not args.no_export,
        )
        print(
            f"[DONE] Completed {summary.completed_steps:,}/{summary.total_steps:,} steps "
            f"(saves={summary.saves_performed}, best_loss={summary.best_loss:.4f}, "
            f"export={summary.final_export_path})",
            flush=True,
        )
    except MeltdownAbortError as exc:
        print(f"[FATAL MELTDOWN] {exc}", file=sys.stderr, flush=True)
        sys.exit(42)


if __name__ == "__main__":
    main()
