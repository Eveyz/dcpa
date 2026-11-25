"""Weights & Biases (`wandb`) telemetry tracker with Google Drive checkpoint run-ID continuity.

Features:
  - Zero hard dependency when disabled: lazy `import wandb` only when `enabled=True`.
  - Rank-0 exclusive (`is_primary=True`) so multi-GPU DDP never spawns duplicate W&B runs.
  - Auto-reads `WANDB_API_KEY` from environment or Colab Secrets (`google.colab.userdata`) when available;
    automatically falls back to `mode="offline"` if no API key is configured in non-interactive sessions
    unless `mode` is explicitly forced.
  - Persists `run_id` into `logs_dir/wandb_run.json` (and checkpoint payloads via `tracker.run_id`) so
    Colab / Vast.ai / DGX Spark preemptions resume the exact same W&B curve (`resume="allow"`).
"""

from __future__ import annotations

import json
import math
import os
import secrets
from pathlib import Path
from typing import Any, Mapping, Optional


def _try_load_colab_secret(secret_name: str) -> Optional[str]:
    """Safely probe `google.colab.userdata` without failing outside Google Colab."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        val = userdata.get(secret_name)
        if val and str(val).strip():
            return str(val).strip()
    except Exception:
        return None
    return None


def resolve_wandb_settings(
    *,
    yaml_wandb_cfg: Optional[Mapping[str, Any]] = None,
    cli_enabled: Optional[bool] = None,
    cli_project: Optional[str] = None,
    cli_entity: Optional[str] = None,
    cli_run_name: Optional[str] = None,
    cli_mode: Optional[str] = None,
    cli_tags: Optional[str] = None,
    cli_log_every: Optional[int] = None,
    default_project: str = "dcpa-turbo",
) -> dict[str, Any]:
    """Merge YAML `wandb:` section, environment variables, and CLI overrides."""
    cfg = dict(yaml_wandb_cfg or {})
    env_mode = os.environ.get("WANDB_MODE", "").strip().lower()
    env_enabled = os.environ.get("DCPA_WANDB", "").strip().lower()

    if cli_enabled is not None:
        enabled = bool(cli_enabled)
    elif env_enabled in ("1", "true", "yes", "on"):
        enabled = True
    elif env_enabled in ("0", "false", "no", "off"):
        enabled = False
    else:
        enabled = bool(cfg.get("enabled", False))

    mode = (cli_mode or env_mode or str(cfg.get("mode", "online"))).strip().lower()
    if mode not in ("online", "offline", "disabled"):
        mode = "online"
    if mode == "disabled":
        enabled = False

    project = (
        cli_project
        or os.environ.get("WANDB_PROJECT", "").strip()
        or str(cfg.get("project", default_project)).strip()
        or default_project
    )
    entity_raw = (
        cli_entity
        or os.environ.get("WANDB_ENTITY", "").strip()
        or str(cfg.get("entity", "") or "").strip()
    )
    entity = entity_raw if entity_raw else None

    run_name_raw = (
        cli_run_name
        or os.environ.get("WANDB_NAME", "").strip()
        or str(cfg.get("run_name", "") or "").strip()
    )
    run_name = run_name_raw if run_name_raw else None

    raw_tags = cfg.get("tags", [])
    tags: list[str] = [str(t).strip() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, (list, tuple)) else []
    if cli_tags:
        for tok in cli_tags.split(","):
            t = tok.strip()
            if t and t not in tags:
                tags.append(t)

    log_every = int(cli_log_every) if cli_log_every is not None else int(cfg.get("log_every_steps", 10))
    return {
        "enabled": enabled,
        "mode": mode,
        "project": project,
        "entity": entity,
        "run_name": run_name,
        "tags": tags,
        "log_every_steps": max(1, log_every),
    }


class WandbTracker:
    """Rank-0-safe Weights & Biases session manager with Google Drive run-ID persistence."""

    META_FILENAME = "wandb_run.json"

    def __init__(
        self,
        *,
        enabled: bool = False,
        is_primary: bool = True,
        project: str = "dcpa-turbo",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        mode: str = "online",
        tags: Optional[list[str]] = None,
        log_every_steps: int = 10,
        logs_dir: Optional[str | Path] = None,
        drive_logs_dir: Optional[str | Path] = None,
        config_payload: Optional[Mapping[str, Any]] = None,
        wandb_module: Any = None,
    ) -> None:
        self.requested_enabled = bool(enabled) and bool(is_primary) and mode != "disabled"
        self.is_primary = bool(is_primary)
        self.project = str(project or "dcpa-turbo")
        self.entity = entity
        self.run_name = run_name
        self.mode = str(mode or "online").lower()
        self.tags = list(tags or [])
        self.log_every_steps = max(1, int(log_every_steps))
        self.logs_dir = Path(logs_dir) if logs_dir is not None else None
        self.drive_logs_dir = Path(drive_logs_dir) if drive_logs_dir is not None else None
        self.config_payload = dict(config_payload or {})
        self._wandb = wandb_module
        self._run: Any = None
        self.active: bool = False
        self.run_id: Optional[str] = run_id

    @classmethod
    def load_persisted_run_id(
        cls,
        logs_dir: Optional[str | Path],
        drive_logs_dir: Optional[str | Path] = None,
    ) -> Optional[str]:
        """Read previously saved `wandb_run.json` from local NVMe logs or Google Drive logs."""
        for base in (logs_dir, drive_logs_dir):
            if base is None:
                continue
            p = Path(base) / cls.META_FILENAME
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    rid = str(data.get("run_id", "")).strip()
                    if rid:
                        return rid
                except Exception:
                    continue
        return None

    def _persist_run_meta(self) -> None:
        if not self.is_primary or not self.run_id or self.logs_dir is None:
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.logs_dir / self.META_FILENAME
        tmp_path = self.logs_dir / f"{self.META_FILENAME}.tmp"
        payload = {
            "run_id": self.run_id,
            "project": self.project,
            "entity": self.entity,
            "run_name": self.run_name,
            "mode": self.mode,
        }
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, meta_path)
        if self.drive_logs_dir is not None:
            try:
                from train.colab_drive_sync import sync_file_to_drive

                sync_file_to_drive(meta_path, self.drive_logs_dir, dest_name=self.META_FILENAME)
            except Exception:
                pass

    def start(self, *, resumed_run_id: Optional[str] = None, resume_step: int = 0) -> Optional[str]:
        """Initialize W&B run (or resume existing run ID) on Rank 0."""
        if not self.requested_enabled:
            return None

        wb = self._wandb
        if wb is None:
            try:
                import wandb as wb_mod  # type: ignore[import-not-found]

                wb = wb_mod
                self._wandb = wb
            except ImportError:
                print(
                    "[WandB][WARN] `wandb` package is not installed (`pip install wandb`); continuing without W&B.",
                    flush=True,
                )
                self.active = False
                return None

        # Probe Colab Secret if WANDB_API_KEY is not already set in environment
        if not os.environ.get("WANDB_API_KEY"):
            secret_key = _try_load_colab_secret("WANDB_API_KEY")
            if secret_key:
                os.environ["WANDB_API_KEY"] = secret_key

        eff_mode = self.mode
        has_api_key = bool(os.environ.get("WANDB_API_KEY", "").strip())
        netrc_path = Path.home() / ".netrc"
        has_netrc = netrc_path.exists() and ("wandb" in netrc_path.read_text(encoding="utf-8", errors="ignore"))
        if eff_mode == "online" and not has_api_key and not has_netrc:
            print(
                "[WandB][INFO] WANDB_API_KEY not found in env/Colab Secrets; switching W&B to `offline` mode "
                "(set WANDB_API_KEY to sync live curves).",
                flush=True,
            )
            eff_mode = "offline"
        self.mode = eff_mode

        eff_id = (
            resumed_run_id
            or self.run_id
            or (self.load_persisted_run_id(self.logs_dir, self.drive_logs_dir) if resume_step > 0 else None)
            or os.environ.get("WANDB_RUN_ID", "").strip()
            or secrets.token_hex(8)
        )
        self.run_id = eff_id

        init_dir = str(self.logs_dir) if self.logs_dir is not None else None
        if init_dir is not None:
            Path(init_dir).mkdir(parents=True, exist_ok=True)

        try:
            self._run = wb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                id=eff_id,
                resume="allow",
                mode=eff_mode,
                tags=self.tags if self.tags else None,
                dir=init_dir,
                config=self.config_payload,
            )
            self.active = self._run is not None
            if self._run is not None and getattr(self._run, "id", None):
                self.run_id = str(self._run.id)
            self._persist_run_meta()
            url_info = getattr(self._run, "url", None) or f"mode={eff_mode}"
            print(
                f"[WandB] Active: project={self.project} | run_id={self.run_id} | "
                f"resumed_step={resume_step} | {url_info}",
                flush=True,
            )
        except Exception as exc:
            print(f"[WandB][WARN] Failed to initialize wandb ({exc}); continuing training without W&B.", flush=True)
            self.active = False
        return self.run_id

    def should_log_step(self, step: int, *, total_steps: int = 0) -> bool:
        if not self.active:
            return False
        return step == 1 or (total_steps > 0 and step == total_steps) or (step % self.log_every_steps == 0)

    def log_train_step(
        self,
        *,
        step: int,
        stage_id: int,
        metrics: Mapping[str, float],
        applied_lrs: Optional[Mapping[str, float]] = None,
        kb_per_sec: Optional[float] = None,
        peak_mem_gb: Optional[float] = None,
        watchdog_lr_mult: Optional[float] = None,
        total_steps: int = 0,
    ) -> None:
        """Log structured DCPA-Turbo step telemetry at `step`."""
        if not self.should_log_step(step, total_steps=total_steps) or self._run is None:
            return

        payload: dict[str, float | int] = {"curriculum/stage_id": int(stage_id)}
        loss_lang = metrics.get("loss_lang")
        if loss_lang is not None and math.isfinite(float(loss_lang)):
            payload["train/bpc"] = float(loss_lang) / math.log(2.0)

        key_map = {
            "loss_total": "train/loss_total",
            "loss_lang": "train/loss_lang",
            "loss_pc": "train/loss_pc",
            "loss_halt": "train/loss_halt",
            "loss_act": "train/loss_act",
            "loss_rate": "train/loss_rate",
            "loss_var": "train/loss_var",
            "loss_moe_bal": "train/loss_moe_bal",
            "grad_norm": "train/grad_norm",
            "spike_rate": "sgsc/spike_rate",
            "compression_ratio": "sgsc/compression_ratio",
            "theta": "sgsc/theta",
            "c_slow_std_min": "sgsc/c_slow_std_min",
            "avg_ponder_steps": "bcg_pc/avg_ponder_steps",
            "ponder_skip_rate": "bcg_pc/ponder_skip_rate",
            "merge_entropy_nats": "bcg_pc/merge_entropy_nats",
            "surprisal_mean": "bcg_pc/surprisal_mean",
            "mean_ach": "hippocampus/mean_ach",
            "mean_beta": "hippocampus/mean_beta",
            "familiar_ach": "hippocampus/familiar_ach",
        }
        for src_k, dst_k in key_map.items():
            if src_k in metrics:
                val = float(metrics[src_k])
                if math.isfinite(val):
                    payload[dst_k] = val

        if applied_lrs:
            active_vals = [float(v) for v in applied_lrs.values() if float(v) > 0.0]
            payload["optim/lr_max"] = max(active_vals) if active_vals else 0.0
            for gk, gv in applied_lrs.items():
                if math.isfinite(float(gv)):
                    payload[f"optim/lr_{gk}"] = float(gv)
        if watchdog_lr_mult is not None and math.isfinite(float(watchdog_lr_mult)):
            payload["optim/watchdog_lr_mult"] = float(watchdog_lr_mult)
        if kb_per_sec is not None and math.isfinite(float(kb_per_sec)):
            payload["perf/kb_per_sec"] = float(kb_per_sec)
        if peak_mem_gb is not None and math.isfinite(float(peak_mem_gb)):
            payload["perf/peak_vram_gb"] = float(peak_mem_gb)

        try:
            self._run.log(payload, step=int(step))
        except Exception:
            pass

    def log_visual_probe(self, *, step: int, probe_result: Any, best_val_metric: float) -> None:
        """Log 500-step zero-grad validation BPC & Wug fast-mapping metrics."""
        if not self.active or self._run is None:
            return
        payload: dict[str, float] = {}
        for attr, key in (
            ("train_bpc", "eval/train_bpc"),
            ("val_bpc_overall", "eval/val_bpc_overall"),
            ("wug_target_bpc", "eval/wug_target_bpc"),
        ):
            val = getattr(probe_result, attr, None)
            if val is not None and math.isfinite(float(val)):
                payload[key] = float(val)
        val_by_dom = getattr(probe_result, "val_bpc_by_domain", None)
        if isinstance(val_by_dom, Mapping):
            for dom_name, dom_bpc in val_by_dom.items():
                if dom_bpc is not None and math.isfinite(float(dom_bpc)):
                    payload[f"eval/val_bpc_{dom_name}"] = float(dom_bpc)
        if math.isfinite(float(best_val_metric)):
            payload["eval/best_metric"] = float(best_val_metric)
        if payload:
            try:
                self._run.log(payload, step=int(step))
            except Exception:
                pass

    def log_watchdog_rollback(
        self,
        *,
        step: int,
        rule: str,
        reason: str,
        rollback_count: int,
        lr_multiplier: float,
    ) -> None:
        """Record self-healing watchdog rollback event."""
        if not self.active or self._run is None:
            return
        try:
            self._run.log(
                {
                    "watchdog/rollback_count": int(rollback_count),
                    "watchdog/lr_multiplier": float(lr_multiplier),
                },
                step=int(step),
            )
            if hasattr(self._wandb, "AlertLevel") and hasattr(self._run, "alert"):
                self._run.alert(
                    title=f"DCPA Watchdog Rule {rule} Rollback (step {step})",
                    text=f"{reason} | rollback_count={rollback_count} | lr_mult={lr_multiplier:.4f}",
                    level=self._wandb.AlertLevel.WARN,
                )
        except Exception:
            pass

    def log_sft_step(
        self,
        *,
        step: int,
        metrics: Mapping[str, float],
        lr: float,
        kb_per_sec: float,
        peak_mem_gb: float,
        eot_grad_norm_max: float,
        total_steps: int = 0,
    ) -> None:
        """Log Stage 5 SFT metrics."""
        if not self.should_log_step(step, total_steps=total_steps) or self._run is None:
            return
        loss_lang = float(metrics.get("loss_lang", 0.0))
        payload: dict[str, float] = {
            "sft/loss_total": float(metrics.get("loss_total", 0.0)),
            "sft/loss_lang": loss_lang,
            "sft/bpc": loss_lang / math.log(2.0),
            "sft/loss_pc": float(metrics.get("loss_pc", 0.0)),
            "sft/loss_halt": float(metrics.get("loss_halt", 0.0)),
            "sft/lr": float(lr),
            "sft/kb_per_sec": float(kb_per_sec),
            "sft/peak_vram_gb": float(peak_mem_gb),
            "sft/eot_grad_norm_max": float(eot_grad_norm_max),
        }
        try:
            self._run.log({k: v for k, v in payload.items() if math.isfinite(v)}, step=int(step))
        except Exception:
            pass

    def finish(self, summary_extra: Optional[Mapping[str, Any]] = None) -> None:
        if not self.active or self._run is None:
            return
        try:
            if summary_extra and hasattr(self._run, "summary"):
                for k, v in summary_extra.items():
                    if v is not None:
                        self._run.summary[k] = v
            self._run.finish()
        except Exception:
            pass
        finally:
            self.active = False
            self._run = None
