"""Self-Healing Watchdog and Preemption-Safe Atomic Checkpoint Manager for DCPA.

Implements Deliverable 3 & Red Line #4:
  1. Atomic disk writes ONLY (`*.pt.tmp` -> `flush()` -> `os.fsync()` -> `os.replace()`).
  2. Dual-track checkpoint rotation under `--ckpt-dir`:
       - `last_healthy.pt` (every `healthy_save_every_steps`, default 1,000, only when healthy)
       - `milestone_step_{step:06d}.pt` (every 5,000 steps and right at every Stage boundary)
       - `best_model.pt` (new best `val/bpc` in Stages 1-3 or `val/wug_target_bpc` in Stage 4)
       - `emergency_interrupt.pt` (on `SIGINT` / `SIGTERM` preemption signals)
  3. Full deterministic checkpoint payload: `step`, `stage_id`, `model_state_dict`,
     `optimizer_state_dict`, `scheduler_state`, `dynamic_loss_multipliers`, `dataloader_cursor`,
     and cross-platform RNG state via `utils/device.py`.
  4. 20-step sliding-window Self-Healing Watchdog intercepting 4 failure classes:
       - Rule A: NaN/Inf in loss, any sub-loss, or gradients -> rollback + skip 10 batches +
                 advance seed (`seed + rollback_count * 9973`) + `lr *= 0.7`.
       - Rule B: 20-step avg `compression_ratio < 2.5` or `> 25.0` -> rollback + `lambda_rate *= 3.0`
                 (cap `50.0`) + `tau_s >= 0.5`.
       - Rule C: 20-step avg `c_slow_std_min < 0.35` -> rollback + `lambda_var *= 2.5` (cap `25.0`).
       - Rule D (Stage 2+): 20-step avg `merge_entropy < 0.08` -> rollback + `tau_m = min(1.2, tau_m + 0.4)`
                 + 10% uniform merge exploration & KL for 1,000 steps.
       - Circuit breaker: 3 consecutive rollbacks failing to advance 200 steps -> autopsy log +
         `MeltdownAbortError` (exit code 42).
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from dcpa.compile_utils import (
    get_clean_state_dict,
    migrate_legacy_moe_optimizer_state_dict,
    migrate_legacy_moe_state_dict,
    strip_orig_mod_prefix,
    unwrap_compiled_module,
)
from utils.device import capture_rng_state, restore_rng_state, seed_everything


class MeltdownAbortError(RuntimeError):
    """Raised when the Self-Healing Watchdog hits its consecutive-rollback circuit breaker."""


def _fsync_dir(dir_path: Path) -> None:
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_torch_save(payload: Mapping[str, Any], target_path: str | Path) -> Path:
    """Atomically serialize a checkpoint dict via `*.pt.tmp` + `flush()` + `os.fsync()` + `os.replace()`."""
    dst = Path(target_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp")
    with tmp.open("wb") as f:
        torch.save(dict(payload), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    _fsync_dir(dst.parent)
    return dst


def atomic_append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append a single JSONL event line with flush + fsync durability."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), ensure_ascii=False) + "\n"
    with dst.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


class PreemptionSignalGuard:
    """Context manager intercepting `SIGINT` and `SIGTERM` for graceful atomic checkpointing."""

    def __init__(self) -> None:
        self.interrupted: bool = False
        self.signal_name: Optional[str] = None
        self._prev_sigint: Any = None
        self._prev_sigterm: Any = None
        self._installed: bool = False

    def _handler(self, signum: int, _frame: Any) -> None:
        self.interrupted = True
        try:
            self.signal_name = signal.Signals(signum).name
        except Exception:
            self.signal_name = str(signum)
        print(
            f"\n[PreemptionGuard] Caught {self.signal_name}; finishing current step before atomic "
            "emergency_interrupt.pt save...",
            flush=True,
        )

    def __enter__(self) -> PreemptionSignalGuard:
        try:
            self._prev_sigint = signal.getsignal(signal.SIGINT)
            self._prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, self._handler)
            signal.signal(signal.SIGTERM, self._handler)
            self._installed = True
        except ValueError:
            # Signal handlers can only be registered on the main thread
            self._installed = False
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        if self._installed:
            try:
                signal.signal(signal.SIGINT, self._prev_sigint)
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            except ValueError:
                pass
            self._installed = False


@dataclass
class WatchdogDirectives:
    """Dynamic hyperparameter overrides currently enforced by the Self-Healing Watchdog."""

    lr_multiplier: float = 1.0
    lambda_rate_multiplier: float = 1.0
    lambda_var_multiplier: float = 1.0
    tau_s_floor: float = 0.0
    tau_m_boost: float = 0.0
    tau_m_cap: float = 1.2
    explore_steps_remaining: int = 0
    merge_uniform_mix: float = 0.0
    merge_kl_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> WatchdogDirectives:
        if not data:
            return cls()
        valid = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**valid)


@dataclass
class WatchdogAction:
    """Decision returned by `SelfHealingWatchdog.check_step(...)`."""

    triggered: bool = False
    rule: Optional[str] = None         # "A", "B", "C", or "D"
    reason: str = ""
    rollback_needed: bool = False
    skip_batches: int = 0
    new_seed: Optional[int] = None
    event_record: dict[str, Any] = field(default_factory=dict)


class SelfHealingWatchdog:
    """20-step sliding-window collapse detector, automatic rollback healer, and circuit breaker."""

    SUB_LOSS_KEYS: tuple[str, ...] = (
        "loss_total",
        "loss_lang",
        "loss_sgsc_prior",
        "loss_pc",
        "loss_var",
        "loss_rate",
        "loss_ponder",
        "loss_halt",
        "loss_ach_reg",
    )

    def __init__(
        self,
        watchdog_cfg: Optional[Mapping[str, Any]] = None,
        base_seed: int = 1337,
        events_log_path: Optional[str | Path] = None,
        enable_event_logging: bool = True,
    ) -> None:
        cfg = dict(watchdog_cfg or {})
        self.window_size: int = int(cfg.get("window_size", 20))
        self.rule_bcd_warmup_steps: int = int(cfg.get("rule_bcd_warmup_steps", 500))
        self.compression_ratio_min: float = float(cfg.get("compression_ratio_min", 2.5))
        self.compression_ratio_max: float = float(cfg.get("compression_ratio_max", 25.0))
        self.concept_std_min_floor: float = float(cfg.get("concept_std_min_floor", 0.35))
        self.merge_entropy_min: float = float(cfg.get("merge_entropy_min", 0.08))

        self.skip_batches_on_rollback: int = int(cfg.get("skip_batches_on_rollback", 10))
        self.seed_stride_on_rollback: int = int(cfg.get("seed_stride_on_rollback", 9973))

        self.rule_a_lr_multiplier: float = float(cfg.get("rule_a_lr_multiplier", 0.7))
        self.rule_b_lambda_rate_multiplier: float = float(cfg.get("rule_b_lambda_rate_multiplier", 3.0))
        self.rule_b_lambda_rate_cap: float = float(cfg.get("rule_b_lambda_rate_cap", 50.0))
        self.rule_b_tau_s_floor: float = float(cfg.get("rule_b_tau_s_floor", 0.5))
        self.rule_c_lambda_var_multiplier: float = float(cfg.get("rule_c_lambda_var_multiplier", 2.5))
        self.rule_c_lambda_var_cap: float = float(cfg.get("rule_c_lambda_var_cap", 25.0))
        self.rule_d_tau_m_delta: float = float(cfg.get("rule_d_tau_m_delta", 0.4))
        self.rule_d_tau_m_cap: float = float(cfg.get("rule_d_tau_m_cap", 1.2))
        self.rule_d_explore_steps: int = int(cfg.get("rule_d_explore_steps", 1000))
        self.rule_d_uniform_mix: float = float(cfg.get("rule_d_uniform_mix", 0.10))
        self.rule_d_merge_kl_weight: float = float(cfg.get("rule_d_merge_kl_weight", 0.10))

        self.max_consecutive_rollbacks: int = int(cfg.get("max_consecutive_rollbacks", 3))
        self.min_progress_steps_to_reset_breaker: int = int(
            cfg.get("min_progress_steps_to_reset_breaker", 200)
        )

        self.base_seed: int = int(base_seed)
        self.enable_event_logging: bool = bool(enable_event_logging)
        self.events_log_path: Path = Path(
            events_log_path
            if events_log_path is not None
            else cfg.get("events_log_path", "artifacts/logs/watchdog_events.jsonl")
        )

        self.directives = WatchdogDirectives(tau_m_cap=self.rule_d_tau_m_cap)
        self.rollback_count: int = 0
        self.consecutive_rollbacks: int = 0
        self.anchor_step_at_rollback: int = -1

        self._win_comp: deque[float] = deque(maxlen=self.window_size)
        self._win_std_min: deque[float] = deque(maxlen=self.window_size)
        self._win_merge_ent: deque[float] = deque(maxlen=self.window_size)

    def clear_window(self) -> None:
        self._win_comp.clear()
        self._win_std_min.clear()
        self._win_merge_ent.clear()

    def effective_tau_s(self, scheduled_tau_s: float) -> float:
        return max(float(scheduled_tau_s), float(self.directives.tau_s_floor))

    def effective_tau_m(self, scheduled_tau_m: float) -> float:
        if self.directives.explore_steps_remaining > 0 and self.directives.tau_m_boost > 0.0:
            return min(self.directives.tau_m_cap, float(scheduled_tau_m) + float(self.directives.tau_m_boost))
        return float(scheduled_tau_m)

    def effective_lambda_rate(self, scheduled_lambda_rate: float) -> float:
        if scheduled_lambda_rate <= 0.0:
            return 0.0
        return min(self.rule_b_lambda_rate_cap, float(scheduled_lambda_rate) * self.directives.lambda_rate_multiplier)

    def effective_lambda_var(self, scheduled_lambda_var: float) -> float:
        if scheduled_lambda_var <= 0.0:
            return 0.0
        return min(self.rule_c_lambda_var_cap, float(scheduled_lambda_var) * self.directives.lambda_var_multiplier)

    def active_merge_uniform_mix(self) -> float:
        if self.directives.explore_steps_remaining > 0:
            return float(self.directives.merge_uniform_mix)
        return 0.0

    def active_merge_kl_weight(self) -> float:
        if self.directives.explore_steps_remaining > 0:
            return float(self.directives.merge_kl_weight)
        return 0.0

    def tick_post_healthy_step(self, completed_step: int) -> None:
        """Decrement temporary exploration timers and reset circuit breaker after sufficient progress."""
        if self.directives.explore_steps_remaining > 0:
            self.directives.explore_steps_remaining -= 1
            if self.directives.explore_steps_remaining == 0:
                self.directives.tau_m_boost = 0.0
                self.directives.merge_uniform_mix = 0.0
                self.directives.merge_kl_weight = 0.0

        if (
            self.consecutive_rollbacks > 0
            and self.anchor_step_at_rollback >= 0
            and (completed_step - self.anchor_step_at_rollback) >= self.min_progress_steps_to_reset_breaker
        ):
            self.consecutive_rollbacks = 0
            self.anchor_step_at_rollback = -1

    def check_step(
        self,
        *,
        step: int,
        stage_id: int,
        metrics: Mapping[str, float],
        grads_finite: bool,
        grad_norm: Optional[float] = None,
        rollback_target_step: Optional[int] = None,
    ) -> WatchdogAction:
        """Inspect current step metrics and gradients against Rules A, B, C, and D."""
        # --- Rule A: Always active (even in first 500 steps) ---
        bad_reasons: list[str] = []
        if not grads_finite:
            bad_reasons.append("non_finite_gradient")
        if grad_norm is not None and not math.isfinite(float(grad_norm)):
            bad_reasons.append(f"grad_norm={grad_norm}")
        for key in self.SUB_LOSS_KEYS:
            if key in metrics:
                val = float(metrics[key])
                if not math.isfinite(val):
                    bad_reasons.append(f"{key}={val}")

        if bad_reasons:
            return self._trigger_rollback(
                rule="A",
                reason="Numerical explosion detected: " + ", ".join(bad_reasons),
                step=step,
                stage_id=stage_id,
                metrics=metrics,
                grad_norm=grad_norm,
                rollback_target_step=rollback_target_step,
            )

        # Update 20-step sliding window on numerically finite steps
        comp = float(metrics.get("compression_ratio", 9.5))
        std_min = float(metrics.get("c_slow_std_min", 1.0))
        merge_ent = float(metrics.get("merge_entropy_nats", 1.0))
        self._win_comp.append(comp)
        self._win_std_min.append(std_min)
        if stage_id in (2, 3):
            self._win_merge_ent.append(merge_ent)

        # Rules B, C, D activate strictly after `rule_bcd_warmup_steps` (default 500)
        if step < self.rule_bcd_warmup_steps:
            return WatchdogAction(triggered=False)

        # --- Rule B: SGSC spike degeneration (20-step moving avg compression ratio) ---
        if stage_id in (1, 2, 3) and len(self._win_comp) >= self.window_size:
            avg_comp = sum(self._win_comp) / len(self._win_comp)
            if avg_comp < self.compression_ratio_min or avg_comp > self.compression_ratio_max:
                return self._trigger_rollback(
                    rule="B",
                    reason=(
                        f"SGSC spike degeneration: 20-step avg compression_ratio={avg_comp:.3f} "
                        f"outside [{self.compression_ratio_min}, {self.compression_ratio_max}]"
                    ),
                    step=step,
                    stage_id=stage_id,
                    metrics=metrics,
                    grad_norm=grad_norm,
                    rollback_target_step=rollback_target_step,
                    window_summary={"avg_compression_ratio": avg_comp},
                )

        # --- Rule C: BCG/SGSC concept space collapse (20-step moving avg concept_std_min) ---
        if stage_id in (1, 2, 3) and len(self._win_std_min) >= self.window_size:
            avg_std_min = sum(self._win_std_min) / len(self._win_std_min)
            if avg_std_min < self.concept_std_min_floor:
                return self._trigger_rollback(
                    rule="C",
                    reason=(
                        f"Concept latent collapse: 20-step avg concept_std_min={avg_std_min:.4f} "
                        f"< {self.concept_std_min_floor}"
                    ),
                    step=step,
                    stage_id=stage_id,
                    metrics=metrics,
                    grad_norm=grad_norm,
                    rollback_target_step=rollback_target_step,
                    window_summary={"avg_concept_std_min": avg_std_min},
                )

        # --- Rule D: Syntactic merge deadlock (Stage 2 & 3 only) ---
        if stage_id in (2, 3) and len(self._win_merge_ent) >= self.window_size:
            avg_ent = sum(self._win_merge_ent) / len(self._win_merge_ent)
            if avg_ent < self.merge_entropy_min:
                return self._trigger_rollback(
                    rule="D",
                    reason=(
                        f"Syntactic merge deadlock: 20-step avg merge_entropy={avg_ent:.4f} "
                        f"< {self.merge_entropy_min}"
                    ),
                    step=step,
                    stage_id=stage_id,
                    metrics=metrics,
                    grad_norm=grad_norm,
                    rollback_target_step=rollback_target_step,
                    window_summary={"avg_merge_entropy": avg_ent},
                )

        return WatchdogAction(triggered=False)

    def is_healthy_for_checkpoint(self, step: int, stage_id: int, metrics: Mapping[str, float]) -> bool:
        """Return True iff the current step is safe to commit as `last_healthy.pt`."""
        for key in self.SUB_LOSS_KEYS:
            if key in metrics and not math.isfinite(float(metrics[key])):
                return False
        if step >= self.rule_bcd_warmup_steps and stage_id in (1, 2, 3):
            comp = float(metrics.get("compression_ratio", 9.5))
            std_min = float(metrics.get("c_slow_std_min", 1.0))
            if comp < self.compression_ratio_min or comp > self.compression_ratio_max:
                return False
            if std_min < self.concept_std_min_floor:
                return False
            if stage_id in (2, 3):
                ent = float(metrics.get("merge_entropy_nats", 1.0))
                if ent < self.merge_entropy_min:
                    return False
        return True

    def _trigger_rollback(
        self,
        *,
        rule: str,
        reason: str,
        step: int,
        stage_id: int,
        metrics: Mapping[str, float],
        grad_norm: Optional[float],
        rollback_target_step: Optional[int],
        window_summary: Optional[Mapping[str, float]] = None,
    ) -> WatchdogAction:
        before_directives = self.directives.to_dict()
        self.rollback_count += 1

        if self.anchor_step_at_rollback < 0:
            self.anchor_step_at_rollback = rollback_target_step if rollback_target_step is not None else step
        self.consecutive_rollbacks += 1

        # Targeted hyperparameter self-healing per rule
        if rule == "A":
            self.directives.lr_multiplier *= self.rule_a_lr_multiplier
        elif rule == "B":
            self.directives.lambda_rate_multiplier = min(
                self.rule_b_lambda_rate_cap,
                self.directives.lambda_rate_multiplier * self.rule_b_lambda_rate_multiplier,
            )
            self.directives.tau_s_floor = max(self.directives.tau_s_floor, self.rule_b_tau_s_floor)
        elif rule == "C":
            self.directives.lambda_var_multiplier = min(
                self.rule_c_lambda_var_cap,
                self.directives.lambda_var_multiplier * self.rule_c_lambda_var_multiplier,
            )
        elif rule == "D":
            self.directives.tau_m_boost = min(
                self.rule_d_tau_m_cap,
                self.directives.tau_m_boost + self.rule_d_tau_m_delta,
            )
            self.directives.explore_steps_remaining = self.rule_d_explore_steps
            self.directives.merge_uniform_mix = self.rule_d_uniform_mix
            self.directives.merge_kl_weight = self.rule_d_merge_kl_weight

        new_seed = self.base_seed + self.rollback_count * self.seed_stride_on_rollback
        self.clear_window()

        circuit_broken = self.consecutive_rollbacks >= self.max_consecutive_rollbacks
        event_record: dict[str, Any] = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "circuit_breaker_abort" if circuit_broken else "self_healing_rollback",
            "rule": rule,
            "reason": reason,
            "triggered_step": int(step),
            "stage_id": int(stage_id),
            "rollback_target_step": rollback_target_step,
            "rollback_count": int(self.rollback_count),
            "consecutive_rollbacks": int(self.consecutive_rollbacks),
            "skip_batches": int(self.skip_batches_on_rollback),
            "new_seed": int(new_seed),
            "grad_norm": None if grad_norm is None else (float(grad_norm) if math.isfinite(float(grad_norm)) else str(grad_norm)),
            "metrics_before": {
                k: (float(v) if math.isfinite(float(v)) else str(v))
                for k, v in metrics.items()
            },
            "window_summary": dict(window_summary or {}),
            "directives_before": before_directives,
            "directives_after": self.directives.to_dict(),
        }
        if self.enable_event_logging:
            atomic_append_jsonl(self.events_log_path, event_record)

        if circuit_broken:
            raise MeltdownAbortError(
                f"Watchdog circuit breaker tripped after {self.consecutive_rollbacks} consecutive "
                f"rollbacks without advancing {self.min_progress_steps_to_reset_breaker} steps "
                f"(last rule={rule} at step={step}: {reason}). Autopsy logged to {self.events_log_path}."
            )

        return WatchdogAction(
            triggered=True,
            rule=rule,
            reason=reason,
            rollback_needed=True,
            skip_batches=self.skip_batches_on_rollback,
            new_seed=new_seed,
            event_record=event_record,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "directives": self.directives.to_dict(),
            "rollback_count": int(self.rollback_count),
            "consecutive_rollbacks": int(self.consecutive_rollbacks),
            "anchor_step_at_rollback": int(self.anchor_step_at_rollback),
            "base_seed": int(self.base_seed),
        }

    def load_state_dict(self, state: Optional[Mapping[str, Any]], *, preserve_active_repairs: bool = False) -> None:
        if not state or preserve_active_repairs:
            return
        self.directives = WatchdogDirectives.from_dict(state.get("directives"))
        self.rollback_count = int(state.get("rollback_count", 0))
        self.consecutive_rollbacks = int(state.get("consecutive_rollbacks", 0))
        self.anchor_step_at_rollback = int(state.get("anchor_step_at_rollback", -1))


def reseed_all_rngs(new_seed: int, device: torch.device) -> random.Random:
    """Advance Python, NumPy, and PyTorch CPU/accelerator RNGs cleanly after a rollback."""
    return seed_everything(new_seed, device=device)


class CheckpointManager:
    """Dual-track preemption-safe atomic checkpoint manager."""

    def __init__(
        self,
        ckpt_dir: str | Path = "artifacts/checkpoints",
        healthy_save_every_steps: int = 1000,
        milestone_every_steps: int = 5000,
        stage_boundaries: Sequence[int] = (15000, 65000, 85000, 100000),
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.healthy_save_every_steps = int(healthy_save_every_steps)
        self.milestone_every_steps = int(milestone_every_steps)
        self.stage_boundaries: set[int] = {int(s) for s in stage_boundaries}
        self.last_healthy_step: Optional[int] = None

    @property
    def last_healthy_path(self) -> Path:
        return self.ckpt_dir / "last_healthy.pt"

    @property
    def best_model_path(self) -> Path:
        return self.ckpt_dir / "best_model.pt"

    @property
    def emergency_interrupt_path(self) -> Path:
        return self.ckpt_dir / "emergency_interrupt.pt"

    def milestone_path(self, step: int) -> Path:
        return self.ckpt_dir / f"milestone_step_{int(step):06d}.pt"

    def is_milestone_step(self, completed_step: int) -> bool:
        if completed_step in self.stage_boundaries:
            return True
        if self.milestone_every_steps > 0 and completed_step > 0 and (completed_step % self.milestone_every_steps == 0):
            return True
        return False

    def is_healthy_save_due(self, completed_step: int) -> bool:
        return (
            self.healthy_save_every_steps > 0
            and completed_step > 0
            and (completed_step % self.healthy_save_every_steps == 0)
        )

    @staticmethod
    def build_payload(
        *,
        step: int,
        stage_id: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: Mapping[str, Any],
        dynamic_loss_multipliers: Mapping[str, Any],
        dataloader_cursor: Mapping[str, Any],
        best_val_metric: float,
        config_dict: Mapping[str, Any],
        hardware_tier: str,
        rng_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return {
            "step": int(step),
            "stage_id": int(stage_id),
            "model_state_dict": get_clean_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state": dict(scheduler_state),
            "dynamic_loss_multipliers": dict(dynamic_loss_multipliers),
            "dataloader_cursor": dict(dataloader_cursor),
            "rng_state": capture_rng_state() if rng_state is None else rng_state,
            "best_val_metric": float(best_val_metric),
            "config": dict(config_dict),
            "hardware_tier": str(hardware_tier),
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def save_healthy(self, payload: Mapping[str, Any]) -> Path:
        path = atomic_torch_save(payload, self.last_healthy_path)
        self.last_healthy_step = int(payload["step"])
        return path

    def save_milestone(self, payload: Mapping[str, Any]) -> Path:
        step = int(payload["step"])
        return atomic_torch_save(payload, self.milestone_path(step))

    def save_best(self, payload: Mapping[str, Any]) -> Path:
        return atomic_torch_save(payload, self.best_model_path)

    def save_emergency(self, payload: Mapping[str, Any]) -> Path:
        return atomic_torch_save(payload, self.emergency_interrupt_path)

    def load_checkpoint(self, path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
        is_cpu = (isinstance(map_location, str) and map_location == "cpu") or (
            isinstance(map_location, torch.device) and map_location.type == "cpu"
        )
        if is_cpu:
            try:
                return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
            except (TypeError, RuntimeError):
                pass
        return torch.load(str(path), map_location=map_location, weights_only=False)

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Select the highest-step valid checkpoint among emergency, last_healthy, and milestones."""
        candidates: list[Path] = []
        for p in (self.emergency_interrupt_path, self.last_healthy_path):
            if p.exists():
                candidates.append(p)
        candidates.extend(sorted(self.ckpt_dir.glob("milestone_step_*.pt"), reverse=True))
        best_path: Optional[Path] = None
        best_step = -1
        for cand in candidates:
            try:
                ckpt = self.load_checkpoint(cand, map_location="cpu")
                st = int(ckpt.get("step", -1))
                del ckpt
                if st > best_step:
                    best_step = st
                    best_path = cand
            except Exception:
                continue
        return best_path

    @staticmethod
    def restore_into(
        ckpt: Mapping[str, Any],
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        controller: Any,
        watchdog: Optional[SelfHealingWatchdog] = None,
        preserve_watchdog_repairs: bool = False,
        restore_rng: bool = True,
    ) -> tuple[int, dict[str, Any]]:
        """Restore model, optimizer, curriculum, watchdog, RNG, and return `(step, dataloader_cursor)`.

        Strips any `_orig_mod.` prefixes and restores cleanly into either compiled or uncompiled
        models with `strict=True` (Performance Audit Task 2.4).
        """
        clean_sd = migrate_legacy_moe_state_dict(strip_orig_mod_prefix(ckpt["model_state_dict"]))
        target_mod = unwrap_compiled_module(model)
        # If individual submodules (e.g. model.bcg_pc) are compiled, unwrap temporarily for strict=True load
        # or remap keys to match active submodule wrapper cleanly.
        target_keys = set(target_mod.state_dict().keys())
        if target_keys == set(clean_sd.keys()):
            target_mod.load_state_dict(clean_sd, strict=True)
        else:
            remapped = {}
            for tk in target_keys:
                canon = ".".join(seg for seg in tk.split(".") if seg != "_orig_mod")
                if canon in clean_sd:
                    remapped[tk] = clean_sd[canon]
            target_mod.load_state_dict(remapped, strict=True)
        # Preserve active YAML config's `base_lr` and runtime `fused` flag per group across resume
        # so LR tuning & fused CUDA AdamW take effect immediately while retaining 100% of AdamW momentum.
        configured_base_lrs = [
            float(g.get("base_lr", g["lr"])) for g in optimizer.param_groups
        ]
        configured_fused = [g.get("fused") for g in optimizer.param_groups]
        # Free existing GPU AdamW moment buffers BEFORE materializing restored moments to prevent 2x VRAM spikes
        optimizer.state.clear()
        opt_sd = migrate_legacy_moe_optimizer_state_dict(ckpt["optimizer_state_dict"], optimizer)
        # Align group `fused` flag before load_state_dict so PyTorch places `step` on the right device!
        if "param_groups" in opt_sd and len(opt_sd["param_groups"]) == len(configured_fused):
            opt_sd = dict(opt_sd)
            patched_groups = []
            for og, f_val in zip(opt_sd["param_groups"], configured_fused):
                pg = dict(og)
                if f_val is not None:
                    pg["fused"] = f_val
                patched_groups.append(pg)
            opt_sd["param_groups"] = patched_groups
        optimizer.load_state_dict(opt_sd)
        for g, cfg_blr, f_val in zip(optimizer.param_groups, configured_base_lrs, configured_fused):
            g["base_lr"] = cfg_blr
            if f_val is not None:
                g["fused"] = f_val
        if hasattr(controller, "load_state_dict"):
            controller.load_state_dict(ckpt.get("scheduler_state"))
        if watchdog is not None:
            watchdog.load_state_dict(
                ckpt.get("dynamic_loss_multipliers"),
                preserve_active_repairs=preserve_watchdog_repairs,
            )
        if restore_rng:
            restore_rng_state(ckpt.get("rng_state"))
        step = int(ckpt.get("step", 0))
        cursor = dict(ckpt.get("dataloader_cursor") or {})
        return step, cursor
