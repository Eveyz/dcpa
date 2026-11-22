"""Real-Time Anti-Collapse Monitor & Automated Emergency Rescue Matrix (Section 5.4).

Monitors all four critical failure modes in real time and applies automated interventions:
  1. Spike shattering or freezing: spike_rate > 0.25 or < 0.04 for 200 steps
     -> dual Lagrangian multiplier update every 50 steps & temporary lambda_rate = 20.0.
  2. Latent representation collapse: mean per-dim std(c_slow) < 0.35
     -> elevate lambda_var to 5.0 and trigger spectral normalization projection on W_sem.
  3. Syntactic merge rigidity (sliding-window collapse): 72-pair entropy < 0.15 nats
     -> enable uniform KL regularizer (0.02) and re-heat tau_m to 0.6.
  4. Hippocampal passive bypass in Stage D: mean ACh < 0.3 or beta < 0.1
     -> enable weak ACh supervision hint during first 2000 steps of Stage D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RescueDirectives:
    lambda_rate_override: Optional[float] = None
    dual_rate_lagrange: float = 0.0
    lambda_var_override: Optional[float] = None
    apply_wsem_spectral_norm: bool = False
    merge_kl_weight: float = 0.0
    tau_m_override: Optional[float] = None
    ach_hint_weight: float = 0.0
    alerts: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.alerts is None:
            self.alerts = []


class AntiCollapseMonitor:
    def __init__(self, target_spike_rate: float = 0.10) -> None:
        self.target_spike_rate = target_spike_rate
        self.bad_spike_streak: int = 0
        self.dual_lagrange: float = 0.0
        self.tau_m_reheat_steps: int = 0
        self.step_count: int = 0

    def observe(
        self,
        metrics: dict[str, float],
        stage: str,
        step_in_stage: int,
    ) -> RescueDirectives:
        self.step_count += 1
        d = RescueDirectives()

        # 1. Spike shattering / freezing guard
        sr = metrics.get("spike_rate", self.target_spike_rate)
        if stage != "D" and (sr > 0.25 or sr < 0.04):
            self.bad_spike_streak += 1
        else:
            self.bad_spike_streak = max(0, self.bad_spike_streak - 2)

        if self.bad_spike_streak >= 200:
            d.lambda_rate_override = 20.0
            if self.step_count % 50 == 0:
                sign = 1.0 if sr > self.target_spike_rate else -1.0
                self.dual_lagrange = max(-5.0, min(5.0, self.dual_lagrange + 0.25 * sign))
            d.dual_rate_lagrange = self.dual_lagrange
            d.alerts.append(f"SPIKE_BOUNDS(rate={sr:.3f}, streak={self.bad_spike_streak})")
        else:
            self.dual_lagrange *= 0.98
            d.dual_rate_lagrange = self.dual_lagrange

        # 2. Latent representation collapse guard
        c_std = metrics.get("c_slow_std_mean", 1.0)
        if c_std < 0.35:
            d.lambda_var_override = 5.0
            d.apply_wsem_spectral_norm = True
            d.alerts.append(f"LATENT_COLLAPSE(c_slow_std={c_std:.3f})")

        # 3. Syntactic merge rigidity guard (Stages B/C/D)
        if stage in ("B", "C", "D"):
            ent = metrics.get("merge_entropy_nats", 1.0)
            if ent < 0.15:
                self.tau_m_reheat_steps = 500
                d.alerts.append(f"MERGE_RIGIDITY(entropy={ent:.3f} nats)")
            if self.tau_m_reheat_steps > 0:
                d.merge_kl_weight = 0.02
                d.tau_m_override = 0.25 + (0.6 - 0.25) * (self.tau_m_reheat_steps / 500.0)
                self.tau_m_reheat_steps -= 1

        # 4. Hippocampal passive bypass guard (Stage D first 2000 steps)
        if stage == "D":
            ach = metrics.get("mean_ach", 0.5)
            beta = metrics.get("mean_beta", 0.5)
            if step_in_stage < 2000 and (ach < 0.3 or beta < 0.1):
                d.ach_hint_weight = 0.25
                d.alerts.append(f"HIP_BYPASS(ach={ach:.3f}, beta={beta:.3f})")

        return d


@torch.no_grad()
def project_spectral_norm(weight: torch.Tensor, max_sigma: float = 1.0, iters: int = 3) -> None:
    """Lightweight power-iteration spectral norm projection on W_sem during collapse rescue."""
    w = weight.float()
    v = torch.randn(w.shape[1], device=w.device, dtype=torch.float32)
    v = v / (v.norm() + 1e-8)
    for _ in range(iters):
        u = w @ v
        u = u / (u.norm() + 1e-8)
        v = w.T @ u
        v = v / (v.norm() + 1e-8)
    sigma = float((u @ (w @ v)).item())
    if sigma > max_sigma:
        weight.mul_(max_sigma / sigma)
