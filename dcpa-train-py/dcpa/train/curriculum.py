"""Four-Stage Developmental Curriculum State Machine (Engineering Doc Section 5.3).

Standard 100,000-step developmental schedule (proportionally scalable via total_steps):
  - Stage A (0%..15%,   0..15k):  SGSC + PMD lower motor loop; tau_s linear 1.0 -> 0.35.
  - Stage B (15%..55%, 15k..55k): BCG-PC concept bottleneck + L_PC + var hinge; tau_m cosine 1.2 -> 0.25.
  - Stage C (55%..85%, 55k..85k): Broca CQ (50% streaming + 50% CQ) + multilingual/code.
  - Stage D (85%..100%, 85k..100k): Freeze sensory embed & theta, activate Hippocampus Gated Delta
                                    with online Category 4 fast-mapping episodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StageConfig:
    stage: str           # "A", "B", "C", or "D"
    step_in_stage: int
    stage_progress: float
    tau_s: float
    tau_m: float
    use_broca_cq: bool
    lr_scale: float


class CurriculumSchedule:
    def __init__(self, total_steps: int = 100_000, warmup_steps: int = 1_000) -> None:
        self.total_steps = max(total_steps, 4)
        self.warmup_steps = min(warmup_steps, max(1, total_steps // 20))
        self.stage_a_end = int(0.15 * self.total_steps)
        self.stage_b_end = int(0.55 * self.total_steps)
        self.stage_c_end = int(0.85 * self.total_steps)

    def get(self, step: int) -> StageConfig:
        # Global cosine LR multiplier with linear warmup
        if step < self.warmup_steps:
            lr_scale = float(step + 1) / float(self.warmup_steps)
        else:
            prog = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
            lr_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

        if step < self.stage_a_end:
            prog = step / max(1, self.stage_a_end)
            tau_s = 1.0 + (0.35 - 1.0) * prog
            return StageConfig(
                stage="A",
                step_in_stage=step,
                stage_progress=prog,
                tau_s=tau_s,
                tau_m=1.2,
                use_broca_cq=False,
                lr_scale=lr_scale,
            )
        if step < self.stage_b_end:
            local = step - self.stage_a_end
            prog = local / max(1, self.stage_b_end - self.stage_a_end)
            # Cosine annealing tau_m: 1.2 -> 0.25
            tau_m = 0.25 + 0.5 * (1.2 - 0.25) * (1.0 + math.cos(math.pi * prog))
            return StageConfig(
                stage="B",
                step_in_stage=local,
                stage_progress=prog,
                tau_s=0.35,
                tau_m=tau_m,
                use_broca_cq=False,
                lr_scale=lr_scale,
            )
        if step < self.stage_c_end:
            local = step - self.stage_b_end
            prog = local / max(1, self.stage_c_end - self.stage_b_end)
            # Alternate 50% streaming + 50% Broca CQ full-graph mode
            return StageConfig(
                stage="C",
                step_in_stage=local,
                stage_progress=prog,
                tau_s=0.35,
                tau_m=0.25,
                use_broca_cq=(step % 2 == 1),
                lr_scale=lr_scale,
            )
        local = step - self.stage_c_end
        prog = min(1.0, local / max(1, self.total_steps - self.stage_c_end))
        return StageConfig(
            stage="D",
            step_in_stage=local,
            stage_progress=prog,
            tau_s=0.35,
            tau_m=0.25,
            use_broca_cq=(step % 2 == 1),
            lr_scale=lr_scale,
        )
