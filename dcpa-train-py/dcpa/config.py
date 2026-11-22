"""Standard hyperparameter dictionary and architectural invariants for DCPA.

Strictly enforces the Engineering Blueprint constraints:
- Raw UTF-8 byte vocabulary (V = 256, zero external tokenizers).
- Constant working memory slot cap K <= 12 (default K = 8).
- d_s and D aligned to multiples of 64 for RoPE and SIMD width.
- Variance hinge weight lambda_var strictly positive to prevent latent collapse.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LANG_DOMAINS: dict[str, int] = {
    "zh": 0,
    "en": 1,
    "ja": 2,
    "python": 3,
    "rust": 4,
    "synthetic": 5,
}


@dataclass
class DCPAConfig:
    # Vocabulary: strictly raw UTF-8 bytes 0..255
    vocab_size: int = 256

    # Core dimensions (must be multiples of 64)
    d_s: int = 192
    d_concept: int = 768

    # Bottlenecked working memory graph (hard constant cap K <= 12)
    k_slots: int = 8
    d_r: int = 32
    d_merge: int = 256

    # Hippocampal sparse dual memory
    d_h: int = 4096
    s_active: int = 32
    dg_seed_lex: int = 42
    dg_seed_sem: int = 43
    gamma_hip: float = 0.999
    lambda_eta: float = 0.95
    kappa_quench: float = 0.95
    c0_ema_momentum: float = 0.995
    householder_max_writes: int = 32
    householder_write_threshold: float = 0.05

    # Sparse micro-experts & adaptive pondering
    num_experts: int = 16
    top_k_experts: int = 2
    expert_hidden: int = 1152
    l_max: int = 4
    tau_skip: float = 0.65
    tau_halt: float = 0.35
    ponder_eta: float = 0.5

    # SGSC LIF spike gate & segment bounds
    lif_lambda: float = 0.85
    lif_beta1: float = 0.25
    lif_beta2: float = 0.75
    lif_theta_init: float = 2.2
    lif_theta_min: float = 0.05
    lif_theta_scale: float = 12.0
    max_seg_len: int = 16
    target_spike_rate: float = 0.10
    lambda_rate: float = 8.0

    # Slot salience conservation
    delta_decay: float = 0.05

    # PMD pragmatic motor decoder
    d_l: int = 64
    num_lang_domains: int = 6
    stack_depth: int = 16
    stack_tag_dim: int = 4
    tau_e: float = 0.5
    tau_q: float = 0.5
    lambda_inh: float = 2.0
    tau_done: float = 0.15
    tau_repair: float = 0.35
    max_repair_retries: int = 2

    # Holographic orthogonal RoPE parameters (exact harmonic period > max_seg_len)
    rope_period: float = 33.0
    rope_num_harmonics: int = 16

    # Joint loss weights
    lambda_var: float = 2.5
    lambda_act: float = 0.04
    lambda_halt: float = 0.5

    # Optimizer defaults
    lr_micro: float = 6e-4
    lr_concept: float = 3e-4
    lr_scalar_gate: float = 1e-4
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0

    # Normalization epsilon
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.vocab_size != 256:
            raise ValueError(
                f"DCPA Commandment #1 violation: vocab_size must be 256 (raw UTF-8 bytes), got {self.vocab_size}"
            )
        if not (2 <= self.k_slots <= 12):
            raise ValueError(
                f"DCPA Commandment #2 violation: k_slots must be in [2, 12], got {self.k_slots}"
            )
        if self.d_s % 64 != 0 or self.d_concept % 64 != 0:
            raise ValueError(
                f"d_s ({self.d_s}) and d_concept ({self.d_concept}) must be multiples of 64 for RoPE and SIMD."
            )
        if self.s_active <= 0 or self.s_active > self.d_h:
            raise ValueError(f"s_active ({self.s_active}) must be in (0, d_h={self.d_h}].")
        if self.lambda_var <= 0.0:
            raise ValueError("lambda_var must be strictly > 0 to prevent latent collapse.")
        if self.stack_depth != 16 or self.stack_tag_dim != 4:
            raise ValueError("PMD differentiable pushdown stack must be 16x4 (128B in bf16).")
        if self.d_l != 64:
            raise ValueError("Language/register control dimension d_l must be 64.")

    @property
    def num_merge_pairs(self) -> int:
        return self.k_slots * (self.k_slots + 1)

    @classmethod
    def base(cls) -> DCPAConfig:
        """DCPA-Base (~95M slow weights, standard engineering blueprint reference)."""
        return cls()

    @classmethod
    def standard_152m(cls) -> DCPAConfig:
        """DCPA-152M Standard Preset (~152.0M slow weights, ~22M active per step)."""
        return cls(
            d_s=192,
            d_concept=768,
            k_slots=8,
            d_r=32,
            d_merge=256,
            d_h=4096,
            s_active=32,
            d_l=64,
            num_experts=16,
            top_k_experts=2,
            expert_hidden=1920,
            l_max=4,
        )

    @classmethod
    def proto_48m(cls) -> DCPAConfig:
        """DCPA-48M Prototype Preset (~47.6M slow weights: d_s=192, D=512, K=8, d_h=4096, s=32)."""
        return cls(
            d_s=192,
            d_concept=512,
            k_slots=8,
            d_r=32,
            d_merge=256,
            d_h=4096,
            s_active=32,
            d_l=64,
            num_experts=8,
            top_k_experts=2,
            expert_hidden=1728,
            l_max=4,
            target_spike_rate=1.0 / 9.5,
            lambda_rate=2.0,
            lambda_var=1.0,
            lambda_halt=1.0,
            lambda_act=0.04,
            lr_micro=6e-4,
            lr_concept=6e-4,
            lr_scalar_gate=1.2e-3,
        )

    @classmethod
    def tiny(cls) -> DCPAConfig:
        """DCPA-Tiny (~48M slow weights, ~14M active per step)."""
        return cls(
            d_s=128,
            d_concept=512,
            k_slots=8,
            d_r=32,
            d_merge=192,
            d_h=2048,
            s_active=24,
            num_experts=8,
            top_k_experts=2,
            expert_hidden=768,
            l_max=4,
            lr_micro=8e-4,
            lr_concept=4e-4,
        )

    @classmethod
    def micro(cls) -> DCPAConfig:
        """DCPA-Micro for rapid local testing and gate verification (all invariants preserved)."""
        return cls(
            d_s=64,
            d_concept=128,
            k_slots=8,
            d_r=32,
            d_merge=64,
            d_h=512,
            s_active=16,
            num_experts=4,
            top_k_experts=2,
            expert_hidden=192,
            l_max=4,
            lr_micro=1e-3,
            lr_concept=5e-4,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> DCPAConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
