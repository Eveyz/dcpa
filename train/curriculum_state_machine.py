"""Four-stage 100,000-step developmental curriculum state machine for DCPA.

Implements Deliverable 2 of the DCPA Training System:
  - Deterministic step-driven `CurriculumController(step)` governing Stage 1..4 topology,
    hard parameter freeze masks (`requires_grad_(False)` + `grad = None` — NEVER grad-zeroing),
    per-group local linear LR warmups (`0.01 -> 1.0`), window size escalation (`max_doc_bytes`),
    domain mix ratios (`[en, zh, ja, py, rs]` + Stage 4 Wug 50%), temperature annealing, and
    dynamic loss weights.
  - Step-0 fixed three-way differentiated `AdamW` parameter grouping (`group_matrix_decay`,
    `group_norm_embed_nodecay`, `group_gates_thresholds`), sub-partitioned by developmental
    ownership (`cortical_io`, `bcg_pc`, `broca_cq`, `hippocampus_adapter`, `hippocampus_gates`)
    so local warmups NEVER rebuild the optimizer or lose AdamW momentum states.
  - Strictly guarantees Hippocampus fast-weight buffers (`m_c_from_l`, `m_l_from_c`, `u_lex`,
    `u_sem`) are NEVER registered in any optimizer parameter group (Red Line #3).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn


DOMAIN_NAMES: tuple[str, ...] = ("en", "zh", "ja", "python", "rust", "javascript")
ACTIVE_DOMAIN_IDS: tuple[int, ...] = (0, 1, 3, 5)

HIPPOCAMPUS_SCALAR_GATE_NAMES: frozenset[str] = frozenset(
    {
        "hippocampus.alpha_1",
        "hippocampus.alpha_2",
        "hippocampus.theta_i",
        "hippocampus.theta_n",
        "hippocampus.alpha_q",
        "hippocampus.gamma_e",
        "hippocampus.theta_q",
        "hippocampus.alpha_out",
        "hippocampus.theta_out",
    }
)

HIPPOCAMPUS_ADAPTER_NAMES: frozenset[str] = frozenset(
    {
        "hippocampus.f_prec_cort.bias",
    }
)

HIPPOCAMPUS_FAST_BUFFER_NAMES: frozenset[str] = frozenset(
    {
        "m_c_from_l",
        "m_l_from_c",
        "u_lex",
        "u_sem",
        "hippocampus.m_c_from_l",
        "hippocampus.m_l_from_c",
        "hippocampus.u_lex",
        "hippocampus.u_sem",
    }
)

CODE_DELIMITER_BYTES: frozenset[int] = frozenset(ord(ch) for ch in "{}()[]:;\n")


@dataclass(frozen=True)
class CurriculumStepState:
    """Complete deterministic curriculum contract for a single optimization step."""

    step: int
    stage_id: int                      # 1, 2, 3, or 4
    stage_name: str
    model_stage: str                   # "A", "B", "C", or "D"
    step_in_stage: int
    stage_length: int
    stage_progress: float              # [0.0, 1.0]
    global_lr_scale: float
    owner_lr_multipliers: dict[str, float]
    max_doc_bytes: int
    domain_weights: dict[int, float]   # {0: en, 1: zh, 2: ja, 3: py, 4: rs}
    wug_ratio: float
    tau_s: float
    tau_m: float
    target_rho: float
    target_spike_rate: float
    lambda_rate: float
    lambda_var: float
    lambda_halt: float
    lambda_pc: float
    lambda_act: float
    code_delimiter_focal_weight: float
    wug_target_weight: float
    lambda_ach_reg: float
    use_broca_cq: bool
    active_owners: tuple[str, ...]
    frozen_owners: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_param_name(name: str) -> str:
    """Strip any `_orig_mod.` wrapper segments introduced by `torch.compile` submodules."""
    if "_orig_mod" not in name:
        return name
    return ".".join(seg for seg in name.split(".") if seg != "_orig_mod")


def classify_parameter(name: str, param: nn.Parameter) -> tuple[str, str]:
    """Classify a model parameter into `(lr_wd_tier, developmental_owner)`.

    Tiers (Deliverable 1):
      - `group_gates_thresholds`: SGSC LIF params, BCG skip/halt/salience scalars,
        Hippocampus scalar gates, PMD charge depletion gates (`w_e_init`, `w_drain`).
      - `group_norm_embed_nodecay`: Embeddings, RMSNorms, all biases, `lang_embed`,
        1D / initial-state parameters (`h0`, `m0`, `s0`, `a0`, `omega0`).
      - `group_matrix_decay`: All remaining `ndim >= 2` linear/expert/bilinear matrices.

    Owners (Deliverable 2):
      - `cortical_io`: SGSC + lower PMD (active Stages 1-3, frozen Stage 4).
      - `bcg_pc`: BCG-PC + slow cortical hippocampus projections (frozen Stage 1, active 2-3, frozen 4).
      - `broca_cq`: PMD Broca CQ precedence/coherence (`w_ord`, `w_coh`) (frozen 1-2, active 3, frozen 4).
      - `hippocampus_adapter`: lightweight inter-layer adapter bias `hippocampus.f_prec_cort.bias`
        (frozen Stage 1, active Stages 2-4).
      - `hippocampus_gates`: all 9 Hippocampus neuromodulatory gating scalars (frozen 1-3, active 4).
    """
    name = _canonical_param_name(name)
    if any(bad in name for bad in ("m_c_from_l", "m_l_from_c", "u_lex", "u_sem")):
        raise RuntimeError(
            f"Red Line #3 violation: Hippocampal fast buffer {name!r} appeared in model.named_parameters()!"
        )

    # 1. Developmental owner
    if name in HIPPOCAMPUS_SCALAR_GATE_NAMES:
        owner = "hippocampus_gates"
    elif name in HIPPOCAMPUS_ADAPTER_NAMES:
        owner = "hippocampus_adapter"
    elif name.startswith("bcg_pc.") or name.startswith("hippocampus."):
        owner = "bcg_pc"
    elif name.startswith("pmd.w_ord.") or name.startswith("pmd.w_coh."):
        owner = "broca_cq"
    elif name.startswith("sgsc.") or name.startswith("pmd."):
        owner = "cortical_io"
    else:
        owner = "cortical_io"

    # 2. Three-way differentiated AdamW LR / weight_decay tier
    is_gate_param = (
        name == "sgsc.theta_raw"
        or name == "bcg_pc.kappa_omega"
        or name in HIPPOCAMPUS_SCALAR_GATE_NAMES
        or name.startswith("pmd.w_e_init.")
        or name.startswith("pmd.w_drain.")
    )
    if is_gate_param:
        tier = "group_gates_thresholds"
    elif (
        param.ndim < 2
        or name.endswith(".bias")
        or "embed" in name
        or "norm" in name
        or name in ("sgsc.h0", "pmd.m0", "bcg_pc.s0", "bcg_pc.a0", "bcg_pc.omega0")
    ):
        tier = "group_norm_embed_nodecay"
    else:
        tier = "group_matrix_decay"

    return tier, owner


def build_three_way_param_groups(
    model: nn.Module,
    optimizer_cfg: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Build fixed Step-0 AdamW parameter groups partitioned by `(tier, owner)`.

    Never rebuilt mid-training, preserving 100% of AdamW momentum states across stages while
    supporting independent per-module local LR warmups in Stage 2 (BCG-PC) and Stage 4 (Hippocampus).
    """
    groups_cfg = (optimizer_cfg or {}).get("groups", {})
    tier_defaults: dict[str, tuple[float, float]] = {
        "group_matrix_decay": (
            float(groups_cfg.get("group_matrix_decay", {}).get("lr", 6.0e-4)),
            float(groups_cfg.get("group_matrix_decay", {}).get("weight_decay", 0.05)),
        ),
        "group_norm_embed_nodecay": (
            float(groups_cfg.get("group_norm_embed_nodecay", {}).get("lr", 6.0e-4)),
            float(groups_cfg.get("group_norm_embed_nodecay", {}).get("weight_decay", 0.0)),
        ),
        "group_gates_thresholds": (
            float(groups_cfg.get("group_gates_thresholds", {}).get("lr", 1.2e-3)),
            float(groups_cfg.get("group_gates_thresholds", {}).get("weight_decay", 0.0)),
        ),
    }

    buckets: dict[tuple[str, str], list[nn.Parameter]] = {}
    bucket_names: dict[tuple[str, str], list[str]] = {}
    seen_ids: set[int] = set()

    for raw_name, param in model.named_parameters():
        pname = _canonical_param_name(raw_name)
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        tier, owner = classify_parameter(pname, param)
        key = (tier, owner)
        buckets.setdefault(key, []).append(param)
        bucket_names.setdefault(key, []).append(pname)

    # Verify fast buffers are absent from optimizer parameter IDs (Red Line #3)
    fast_buf_ids: set[int] = set()
    for bname, buf in model.named_buffers():
        if bname in HIPPOCAMPUS_FAST_BUFFER_NAMES or bname.endswith(("m_c_from_l", "m_l_from_c", "u_lex", "u_sem")):
            fast_buf_ids.add(id(buf))
    overlap = seen_ids.intersection(fast_buf_ids)
    if overlap:
        raise RuntimeError("Red Line #3 violation: Hippocampus fast buffer leaked into optimizer params!")

    ordered_keys = sorted(buckets.keys())
    param_groups: list[dict[str, Any]] = []
    for tier, owner in ordered_keys:
        base_lr, wd = tier_defaults[tier]
        param_groups.append(
            {
                "name": f"{tier}:{owner}",
                "tier": tier,
                "owner": owner,
                "param_names": tuple(bucket_names[(tier, owner)]),
                "params": buckets[(tier, owner)],
                "base_lr": base_lr,
                "lr": base_lr,
                "weight_decay": wd,
            }
        )
    return param_groups


def create_dcpa_optimizer(
    model: nn.Module,
    optimizer_cfg: Optional[Mapping[str, Any]] = None,
) -> torch.optim.AdamW:
    """Create the fixed Step-0 three-way AdamW optimizer (`betas=(0.9, 0.95), eps=1e-8`)."""
    cfg = optimizer_cfg or {}
    betas_raw = cfg.get("betas", [0.9, 0.95])
    betas = (float(betas_raw[0]), float(betas_raw[1]))
    eps = float(cfg.get("eps", 1.0e-8))
    groups = build_three_way_param_groups(model, optimizer_cfg=cfg)
    return torch.optim.AdamW(groups, betas=betas, eps=eps)


def stage_owner_sets(stage_id: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return `(active_owners, frozen_owners)` for developmental stage `stage_id` in {1, 2, 3, 4}."""
    if stage_id == 1:
        return (
            ("cortical_io",),
            ("bcg_pc", "broca_cq", "hippocampus_adapter", "hippocampus_gates"),
        )
    if stage_id == 2:
        return (
            ("cortical_io", "bcg_pc", "hippocampus_adapter"),
            ("broca_cq", "hippocampus_gates"),
        )
    if stage_id == 3:
        return (
            ("cortical_io", "bcg_pc", "broca_cq", "hippocampus_adapter"),
            ("hippocampus_gates",),
        )
    if stage_id == 4:
        return (
            ("hippocampus_gates", "hippocampus_adapter"),
            ("cortical_io", "bcg_pc", "broca_cq"),
        )
    raise ValueError(f"Unsupported stage_id={stage_id}; expected 1..4")


def expected_stage_param_sets(model: nn.Module, stage_id: int) -> tuple[set[str], set[str]]:
    """Return `(expected_trainable_names, expected_frozen_names)` for audit verification."""
    active_owners, _ = stage_owner_sets(stage_id)
    active_set = set(active_owners)
    trainable: set[str] = set()
    frozen: set[str] = set()
    for raw_name, param in model.named_parameters():
        pname = _canonical_param_name(raw_name)
        _, owner = classify_parameter(pname, param)
        if owner in active_set:
            trainable.add(pname)
        else:
            frozen.add(pname)
    return trainable, frozen


def apply_stage_freeze(model: nn.Module, stage_id: int) -> dict[str, int]:
    """Enforce Red Line #2 hard freezing: `p.requires_grad_(False)` AND `p.grad = None`.

    NEVER fakes freezing by zeroing gradients on `requires_grad=True` tensors (which would allow
    AdamW decoupled weight decay to silently corrode frozen weights toward zero).
    """
    active_owners, _ = stage_owner_sets(stage_id)
    active_set = set(active_owners)
    n_active_tensors = 0
    n_frozen_tensors = 0
    n_active_elems = 0
    n_frozen_elems = 0

    for pname, param in model.named_parameters():
        _, owner = classify_parameter(pname, param)
        if owner in active_set:
            param.requires_grad_(True)
            n_active_tensors += 1
            n_active_elems += param.numel()
        else:
            param.requires_grad_(False)
            param.grad = None
            n_frozen_tensors += 1
            n_frozen_elems += param.numel()

    return {
        "active_tensors": n_active_tensors,
        "frozen_tensors": n_frozen_tensors,
        "active_params": n_active_elems,
        "frozen_params": n_frozen_elems,
    }


def hash_named_parameters(model: nn.Module, param_names: Optional[Sequence[str] | set[str]] = None) -> dict[str, str]:
    """Compute exact `blake2b` raw-byte digests for selected model parameters (freeze corrosion audit)."""
    target = {_canonical_param_name(n) for n in param_names} if param_names is not None else None
    digests: dict[str, str] = {}
    for raw_name, param in model.named_parameters():
        pname = _canonical_param_name(raw_name)
        if target is not None and pname not in target:
            continue
        raw = param.detach().to(device="cpu").contiguous().numpy().tobytes()
        digests[pname] = hashlib.blake2b(raw).hexdigest()
    return digests


def _cosine_interp(start: float, end: float, frac: float) -> float:
    f = min(max(frac, 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * f))


def _linear_interp(start: float, end: float, frac: float) -> float:
    f = min(max(frac, 0.0), 1.0)
    return start + (end - start) * f


def _scale_duration(nominal_steps: int, nominal_stage_len: int, actual_stage_len: int) -> int:
    """Keep exact step counts at 100k scale, proportionally scaling for compressed lifecycle tests."""
    if actual_stage_len == nominal_stage_len:
        return nominal_steps
    if nominal_steps <= 0 or actual_stage_len <= 0:
        return 0
    scaled = int(round(nominal_steps * (actual_stage_len / max(nominal_stage_len, 1))))
    return max(1, min(scaled, actual_stage_len))


def _weights_list_to_dict(vals: Sequence[float] | Mapping[Any, float]) -> dict[int, float]:
    if isinstance(vals, Mapping):
        return {int(k): float(v) for k, v in vals.items() if float(v) > 0.0}
    if len(vals) == 4:
        # 4-domain v2 layout: [en(0), zh(1), python(3), javascript(5)]
        return {0: float(vals[0]), 1: float(vals[1]), 3: float(vals[2]), 5: float(vals[3])}
    if len(vals) == 6:
        # Full 6-slot layout: [en(0), zh(1), ja(2), python(3), rust(4), javascript(5)]
        return {i: float(v) for i, v in enumerate(vals) if float(v) > 0.0}
    if len(vals) == 5:
        return {i: float(v) for i, v in enumerate(vals) if float(v) > 0.0}
    raise ValueError(f"Expected 4 [en,zh,py,js], 5, or 6 domain weights, got {vals}")


class CurriculumController:
    """Deterministic step-driven 4-stage developmental curriculum state machine."""

    NOMINAL_BOUNDARIES: tuple[int, int, int, int, int] = (0, 15000, 65000, 85000, 100000)

    def __init__(
        self,
        curriculum_cfg: Optional[Mapping[str, Any]] = None,
        stage_boundaries: Optional[Sequence[int]] = None,
    ) -> None:
        cfg = dict(curriculum_cfg or {})
        raw_bounds = (
            list(stage_boundaries)
            if stage_boundaries is not None
            else list(cfg.get("stage_boundaries", self.NOMINAL_BOUNDARIES))
        )
        if len(raw_bounds) != 5 or raw_bounds[0] != 0 or any(raw_bounds[i] >= raw_bounds[i + 1] for i in range(4)):
            raise ValueError(
                f"stage_boundaries must be 5 strictly increasing ints starting at 0, got {raw_bounds}"
            )
        self.boundaries: tuple[int, int, int, int, int] = tuple(int(x) for x in raw_bounds)  # type: ignore[assignment]
        self.total_steps: int = self.boundaries[-1]

        self.s1_cfg = dict(cfg.get("stage1", {}))
        self.s2_cfg = dict(cfg.get("stage2", {}))
        self.s3_cfg = dict(cfg.get("stage3", {}))
        self.s4_cfg = dict(cfg.get("stage4", {}))

        # Nominal stage lengths from the 100k specification
        self._nom_s1 = 15000
        self._nom_s2 = 50000
        self._nom_s3 = 20000
        self._nom_s4 = 15000

        # Active step tracking for serialization
        self._last_step: int = 0
        self._last_stage_id: int = 0

    def stage_boundary_milestones(self) -> tuple[int, ...]:
        """Step indices right at each Stage boundary (`[15000, 65000, 85000, 100000]`)."""
        return tuple(self.boundaries[1:])

    def stage_for_step(self, step: int) -> int:
        """Map 0-based optimization step index to developmental stage ID in {1, 2, 3, 4}."""
        s = max(0, int(step))
        if s < self.boundaries[1]:
            return 1
        if s < self.boundaries[2]:
            return 2
        if s < self.boundaries[3]:
            return 3
        return 4

    def get(self, step: int) -> CurriculumStepState:
        """Compute exact deterministic `CurriculumStepState` for 0-based `step`."""
        s = max(0, int(step))
        stage_id = self.stage_for_step(s)
        st_start = self.boundaries[stage_id - 1]
        st_end = self.boundaries[stage_id]
        st_len = max(st_end - st_start, 1)
        step_in_stage = min(max(s - st_start, 0), st_len - 1)
        progress = step_in_stage / float(max(st_len - 1, 1)) if st_len > 1 else 1.0

        active_owners, frozen_owners = stage_owner_sets(stage_id)
        owner_mults: dict[str, float] = {
            "cortical_io": 0.0,
            "bcg_pc": 0.0,
            "broca_cq": 0.0,
            "hippocampus_adapter": 0.0,
            "hippocampus_gates": 0.0,
        }
        for own in active_owners:
            owner_mults[own] = 1.0

        if stage_id == 1:
            warmup_steps = _scale_duration(
                int(self.s1_cfg.get("global_warmup_steps", 2000)), self._nom_s1, st_len
            )
            global_lr = min(1.0, float(step_in_stage + 1) / float(max(warmup_steps, 1)))
            tau_s = _linear_interp(
                float(self.s1_cfg.get("tau_s_start", 1.0)),
                float(self.s1_cfg.get("tau_s_end", 0.35)),
                progress,
            )
            target_rho = float(self.s1_cfg.get("target_rho", 9.5))
            state = CurriculumStepState(
                step=s,
                stage_id=1,
                stage_name=str(self.s1_cfg.get("name", "sensorimotor_spelling_autoencoder")),
                model_stage="A",
                step_in_stage=step_in_stage,
                stage_length=st_len,
                stage_progress=progress,
                global_lr_scale=global_lr,
                owner_lr_multipliers=owner_mults,
                max_doc_bytes=int(self.s1_cfg.get("max_doc_bytes", 512)),
                domain_weights=_weights_list_to_dict(
                    self.s1_cfg.get("domain_weights", [0.35, 0.35, 0.15, 0.15])
                ),
                wug_ratio=float(self.s1_cfg.get("wug_ratio", 0.0)),
                tau_s=tau_s,
                tau_m=1.5,
                target_rho=target_rho,
                target_spike_rate=1.0 / target_rho,
                lambda_rate=float(self.s1_cfg.get("lambda_rate", 2.0)),
                lambda_var=float(self.s1_cfg.get("lambda_var", 1.0)),
                lambda_halt=float(self.s1_cfg.get("lambda_halt", 1.0)),
                lambda_pc=float(self.s1_cfg.get("lambda_pc", 0.0)),
                lambda_act=float(self.s1_cfg.get("lambda_act", 0.0)),
                code_delimiter_focal_weight=1.0,
                wug_target_weight=1.0,
                lambda_ach_reg=0.0,
                use_broca_cq=False,
                active_owners=active_owners,
                frozen_owners=frozen_owners,
            )
        elif stage_id == 2:
            # Local linear warmup 0.01 -> 1.0 for newly unfrozen BCG-PC params over first 1,000 steps
            bcg_warmup = _scale_duration(
                int(self.s2_cfg.get("bcg_local_warmup_steps", 1000)), self._nom_s2, st_len
            )
            warmup_start = float(self.s2_cfg.get("local_warmup_start_factor", 0.01))
            if step_in_stage < bcg_warmup:
                frac_w = float(step_in_stage) / float(max(bcg_warmup - 1, 1)) if bcg_warmup > 1 else 1.0
                bcg_local = _linear_interp(warmup_start, 1.0, frac_w)
            else:
                bcg_local = 1.0
            owner_mults["bcg_pc"] = bcg_local
            owner_mults["hippocampus_adapter"] = bcg_local

            # Dynamic window schedule: 512 (0..40%) -> 1024 (40..70%) -> 2048 (70..100%)
            if st_len == self._nom_s2 and "window_schedule" in self.s2_cfg:
                max_doc = 512
                for thr_step, win_bytes in self.s2_cfg["window_schedule"]:
                    if s >= int(thr_step):
                        max_doc = int(win_bytes)
            else:
                if progress < 0.40:
                    max_doc = 512
                elif progress < 0.70:
                    max_doc = 1024
                else:
                    max_doc = 2048

            tau_m = _cosine_interp(
                float(self.s2_cfg.get("tau_m_start", 1.5)),
                float(self.s2_cfg.get("tau_m_end", 0.25)),
                progress,
            )
            tau_s = float(self.s2_cfg.get("tau_s", 0.30))

            pc_ramp = _scale_duration(
                int(self.s2_cfg.get("lambda_pc_ramp_steps", 3000)), self._nom_s2, st_len
            )
            pc_target = float(self.s2_cfg.get("lambda_pc_target", 1.0))
            lambda_pc = pc_target * min(1.0, float(step_in_stage + 1) / float(max(pc_ramp, 1)))

            act_delay = _scale_duration(
                int(self.s2_cfg.get("lambda_act_delay_steps", 15000)), self._nom_s2, st_len
            )
            act_ramp = _scale_duration(
                int(self.s2_cfg.get("lambda_act_ramp_steps", 15000)), self._nom_s2, st_len
            )
            act_target = float(self.s2_cfg.get("lambda_act_target", 0.04))
            if step_in_stage < act_delay:
                lambda_act = 0.0
            else:
                lambda_act = act_target * min(
                    1.0, float(step_in_stage - act_delay + 1) / float(max(act_ramp, 1))
                )

            target_rho = float(self.s2_cfg.get("target_rho", 9.5))
            state = CurriculumStepState(
                step=s,
                stage_id=2,
                stage_name=str(self.s2_cfg.get("name", "bottlenecked_concept_merge_and_pondering")),
                model_stage="B",
                step_in_stage=step_in_stage,
                stage_length=st_len,
                stage_progress=progress,
                global_lr_scale=1.0,
                owner_lr_multipliers=owner_mults,
                max_doc_bytes=max_doc,
                domain_weights=_weights_list_to_dict(
                    self.s2_cfg.get("domain_weights", [0.32, 0.36, 0.16, 0.16])
                ),
                wug_ratio=float(self.s2_cfg.get("wug_ratio", 0.0)),
                tau_s=tau_s,
                tau_m=tau_m,
                target_rho=target_rho,
                target_spike_rate=1.0 / target_rho,
                lambda_rate=float(self.s2_cfg.get("lambda_rate", 2.0)),
                lambda_var=float(self.s2_cfg.get("lambda_var", 1.0)),
                lambda_halt=float(self.s2_cfg.get("lambda_halt", 1.0)),
                lambda_pc=lambda_pc,
                lambda_act=lambda_act,
                code_delimiter_focal_weight=1.0,
                wug_target_weight=1.0,
                lambda_ach_reg=0.0,
                use_broca_cq=False,
                active_owners=active_owners,
                frozen_owners=frozen_owners,
            )
        elif stage_id == 3:
            end_ratio = float(self.s3_cfg.get("lr_cosine_end_ratio", 0.25))
            global_lr = _cosine_interp(1.0, end_ratio, progress)
            target_rho = float(self.s3_cfg.get("target_rho", 9.5))
            broca_prob = float(self.s3_cfg.get("broca_cq_prob", 0.5))
            if broca_prob <= 0.0:
                use_cq = False
            elif broca_prob >= 1.0:
                use_cq = True
            else:
                cq_period = max(2, int(round(1.0 / broca_prob)))
                use_cq = (s % cq_period) == (cq_period - 1)
            state = CurriculumStepState(
                step=s,
                stage_id=3,
                stage_name=str(self.s3_cfg.get("name", "cross_lingual_reordering_and_code_stack")),
                model_stage="C",
                step_in_stage=step_in_stage,
                stage_length=st_len,
                stage_progress=progress,
                global_lr_scale=global_lr,
                owner_lr_multipliers=owner_mults,
                max_doc_bytes=int(self.s3_cfg.get("max_doc_bytes", 4096)),
                domain_weights=_weights_list_to_dict(
                    self.s3_cfg.get("domain_weights", [0.26, 0.28, 0.23, 0.23])
                ),
                wug_ratio=float(self.s3_cfg.get("wug_ratio", 0.0)),
                tau_s=float(self.s3_cfg.get("tau_s", 0.25)),
                tau_m=float(self.s3_cfg.get("tau_m", 0.25)),
                target_rho=target_rho,
                target_spike_rate=1.0 / target_rho,
                lambda_rate=float(self.s3_cfg.get("lambda_rate", 2.0)),
                lambda_var=float(self.s3_cfg.get("lambda_var", 1.0)),
                lambda_halt=float(self.s3_cfg.get("lambda_halt", 1.0)),
                lambda_pc=float(self.s3_cfg.get("lambda_pc", 1.0)),
                lambda_act=float(self.s3_cfg.get("lambda_act", 0.04)),
                code_delimiter_focal_weight=float(self.s3_cfg.get("code_delimiter_focal_weight", 1.5)),
                wug_target_weight=1.0,
                lambda_ach_reg=0.0,
                use_broca_cq=use_cq,
                active_owners=active_owners,
                frozen_owners=frozen_owners,
            )
        else:
            # Stage 4 (85k..100k): freeze 95%+ slow weights, unfreeze Hippocampus gates + adapter
            # with 500-step local linear warmup (0.01 -> 1.0), then gentle cosine tail to 0.25.
            hip_warmup = _scale_duration(
                int(self.s4_cfg.get("hip_local_warmup_steps", 500)), self._nom_s4, st_len
            )
            warmup_start = float(self.s4_cfg.get("local_warmup_start_factor", 0.01))
            if step_in_stage < hip_warmup:
                frac_w = float(step_in_stage) / float(max(hip_warmup - 1, 1)) if hip_warmup > 1 else 1.0
                hip_local = _linear_interp(warmup_start, 1.0, frac_w)
            else:
                tail_frac = float(step_in_stage - hip_warmup) / float(max(st_len - hip_warmup - 1, 1))
                hip_local = _cosine_interp(1.0, 0.25, tail_frac)
            owner_mults["hippocampus_gates"] = hip_local
            owner_mults["hippocampus_adapter"] = hip_local

            target_rho = 9.5
            state = CurriculumStepState(
                step=s,
                stage_id=4,
                stage_name=str(self.s4_cfg.get("name", "hippocampal_oneshot_meta_plasticity")),
                model_stage="D",
                step_in_stage=step_in_stage,
                stage_length=st_len,
                stage_progress=progress,
                global_lr_scale=1.0,
                owner_lr_multipliers=owner_mults,
                max_doc_bytes=int(self.s4_cfg.get("max_doc_bytes", 1024)),
                domain_weights=_weights_list_to_dict(
                    self.s4_cfg.get("domain_weights", [0.25, 0.25, 0.25, 0.25])
                ),
                wug_ratio=float(self.s4_cfg.get("wug_ratio", 0.50)),
                tau_s=float(self.s4_cfg.get("tau_s", 0.25)),
                tau_m=float(self.s4_cfg.get("tau_m", 0.25)),
                target_rho=target_rho,
                target_spike_rate=1.0 / target_rho,
                lambda_rate=float(self.s4_cfg.get("lambda_rate", 0.0)),
                lambda_var=float(self.s4_cfg.get("lambda_var", 0.0)),
                lambda_halt=float(self.s4_cfg.get("lambda_halt", 0.0)),
                lambda_pc=float(self.s4_cfg.get("lambda_pc", 0.0)),
                lambda_act=float(self.s4_cfg.get("lambda_act", 0.0)),
                code_delimiter_focal_weight=1.0,
                wug_target_weight=float(self.s4_cfg.get("wug_target_weight", 3.0)),
                lambda_ach_reg=float(self.s4_cfg.get("lambda_ach_reg", 0.5)),
                use_broca_cq=False,
                active_owners=active_owners,
                frozen_owners=frozen_owners,
            )

        self._last_step = s
        self._last_stage_id = stage_id
        return state

    def apply_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        state: CurriculumStepState,
        watchdog_lr_multiplier: float = 1.0,
    ) -> dict[str, float]:
        """Set effective LR on every fixed AdamW parameter group without rebuilding optimizer."""
        applied: dict[str, float] = {}
        for group in optimizer.param_groups:
            owner = str(group.get("owner", "cortical_io"))
            base_lr = float(group.get("base_lr", group["lr"]))
            local_mult = float(state.owner_lr_multipliers.get(owner, 0.0))
            eff_lr = base_lr * state.global_lr_scale * local_mult * float(watchdog_lr_multiplier)
            group["lr"] = eff_lr
            applied[str(group.get("name", owner))] = eff_lr
        return applied

    def state_dict(self) -> dict[str, Any]:
        return {
            "boundaries": list(self.boundaries),
            "last_step": int(self._last_step),
            "last_stage_id": int(self._last_stage_id),
        }

    def load_state_dict(self, state: Optional[Mapping[str, Any]]) -> None:
        if not state:
            return
        self._last_step = int(state.get("last_step", 0))
        self._last_stage_id = int(state.get("last_stage_id", 0))


def build_byte_loss_weights(
    x_bytes: torch.Tensor,
    byte_lang_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    wug_target_mask: torch.Tensor,
    state: CurriculumStepState,
) -> Optional[torch.Tensor]:
    """Build exact per-byte loss weighting tensor `[1, T]` for Stages 3 and 4.

    - Stages 1 & 2: returns `None` (uniform 1.0 over all bytes).
    - Stage 3: applies `1.5x` focal CE weight on code delimiter bytes (`{}()[]:;\\n`) inside
      Python (`lang_id=3`) and Rust (`lang_id=4`) segments to accelerate pushdown stack convergence.
    - Stage 4: zeros out Turn 1 (`loss_mask == False`), keeps `1.0` on Turn 2 & familiar replay,
      and boosts `wug_target_mask` bytes to `3.0x`.
    """
    device = x_bytes.device
    if state.stage_id == 4:
        w = loss_mask.to(device=device, dtype=torch.float32)
        if state.wug_target_weight != 1.0:
            tgt = wug_target_mask.to(device=device, dtype=torch.bool) & loss_mask.to(device=device, dtype=torch.bool)
            w = torch.where(tgt, torch.full_like(w, float(state.wug_target_weight)), w)
        return w.unsqueeze(0)

    if state.stage_id == 3:
        b = x_bytes.to(device=device, dtype=torch.int64)
        m_bool = loss_mask.to(device=device, dtype=torch.bool)
        w = m_bool.to(dtype=torch.float32)
        if state.code_delimiter_focal_weight > 1.0:
            lids = byte_lang_ids.to(device=device, dtype=torch.int64)
            is_code = (lids == 3) | (lids == 4) | (lids == 5)
            is_delim = torch.zeros_like(is_code)
            for code_pt in CODE_DELIMITER_BYTES:
                is_delim = is_delim | (b == code_pt)
            boost = is_code & is_delim & m_bool
            w = torch.where(
                boost,
                torch.full_like(w, float(state.code_delimiter_focal_weight)),
                w,
            )
        # Boost EOT (0x04) and tool-call framing bytes (0x11 / 0x13) on supervised capability spans
        is_eot = (b == 0x04) & m_bool
        is_tool = ((b == 0x11) | (b == 0x13)) & m_bool
        w = torch.where(is_eot, torch.maximum(w, torch.full_like(w, 1.5)), w)
        w = torch.where(is_tool, torch.maximum(w, torch.full_like(w, 2.0)), w)
        return w.unsqueeze(0)

    if state.stage_id == 2 and not bool(loss_mask.all().item()):
        return loss_mask.to(device=device, dtype=torch.float32).unsqueeze(0)

    return None
