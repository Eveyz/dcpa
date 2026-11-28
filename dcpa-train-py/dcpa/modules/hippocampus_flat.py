"""WS-4: Sandwiched Chunk-Folded Hippocampus & Dual-Morpheme Whitened Dentate Gyrus (DCPA-Flat v3.3).

Implements R4 of `docs/DCPA_FLAT_V3_3_SPEC.md` and `HippocampusChunkState` from `dcpa.interfaces`:
  1. `DualMorphemeWhitenedDG`:
     - Non-commutative orthogonal rotation dual-morpheme binding (`bind(a, b) != bind(b, a)`),
       resetting `f_prev` cleanly on document boundaries (`seg_reset`).
     - Near-zero-initialized residual whitening preconditioner `W_whiten` placed before frozen
       block-QR orthogonal random projections `R_lex [d_h, d_s]` and `R_sem [d_h, D]`, with
       `freeze_whiten(True)` for Stage 5 / inference stationarity.
     - Non-gradient online homeostatic excitability bias buffers `b_homeo` / `b_homeo_sem`.
     - Static-shape sparse pairwise inner-product hinge decorrelation loss `L_dg` (zero `nonzero`).
  2. `HippocampusFlat`:
     - Lane-resident fixed-shape `HippocampusChunkState` with lazy doc-ID masking (zero `W_max`
       eviction, zero Python list state containers).
     - `forward_interleaved_with_carrier_g2`: R4 causal chunk-interleaved `HIPR <-> Carrier G2`
       loop with Write-Before-Merge three-factor crystallization, fixed-shape intra-chunk delta
       buffers, and doc-isolated chunk-end folding into `m_c_base`.
     - `read_expression_chunk_triangular`: doc-isolated chunk-level unit-lower-triangular WY /
       Gated-Delta parallel solver (`torch.linalg.solve_triangular`) for `M^{L<-C}`, mathematically
       bit-equivalent (`< 1e-5` fp32) to isolated per-document sequential fast weights.
     - `step_streaming`: single-step exact sequential Gated-Delta path for Gate 6 parity.

Global v3.3 hot-path red lines strictly enforced:
  - ZERO `torch.nonzero`, ZERO `repeat_interleave`, ZERO dynamic list state containers,
    ZERO `.item()`, `.cpu()`, or `.tolist()`.
  - Decoupled from `graph_carrier.py` via `carrier_step_fn` callable protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.interfaces import (
    CarrierGraphState,
    CarrierScanOut,
    DCPAFlatConfig,
    HippocampusChunkState,
)
from dcpa.ops.fp32 import fp32_island

EMPTY_DOC_SENTINEL: int = -999999

CarrierStepFn = Callable[
    [
        CarrierGraphState,  # state
        torch.Tensor,       # x_in [P, D]
        torch.Tensor,       # reset [P] bool
        torch.Tensor,       # valid [P] bool
        torch.Tensor,       # pin_trig [P] bool
        torch.Tensor,       # unpin_trig [P] bool
    ],
    tuple[
        CarrierGraphState,  # next_state
        CarrierGraphState,  # prior_state (G2_prior)
        torch.Tensor,       # evict_weight [P, K] float32
        torch.Tensor,       # evict_target [P, K, D]
        Any,                # slot_assign_or_merge_info
    ],
]


def deterministic_orthogonal_projection(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Create deterministic frozen `[rows, cols]` matrix with block-QR orthogonal blocks & unit-norm rows."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    blocks: list[torch.Tensor] = []
    remaining = int(rows)
    while remaining > 0:
        mat = torch.randn((cols, cols), generator=gen, dtype=torch.float32)
        q, r = torch.linalg.qr(mat)
        diag_sign = torch.sign(torch.diag(r))
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        q = q * diag_sign.unsqueeze(0)
        take = min(remaining, cols)
        blocks.append(q[:take, :])
        remaining -= take

    full = torch.cat(blocks, dim=0)
    full = full / (full.norm(dim=-1, keepdim=True) + 1e-8)
    return full.contiguous()


def deterministic_orthogonal_rotation(dim: int, seed: int) -> torch.Tensor:
    """Create deterministic frozen Haar-orthogonal `[dim, dim]` matrix (`Q^T Q = I`, `Q != I`)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    mat = torch.randn((dim, dim), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)
    return q.contiguous()


@dataclass(frozen=True)
class IntraChunkWriteBuffer:
    """Fixed-shape intra-chunk delta buffer for `M^{C<-L}` (zero dynamic lists)."""

    k_idx: torch.Tensor         # [P, C, K, s] int64
    k_val: torch.Tensor         # [P, C, K, s] float32
    delta: torch.Tensor         # [P, C, K, D] float32
    write_doc_id: torch.Tensor  # [P, C]       int32
    write_step: torch.Tensor    # [C]          int64 (0..C-1)
    write_group: Optional[torch.Tensor] = None  # [P, C] int32 intra-chunk doc instance id
    beta: Optional[torch.Tensor] = None         # [P, C, K] float32


@dataclass(frozen=True)
class HippocampusFlatReadOut:
    """Comprehension-side `HIPR` readout output."""

    c_fused: torch.Tensor     # [P, D]
    prec_bu: torch.Tensor     # [P, D] float32
    psi_idx: torch.Tensor     # [P, s] int64
    psi_val: torch.Tensor     # [P, s] float32
    ach_signal: torch.Tensor  # [P] float32
    gamma_slow: torch.Tensor  # [P] float32
    gamma_hip: torch.Tensor   # [P] float32
    c_hip: torch.Tensor       # [P, D] float32


@dataclass(frozen=True)
class HippocampusWriteEvents:
    """Fixed-shape sequence of Write-Before-Merge events for chunk-triangular `M^{L<-C}`."""

    phi_idx: torch.Tensor    # [P, S, K, s]   int64
    phi_val: torch.Tensor    # [P, S, K, s]   float32
    target_f: torch.Tensor   # [P, S, K, d_s] float32
    beta: torch.Tensor       # [P, S, K]      float32
    doc_id: torch.Tensor     # [P, S]         int32
    seg_reset: torch.Tensor  # [P, S]         bool
    seg_valid: torch.Tensor  # [P, S]         bool


@dataclass(frozen=True)
class HippocampusInterleavedOut:
    """Output of R4 `HIPR <-> Carrier G2` causal chunk-interleaved forward pass."""

    g2_scan: CarrierScanOut
    write_events: HippocampusWriteEvents
    final_hip_state: HippocampusChunkState
    c_fused_seq: torch.Tensor    # [P, S, D]
    prec_bu_seq: torch.Tensor    # [P, S, D] float32
    c_hip_seq: torch.Tensor      # [P, S, D] float32
    ach_seq: torch.Tensor        # [P, S] float32
    gamma_hip_seq: torch.Tensor  # [P, S] float32
    l_dg: torch.Tensor           # scalar float32
    mean_ach: torch.Tensor       # scalar float32
    mean_beta: torch.Tensor      # scalar float32
    max_beta_seq: torch.Tensor   # [P, S] float32 (for WS-6 sleep capture `beta > 0.25`)


@dataclass(frozen=True)
class HippocampusStreamStepOut:
    """Single-step streaming output for Gate 6 parity (`Python <-> dcpa-core-rs`)."""

    next_g2_state: CarrierGraphState
    prior_g2_state: CarrierGraphState
    next_hip_state: HippocampusChunkState
    next_prev_stem: torch.Tensor      # [P, d_s]
    read_out: HippocampusFlatReadOut
    f_hip: Optional[torch.Tensor]     # [P, d_s] if q_concept was supplied
    rho_out: Optional[torch.Tensor]   # [P] float32 if q_concept was supplied
    beta: torch.Tensor                # [P, K] float32
    max_beta: torch.Tensor            # [P] float32


class DualMorphemeWhitenedDG(nn.Module):
    """Non-commutative dual-morpheme binding + residual whitening + sparse k-WTA Dentate Gyrus."""

    def __init__(self, cfg: DCPAFlatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_s = cfg.d_s
        D = cfg.d_concept
        d_h = cfg.d_h
        h_w = cfg.dg_whiten_hidden

        # Frozen deterministic orthogonal rotation for non-commutative dual-morpheme binding
        self.register_buffer("q_rot", deterministic_orthogonal_rotation(d_s, cfg.dg_seed_lex + 10007))

        # Frozen deterministic block-QR orthogonal random projections (NEVER trainable)
        self.register_buffer("r_lex", deterministic_orthogonal_projection(d_h, d_s, cfg.dg_seed_lex))
        self.register_buffer("r_sem", deterministic_orthogonal_projection(d_h, D, cfg.dg_seed_sem))

        # Online non-gradient homeostatic excitability bias buffers
        self.register_buffer("b_homeo", torch.zeros((d_h,), dtype=torch.float32))
        self.register_buffer("b_homeo_sem", torch.zeros((d_h,), dtype=torch.float32))

        # Residual whitening preconditioner placed BEFORE frozen R_lex
        # Initialized near zero so step 0 starts at exact orthogonal baseline while keeping grad non-zero
        self.w_whiten_1 = nn.Linear(d_s, h_w, bias=True)
        self.w_whiten_2 = nn.Linear(h_w, d_s, bias=True)
        nn.init.kaiming_uniform_(self.w_whiten_1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_whiten_1.bias)
        nn.init.normal_(self.w_whiten_2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.w_whiten_2.bias)

    def whiten_parameters(self) -> list[nn.Parameter]:
        return [
            self.w_whiten_1.weight,
            self.w_whiten_1.bias,
            self.w_whiten_2.weight,
            self.w_whiten_2.bias,
        ]

    def freeze_whiten(self, freeze: bool = True) -> None:
        """Freeze or unfreeze `W_whiten` (frozen in Stage 5 / inference so DG geometry stays stationary)."""
        req_grad = not bool(freeze)
        for p in self.whiten_parameters():
            p.requires_grad_(req_grad)

    def bind_dual_morpheme(
        self,
        f_cur: torch.Tensor,
        f_prev: Optional[torch.Tensor] = None,
        seg_reset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Non-commutative orthogonal rotation binding: `(f_cur + Q_rot(f_prev)) / sqrt(2)`.

        When `seg_reset` is True (or `f_prev` is None), previous morpheme is zeroed so the first
        morpheme of each document is strictly isolated from prior documents and preserves unit scale.
        """
        cur_f = f_cur.float()
        if f_prev is None:
            return cur_f

        rot_prev = F.linear(f_prev.float(), self.q_rot.float())
        inv_sqrt2 = math.sqrt(0.5)
        if seg_reset is None:
            return (cur_f + rot_prev) * inv_sqrt2

        keep_prev = (~seg_reset).unsqueeze(-1)
        rot_eff = torch.where(keep_prev, rot_prev, torch.zeros_like(rot_prev))
        scale = torch.where(
            keep_prev,
            torch.full_like(cur_f[..., :1], inv_sqrt2),
            torch.ones_like(cur_f[..., :1]),
        )
        return (cur_f + rot_eff) * scale

    def apply_whiten(self, f_bound: torch.Tensor) -> torch.Tensor:
        """Residual whitening preconditioner: `f_w = f_bound + W_whiten_2(silu(W_whiten_1(f_bound)))`."""
        x = f_bound.float()
        h = F.silu(
            F.linear(
                x,
                self.w_whiten_1.weight.float(),
                self.w_whiten_1.bias.float() if self.w_whiten_1.bias is not None else None,
            )
        )
        delta = F.linear(
            h,
            self.w_whiten_2.weight.float(),
            self.w_whiten_2.bias.float() if self.w_whiten_2.bias is not None else None,
        )
        return x + delta

    def dg_lex(
        self,
        f_cur: torch.Tensor,
        f_prev: Optional[torch.Tensor] = None,
        seg_reset: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual-morpheme whitened lexical k-WTA DG -> `(psi_idx [..., s] int64, psi_val [..., s] fp32)`."""
        with fp32_island(f_cur.device):
            f_bound = self.bind_dual_morpheme(f_cur, f_prev=f_prev, seg_reset=seg_reset)
            f_w = self.apply_whiten(f_bound)
            proj = F.linear(f_w, self.r_lex.float())
            scores = proj + self.b_homeo.float()
            top_scores, top_idx = torch.topk(scores, k=self.cfg.s_active, dim=-1, sorted=True)
            has_energy = (f_bound.pow(2).sum(dim=-1, keepdim=True) > 1e-12).to(torch.float32)
            psi_tilde = F.relu(top_scores) * has_energy
            norm = psi_tilde.norm(dim=-1, keepdim=True) + 1e-8
            psi_val = psi_tilde / norm
        return top_idx, psi_val

    def dg_sem(self, concept: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Semantic k-WTA DG for expression lookup -> `(phi_idx [..., s] int64, phi_val [..., s] fp32)`."""
        with fp32_island(concept.device):
            c_f = concept.float()
            proj = F.linear(c_f, self.r_sem.float())
            scores = proj + self.b_homeo_sem.float()
            top_scores, top_idx = torch.topk(scores, k=self.cfg.s_active, dim=-1, sorted=True)
            has_energy = (c_f.pow(2).sum(dim=-1, keepdim=True) > 1e-12).to(torch.float32)
            phi_tilde = F.relu(top_scores) * has_energy
            norm = phi_tilde.norm(dim=-1, keepdim=True) + 1e-8
            phi_val = phi_tilde / norm
        return top_idx, phi_val

    @torch.no_grad()
    def update_homeostasis(
        self,
        psi_idx: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        *,
        is_sem: bool = False,
        lr: Optional[float] = None,
    ) -> None:
        """Adjust `b_homeo` (or `b_homeo_sem`) without autograd to keep firing rate uniform across `d_h`."""
        s_active = psi_idx.shape[-1]
        flat_idx = psi_idx.reshape(-1, s_active)
        N = flat_idx.shape[0]
        if N == 0:
            return

        if valid_mask is None:
            weights = torch.ones((N, s_active), device=psi_idx.device, dtype=torch.float32)
            denom = torch.tensor(float(N), device=psi_idx.device, dtype=torch.float32)
            has_valid = torch.tensor(1.0, device=psi_idx.device, dtype=torch.float32)
        else:
            vm = valid_mask.reshape(N).to(dtype=torch.float32)
            weights = vm.unsqueeze(-1).expand(N, s_active)
            valid_sum = vm.sum()
            denom = valid_sum.clamp_min(1.0)
            has_valid = (valid_sum > 0.0).to(torch.float32)

        counts = torch.zeros((self.cfg.d_h,), device=psi_idx.device, dtype=torch.float32)
        counts.scatter_add_(0, flat_idx.reshape(-1), weights.reshape(-1))
        emp_rate = counts / denom
        target_rate = float(self.cfg.s_active) / float(self.cfg.d_h)
        step_lr = float(self.cfg.dg_homeo_lr if lr is None else lr)
        delta = (target_rate - emp_rate) * (float(self.cfg.d_h) / float(self.cfg.s_active)) * has_valid

        buf = self.b_homeo_sem if is_sem else self.b_homeo
        buf.add_(step_lr * delta).clamp_(-1.0, 1.0)

    def compute_decorrelation_loss(
        self,
        psi_idx: torch.Tensor,
        psi_val: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        max_samples: int = 512,
        margin: float = 0.05,
    ) -> torch.Tensor:
        """Static-shape sparse pairwise inner-product hinge loss `L_dg` on up to `max_samples` keys.

        Strictly ZERO `torch.nonzero` and ZERO host sync: uses deterministic hash-prioritized
        `topk` selection weighted by `valid_mask` and dense `[K_sub, d_h]` Gram multiplication.
        """
        with fp32_island(psi_val.device):
            s_active = psi_idx.shape[-1]
            flat_idx = psi_idx.reshape(-1, s_active)
            flat_val = psi_val.reshape(-1, s_active).float()
            N = flat_idx.shape[0]
            device = psi_idx.device

            if N <= 1:
                return flat_val.sum() * 0.0

            if valid_mask is None:
                flat_valid = torch.ones((N,), device=device, dtype=torch.float32)
            else:
                flat_valid = valid_mask.reshape(N).to(dtype=torch.float32)

            k_sub = min(N, int(max_samples))
            if N > k_sub:
                # Deterministic quasi-random hash in [0, 0.5) so valid samples across all lanes are evenly picked
                perm_hash = ((torch.arange(N, device=device, dtype=torch.int64) * 2654435761) % N).to(torch.float32)
                priority = flat_valid + perm_hash / (2.0 * float(N))
                _, sel = torch.topk(priority, k=k_sub, dim=0, sorted=False)
                sub_idx = flat_idx.index_select(0, sel)
                sub_val = flat_val.index_select(0, sel)
                sub_valid = flat_valid.index_select(0, sel)
            else:
                sub_idx = flat_idx
                sub_val = flat_val
                sub_valid = flat_valid

            dense = torch.zeros((k_sub, self.cfg.d_h), device=device, dtype=torch.float32)
            dense = dense.scatter_add(1, sub_idx, sub_val)
            gram = dense @ dense.transpose(0, 1)  # [k_sub, k_sub]

            off_diag = 1.0 - torch.eye(k_sub, device=device, dtype=torch.float32)
            pair_mask = (sub_valid.unsqueeze(1) * sub_valid.unsqueeze(0)) * off_diag
            hinge = F.relu(gram - float(margin)).pow(2) * pair_mask
            denom = pair_mask.sum().clamp_min(1.0)
            return hinge.sum() / denom


class HippocampusFlat(nn.Module):
    """DCPA-Flat v3.3 chunk-folded sandwiched Hippocampus with doc-isolated triangular expression."""

    def __init__(self, cfg: DCPAFlatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_concept

        self.dg = DualMorphemeWhitenedDG(cfg)

        # Scalar gating parameters (dedicated optimizer group: lr_scalar_gate, weight_decay=0.0)
        self.alpha_1 = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.alpha_2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.theta_i = nn.Parameter(torch.tensor(2.5, dtype=torch.float32))
        self.theta_n = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.alpha_q = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.gamma_e = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.theta_q = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))

        self.alpha_out = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))
        self.theta_out = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        # Bottom-up cortical precision projection pi_cort = Softplus(f_prec_cort(c_slow)) + 1e-4
        self.f_prec_cort = nn.Linear(D, D, bias=True)
        nn.init.normal_(self.f_prec_cort.weight, std=0.01)
        nn.init.constant_(self.f_prec_cort.bias, 0.5413)

    def scalar_gate_parameters(self) -> list[nn.Parameter]:
        return [
            self.alpha_1,
            self.alpha_2,
            self.theta_i,
            self.theta_n,
            self.alpha_q,
            self.gamma_e,
            self.theta_q,
            self.alpha_out,
            self.theta_out,
        ]

    def init_chunk_state(
        self,
        P: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> HippocampusChunkState:
        """Allocate lane-resident `HippocampusChunkState` (fast weights strictly fp32, sentinel doc_id)."""
        _ = dtype  # Fast-weight precision island is always float32 per `HippocampusChunkState` contract
        dev = self.dg.r_lex.device if device is None else device
        d_h = self.cfg.d_h
        D = self.cfg.d_concept
        d_s = self.cfg.d_s
        K = self.cfg.k_slots
        s = self.cfg.s_active
        return HippocampusChunkState(
            m_c_base=torch.zeros((P, d_h, D), device=dev, dtype=torch.float32),
            m_l_base=torch.zeros((P, d_h, d_s), device=dev, dtype=torch.float32),
            u_lex=torch.zeros((P, d_h), device=dev, dtype=torch.float32),
            u_sem=torch.zeros((P, d_h), device=dev, dtype=torch.float32),
            base_doc_id=torch.full((P,), EMPTY_DOC_SENTINEL, device=dev, dtype=torch.int32),
            elig_idx=torch.zeros((P, K, s), device=dev, dtype=torch.int64),
            elig_val=torch.zeros((P, K, s), device=dev, dtype=torch.float32),
            elig_f=torch.zeros((P, K, d_s), device=dev, dtype=torch.float32),
            elig_eta=torch.zeros((P, K), device=dev, dtype=torch.float32),
        )

    def _compute_pi_cort(self, c_slow: torch.Tensor) -> torch.Tensor:
        return (
            F.softplus(
                F.linear(
                    c_slow.float(),
                    self.f_prec_cort.weight.float(),
                    self.f_prec_cort.bias.float() if self.f_prec_cort.bias is not None else None,
                )
            )
            + 1e-4
        )

    def _compute_gamma_slow(self, sem_norm: torch.Tensor, mean_info: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.alpha_1 * (self.theta_i - mean_info.float())
            + self.alpha_2 * (sem_norm.float() - self.theta_n)
        )

    def _compute_three_factor_beta(
        self,
        elig_eta: torch.Tensor,
        evict_weight: torch.Tensor,
        evict_target: torch.Tensor,
        edges: torch.Tensor,
        valid: torch.Tensor,
        write_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute crystallization `q_cryst [P, K]` and effective write strength `beta [P, K]`."""
        slot_norm = evict_target.float().norm(dim=-1)               # [P, K]
        edge_norm_sum = edges.float().norm(dim=-1).sum(dim=-1)      # [P, K]
        q_cryst = torch.sigmoid(
            self.alpha_q * (slot_norm + self.gamma_e * edge_norm_sum - self.theta_q)
        )
        evict_w = evict_weight.float().clamp(0.0, 1.0)
        beta_raw = elig_eta.float() * q_cryst * evict_w
        valid_f = valid.to(torch.float32).unsqueeze(-1)
        active = (beta_raw.detach() > float(write_threshold)).to(torch.float32) * valid_f
        beta = beta_raw * active
        return q_cryst, beta

    @staticmethod
    def _parse_slot_info(
        slot_assign_or_merge_info: Any,
        next_slots: torch.Tensor,
        x_in: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Extract `(slot_star [P], win_i, win_j, merge_active)` from flexible carrier step metadata."""
        P, K, _ = next_slots.shape
        device = next_slots.device

        slot_star: Optional[torch.Tensor] = None
        win_i: Optional[torch.Tensor] = None
        win_j: Optional[torch.Tensor] = None
        merge_active: Optional[torch.Tensor] = None

        if isinstance(slot_assign_or_merge_info, torch.Tensor):
            if slot_assign_or_merge_info.ndim == 2:
                slot_star = slot_assign_or_merge_info.argmax(dim=-1).to(torch.int64)
            elif slot_assign_or_merge_info.ndim == 1:
                slot_star = slot_assign_or_merge_info.to(torch.int64)
        elif isinstance(slot_assign_or_merge_info, dict):
            sa = slot_assign_or_merge_info.get("slot_assign", None)
            if isinstance(sa, torch.Tensor):
                slot_star = sa.argmax(dim=-1).to(torch.int64) if sa.ndim == 2 else sa.to(torch.int64)
            wi = slot_assign_or_merge_info.get("win_i", slot_assign_or_merge_info.get("merge_dst", None))
            wj = slot_assign_or_merge_info.get("win_j", slot_assign_or_merge_info.get("merge_src", None))
            if isinstance(wi, torch.Tensor) and isinstance(wj, torch.Tensor):
                win_i = wi.to(torch.int64)
                win_j = wj.to(torch.int64)
                ma = slot_assign_or_merge_info.get("merge_active", slot_assign_or_merge_info.get("sigma_full", None))
                if isinstance(ma, torch.Tensor):
                    merge_active = ma if ma.dtype == torch.bool else (ma.float() > 0.5)
        elif isinstance(slot_assign_or_merge_info, (tuple, list)):
            if len(slot_assign_or_merge_info) == 2:
                a, b = slot_assign_or_merge_info
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    if a.ndim == 1 and b.ndim == 1 and not a.is_floating_point() and not b.is_floating_point():
                        win_i = a.to(torch.int64)
                        win_j = b.to(torch.int64)
                    elif a.ndim == 2:
                        slot_star = a.argmax(dim=-1).to(torch.int64)
            elif len(slot_assign_or_merge_info) == 3:
                a, b, c = slot_assign_or_merge_info
                if isinstance(a, torch.Tensor) and a.ndim == 2:
                    slot_star = a.argmax(dim=-1).to(torch.int64)
                    if isinstance(b, torch.Tensor) and isinstance(c, torch.Tensor):
                        win_i = b.to(torch.int64)
                        win_j = c.to(torch.int64)
                elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and isinstance(c, torch.Tensor):
                    win_i = a.to(torch.int64)
                    win_j = b.to(torch.int64)
                    merge_active = c if c.dtype == torch.bool else (c.float() > 0.5)

        if slot_star is None:
            if win_j is not None:
                slot_star = win_j
            else:
                sims = (next_slots.float() * x_in.float().unsqueeze(1)).sum(dim=-1)
                slot_star = sims.argmax(dim=-1).to(torch.int64)

        slot_star = slot_star.clamp(0, K - 1)
        if win_i is not None and win_j is not None:
            win_i = win_i.clamp(0, K - 1)
            win_j = win_j.clamp(0, K - 1)
            if merge_active is None:
                merge_active = torch.ones((P,), device=device, dtype=torch.bool)
        return slot_star, win_i, win_j, merge_active

    def _update_eligibility_after_write(
        self,
        elig_idx: torch.Tensor,
        elig_val: torch.Tensor,
        elig_f: torch.Tensor,
        elig_eta: torch.Tensor,
        beta: torch.Tensor,
        evict_weight: torch.Tensor,
        psi_idx: torch.Tensor,
        psi_val: torch.Tensor,
        f_cur: torch.Tensor,
        ach: torch.Tensor,
        valid: torch.Tensor,
        next_slots: torch.Tensor,
        x_in: torch.Tensor,
        slot_assign_or_merge_info: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quench consumed eligibility, apply branchless argmax merge inheritance, and tag target slot."""
        P, K, s = elig_idx.shape
        d_s = elig_f.shape[-1]
        device = elig_idx.device
        kappa = float(getattr(self.cfg, "kappa_quench", 0.95))
        lam = float(self.cfg.lambda_eta)

        # 1. Quench written slots & decay eligibility traces
        eta_q = ((1.0 - kappa * beta.detach().float()) * elig_eta.float()).clamp(0.0, 1.0)
        eta_dec = eta_q * lam

        new_idx = elig_idx
        new_val = elig_val.float()
        new_f = elig_f.float()
        new_eta = eta_dec

        slot_star, win_i, win_j, merge_active = self._parse_slot_info(slot_assign_or_merge_info, next_slots, x_in)
        slot_ids = torch.arange(K, device=device, dtype=torch.int64).unsqueeze(0)  # [1, K]

        # 2. Optional discrete argmax asymmetric inheritance on merge pair (preserves s-sparse unit norm)
        if win_i is not None and win_j is not None and merge_active is not None:
            idx_i = win_i.view(P, 1, 1)
            idx_j = win_j.view(P, 1, 1)
            eta_i = new_eta.gather(1, win_i.unsqueeze(1)).squeeze(1)
            eta_j = new_eta.gather(1, win_j.unsqueeze(1)).squeeze(1)
            p_star = torch.where(eta_i >= eta_j, win_i, win_j)

            p_idx = new_idx.gather(1, p_star.view(P, 1, 1).expand(P, 1, s))
            p_val = new_val.gather(1, p_star.view(P, 1, 1).expand(P, 1, s))
            p_f = new_f.gather(1, p_star.view(P, 1, 1).expand(P, 1, d_s))
            p_eta = new_eta.gather(1, p_star.unsqueeze(1))

            dst_mask = (merge_active & valid).unsqueeze(1) & (slot_ids == win_i.unsqueeze(1))
            new_idx = torch.where(dst_mask.unsqueeze(-1), p_idx, new_idx)
            new_val = torch.where(dst_mask.unsqueeze(-1), p_val, new_val)
            new_f = torch.where(dst_mask.unsqueeze(-1), p_f, new_f)
            new_eta = torch.where(dst_mask, p_eta, new_eta)
            _ = (idx_i, idx_j)

        # 3. Assign incoming morpheme to target slot when novel enough or slot was freshly evicted
        is_target = slot_ids == slot_star.unsqueeze(1)  # [P, K]
        take_new = valid.unsqueeze(1) & is_target & (
            (ach.float().unsqueeze(1) >= new_eta) | (evict_weight.float() > 0.5)
        )
        new_idx = torch.where(take_new.unsqueeze(-1), psi_idx.unsqueeze(1), new_idx)
        new_val = torch.where(take_new.unsqueeze(-1), psi_val.float().unsqueeze(1), new_val)
        new_f = torch.where(take_new.unsqueeze(-1), f_cur.float().unsqueeze(1), new_f)
        new_eta = torch.where(take_new, ach.float().unsqueeze(1), new_eta)

        # Preserve state strictly untouched on invalid lanes
        v_mask = valid.unsqueeze(1)
        out_idx = torch.where(v_mask.unsqueeze(-1), new_idx, elig_idx)
        out_val = torch.where(v_mask.unsqueeze(-1), new_val, elig_val.float())
        out_f = torch.where(v_mask.unsqueeze(-1), new_f, elig_f.float())
        out_eta = torch.where(v_mask, new_eta, elig_eta.float())
        return out_idx, out_val, out_f, out_eta

    def _read_c_from_base_and_buf_single(
        self,
        q_idx: torch.Tensor,
        q_val: torch.Tensor,
        m_c_base: torch.Tensor,
        base_match: torch.Tensor,
        cur_doc_id: torch.Tensor,
        cur_step: int,
        intra_buf: Optional[Union[IntraChunkWriteBuffer, tuple[torch.Tensor, ...]]],
        cur_doc_group: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Read `M^{C<-L} @ q` from `m_c_base` (lazy doc-masked) + strictly prior intra-chunk deltas."""
        P, s = q_idx.shape
        D = self.cfg.d_concept
        d_h = self.cfg.d_h
        gamma = float(self.cfg.gamma_hip)
        device = q_idx.device

        cols = m_c_base.gather(1, q_idx.unsqueeze(-1).expand(P, s, D))
        base_decay = gamma ** int(cur_step)
        c_hip = torch.einsum("ps,psd->pd", q_val.float(), cols) * (
            base_match.to(torch.float32).unsqueeze(-1) * base_decay
        )

        if intra_buf is None:
            return c_hip

        if isinstance(intra_buf, IntraChunkWriteBuffer):
            k_idx = intra_buf.k_idx
            k_val = intra_buf.k_val
            delta = intra_buf.delta
            write_doc_id = intra_buf.write_doc_id
            write_step = intra_buf.write_step
            write_group = intra_buf.write_group
        else:
            k_idx, k_val, delta, write_doc_id, write_step = intra_buf[:5]
            write_group = intra_buf[5] if len(intra_buf) > 5 else None

        C_buf = k_idx.shape[1]
        K_buf = k_idx.shape[2]
        # Exact sparse inner products via scatter_add -> gather (1.5 MB peak, zero [s, s] blowup)
        q_dense = torch.zeros((P, d_h), device=device, dtype=torch.float32).scatter_add(1, q_idx, q_val.float())
        gathered_q = q_dense.gather(1, k_idx.reshape(P, C_buf * K_buf * s)).view(P, C_buf, K_buf, s)
        dots = (gathered_q * k_val.float()).sum(dim=-1)  # [P, C_buf, K_buf]

        step_mask = write_step < int(cur_step)  # [C_buf]
        doc_mask = write_doc_id == cur_doc_id.unsqueeze(1)  # [P, C_buf]
        if write_group is not None and cur_doc_group is not None:
            doc_mask = doc_mask & (write_group == cur_doc_group.unsqueeze(1))
        valid_pair = doc_mask & step_mask.unsqueeze(0)  # [P, C_buf]

        step_diff = (int(cur_step) - write_step).clamp_min(0).to(torch.float32)
        decay = torch.pow(gamma, step_diff)  # [C_buf]
        weights = dots * (valid_pair.to(torch.float32) * decay.unsqueeze(0)).unsqueeze(-1)  # [P, C_buf, K_buf]
        c_recent = torch.einsum("pck,pckd->pd", weights, delta.float())
        return c_hip + c_recent

    def _read_c_from_base_and_buf_multi(
        self,
        e_idx: torch.Tensor,
        e_val: torch.Tensor,
        m_c_base: torch.Tensor,
        base_match: torch.Tensor,
        cur_doc_id: torch.Tensor,
        cur_step: int,
        intra_buf: IntraChunkWriteBuffer,
        cur_doc_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read `M^{C<-L} @ e_k` for all `K` slots & return `(old_c [P, K, D], e_dense [P, K, d_h])`."""
        P, K, s = e_idx.shape
        D = self.cfg.d_concept
        d_h = self.cfg.d_h
        gamma = float(self.cfg.gamma_hip)
        device = e_idx.device

        flat_idx = e_idx.reshape(P, K * s).unsqueeze(-1).expand(P, K * s, D)
        cols = m_c_base.gather(1, flat_idx).view(P, K, s, D)
        base_decay = gamma ** int(cur_step)
        old_base = torch.einsum("pks,pksd->pkd", e_val.float(), cols) * (
            base_match.to(torch.float32).view(P, 1, 1) * base_decay
        )

        e_dense = torch.zeros((P, K, d_h), device=device, dtype=torch.float32).scatter_add(2, e_idx, e_val.float())

        C_buf = intra_buf.k_idx.shape[1]
        K_buf = intra_buf.k_idx.shape[2]
        flat_buf_idx = intra_buf.k_idx.reshape(P, 1, C_buf * K_buf * s).expand(P, K, C_buf * K_buf * s)
        gathered_e = e_dense.gather(2, flat_buf_idx).view(P, K, C_buf, K_buf, s)
        dots = (gathered_e * intra_buf.k_val.float().unsqueeze(1)).sum(dim=-1)  # [P, K, C_buf, K_buf]

        step_mask = intra_buf.write_step < int(cur_step)  # [C_buf]
        doc_mask = intra_buf.write_doc_id == cur_doc_id.unsqueeze(1)  # [P, C_buf]
        if intra_buf.write_group is not None:
            doc_mask = doc_mask & (intra_buf.write_group == cur_doc_group.unsqueeze(1))
        valid_pair = doc_mask & step_mask.unsqueeze(0)  # [P, C_buf]

        step_diff = (int(cur_step) - intra_buf.write_step).clamp_min(0).to(torch.float32)
        decay = torch.pow(gamma, step_diff)
        weights = dots * (valid_pair.to(torch.float32) * decay.unsqueeze(0)).view(P, 1, C_buf, 1)
        old_recent = torch.einsum("pqck,pckd->pqd", weights, intra_buf.delta.float())
        return old_base + old_recent, e_dense

    @staticmethod
    def _update_familiarity_step(
        u: torch.Tensor,
        key_idx: torch.Tensor,
        key_val: torch.Tensor,
        beta: torch.Tensor,
        valid: torch.Tensor,
        gamma_hip: float,
    ) -> torch.Tensor:
        """Exact commutative log-retention familiarity update across `K` slots + end-of-step decay."""
        P, K, s = key_idx.shape
        active = (key_val.float() > 0.0).to(torch.float32)
        b_eff = beta.float() * valid.to(torch.float32).unsqueeze(-1)
        retain = (1.0 - b_eff.unsqueeze(-1) * active).clamp(1e-7, 1.0)
        log_ret = torch.zeros_like(u).scatter_add(
            1,
            key_idx.reshape(P, K * s),
            torch.log(retain).reshape(P, K * s),
        )
        u_updated = (1.0 - (1.0 - u) * torch.exp(log_ret)).clamp(0.0, 1.0)
        u_decayed = u_updated * float(gamma_hip)
        return torch.where(valid.unsqueeze(-1), u_decayed, u)

    def read_comprehension_from_state(
        self,
        psi_idx: torch.Tensor,
        psi_val: torch.Tensor,
        c_slow: torch.Tensor,
        sem_norm: torch.Tensor,
        mean_info: torch.Tensor,
        cur_doc_id: torch.Tensor,
        cur_step: int,
        chunk_state: HippocampusChunkState,
        intra_buf: Optional[Union[IntraChunkWriteBuffer, tuple[torch.Tensor, ...]]] = None,
        *,
        base_active_mask: Optional[torch.Tensor] = None,
        cur_doc_group: Optional[torch.Tensor] = None,
        u_lex_override: Optional[torch.Tensor] = None,
        pi_cort_override: Optional[torch.Tensor] = None,
        gamma_slow_override: Optional[torch.Tensor] = None,
        ach_override: Optional[float] = None,
    ) -> HippocampusFlatReadOut:
        """Query `m_c_base` (doc-masked) + strictly prior intra-chunk delta writes in fp32."""
        orig_dtype = c_slow.dtype
        with fp32_island(c_slow.device):
            base_match = cur_doc_id == chunk_state.base_doc_id
            if base_active_mask is not None:
                base_match = base_match & base_active_mask

            c_hip = self._read_c_from_base_and_buf_single(
                q_idx=psi_idx,
                q_val=psi_val,
                m_c_base=chunk_state.m_c_base,
                base_match=base_match,
                cur_doc_id=cur_doc_id,
                cur_step=cur_step,
                intra_buf=intra_buf,
                cur_doc_group=cur_doc_group,
            )

            if u_lex_override is not None:
                u_lex_eff = u_lex_override.float()
            else:
                base_decay = float(self.cfg.gamma_hip) ** int(cur_step)
                u_lex_eff = chunk_state.u_lex.float() * (base_match.to(torch.float32).unsqueeze(-1) * base_decay)

            gamma_hip = (psi_val.float().pow(2) * u_lex_eff.gather(1, psi_idx)).sum(dim=-1).clamp(0.0, 1.0)
            gamma_slow = (
                gamma_slow_override.float()
                if gamma_slow_override is not None
                else self._compute_gamma_slow(sem_norm, mean_info)
            )
            ach = (1.0 - gamma_slow) * (1.0 - gamma_hip)
            if ach_override is not None:
                ach = torch.full_like(ach, float(ach_override))

            mu = gamma_hip / (gamma_slow + gamma_hip + self.cfg.eps)
            c_fused = ((1.0 - mu).unsqueeze(-1) * c_slow.float() + mu.unsqueeze(-1) * c_hip).to(orig_dtype)
            pi_cort = pi_cort_override.float() if pi_cort_override is not None else self._compute_pi_cort(c_slow)
            prec_bu = (1.0 - ach).unsqueeze(-1) * pi_cort

        return HippocampusFlatReadOut(
            c_fused=c_fused,
            prec_bu=prec_bu,
            psi_idx=psi_idx,
            psi_val=psi_val.float(),
            ach_signal=ach,
            gamma_slow=gamma_slow,
            gamma_hip=gamma_hip,
            c_hip=c_hip,
        )

    def forward_interleaved_with_carrier_g2(
        self,
        x3: torch.Tensor,
        f_stem: torch.Tensor,
        sem_norm: torch.Tensor,
        mean_info: torch.Tensor,
        seg_doc_id: torch.Tensor,
        seg_reset: torch.Tensor,
        seg_valid: torch.Tensor,
        seg_pin_trig: torch.Tensor,
        seg_unpin_trig: torch.Tensor,
        g2_state: CarrierGraphState,
        carrier_step_fn: CarrierStepFn,
        *,
        hip_enabled: bool = True,
        chunk_state: Optional[HippocampusChunkState] = None,
        init_prev_stem: Optional[torch.Tensor] = None,
        write_threshold: Optional[float] = None,
        detach_chunk_base: bool = True,
        update_homeostasis: bool = False,
    ) -> HippocampusInterleavedOut:
        """R4 `HIPR <-> Carrier G2` causal chunk-interleaved forward scan (`C = cfg.hip_chunk_size`)."""
        P, S, D = x3.shape
        d_s = self.cfg.d_s
        K = self.cfg.k_slots
        s_act = self.cfg.s_active
        d_h = self.cfg.d_h
        C = int(self.cfg.hip_chunk_size)
        gamma = float(self.cfg.gamma_hip)
        w_thresh = float(self.cfg.hip_write_threshold if write_threshold is None else write_threshold)
        device = x3.device

        state_hip = self.init_chunk_state(P, device=device) if chunk_state is None else chunk_state
        cur_g2 = g2_state

        # Build effective doc reset mask catching both explicit seg_reset and implicit doc_id change
        doc_changed = torch.cat(
            [
                seg_doc_id[:, :1] != state_hip.base_doc_id.unsqueeze(1),
                seg_doc_id[:, 1:] != seg_doc_id[:, :-1],
            ],
            dim=1,
        )
        eff_reset_all = seg_reset | doc_changed

        # Precompute dual-morpheme lexical DG keys and cortical terms in ONE batched pass
        prev0 = torch.zeros_like(f_stem[:, :1, :]) if init_prev_stem is None else init_prev_stem.unsqueeze(1)
        f_prev_all = torch.cat([prev0, f_stem[:, :-1, :]], dim=1)
        psi_idx_all, psi_val_all = self.dg.dg_lex(f_stem, f_prev=f_prev_all, seg_reset=eff_reset_all)
        l_dg = self.dg.compute_decorrelation_loss(psi_idx_all, psi_val_all, valid_mask=seg_valid)
        if update_homeostasis and self.training:
            self.dg.update_homeostasis(psi_idx_all, valid_mask=seg_valid, is_sem=False)

        with fp32_island(device):
            pi_cort_all = self._compute_pi_cort(x3)  # [P, S, D]
            gamma_slow_all = self._compute_gamma_slow(sem_norm, mean_info)  # [P, S]

        # Fast bypass path for Stage 1-3 (`hip_enabled=False`)
        if not hip_enabled:
            prior_slots_list: list[torch.Tensor] = []
            prior_edges_list: list[torch.Tensor] = []
            prior_occ_list: list[torch.Tensor] = []
            evict_w_list: list[torch.Tensor] = []
            evict_t_list: list[torch.Tensor] = []

            for s_idx in range(S):
                cur_g2, prior_g2, ev_w, ev_t, _ = carrier_step_fn(
                    cur_g2,
                    x3[:, s_idx, :],
                    seg_reset[:, s_idx],
                    seg_valid[:, s_idx],
                    seg_pin_trig[:, s_idx],
                    seg_unpin_trig[:, s_idx],
                )
                prior_slots_list.append(prior_g2.slots)
                prior_edges_list.append(prior_g2.edges)
                prior_occ_list.append(prior_g2.occ)
                evict_w_list.append(ev_w.float())
                evict_t_list.append(ev_t)

            g2_scan = CarrierScanOut(
                slots_seq=torch.stack(prior_slots_list, dim=1),
                edges_seq=torch.stack(prior_edges_list, dim=1),
                occ_seq=torch.stack(prior_occ_list, dim=1),
                final_state=cur_g2,
                evict_weight=torch.stack(evict_w_list, dim=1),
                evict_target=torch.stack(evict_t_list, dim=1),
            )
            zeros_phi_idx = torch.zeros((P, S, K, s_act), device=device, dtype=torch.int64)
            zeros_phi_val = torch.zeros((P, S, K, s_act), device=device, dtype=torch.float32)
            zeros_tgt_f = torch.zeros((P, S, K, d_s), device=device, dtype=torch.float32)
            zeros_beta = torch.zeros((P, S, K), device=device, dtype=torch.float32)
            write_events = HippocampusWriteEvents(
                phi_idx=zeros_phi_idx,
                phi_val=zeros_phi_val,
                target_f=zeros_tgt_f,
                beta=zeros_beta,
                doc_id=seg_doc_id,
                seg_reset=eff_reset_all,
                seg_valid=seg_valid,
            )
            zero_scalar = torch.zeros((), device=device, dtype=torch.float32)
            return HippocampusInterleavedOut(
                g2_scan=g2_scan,
                write_events=write_events,
                final_hip_state=state_hip,
                c_fused_seq=x3,
                prec_bu_seq=pi_cort_all,
                c_hip_seq=torch.zeros((P, S, D), device=device, dtype=torch.float32),
                ach_seq=torch.zeros((P, S), device=device, dtype=torch.float32),
                gamma_hip_seq=torch.zeros((P, S), device=device, dtype=torch.float32),
                l_dg=l_dg,
                mean_ach=zero_scalar,
                mean_beta=zero_scalar,
                max_beta_seq=torch.zeros((P, S), device=device, dtype=torch.float32),
            )

        # Active chunk-interleaved HIPR <-> Carrier G2 loop
        m_c_base = state_hip.m_c_base.float()
        m_l_base = state_hip.m_l_base.float()
        u_lex = state_hip.u_lex.float()
        u_sem = state_hip.u_sem.float()
        base_doc_id = state_hip.base_doc_id
        elig_idx = state_hip.elig_idx
        elig_val = state_hip.elig_val.float()
        elig_f = state_hip.elig_f.float()
        elig_eta = state_hip.elig_eta.float()

        eye_k = torch.eye(K, device=device, dtype=torch.float32).unsqueeze(0).expand(P, -1, -1)

        all_prior_slots: list[torch.Tensor] = []
        all_prior_edges: list[torch.Tensor] = []
        all_prior_occ: list[torch.Tensor] = []
        all_evict_w: list[torch.Tensor] = []
        all_evict_t: list[torch.Tensor] = []
        all_c_fused: list[torch.Tensor] = []
        all_prec_bu: list[torch.Tensor] = []
        all_c_hip: list[torch.Tensor] = []
        all_ach: list[torch.Tensor] = []
        all_gamma_hip: list[torch.Tensor] = []

        all_phi_idx: list[torch.Tensor] = []
        all_phi_val: list[torch.Tensor] = []
        all_target_f: list[torch.Tensor] = []
        all_beta: list[torch.Tensor] = []

        for s0 in range(0, S, C):
            s1 = min(s0 + C, S)
            Cc = s1 - s0

            # Fixed-shape intra-chunk write buffer for M^{C<-L} (ZERO dynamic lists!)
            buf_k_idx = torch.zeros((P, Cc, K, s_act), device=device, dtype=torch.int64)
            buf_k_val = torch.zeros((P, Cc, K, s_act), device=device, dtype=torch.float32)
            buf_delta = torch.zeros((P, Cc, K, D), device=device, dtype=torch.float32)
            buf_beta = torch.zeros((P, Cc, K), device=device, dtype=torch.float32)
            buf_doc_id = torch.full((P, Cc), EMPTY_DOC_SENTINEL, device=device, dtype=torch.int32)
            buf_group = torch.full((P, Cc), -1, device=device, dtype=torch.int32)
            buf_step = torch.arange(Cc, device=device, dtype=torch.int64)
            eye_c = torch.eye(Cc, device=device, dtype=torch.float32)
            eye_c_bool = eye_c > 0.5

            orig_base_doc = base_doc_id
            base_active = torch.ones((P,), device=device, dtype=torch.bool)
            cur_doc_group = torch.zeros((P,), device=device, dtype=torch.int32)

            chunk_state_view = HippocampusChunkState(
                m_c_base=m_c_base,
                m_l_base=m_l_base,
                u_lex=u_lex,
                u_sem=u_sem,
                base_doc_id=orig_base_doc,
                elig_idx=elig_idx,
                elig_val=elig_val,
                elig_f=elig_f,
                elig_eta=elig_eta,
            )

            chunk_evict_targets: list[torch.Tensor] = []

            for s_loc in range(Cc):
                s_abs = s0 + s_loc
                doc_s = seg_doc_id[:, s_abs]
                rst_s = eff_reset_all[:, s_abs]
                val_s = seg_valid[:, s_abs]

                # 1. Lazy doc-boundary masking/reset on seg_reset (zero cross-doc bleed!)
                base_active = base_active & (~rst_s) & (doc_s == orig_base_doc)
                cur_doc_group = cur_doc_group + rst_s.to(torch.int32)

                rst_1 = rst_s.unsqueeze(-1)
                rst_2 = rst_s.view(P, 1, 1)
                u_lex = torch.where(rst_1, torch.zeros_like(u_lex), u_lex)
                elig_idx = torch.where(rst_2, torch.zeros_like(elig_idx), elig_idx)
                elig_val = torch.where(rst_2, torch.zeros_like(elig_val), elig_val)
                elig_f = torch.where(rst_2, torch.zeros_like(elig_f), elig_f)
                elig_eta = torch.where(rst_1, torch.zeros_like(elig_eta), elig_eta)

                intra_buf = IntraChunkWriteBuffer(
                    k_idx=buf_k_idx,
                    k_val=buf_k_val,
                    delta=buf_delta,
                    write_doc_id=buf_doc_id,
                    write_step=buf_step,
                    write_group=buf_group,
                    beta=buf_beta,
                )

                # 2. HIPR comprehension read using m_c_base + strictly prior (< s_loc) same-doc writes
                psi_i_s = psi_idx_all[:, s_abs, :]
                psi_v_s = psi_val_all[:, s_abs, :]
                read_out = self.read_comprehension_from_state(
                    psi_idx=psi_i_s,
                    psi_val=psi_v_s,
                    c_slow=x3[:, s_abs, :],
                    sem_norm=sem_norm[:, s_abs],
                    mean_info=mean_info[:, s_abs],
                    cur_doc_id=doc_s,
                    cur_step=s_loc,
                    chunk_state=chunk_state_view,
                    intra_buf=intra_buf,
                    base_active_mask=base_active,
                    cur_doc_group=cur_doc_group,
                    u_lex_override=u_lex,
                    pi_cort_override=pi_cort_all[:, s_abs, :],
                    gamma_slow_override=gamma_slow_all[:, s_abs],
                )

                # 3. Advance Carrier G2 one step with fused concept (strictly causal G2_prior!)
                cur_g2, prior_g2, ev_w, ev_t, slot_info = carrier_step_fn(
                    cur_g2,
                    read_out.c_fused,
                    rst_s,
                    val_s,
                    seg_pin_trig[:, s_abs],
                    seg_unpin_trig[:, s_abs],
                )

                # 4. Write-Before-Merge: 3-factor crystallization & exact K-slot WY Gated-Delta
                with fp32_island(device):
                    _, beta_s = self._compute_three_factor_beta(
                        elig_eta=elig_eta,
                        evict_weight=ev_w,
                        evict_target=ev_t,
                        edges=prior_g2.edges,
                        valid=val_s,
                        write_threshold=w_thresh,
                    )

                    old_c, e_dense = self._read_c_from_base_and_buf_multi(
                        e_idx=elig_idx,
                        e_val=elig_val,
                        m_c_base=m_c_base,
                        base_match=base_active,
                        cur_doc_id=doc_s,
                        cur_step=s_loc,
                        intra_buf=intra_buf,
                        cur_doc_group=cur_doc_group,
                    )
                    rhs_c = beta_s.unsqueeze(-1) * (ev_t.float() - old_c)  # [P, K, D]
                    gram_c = torch.bmm(e_dense, e_dense.transpose(1, 2))   # [P, K, K]
                    l_c = torch.tril(beta_s.unsqueeze(-1) * gram_c, diagonal=-1)
                    delta_c = torch.linalg.solve_triangular(
                        eye_k + l_c, rhs_c, upper=False, unitriangular=True
                    )  # [P, K, D]

                    # Functional out-of-place update of fixed-shape intra-chunk buffer
                    step_b4 = eye_c_bool[s_loc].view(1, Cc, 1, 1)
                    step_b2 = eye_c_bool[s_loc].view(1, Cc)
                    step_f4 = eye_c[s_loc].view(1, Cc, 1, 1)
                    step_f3 = eye_c[s_loc].view(1, Cc, 1)

                    buf_k_idx = torch.where(step_b4, elig_idx.unsqueeze(1), buf_k_idx)
                    buf_k_val = torch.where(step_b4, elig_val.unsqueeze(1), buf_k_val)
                    buf_delta = buf_delta + step_f4 * delta_c.unsqueeze(1)
                    buf_beta = buf_beta + step_f3 * beta_s.unsqueeze(1)
                    buf_doc_id = torch.where(step_b2, doc_s.unsqueeze(1), buf_doc_id)
                    buf_group = torch.where(step_b2, cur_doc_group.unsqueeze(1), buf_group)

                    # Snapshot pre-update elig_f for expression write events BEFORE eligibility update
                    tgt_f_s = elig_f

                    # Update u_lex (exact commutative log-retention + end-of-step gamma_hip decay)
                    u_lex = self._update_familiarity_step(
                        u=u_lex,
                        key_idx=elig_idx,
                        key_val=elig_val,
                        beta=beta_s,
                        valid=val_s,
                        gamma_hip=gamma,
                    )

                    # Update slot eligibility traces after Write-Before-Merge
                    elig_idx, elig_val, elig_f, elig_eta = self._update_eligibility_after_write(
                        elig_idx=elig_idx,
                        elig_val=elig_val,
                        elig_f=elig_f,
                        elig_eta=elig_eta,
                        beta=beta_s,
                        evict_weight=ev_w,
                        psi_idx=psi_i_s,
                        psi_val=psi_v_s,
                        f_cur=f_stem[:, s_abs, :],
                        ach=read_out.ach_signal,
                        valid=val_s,
                        next_slots=cur_g2.slots,
                        x_in=read_out.c_fused,
                        slot_assign_or_merge_info=slot_info,
                    )

                all_prior_slots.append(prior_g2.slots)
                all_prior_edges.append(prior_g2.edges)
                all_prior_occ.append(prior_g2.occ)
                all_evict_w.append(ev_w.float())
                all_evict_t.append(ev_t)
                all_c_fused.append(read_out.c_fused)
                all_prec_bu.append(read_out.prec_bu)
                all_c_hip.append(read_out.c_hip)
                all_ach.append(read_out.ach_signal)
                all_gamma_hip.append(read_out.gamma_hip)
                chunk_evict_targets.append(ev_t)
                all_target_f.append(tgt_f_s)
                all_beta.append(beta_s)

            # Per-chunk batched semantic DG for expression write keys (peak VRAM <= 1 chunk)
            chunk_ev_t_stacked = torch.stack(chunk_evict_targets, dim=1)  # [P, Cc, K, D]
            phi_i_c, phi_v_c = self.dg.dg_sem(chunk_ev_t_stacked)         # [P, Cc, K, s]
            all_phi_idx.append(phi_i_c)
            all_phi_val.append(phi_v_c)

            # Chunk-end fold into m_c_base ONLY for writes matching the lane's ending doc & group
            with fp32_island(device):
                M = Cc * K
                last_doc = seg_doc_id[:, s1 - 1]
                last_grp = cur_doc_group
                base_survives = base_active & (last_doc == orig_base_doc)
                m_c_carry = m_c_base * (base_survives.to(torch.float32) * (gamma ** Cc)).view(P, 1, 1)

                k_dense_c = torch.zeros((P, M, d_h), device=device, dtype=torch.float32).scatter_add(
                    2, buf_k_idx.reshape(P, M, s_act), buf_k_val.reshape(P, M, s_act)
                )
                doc_m = buf_doc_id.unsqueeze(2).expand(P, Cc, K).reshape(P, M)
                grp_m = buf_group.unsqueeze(2).expand(P, Cc, K).reshape(P, M)
                step_m = buf_step.unsqueeze(1).expand(Cc, K).reshape(M)

                fold_mask = ((doc_m == last_doc.unsqueeze(1)) & (grp_m == last_grp.unsqueeze(1))).to(torch.float32)
                fold_decay = torch.pow(gamma, (Cc - step_m).to(torch.float32))
                fold_deltas = buf_delta.reshape(P, M, D) * (fold_mask * fold_decay.unsqueeze(0)).unsqueeze(-1)
                m_c_folded = m_c_carry + torch.bmm(k_dense_c.transpose(1, 2), fold_deltas)
                m_c_base = m_c_folded.detach() if detach_chunk_base else m_c_folded
                base_doc_id = last_doc

        g2_scan = CarrierScanOut(
            slots_seq=torch.stack(all_prior_slots, dim=1),
            edges_seq=torch.stack(all_prior_edges, dim=1),
            occ_seq=torch.stack(all_prior_occ, dim=1),
            final_state=cur_g2,
            evict_weight=torch.stack(all_evict_w, dim=1),
            evict_target=torch.stack(all_evict_t, dim=1),
        )
        beta_seq = torch.stack(all_beta, dim=1)  # [P, S, K]
        write_events = HippocampusWriteEvents(
            phi_idx=torch.cat(all_phi_idx, dim=1),
            phi_val=torch.cat(all_phi_val, dim=1),
            target_f=torch.stack(all_target_f, dim=1),
            beta=beta_seq,
            doc_id=seg_doc_id,
            seg_reset=eff_reset_all,
            seg_valid=seg_valid,
        )
        ach_seq = torch.stack(all_ach, dim=1)
        valid_f = seg_valid.to(torch.float32)
        valid_denom = valid_f.sum().clamp_min(1.0)
        mean_ach = (ach_seq * valid_f).sum() / valid_denom
        mean_beta = (beta_seq * valid_f.unsqueeze(-1)).sum() / (valid_denom * float(K))
        max_beta_seq = beta_seq.amax(dim=-1) * valid_f

        final_hip_state = HippocampusChunkState(
            m_c_base=m_c_base,
            m_l_base=m_l_base,
            u_lex=u_lex,
            u_sem=u_sem,
            base_doc_id=base_doc_id,
            elig_idx=elig_idx,
            elig_val=elig_val,
            elig_f=elig_f,
            elig_eta=elig_eta,
        )

        return HippocampusInterleavedOut(
            g2_scan=g2_scan,
            write_events=write_events,
            final_hip_state=final_hip_state,
            c_fused_seq=torch.stack(all_c_fused, dim=1),
            prec_bu_seq=torch.stack(all_prec_bu, dim=1),
            c_hip_seq=torch.stack(all_c_hip, dim=1),
            ach_seq=ach_seq,
            gamma_hip_seq=torch.stack(all_gamma_hip, dim=1),
            l_dg=l_dg,
            mean_ach=mean_ach,
            mean_beta=mean_beta,
            max_beta_seq=max_beta_seq,
        )

    def read_expression_chunk_triangular(
        self,
        q_concept: torch.Tensor,
        seg_doc_id: torch.Tensor,
        seg_valid: torch.Tensor,
        write_events: HippocampusWriteEvents,
        chunk_state_init: Optional[HippocampusChunkState] = None,
        *,
        seg_reset: Optional[torch.Tensor] = None,
        hip_enabled: bool = True,
        detach_chunk_base: bool = True,
        update_homeostasis: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Doc-isolated chunk-level unit-lower-triangular WY/Gated-Delta solver for `M^{L<-C}`.

        Solves `(I + L) Delta = B` in fp32 via `torch.linalg.solve_triangular(..., upper=False, unitriangular=True)`
        per chunk and reads out strictly causal `(f_hip [P, S, d_s], rho_out [P, S])` in parallel.
        Mathematically bit-equivalent (`< 1e-5`) to isolated per-document sequential Gated-Delta fast weights.
        """
        P, S, _ = q_concept.shape
        orig_dtype = q_concept.dtype
        device = q_concept.device
        d_s = self.cfg.d_s
        d_h = self.cfg.d_h
        K = self.cfg.k_slots
        s_act = self.cfg.s_active
        C = int(self.cfg.hip_chunk_size)
        gamma = float(self.cfg.gamma_hip)

        if not hip_enabled:
            return (
                torch.zeros((P, S, d_s), device=device, dtype=orig_dtype),
                torch.zeros((P, S), device=device, dtype=torch.float32),
            )

        state_init = self.init_chunk_state(P, device=device) if chunk_state_init is None else chunk_state_init
        m_l_base = state_init.m_l_base.float()
        u_sem = state_init.u_sem.float()
        base_doc_id = state_init.base_doc_id

        doc_changed = torch.cat(
            [
                torch.zeros((P, 1), device=device, dtype=torch.bool),
                seg_doc_id[:, 1:] != seg_doc_id[:, :-1],
            ],
            dim=1,
        )
        reset_eff = (write_events.seg_reset | doc_changed) if seg_reset is None else (seg_reset | doc_changed)

        f_hip_chunks: list[torch.Tensor] = []
        rho_out_chunks: list[torch.Tensor] = []

        with fp32_island(device):
            for s0 in range(0, S, C):
                s1 = min(s0 + C, S)
                Cc = s1 - s0
                M = Cc * K

                q_c = q_concept[:, s0:s1, :]
                doc_c = seg_doc_id[:, s0:s1]
                val_c = seg_valid[:, s0:s1]
                rst_c = reset_eff[:, s0:s1]
                phi_idx_c = write_events.phi_idx[:, s0:s1]
                phi_val_c = write_events.phi_val[:, s0:s1]
                tgt_f_c = write_events.target_f[:, s0:s1]
                beta_c = write_events.beta[:, s0:s1]

                q_idx_c, q_val_c = self.dg.dg_sem(q_c)
                if update_homeostasis and self.training:
                    self.dg.update_homeostasis(q_idx_c, valid_mask=val_c, is_sem=True)

                doc_group = torch.cumsum(rst_c.to(torch.int32), dim=1)  # [P, Cc]
                use_base = (doc_group == 0) & (doc_c == base_doc_id.unsqueeze(1))  # [P, Cc]

                k_dense = torch.zeros((P, M, d_h), device=device, dtype=torch.float32).scatter_add(
                    2, phi_idx_c.reshape(P, M, s_act), phi_val_c.reshape(P, M, s_act).float()
                )
                q_dense = torch.zeros((P, Cc, d_h), device=device, dtype=torch.float32).scatter_add(
                    2, q_idx_c, q_val_c.float()
                )

                step_idx_c = torch.arange(Cc, device=device, dtype=torch.int64)
                write_step = step_idx_c.unsqueeze(1).expand(Cc, K).reshape(M)
                base_decay_c = torch.pow(gamma, step_idx_c.to(torch.float32))
                base_decay_m = torch.pow(gamma, write_step.to(torch.float32))

                use_base_m = use_base.unsqueeze(2).expand(P, Cc, K).reshape(P, M).to(torch.float32)
                old_base_m = torch.bmm(k_dense, m_l_base) * (use_base_m * base_decay_m.unsqueeze(0)).unsqueeze(-1)

                beta_m = (beta_c.float() * val_c.to(torch.float32).unsqueeze(-1)).reshape(P, M)
                tgt_m = tgt_f_c.float().reshape(P, M, d_s)
                rhs_m = beta_m.unsqueeze(-1) * (tgt_m - old_base_m)

                # Doc-isolated strictly lower-triangular Gram system (I + L) Delta = B
                gram_kk = torch.bmm(k_dense, k_dense.transpose(1, 2))  # [P, M, M]
                doc_m = doc_c.unsqueeze(2).expand(P, Cc, K).reshape(P, M)
                grp_m = doc_group.unsqueeze(2).expand(P, Cc, K).reshape(P, M)
                same_doc_mm = (doc_m.unsqueeze(2) == doc_m.unsqueeze(1)) & (grp_m.unsqueeze(2) == grp_m.unsqueeze(1))
                step_diff_mm = (write_step.unsqueeze(1) - write_step.unsqueeze(0)).clamp_min(0).to(torch.float32)
                decay_mm = torch.pow(gamma, step_diff_mm)
                strict_lower = torch.tril(torch.ones((M, M), device=device, dtype=torch.bool), diagonal=-1)

                l_mat = (
                    beta_m.unsqueeze(2)
                    * gram_kk
                    * (same_doc_mm & strict_lower.unsqueeze(0)).to(torch.float32)
                    * decay_mm.unsqueeze(0)
                )
                eye_m = torch.eye(M, device=device, dtype=torch.float32).unsqueeze(0).expand(P, -1, -1)
                deltas_m = torch.linalg.solve_triangular(eye_m + l_mat, rhs_m, upper=False, unitriangular=True)

                # Parallel strictly causal readout (queries see writes with write_step < s_query in same doc)
                f_base_c = torch.bmm(q_dense, m_l_base) * (
                    use_base.to(torch.float32) * base_decay_c.unsqueeze(0)
                ).unsqueeze(-1)
                gram_qk = torch.bmm(q_dense, k_dense.transpose(1, 2))  # [P, Cc, M]
                same_doc_qm = (doc_c.unsqueeze(2) == doc_m.unsqueeze(1)) & (
                    doc_group.unsqueeze(2) == grp_m.unsqueeze(1)
                )
                strict_prior_qm = write_step.unsqueeze(0) < step_idx_c.unsqueeze(1)  # [Cc, M]
                step_diff_qm = (step_idx_c.unsqueeze(1) - write_step.unsqueeze(0)).clamp_min(0).to(torch.float32)
                decay_qm = torch.pow(gamma, step_diff_qm)
                w_qk = (
                    gram_qk
                    * (same_doc_qm & strict_prior_qm.unsqueeze(0)).to(torch.float32)
                    * decay_qm.unsqueeze(0)
                )
                f_recent_c = torch.bmm(w_qk, deltas_m)
                f_hip_c = (f_base_c + f_recent_c) * val_c.to(torch.float32).unsqueeze(-1)
                f_hip_chunks.append(f_hip_c.to(orig_dtype))

                # Exact doc-isolated semantic familiarity trace u_sem & rho_out gate
                gamma_sem_steps: list[torch.Tensor] = []
                for s_loc in range(Cc):
                    need_zero = (rst_c[:, 0] | (doc_c[:, 0] != base_doc_id)) if s_loc == 0 else rst_c[:, s_loc]
                    u_sem = torch.where(need_zero.unsqueeze(-1), torch.zeros_like(u_sem), u_sem)
                    u_g = u_sem.gather(1, q_idx_c[:, s_loc])
                    g_sem_s = (q_val_c[:, s_loc].float().pow(2) * u_g).sum(dim=-1).clamp(0.0, 1.0)
                    gamma_sem_steps.append(g_sem_s)
                    u_sem = self._update_familiarity_step(
                        u=u_sem,
                        key_idx=phi_idx_c[:, s_loc],
                        key_val=phi_val_c[:, s_loc],
                        beta=beta_c[:, s_loc],
                        valid=val_c[:, s_loc],
                        gamma_hip=gamma,
                    )
                gamma_sem_c = torch.stack(gamma_sem_steps, dim=1)
                rho_out_c = torch.sigmoid(self.alpha_out * (gamma_sem_c - self.theta_out)) * val_c.to(torch.float32)
                rho_out_chunks.append(rho_out_c)

                # Chunk-end fold into m_l_base ONLY for writes matching ending doc & group
                last_doc = doc_c[:, Cc - 1]
                last_grp = doc_group[:, Cc - 1]
                base_survives = (last_grp == 0) & (last_doc == base_doc_id)
                m_l_carry = m_l_base * (base_survives.to(torch.float32) * (gamma ** Cc)).view(P, 1, 1)

                fold_mask = ((doc_m == last_doc.unsqueeze(1)) & (grp_m == last_grp.unsqueeze(1))).to(torch.float32)
                fold_decay = torch.pow(gamma, (Cc - write_step).to(torch.float32))
                fold_deltas = deltas_m * (fold_mask * fold_decay.unsqueeze(0)).unsqueeze(-1)
                m_l_folded = m_l_carry + torch.bmm(k_dense.transpose(1, 2), fold_deltas)
                m_l_base = m_l_folded.detach() if detach_chunk_base else m_l_folded
                base_doc_id = last_doc

        return torch.cat(f_hip_chunks, dim=1), torch.cat(rho_out_chunks, dim=1)

    def step_streaming(
        self,
        x3: torch.Tensor,
        f_stem: torch.Tensor,
        prev_stem: Optional[torch.Tensor],
        sem_norm: torch.Tensor,
        mean_info: torch.Tensor,
        doc_id: torch.Tensor,
        reset: torch.Tensor,
        valid: torch.Tensor,
        pin_trig: torch.Tensor,
        unpin_trig: torch.Tensor,
        g2_state: CarrierGraphState,
        hip_state: HippocampusChunkState,
        carrier_step_fn: CarrierStepFn,
        *,
        q_concept: Optional[torch.Tensor] = None,
        hip_enabled: bool = True,
        write_threshold: Optional[float] = None,
    ) -> HippocampusStreamStepOut:
        """Single-step exact sequential Gated-Delta streaming inference (Gate 6 parity)."""
        P, D = x3.shape
        d_s = self.cfg.d_s
        K = self.cfg.k_slots
        s_act = self.cfg.s_active
        gamma = float(self.cfg.gamma_hip)
        w_thresh = float(self.cfg.hip_write_threshold if write_threshold is None else write_threshold)
        device = x3.device

        do_reset = reset | (doc_id != hip_state.base_doc_id)
        rst_1 = do_reset.unsqueeze(-1)
        rst_2 = do_reset.view(P, 1, 1)

        m_c = torch.where(rst_2, torch.zeros_like(hip_state.m_c_base), hip_state.m_c_base.float())
        m_l = torch.where(rst_2, torch.zeros_like(hip_state.m_l_base), hip_state.m_l_base.float())
        u_lex = torch.where(rst_1, torch.zeros_like(hip_state.u_lex), hip_state.u_lex.float())
        u_sem = torch.where(rst_1, torch.zeros_like(hip_state.u_sem), hip_state.u_sem.float())
        elig_idx = torch.where(rst_2, torch.zeros_like(hip_state.elig_idx), hip_state.elig_idx)
        elig_val = torch.where(rst_2, torch.zeros_like(hip_state.elig_val), hip_state.elig_val.float())
        elig_f = torch.where(rst_2, torch.zeros_like(hip_state.elig_f), hip_state.elig_f.float())
        elig_eta = torch.where(rst_1, torch.zeros_like(hip_state.elig_eta), hip_state.elig_eta.float())

        psi_idx, psi_val = self.dg.dg_lex(f_stem, f_prev=prev_stem, seg_reset=do_reset)

        if not hip_enabled:
            with fp32_island(device):
                pi_cort = self._compute_pi_cort(x3)
                gamma_slow = self._compute_gamma_slow(sem_norm, mean_info)
            next_g2, prior_g2, _, _, _ = carrier_step_fn(g2_state, x3, do_reset, valid, pin_trig, unpin_trig)
            read_out = HippocampusFlatReadOut(
                c_fused=x3,
                prec_bu=pi_cort,
                psi_idx=psi_idx,
                psi_val=psi_val,
                ach_signal=torch.zeros((P,), device=device, dtype=torch.float32),
                gamma_slow=gamma_slow,
                gamma_hip=torch.zeros((P,), device=device, dtype=torch.float32),
                c_hip=torch.zeros((P, D), device=device, dtype=torch.float32),
            )
            f_hip_out = torch.zeros((P, d_s), device=device, dtype=q_concept.dtype) if q_concept is not None else None
            rho_out = torch.zeros((P,), device=device, dtype=torch.float32) if q_concept is not None else None
            prev_base = torch.zeros_like(f_stem) if prev_stem is None else prev_stem
            return HippocampusStreamStepOut(
                next_g2_state=next_g2,
                prior_g2_state=prior_g2,
                next_hip_state=hip_state,
                next_prev_stem=torch.where(valid.unsqueeze(-1), f_stem, prev_base),
                read_out=read_out,
                f_hip=f_hip_out,
                rho_out=rho_out,
                beta=torch.zeros((P, K), device=device, dtype=torch.float32),
                max_beta=torch.zeros((P,), device=device, dtype=torch.float32),
            )

        with fp32_island(device):
            # 1. Optional strictly causal expression readout BEFORE this step's writes
            f_hip_out: Optional[torch.Tensor] = None
            rho_out: Optional[torch.Tensor] = None
            if q_concept is not None:
                q_idx, q_val = self.dg.dg_sem(q_concept)
                cols_l = m_l.gather(1, q_idx.unsqueeze(-1).expand(P, s_act, d_s))
                f_hip_raw = torch.einsum("ps,psd->pd", q_val.float(), cols_l) * valid.to(torch.float32).unsqueeze(-1)
                f_hip_out = f_hip_raw.to(q_concept.dtype)
                g_sem = (q_val.float().pow(2) * u_sem.gather(1, q_idx)).sum(dim=-1).clamp(0.0, 1.0)
                rho_out = torch.sigmoid(self.alpha_out * (g_sem - self.theta_out)) * valid.to(torch.float32)

            # 2. Comprehension read HIPR strictly BEFORE this step's writes
            cur_state = HippocampusChunkState(
                m_c_base=m_c,
                m_l_base=m_l,
                u_lex=u_lex,
                u_sem=u_sem,
                base_doc_id=doc_id,
                elig_idx=elig_idx,
                elig_val=elig_val,
                elig_f=elig_f,
                elig_eta=elig_eta,
            )
            read_out = self.read_comprehension_from_state(
                psi_idx=psi_idx,
                psi_val=psi_val,
                c_slow=x3,
                sem_norm=sem_norm,
                mean_info=mean_info,
                cur_doc_id=doc_id,
                cur_step=0,
                chunk_state=cur_state,
                intra_buf=None,
                u_lex_override=u_lex,
            )

        # 3. Carrier G2 single step
        next_g2, prior_g2, ev_w, ev_t, slot_info = carrier_step_fn(
            g2_state,
            read_out.c_fused,
            do_reset,
            valid,
            pin_trig,
            unpin_trig,
        )

        # 4. Exact sequential K-slot Gated-Delta Write-Before-Merge + end-of-step decay
        with fp32_island(device):
            _, beta = self._compute_three_factor_beta(
                elig_eta=elig_eta,
                evict_weight=ev_w,
                evict_target=ev_t,
                edges=prior_g2.edges,
                valid=valid,
                write_threshold=w_thresh,
            )
            phi_idx, phi_val = self.dg.dg_sem(ev_t)  # [P, K, s]

            for k in range(K):
                bk = beta[:, k : k + 1]  # [P, 1]

                # Comprehension M^{C<-L}
                ek_idx = elig_idx[:, k, :]
                ek_val = elig_val[:, k, :]
                cols_c = m_c.gather(1, ek_idx.unsqueeze(-1).expand(P, s_act, D))
                old_c = torch.einsum("ps,psd->pd", ek_val, cols_c)
                dc = bk * (ev_t[:, k, :].float() - old_c)
                m_c = m_c.scatter_add(
                    1,
                    ek_idx.unsqueeze(-1).expand(P, s_act, D),
                    ek_val.unsqueeze(-1) * dc.unsqueeze(1),
                )

                # Expression M^{L<-C}
                pk_idx = phi_idx[:, k, :]
                pk_val = phi_val[:, k, :]
                cols_l = m_l.gather(1, pk_idx.unsqueeze(-1).expand(P, s_act, d_s))
                old_l = torch.einsum("ps,psd->pd", pk_val, cols_l)
                dl = bk * (elig_f[:, k, :].float() - old_l)
                m_l = m_l.scatter_add(
                    1,
                    pk_idx.unsqueeze(-1).expand(P, s_act, d_s),
                    pk_val.unsqueeze(-1) * dl.unsqueeze(1),
                )

            u_lex = self._update_familiarity_step(
                u=u_lex, key_idx=elig_idx, key_val=elig_val, beta=beta, valid=valid, gamma_hip=gamma
            )
            u_sem = self._update_familiarity_step(
                u=u_sem, key_idx=phi_idx, key_val=phi_val, beta=beta, valid=valid, gamma_hip=gamma
            )

            valid_2 = valid.view(P, 1, 1)
            m_c = torch.where(valid_2, m_c * gamma, m_c)
            m_l = torch.where(valid_2, m_l * gamma, m_l)

            elig_idx, elig_val, elig_f, elig_eta = self._update_eligibility_after_write(
                elig_idx=elig_idx,
                elig_val=elig_val,
                elig_f=elig_f,
                elig_eta=elig_eta,
                beta=beta,
                evict_weight=ev_w,
                psi_idx=psi_idx,
                psi_val=psi_val,
                f_cur=f_stem,
                ach=read_out.ach_signal,
                valid=valid,
                next_slots=next_g2.slots,
                x_in=read_out.c_fused,
                slot_assign_or_merge_info=slot_info,
            )

        next_hip = HippocampusChunkState(
            m_c_base=m_c,
            m_l_base=m_l,
            u_lex=u_lex,
            u_sem=u_sem,
            base_doc_id=doc_id,
            elig_idx=elig_idx,
            elig_val=elig_val,
            elig_f=elig_f,
            elig_eta=elig_eta,
        )
        prev_base = torch.zeros_like(f_stem) if prev_stem is None else prev_stem
        next_prev = torch.where(valid.unsqueeze(-1), f_stem, prev_base)
        return HippocampusStreamStepOut(
            next_g2_state=next_g2,
            prior_g2_state=prior_g2,
            next_hip_state=next_hip,
            next_prev_stem=next_prev,
            read_out=read_out,
            f_hip=f_hip_out,
            rho_out=rho_out,
            beta=beta,
            max_beta=beta.amax(dim=-1) * valid.to(torch.float32),
        )


__all__ = [
    "EMPTY_DOC_SENTINEL",
    "CarrierStepFn",
    "deterministic_orthogonal_projection",
    "deterministic_orthogonal_rotation",
    "IntraChunkWriteBuffer",
    "HippocampusFlatReadOut",
    "HippocampusWriteEvents",
    "HippocampusInterleavedOut",
    "HippocampusStreamStepOut",
    "DualMorphemeWhitenedDG",
    "HippocampusFlat",
]
