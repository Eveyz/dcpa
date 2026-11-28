"""WS-3A: True-Contractive Closed-Unit-Ball Concept Graph Carrier (DCPA-Flat v3.3).

Implements `ContractiveGraphCarrier` for both dual concept-graph carriers:
  - `G1` (`pin_enabled=False, return_prior=False`): free-flowing post-update syntax/entity graph.
  - `G2` (`pin_enabled=True, return_prior=True`): strictly causal pre-update discourse/proposition
    graph with Slot-0 Thesis Pinning (R3) and Write-Before-Merge eviction signaling for WS-4
    Hippocampus (`evict_weight`, `evict_target`).

Hard architectural invariants enforced across every step:
  1. Closed unit ball slots `||S_k||_2 <= 1.0` with ZERO RMSNorm on the recurrent slot backbone.
  2. Bounded directed edges `A in (-1, 1)^{K x K x d_r}` with explicit per-step spectral decay
     `gamma_A = cfg.edge_decay <= 0.99` (R3).
  3. Bounded float32 occupancy `rho in [occ_min, occ_max]` (`[0.05, 0.95]`) with R3 relaxed
     fullness temperature `full_gate_temp = 0.10` and Probabilistic-OR merge cap `0.90`.
  4. Configurable gradient guard `carrier_grad_guard in {"none", "router_sg", "soft_clip"}` (R3).
  5. Static `c_scan`-chunked execution (`forward_chunk`) passing
     `torch.compile(..., fullgraph=True, dynamic=False)` with zero `.item()`, `.cpu()`, `.tolist()`,
     `torch.nonzero`, `repeat_interleave`, dynamic boolean slicing, or step-growing Python lists.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from dcpa.interfaces import (
    CarrierGradGuardMode,
    CarrierGraphState,
    CarrierScanOut,
    DCPAFlatConfig,
)
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.rmsnorm import RMSNorm


def _project_closed_unit_ball(v: torch.Tensor, max_norm: float = 0.995, eps: float = 1e-6) -> torch.Tensor:
    """Firm-nonexpansive Euclidean projection onto closed ball `B(0, max_norm)` (1-Lipschitz, NOT RMSNorm).

    - Inside `||v||_2 <= max_norm`: exact identity (`df/dv = I`, zero small-vector amplification).
    - Outside `||v||_2 > max_norm`: projects radially to `max_norm` (`||df/dv||_2 <= 1.0`).
    """
    orig_dtype = v.dtype
    with fp32_island(v.device):
        vf = v.float()
        norm = torch.sqrt(vf.pow(2).sum(dim=-1, keepdim=True) + eps)
        scale = float(max_norm) / torch.clamp(norm, min=float(max_norm))
        out = vf * scale
    return out.to(orig_dtype)


class _SoftClipCarryGrad(torch.autograd.Function):
    """Exact bit-identical forward carry with smooth per-lane backward RMS soft-clipping.

    Forward: exact identity (`0.0` ULP drift, preserves Slot-0 pinning & Gate 6 parity).
    Backward: smooth 1-Lipschitz norm bound `g / sqrt(1 + (rms / max_rms)^2)` when `rms` grows large.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, max_rms: float) -> torch.Tensor:
        ctx.max_rms = float(max_rms)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        g = grad_out.float()
        if g.ndim > 1:
            reduce_dims = tuple(range(1, g.ndim))
            g_max = g.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
            g_scaled = g / g_max
            rms = g_max * torch.sqrt(g_scaled.pow(2).mean(dim=reduce_dims, keepdim=True) + 1e-12)
        else:
            rms = g.abs().clamp_min(1e-12)
        ratio = rms / max(ctx.max_rms, 1e-6)
        scale = torch.rsqrt(1.0 + ratio * ratio)
        return (g * scale).to(grad_out.dtype), None


def _build_ordered_pairs(
    k_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-build static constant tables for all `K*(K-1)` (= 56 for K=8) ordered pairs `(i, j)` (`i != j`).

    Returns:
      - `pair_i`, `pair_j`: `[M_pairs]` int64 primary/secondary slot indices
      - `pair_u`, `pair_v`: `[M_pairs]` int64 `min(i, j)` (merged target) and `max(i, j)` (freed target)
      - `pair_uses_slot0`: `[M_pairs]` bool (True for the `2*(K-1) = 14` pairs involving slot 0)
      - `u_oh`, `v_oh`: `[M_pairs, K]` float32 one-hot scatter bases for `(u, v)`
    """
    pairs: list[tuple[int, int]] = []
    for i in range(k_slots):
        for j in range(k_slots):
            if i != j:
                pairs.append((i, j))

    m_pairs = len(pairs)
    pair_i = torch.empty(m_pairs, dtype=torch.int64)
    pair_j = torch.empty(m_pairs, dtype=torch.int64)
    pair_u = torch.empty(m_pairs, dtype=torch.int64)
    pair_v = torch.empty(m_pairs, dtype=torch.int64)
    pair_uses_slot0 = torch.zeros(m_pairs, dtype=torch.bool)
    u_oh = torch.zeros((m_pairs, k_slots), dtype=torch.float32)
    v_oh = torch.zeros((m_pairs, k_slots), dtype=torch.float32)

    for p, (i, j) in enumerate(pairs):
        u = min(i, j)
        v = max(i, j)
        pair_i[p] = i
        pair_j[p] = j
        pair_u[p] = u
        pair_v[p] = v
        pair_uses_slot0[p] = (i == 0) or (j == 0)
        u_oh[p, u] = 1.0
        v_oh[p, v] = 1.0

    return pair_i, pair_j, pair_u, pair_v, pair_uses_slot0, u_oh, v_oh


class ContractiveGraphCarrier(nn.Module):
    """Two-stage branchless contractive concept-graph carrier (`G1` or `G2`) on the closed unit ball."""

    def __init__(
        self,
        cfg: DCPAFlatConfig,
        pin_enabled: bool = False,
        return_prior: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.pin_enabled = bool(pin_enabled)
        self.return_prior = bool(return_prior)
        self.k_slots = int(cfg.k_slots)
        self.d_concept = int(cfg.d_concept)
        self.d_r = int(cfg.d_r)
        self.d_merge = int(cfg.d_merge)
        self.c_scan = int(cfg.c_scan)
        self.occ_min = float(cfg.occ_min)
        self.occ_max = float(cfg.occ_max)
        self.occ_relax = float(cfg.occ_relax)
        self.slot_keep_max = float(cfg.slot_keep_max)
        self.full_gate_center = float(cfg.full_gate_center)
        self.full_gate_temp = float(cfg.full_gate_temp)
        self.edge_decay = float(cfg.edge_decay)
        self.grad_guard: CarrierGradGuardMode = cfg.carrier_grad_guard
        self.grad_clip_rms = float(cfg.carrier_grad_clip_rms)
        self.eps = float(cfg.eps)
        # Occupancy assigned to a newly written slot (strictly inside [occ_min, occ_max])
        self.write_occ = float(self.occ_min + 0.80 * (self.occ_max - self.occ_min))
        self.num_pairs = self.k_slots * (self.k_slots - 1)

        K = self.k_slots
        D = self.d_concept
        d_r = self.d_r
        d_m = self.d_merge

        # Learnable document-start initial graph state G_init (projected into closed unit ball)
        self.init_slots_raw = nn.Parameter(torch.empty((K, D)))
        self.init_edges_raw = nn.Parameter(torch.zeros((K, K, d_r)))

        # Read-only input side-path RMSNorm (NEVER applied to recurrent slots backbone!)
        self.norm_in = RMSNorm(D, eps=self.eps)
        self.w_in = nn.Linear(D, D, bias=False)

        # Stage 1 (O(K) Coreference Assimilation) scalar retention & gate parameters
        self.w_lam = nn.Linear(D, K, bias=True)
        self.assim_scale = nn.Parameter(torch.tensor(2.5, dtype=torch.float32))
        self.assim_threshold = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))
        self.lam_sim_gain = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # Low-rank bilinear edge projections (output bounded by tanh in (-1, 1))
        self.w_edge_in_src = nn.Linear(D, d_r, bias=False)
        self.w_edge_in_dst = nn.Linear(D, d_r, bias=False)
        self.w_edge_slot_src = nn.Linear(D, d_r, bias=False)
        self.w_edge_slot_dst = nn.Linear(D, d_r, bias=False)
        self.edge_write_logit = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

        # Stage 2B (56-pair Probabilistic-OR Merge) factorized scorer explicitly taking
        # (s_i, s_j, a_ij, rho_tilde_i, rho_tilde_j) in O(K*D*d_m + K^2*d_m)
        self.w_pair_si = nn.Linear(D, d_m, bias=False)
        self.w_pair_sj = nn.Linear(D, d_m, bias=False)
        self.w_pair_edge = nn.Linear(d_r, d_m, bias=False)
        self.w_pair_rho_i = nn.Parameter(torch.zeros(d_m))
        self.w_pair_rho_j = nn.Parameter(torch.zeros(d_m))
        self.pair_bias = nn.Parameter(torch.zeros(d_m))
        self.w_pair_score = nn.Linear(d_m, 1, bias=True)
        self.pair_cos_gain = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # Bounded convex/orthogonal merge blend & small residual refinement inside unit ball
        self.w_merge_beta = nn.Linear(d_m, 1, bias=True)
        self.w_merge_delta = nn.Linear(d_m, D, bias=False)

        # Static prebuilt constant buffers (zero dynamic allocations in hot paths)
        pair_i, pair_j, pair_u, pair_v, pair_uses_slot0, u_oh, v_oh = _build_ordered_pairs(K)
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)
        self.register_buffer("pair_u", pair_u, persistent=False)
        self.register_buffer("pair_v", pair_v, persistent=False)
        self.register_buffer("pair_uses_slot0", pair_uses_slot0, persistent=False)
        self.register_buffer("u_oh", u_oh, persistent=False)
        self.register_buffer("v_oh", v_oh, persistent=False)
        self.register_buffer("slot_idx", torch.arange(K, dtype=torch.int64), persistent=False)
        off_diag = 1.0 - torch.eye(K, dtype=torch.float32)
        self.register_buffer("off_diag_mask", off_diag.view(1, K, K, 1), persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        K, D = self.k_slots, self.d_concept
        # Orthogonal unit-ball initial slots at norm ~0.35 (strictly in [0.15, 0.50])
        with torch.no_grad():
            q = torch.randn((K, D), dtype=torch.float32)
            q = F.normalize(q, dim=-1) * 0.35
            self.init_slots_raw.copy_(q)
            self.init_edges_raw.zero_()

        nn.init.normal_(self.w_in.weight, std=1.0 / math.sqrt(float(D)))
        nn.init.normal_(self.w_lam.weight, std=0.02)
        nn.init.constant_(self.w_lam.bias, 1.5)  # initial lam ~ 0.82 * slot_keep_max

        edge_std = 0.5 / math.sqrt(float(D))
        nn.init.normal_(self.w_edge_in_src.weight, std=edge_std)
        nn.init.normal_(self.w_edge_in_dst.weight, std=edge_std)
        nn.init.normal_(self.w_edge_slot_src.weight, std=edge_std)
        nn.init.normal_(self.w_edge_slot_dst.weight, std=edge_std)

        # Well-conditioned small pair-scorer init keeps unguarded 512-step Jacobians contractive
        pair_std = 0.35 / math.sqrt(float(D))
        nn.init.normal_(self.w_pair_si.weight, std=pair_std)
        nn.init.normal_(self.w_pair_sj.weight, std=pair_std)
        nn.init.normal_(self.w_pair_edge.weight, std=0.35 / math.sqrt(float(self.d_r)))
        nn.init.normal_(self.w_pair_rho_i, std=0.05)
        nn.init.normal_(self.w_pair_rho_j, std=0.05)
        nn.init.zeros_(self.pair_bias)
        nn.init.normal_(self.w_pair_score.weight, std=0.02)
        nn.init.zeros_(self.w_pair_score.bias)

        nn.init.normal_(self.w_merge_beta.weight, std=0.02)
        nn.init.constant_(self.w_merge_beta.bias, 0.4)  # head-biased initial beta ~ 0.73
        nn.init.normal_(self.w_merge_delta.weight, std=0.01)

    def _get_init_tensors(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        init_slots = _project_closed_unit_ball(self.init_slots_raw, max_norm=0.50, eps=self.eps).to(
            device=device, dtype=dtype
        )
        init_edges = (0.5 * torch.tanh(self.init_edges_raw.float())).to(device=device, dtype=dtype)
        init_edges = init_edges * self.off_diag_mask.squeeze(0).to(device=device, dtype=dtype)
        slots_b = init_slots.unsqueeze(0).expand(batch_size, -1, -1)
        edges_b = init_edges.unsqueeze(0).expand(batch_size, -1, -1, -1)
        occ_b = torch.full((batch_size, self.k_slots), self.occ_min, device=device, dtype=torch.float32)
        pinned_b = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        return slots_b, edges_b, occ_b, pinned_b

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> CarrierGraphState:
        """Create initial document-start graph state `G_init` inside the closed unit ball."""
        dev = self.init_slots_raw.device if device is None else device
        dt = self.init_slots_raw.dtype if dtype is None else dtype
        slots_b, edges_b, occ_b, pinned_b = self._get_init_tensors(batch_size, dev, dt)
        return CarrierGraphState(
            slots=slots_b.clone(),
            edges=edges_b.clone(),
            occ=occ_b,
            slot0_pinned=pinned_b,
        )

    def step_single(
        self,
        state: CarrierGraphState,
        x_in: torch.Tensor,
        seg_reset: torch.Tensor,
        seg_valid: torch.Tensor,
        pin_trig: Optional[torch.Tensor] = None,
        unpin_trig: Optional[torch.Tensor] = None,
        hip_inject: Optional[torch.Tensor] = None,
    ) -> tuple[
        CarrierGraphState,
        CarrierGraphState,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Execute one strictly branchless contractive graph transition step.

        Args:
          state:      Previous `CarrierGraphState` (`[P, K, D]`, `[P, K, K, d_r]`, `[P, K]` fp32, `[P]` bool).
          x_in:       `[P, D]` input concept vector at current segment.
          seg_reset:  `[P]` bool pre-update document reset switch.
          seg_valid:  `[P]` bool valid segment mask (False freezes state).
          pin_trig:   Optional `[P]` bool (`\\x1dE\\n` trigger, active only when `pin_enabled=True`).
          unpin_trig: Optional `[P]` bool (`0x04`/`0x1E` trigger, active only when `pin_enabled=True`).
          hip_inject: Optional `[P, D]` WS-4 hippocampal causal read injection (G2).

        Returns:
          `(next_state, out_state, evict_weight [P, K], evict_target [P, K, D], empty_write [P], has_empty [P])`
          where `out_state` is `prior_state` (post-reset pre-update) when `return_prior=True`, else `next_state`.
        """
        P, D = x_in.shape
        K = self.k_slots
        d_r = self.d_r
        dev = x_in.device
        orig_dtype = x_in.dtype

        # 0. Pre-update document reset to learnable G_init + optional unpin (G2 only)
        init_slots_b, init_edges_b, init_occ_b, init_pinned_b = self._get_init_tensors(P, dev, orig_dtype)
        reset_b = seg_reset.bool()
        valid_b = seg_valid.bool()

        slots_pre = torch.where(reset_b[:, None, None], init_slots_b, state.slots.to(orig_dtype))
        edges_pre = torch.where(reset_b[:, None, None, None], init_edges_b, state.edges.to(orig_dtype))
        occ_pre = torch.where(reset_b[:, None], init_occ_b, state.occ.float())
        if self.pin_enabled:
            pinned_pre = torch.where(reset_b, init_pinned_b, state.slot0_pinned.bool())
            if unpin_trig is not None:
                pinned_pre = pinned_pre & (~unpin_trig.bool())
        else:
            pinned_pre = init_pinned_b

        # Apply optional smooth carry-gradient guard (`soft_clip` mode) on recurrent backbone
        if self.grad_guard == "soft_clip" and self.training:
            slots_pre = _SoftClipCarryGrad.apply(slots_pre, self.grad_clip_rms)
            edges_pre = _SoftClipCarryGrad.apply(edges_pre, self.grad_clip_rms)
            occ_pre = _SoftClipCarryGrad.apply(occ_pre, self.grad_clip_rms)

        # Strictly causal post-reset, pre-update snapshot (recorded as G2_prior!)
        prior_state = CarrierGraphState(
            slots=slots_pre,
            edges=edges_pre,
            occ=occ_pre,
            slot0_pinned=pinned_pre,
        )

        # Optional stop-grad view for routing/merge scorers (`router_sg` mode) while keeping
        # the recurrent state backbone 100% grad-flowing
        if self.grad_guard == "router_sg":
            slots_r = slots_pre.detach()
            edges_r = edges_pre.detach()
            occ_r = occ_pre.detach()
        else:
            slots_r = slots_pre
            edges_r = edges_pre
            occ_r = occ_pre

        # 1. Input projection onto closed unit ball B_D (read-only RMSNorm on input side-path only)
        x_eff = x_in if hip_inject is None else (x_in + hip_inject.to(orig_dtype))
        x_n = self.norm_in(x_eff)
        v_in = self.w_in(x_n)

        # Low-rank bilinear edge vectors for incoming concept & existing slots
        e_in_src = self.w_edge_in_src(x_n)  # [P, d_r]
        e_in_dst = self.w_edge_in_dst(x_n)  # [P, d_r]
        e_slot_src = self.w_edge_slot_src(slots_r)  # [P, K, d_r]
        e_slot_dst = self.w_edge_slot_dst(slots_r)  # [P, K, d_r]

        # Factorized Stage 2B pair scorer hidden activations: O(K*D*d_m + K^2*d_m)
        h_si = self.w_pair_si(slots_r)  # [P, K, d_m]
        h_sj = self.w_pair_sj(slots_r)  # [P, K, d_m]

        # Pre-compute Stage 1 scalar retention logits from input side-path
        lam_in_logits = self.w_lam(x_n)  # [P, K]

        # Free-slot mask: exclude Slot-0 while pinned (G2 only)
        if self.pin_enabled:
            free_mask = (self.slot_idx.unsqueeze(0) != 0) | (~pinned_pre.unsqueeze(1))  # [P, K]
        else:
            free_mask = torch.ones((P, K), device=dev, dtype=torch.bool)

        with fp32_island(dev):
            slots_pre_f = slots_pre.float()
            slots_r_f = slots_r.float()
            edges_dec_f = edges_pre.float() * self.edge_decay  # R3 explicit edge contraction
            edges_dec_r_f = edges_r.float() * self.edge_decay

            # Occupancy relaxation: rho_tilde = occ_min + occ_relax * (rho_pre - occ_min)
            rho_tilde = (self.occ_min + self.occ_relax * (occ_pre.float() - self.occ_min)).clamp(
                self.occ_min, self.occ_max
            )
            rho_tilde_r = (self.occ_min + self.occ_relax * (occ_r.float() - self.occ_min)).clamp(
                self.occ_min, self.occ_max
            )

            # Smooth closed-unit-ball input vector: ||u_in||_2 < 0.96 <= 1.0, smooth Jacobian at 0
            v_f = v_in.float()
            u_in_f = (0.96 * v_f) / torch.sqrt(v_f.pow(2).sum(dim=-1, keepdim=True) + 0.04)  # [P, D]

            # Bounded bilinear candidate directed edges in (-1, 1)
            e_out_cand = torch.tanh(e_in_src.float().unsqueeze(1) * e_slot_dst.float())  # [P, K, d_r]
            e_in_cand = torch.tanh(e_slot_src.float() * e_in_dst.float().unsqueeze(1))   # [P, K, d_r]
            eta_e = 0.25 * torch.sigmoid(self.edge_write_logit.float())

            # ------------------------------------------------------------------
            # Stage 1: O(K) Coreference Assimilation
            # ------------------------------------------------------------------
            sim = torch.einsum("pd,pkd->pk", u_in_f, slots_r_f)  # [P, K] in [-1, 1]
            occ_frac = ((rho_tilde_r - self.occ_min) / max(self.occ_max - self.occ_min, 1e-6)).clamp(0.0, 1.0)
            # Empty slots (rho_tilde == occ_min) have occ_active == 0 -> never hijack direct write
            occ_active = (occ_frac * 4.0).clamp(0.0, 1.0)  # [P, K]
            assim_score = sim * occ_active  # [P, K]
            assim_logits = torch.where(free_mask, assim_score * 2.0, torch.full_like(assim_score, -1.0e9))

            k_star = assim_logits.argmax(dim=-1)  # [P]
            p_assim = F.softmax(assim_logits, dim=-1)  # [P, K]
            oh_assim = F.one_hot(k_star, num_classes=K).float()  # [P, K]
            w_assim = oh_assim + (p_assim - p_assim.detach())  # [P, K] STE

            best_sim = (w_assim * assim_score).sum(dim=-1, keepdim=True)  # [P, 1]
            best_occ_active = (oh_assim * occ_active).sum(dim=-1, keepdim=True)  # [P, 1]
            g_assim = best_occ_active * torch.sigmoid(
                self.assim_scale.float().clamp(0.5, 6.0) * (best_sim - self.assim_threshold.float())
            )  # [P, 1] in [0, 1]

            # Per-slot scalar retention lam_k in (0, slot_keep_max=0.990]
            lam = self.slot_keep_max * torch.sigmoid(
                lam_in_logits.float() + self.lam_sim_gain.float().clamp(-2.0, 2.0) * sim
            )  # [P, K]

            # Convex closed-ball update on k*: ||lam*s + (1-lam)*u|| <= 1.0 guaranteed, zero RMSNorm
            s_assim_all = lam.unsqueeze(-1) * slots_pre_f + (1.0 - lam.unsqueeze(-1)) * u_in_f.unsqueeze(1)
            slots_s1 = slots_pre_f + w_assim.unsqueeze(-1) * (s_assim_all - slots_pre_f)

            rho_boost = (rho_tilde + 0.35 * (self.occ_max - rho_tilde)).clamp(self.occ_min, self.occ_max)
            occ_s1 = rho_tilde + w_assim * (rho_boost - rho_tilde)

            # Bounded convex edge refresh on assimilated slot k*
            off_diag_f = self.off_diag_mask.float()
            w_a_row = (eta_e * w_assim)[:, :, None, None]  # [P, K, 1, 1]
            w_a_col = (eta_e * w_assim)[:, None, :, None]  # [P, 1, K, 1]
            edges_s1 = (1.0 - w_a_row) * edges_dec_f + w_a_row * e_out_cand.unsqueeze(1)
            edges_s1 = (1.0 - w_a_col) * edges_s1 + w_a_col * e_in_cand.unsqueeze(2)
            edges_s1 = edges_s1 * off_diag_f

            # ------------------------------------------------------------------
            # Stage 2: Empty-Slot Direct Write vs 56-Pair Probabilistic-OR Merge
            # ------------------------------------------------------------------
            rho_for_min = torch.where(free_mask, rho_tilde_r, torch.full_like(rho_tilde_r, 2.0))
            min_free_rho, k_cold = rho_for_min.min(dim=-1)  # [P], [P]
            sigma_full = torch.sigmoid(
                (min_free_rho - self.full_gate_center) / self.full_gate_temp
            )  # [P] R3 relaxed temp 0.10
            has_empty_slot = (min_free_rho < self.full_gate_center).float()

            # Branch 2A: Lossless empty-slot direct write into coldest free slot k_cold
            w_cold = F.one_hot(k_cold, num_classes=K).float()  # [P, K]
            slots_empty = slots_pre_f + w_cold.unsqueeze(-1) * (u_in_f.unsqueeze(1) - slots_pre_f)
            occ_empty = rho_tilde + w_cold * (self.write_occ - rho_tilde)

            w_c_row = w_cold[:, :, None, None]
            w_c_col = w_cold[:, None, :, None]
            edges_empty = (1.0 - w_c_row) * edges_dec_f + w_c_row * e_out_cand.unsqueeze(1)
            edges_empty = (1.0 - w_c_col) * edges_empty + w_c_col * e_in_cand.unsqueeze(2)
            edges_empty = edges_empty * off_diag_f

            # Branch 2B: Full-load 56-pair (K*(K-1)) Probabilistic-OR Merge
            edges_pairs_r = edges_dec_r_f[:, self.pair_i, self.pair_j, :]  # [P, M_pairs, d_r]
            h_edge = F.linear(edges_pairs_r, self.w_pair_edge.weight.float())  # [P, M_pairs, d_m]
            rho_pi = rho_tilde_r[:, self.pair_i].unsqueeze(-1)  # [P, M_pairs, 1]
            rho_pj = rho_tilde_r[:, self.pair_j].unsqueeze(-1)  # [P, M_pairs, 1]
            h_pair = (
                h_si.float()[:, self.pair_i, :]
                + h_sj.float()[:, self.pair_j, :]
                + h_edge
                + rho_pi * self.w_pair_rho_i.float().view(1, 1, -1)
                + rho_pj * self.w_pair_rho_j.float().view(1, 1, -1)
                + self.pair_bias.float().view(1, 1, -1)
            )  # [P, M_pairs, d_m]
            h_pair_act = F.silu(h_pair)

            # Direct slot-pair dot similarity boosts semantically related merges
            pair_cos = (slots_r_f[:, self.pair_i, :] * slots_r_f[:, self.pair_j, :]).sum(dim=-1)  # [P, M_pairs]
            raw_pair_logits = (
                F.linear(h_pair_act, self.w_pair_score.weight.float(), self.w_pair_score.bias.float()).squeeze(-1)
                + self.pair_cos_gain.float().clamp(-2.0, 2.0) * pair_cos
            )
            # Soft-cap pair logits in (-4, 4) to prevent runaway Jacobian amplification over 512 steps
            pair_logits = 4.0 * torch.tanh(raw_pair_logits * 0.25)

            if self.pin_enabled:
                # Mask out all 14 pairs involving Slot-0 while pinned (leaving 42 valid pairs for K=8)
                pair_valid = ~(pinned_pre.unsqueeze(1) & self.pair_uses_slot0.unsqueeze(0))
                pair_logits = torch.where(pair_valid, pair_logits, torch.full_like(pair_logits, -1.0e9))

            win_p = pair_logits.argmax(dim=-1)  # [P]
            p_merge = F.softmax(pair_logits, dim=-1)  # [P, M_pairs]
            oh_merge = F.one_hot(win_p, num_classes=self.num_pairs).float()  # [P, M_pairs]
            w_pair = oh_merge + (p_merge - p_merge.detach())  # [P, M_pairs] STE

            # Gather winning pair (i*, j*) and target slots u* = min(i*, j*), v* = max(i*, j*)
            win_i = self.pair_i[win_p]  # [P]
            win_j = self.pair_j[win_p]  # [P]

            idx_i_d = win_i.view(P, 1, 1).expand(P, 1, D)
            idx_j_d = win_j.view(P, 1, 1).expand(P, 1, D)
            s_i_star = slots_pre_f.gather(1, idx_i_d).squeeze(1)  # [P, D]
            s_j_star = slots_pre_f.gather(1, idx_j_d).squeeze(1)  # [P, D]
            rho_i_star = rho_tilde.gather(1, win_i.unsqueeze(1)).squeeze(1)  # [P]
            rho_j_star = rho_tilde.gather(1, win_j.unsqueeze(1)).squeeze(1)  # [P]

            h_star_act = h_pair_act.gather(
                1, win_p.view(P, 1, 1).expand(P, 1, self.d_merge)
            ).squeeze(1)  # [P, d_m]

            # Orthogonal-projection head-biased blend inside closed unit ball:
            # ||sqrt(beta)*s_i + sqrt(1-beta)*s_j|| preserves norm on orthogonal slots while
            # maintaining linear state-transition operator norm ||J||_2 <= slot_keep_max <= 0.990!
            beta_logit = F.linear(
                h_star_act, self.w_merge_beta.weight.float(), self.w_merge_beta.bias.float()
            )  # [P, 1]
            beta_m = 0.55 + 0.30 * torch.sigmoid(beta_logit)  # [P, 1] in [0.55, 0.85]
            delta_m = 0.12 * torch.tanh(F.linear(h_star_act, self.w_merge_delta.weight.float()))  # [P, D]
            s_comb = (
                self.slot_keep_max * (torch.sqrt(beta_m) * s_i_star + torch.sqrt(1.0 - beta_m) * s_j_star)
                + delta_m
            )
            s_merged = _project_closed_unit_ball(s_comb, max_norm=0.995, eps=self.eps)  # [P, D]

            # Bounded Probabilistic-OR occupancy for merged parent slot u* (capped at 0.90)
            rho_or = (rho_i_star + rho_j_star - rho_i_star * rho_j_star).clamp(self.occ_min, 0.90)  # [P]

            # Differentiable STE scatter onto (u*, v*)
            u_oh_f = self.u_oh.float()
            v_oh_f = self.v_oh.float()
            w_u = torch.matmul(w_pair, u_oh_f)  # [P, K] (exact one-hot at u* in forward)
            w_v = torch.matmul(w_pair, v_oh_f)  # [P, K] (exact one-hot at v* in forward)
            w_keep = 1.0 - w_u - w_v            # [P, K]

            slots_merge = (
                w_keep.unsqueeze(-1) * slots_pre_f
                + w_u.unsqueeze(-1) * s_merged.unsqueeze(1)
                + w_v.unsqueeze(-1) * u_in_f.unsqueeze(1)
            )
            occ_merge = w_keep * rho_tilde + w_u * rho_or.unsqueeze(1) + w_v * self.write_occ

            # Edge contraction & merge for Branch 2B: average incident edges of (i*, j*) into u*,
            # write fresh bilinear edges for freed slot v*, zero self-loops
            idx_i_row = win_i.view(P, 1, 1, 1).expand(P, 1, K, d_r)
            idx_j_row = win_j.view(P, 1, 1, 1).expand(P, 1, K, d_r)
            row_u = 0.5 * (edges_dec_f.gather(1, idx_i_row) + edges_dec_f.gather(1, idx_j_row))  # [P, 1, K, d_r]
            edges_row = (
                w_keep[:, :, None, None] * edges_dec_f
                + w_u[:, :, None, None] * row_u
                + w_v[:, :, None, None] * e_out_cand.unsqueeze(1)
            )
            idx_i_col = win_i.view(P, 1, 1, 1).expand(P, K, 1, d_r)
            idx_j_col = win_j.view(P, 1, 1, 1).expand(P, K, 1, d_r)
            col_u = 0.5 * (edges_row.gather(2, idx_i_col) + edges_row.gather(2, idx_j_col))  # [P, K, 1, d_r]
            edges_merge = (
                w_keep[:, None, :, None] * edges_row
                + w_u[:, None, :, None] * col_u
                + w_v[:, None, :, None] * e_in_cand.unsqueeze(2)
            )
            edges_merge = edges_merge * off_diag_f

            # ------------------------------------------------------------------
            # Write-Before-Merge eviction signal (G2 -> WS-4 Hippocampus)
            # ------------------------------------------------------------------
            oh_u = torch.matmul(oh_merge, u_oh_f)  # [P, K]
            oh_v = torch.matmul(oh_merge, v_oh_f)  # [P, K]
            evict_intensity = (sigma_full.unsqueeze(1) * (1.0 - g_assim)) * (oh_u + oh_v)  # [P, K]
            evict_weight = torch.where(valid_b.unsqueeze(1), evict_intensity, torch.zeros_like(evict_intensity))

            # ------------------------------------------------------------------
            # Smooth interpolation: Stage 1 (assim) vs Stage 2 (empty vs merge)
            # ------------------------------------------------------------------
            sf_s = sigma_full.view(P, 1, 1)
            sf_o = sigma_full.view(P, 1)
            sf_e = sigma_full.view(P, 1, 1, 1)
            slots_s2 = (1.0 - sf_s) * slots_empty + sf_s * slots_merge
            occ_s2 = (1.0 - sf_o) * occ_empty + sf_o * occ_merge
            edges_s2 = (1.0 - sf_e) * edges_empty + sf_e * edges_merge

            ga_s = g_assim.view(P, 1, 1)
            ga_o = g_assim
            ga_e = g_assim.view(P, 1, 1, 1)
            slots_upd = ga_s * slots_s1 + (1.0 - ga_s) * slots_s2
            occ_upd = (ga_o * occ_s1 + (1.0 - ga_o) * occ_s2).clamp(self.occ_min, self.occ_max)
            edges_upd = (ga_e * edges_s1 + (1.0 - ga_e) * edges_s2).clamp(-0.9999, 0.9999)

            # Bit-exact preservation of Slot-0 while already pinned (lam_0 = 1.0)
            if self.pin_enabled:
                pin_slot0_mask = pinned_pre.unsqueeze(1) & (self.slot_idx.unsqueeze(0) == 0)  # [P, K]
                slots_upd = torch.where(pin_slot0_mask.unsqueeze(-1), slots_pre_f, slots_upd)
                occ_upd = torch.where(pin_slot0_mask, occ_pre.float(), occ_upd)

                # Post-update Slot-0 pinning trigger (\x1dE\n -> write graph summary to Slot-0)
                if pin_trig is not None:
                    do_pin = valid_b & pin_trig.bool() & (~pinned_pre)
                    occ_w = occ_upd / occ_upd.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                    g_sum = torch.einsum("pk,pkd->pd", occ_w, slots_upd)
                    g_sum_proj = (0.92 * g_sum) / torch.sqrt(g_sum.pow(2).sum(dim=-1, keepdim=True) + 0.01)
                    do_pin_s0 = do_pin.unsqueeze(1) & (self.slot_idx.unsqueeze(0) == 0)
                    slots_upd = torch.where(
                        do_pin_s0.unsqueeze(-1), g_sum_proj.unsqueeze(1).expand(P, K, D), slots_upd
                    )
                    occ_pin_val = torch.clamp(occ_upd[:, 0:1], min=0.85).expand(P, K)
                    occ_upd = torch.where(do_pin_s0, occ_pin_val, occ_upd)
                    pinned_next = pinned_pre | do_pin
                else:
                    pinned_next = pinned_pre
            else:
                pinned_next = pinned_pre

            # Gate C diagnostic: empty-slot direct write indicator on steps with an available empty slot
            empty_write = (
                valid_b.float()
                * has_empty_slot
                * (g_assim.squeeze(-1) < 0.5).float()
                * (sigma_full < 0.5).float()
            )
            has_empty_valid = valid_b.float() * has_empty_slot

        # Freeze padded steps (`~seg_valid`) bit-exactly
        slots_next = torch.where(valid_b[:, None, None], slots_upd.to(orig_dtype), slots_pre)
        edges_next = torch.where(valid_b[:, None, None, None], edges_upd.to(orig_dtype), edges_pre)
        occ_next = torch.where(valid_b[:, None], occ_upd.float(), occ_pre.float())
        pinned_next = torch.where(valid_b, pinned_next, pinned_pre)

        next_state = CarrierGraphState(
            slots=slots_next,
            edges=edges_next,
            occ=occ_next,
            slot0_pinned=pinned_next,
        )
        out_state = prior_state if self.return_prior else next_state
        return next_state, out_state, evict_weight, slots_pre, empty_write, has_empty_valid

    def forward_chunk(
        self,
        state: CarrierGraphState,
        x_chunk: torch.Tensor,
        reset_chunk: torch.Tensor,
        valid_chunk: torch.Tensor,
        pin_chunk: Optional[torch.Tensor] = None,
        unpin_chunk: Optional[torch.Tensor] = None,
        hip_chunk: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        CarrierGraphState,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Static-length `C = c_scan` chunk scan passing `torch.compile(fullgraph=True, dynamic=False)`.

        Uses fixed-size `[None] * C` containers (zero step-growing Python lists, zero `.append()`).
        """
        C = int(x_chunk.shape[1])
        slots_arr: list[Optional[torch.Tensor]] = [None] * C
        edges_arr: list[Optional[torch.Tensor]] = [None] * C
        occ_arr: list[Optional[torch.Tensor]] = [None] * C
        ew_arr: list[Optional[torch.Tensor]] = [None] * C
        et_arr: list[Optional[torch.Tensor]] = [None] * C

        cur = state
        ew_num = torch.zeros((), device=x_chunk.device, dtype=torch.float32)
        ew_den = torch.zeros((), device=x_chunk.device, dtype=torch.float32)

        for t in range(C):
            p_t = None if pin_chunk is None else pin_chunk[:, t]
            u_t = None if unpin_chunk is None else unpin_chunk[:, t]
            h_t = None if hip_chunk is None else hip_chunk[:, t]
            cur, out_t, ew_t, et_t, is_ew, has_e = self.step_single(
                cur,
                x_chunk[:, t],
                reset_chunk[:, t],
                valid_chunk[:, t],
                pin_trig=p_t,
                unpin_trig=u_t,
                hip_inject=h_t,
            )
            slots_arr[t] = out_t.slots
            edges_arr[t] = out_t.edges
            occ_arr[t] = out_t.occ
            ew_arr[t] = ew_t
            et_arr[t] = et_t
            ew_num = ew_num + is_ew.sum()
            ew_den = ew_den + has_e.sum()

        slots_seq = torch.stack(slots_arr, dim=1)  # type: ignore[arg-type]
        edges_seq = torch.stack(edges_arr, dim=1)  # type: ignore[arg-type]
        occ_seq = torch.stack(occ_arr, dim=1)      # type: ignore[arg-type]
        evict_w_seq = torch.stack(ew_arr, dim=1)   # type: ignore[arg-type]
        evict_t_seq = torch.stack(et_arr, dim=1)   # type: ignore[arg-type]
        return slots_seq, edges_seq, occ_seq, cur, evict_w_seq, evict_t_seq, ew_num, ew_den

    def _chunk_ckpt_wrapper(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        occ: torch.Tensor,
        pinned: torch.Tensor,
        x_chunk: torch.Tensor,
        reset_chunk: torch.Tensor,
        valid_chunk: torch.Tensor,
        pin_chunk: Optional[torch.Tensor],
        unpin_chunk: Optional[torch.Tensor],
        hip_chunk: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        st = CarrierGraphState(slots=slots, edges=edges, occ=occ, slot0_pinned=pinned)
        s_seq, e_seq, o_seq, nxt, ew_seq, et_seq, ew_num, ew_den = self.forward_chunk(
            st, x_chunk, reset_chunk, valid_chunk, pin_chunk, unpin_chunk, hip_chunk
        )
        return (
            s_seq,
            e_seq,
            o_seq,
            nxt.slots,
            nxt.edges,
            nxt.occ,
            nxt.slot0_pinned,
            ew_seq,
            et_seq,
            ew_num,
            ew_den,
        )

    def forward_scan(
        self,
        x_seq: torch.Tensor,
        seg_reset: torch.Tensor,
        seg_valid: torch.Tensor,
        init_state: Optional[CarrierGraphState] = None,
        seg_pin_trig: Optional[torch.Tensor] = None,
        seg_unpin_trig: Optional[torch.Tensor] = None,
        hip_inject: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_checkpoint: bool = False,
    ) -> CarrierScanOut:
        """Chunked outer loop over `S // c_scan` static chunks returning frozen `CarrierScanOut`."""
        P, S, _ = x_seq.shape
        c_sz = int(self.c_scan if chunk_size is None else chunk_size)
        if S % c_sz != 0:
            raise ValueError(f"Sequence length S={S} must be divisible by chunk_size={c_sz}")

        n_chunks = S // c_sz
        cur = (
            self.init_state(P, device=x_seq.device, dtype=x_seq.dtype)
            if init_state is None
            else init_state
        )

        slots_chunks: list[Optional[torch.Tensor]] = [None] * n_chunks
        edges_chunks: list[Optional[torch.Tensor]] = [None] * n_chunks
        occ_chunks: list[Optional[torch.Tensor]] = [None] * n_chunks
        ew_chunks: list[Optional[torch.Tensor]] = [None] * n_chunks
        et_chunks: list[Optional[torch.Tensor]] = [None] * n_chunks
        total_ew_num = torch.zeros((), device=x_seq.device, dtype=torch.float32)
        total_ew_den = torch.zeros((), device=x_seq.device, dtype=torch.float32)

        do_ckpt = bool(use_checkpoint and self.training and torch.is_grad_enabled())

        for ci in range(n_chunks):
            s0 = ci * c_sz
            s1 = s0 + c_sz
            xc = x_seq[:, s0:s1]
            rc = seg_reset[:, s0:s1]
            vc = seg_valid[:, s0:s1]
            pc = None if seg_pin_trig is None else seg_pin_trig[:, s0:s1]
            uc = None if seg_unpin_trig is None else seg_unpin_trig[:, s0:s1]
            hc = None if hip_inject is None else hip_inject[:, s0:s1]

            if do_ckpt:
                (
                    s_seq,
                    e_seq,
                    o_seq,
                    ns,
                    ne,
                    no,
                    npin,
                    ew_seq,
                    et_seq,
                    ew_num,
                    ew_den,
                ) = torch_checkpoint(
                    self._chunk_ckpt_wrapper,
                    cur.slots,
                    cur.edges,
                    cur.occ,
                    cur.slot0_pinned,
                    xc,
                    rc,
                    vc,
                    pc,
                    uc,
                    hc,
                    use_reentrant=False,
                )
                cur = CarrierGraphState(slots=ns, edges=ne, occ=no, slot0_pinned=npin)
            else:
                s_seq, e_seq, o_seq, cur, ew_seq, et_seq, ew_num, ew_den = self.forward_chunk(
                    cur, xc, rc, vc, pc, uc, hc
                )

            slots_chunks[ci] = s_seq
            edges_chunks[ci] = e_seq
            occ_chunks[ci] = o_seq
            ew_chunks[ci] = ew_seq
            et_chunks[ci] = et_seq
            total_ew_num = total_ew_num + ew_num
            total_ew_den = total_ew_den + ew_den

        slots_all = torch.cat(slots_chunks, dim=1)  # type: ignore[arg-type]
        edges_all = torch.cat(edges_chunks, dim=1)  # type: ignore[arg-type]
        occ_all = torch.cat(occ_chunks, dim=1)      # type: ignore[arg-type]
        empty_write_rate = total_ew_num / total_ew_den.clamp_min(1.0)

        emit_evict = self.pin_enabled or self.return_prior
        evict_weight = torch.cat(ew_chunks, dim=1) if emit_evict else None  # type: ignore[arg-type]
        evict_target = torch.cat(et_chunks, dim=1) if emit_evict else None  # type: ignore[arg-type]

        return CarrierScanOut(
            slots_seq=slots_all,
            edges_seq=edges_all,
            occ_seq=occ_all,
            final_state=cur,
            evict_weight=evict_weight,
            evict_target=evict_target,
            empty_write_rate=empty_write_rate,
        )


__all__ = [
    "ContractiveGraphCarrier",
]
