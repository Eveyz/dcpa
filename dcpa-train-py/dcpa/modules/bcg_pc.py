"""Module 2: Bottlenecked Concept-Graph Predictive Core (BCG-PC).

Implements Engineering Blueprint Section 2.3 & Theory Doc Part 2.II:
  - Fixed working memory graph G = (S: [K, D], A: [K, K, d_r], omega: [K]), hard-capped at K=8.
  - Zero KV Cache, O(1) dynamic memory with respect to sequence length T.
  - Top-down prior prediction c_hat & diagonal precision pi_m (forced float32).
  - Bayesian precision fusion + Masked Unrolled Pondering (L_max = 4) with Top-2 2-layer SwiGLU micro-experts.
  - Candidate slot expansion (K -> K+1) with low-rank bilinear directed edges.
  - Prebuilt constant [72, 8, 9] perm_bank + Gumbel-STE K+1 -> K syntactic merge bottleneck.
  - Exact LogSumExp slot salience conservation equation in float32.
  - EMA neutral concept center c_0 maintenance (momentum = 0.995).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.config import DCPAConfig
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.rmsnorm import RMSNorm


@dataclass
class ConceptGraphState:
    slots: torch.Tensor     # [B, K, D]
    edges: torch.Tensor     # [B, K, K, d_r]
    salience: torch.Tensor  # [B, K] float32
    slot0_pinned: Optional[torch.Tensor] = None  # [B] float32 (0.0 or 1.0, Slot-0 Thesis Pinning)


@dataclass
class BCGPCStepOutput:
    z_star: torch.Tensor            # [B, D] disambiguated concept
    g_ctx: torch.Tensor             # [B, D] pre-step graph context readout
    c_hat: torch.Tensor             # [B, D] top-down prior prediction
    pi_m: torch.Tensor              # [B, D] float32 diagonal precision
    surprisal: torch.Tensor         # [B] float32 normalized residual E_m
    ponder_steps: torch.Tensor      # [B] float32 executed pondering steps (with STE surrogate)
    state: ConceptGraphState        # updated (S_m, A_m, omega_m, slot0_pinned_m)
    s_bar_post: torch.Tensor        # [B, K, D] post-merge message-passed slots (for Hippocampus q_{m,i})
    win_pair_idx: torch.Tensor      # [B] int64 winning ordered pair index in 0..71
    win_i: torch.Tensor             # [B] int64 i* of winning pair
    win_j: torch.Tensor             # [B] int64 j* of winning pair
    merge_probs: torch.Tensor       # [B, 72] float32 pre-hard softmax probabilities (for entropy monitor/KL)
    merge_alpha: torch.Tensor       # [B, 72] float32 hard one-hot forward STE merge selection
    moe_bal_loss: Optional[torch.Tensor] = None  # scalar fp32 Switch-style load balancing N_E * sum(f_e * P_e)


def build_perm_bank(k_slots: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-build read-only [K*(K+1), K, K+1] permutation/contraction tensor per Engineering Doc 2.3.

    For ordered pair p = (i, j) with 0 <= i != j <= K:
      u = min(i, j)  (< K)
      v = max(i, j)  (<= K)
      - Row u has 0.5 at column i and 0.5 at column j (base for merged parent node & edge average).
      - If v < K: row v has 1.0 at column K (places new concept z_m^* into freed slot v).
      - Every remaining row r in {0..K-1} \\ {u, v} has 1.0 at column r (in-place identity).
    """
    pairs: list[tuple[int, int]] = []
    for i in range(k_slots + 1):
        for j in range(k_slots + 1):
            if i != j:
                pairs.append((i, j))

    num_pairs = len(pairs)
    perm_bank = torch.zeros((num_pairs, k_slots, k_slots + 1), dtype=torch.float32)
    pair_i = torch.empty(num_pairs, dtype=torch.int64)
    pair_j = torch.empty(num_pairs, dtype=torch.int64)
    pair_u = torch.empty(num_pairs, dtype=torch.int64)
    pair_v = torch.empty(num_pairs, dtype=torch.int64)

    for p, (i, j) in enumerate(pairs):
        u = min(i, j)
        v = max(i, j)
        pair_i[p] = i
        pair_j[p] = j
        pair_u[p] = u
        pair_v[p] = v

        perm_bank[p, u, i] = 0.5
        perm_bank[p, u, j] = 0.5
        if v < k_slots:
            perm_bank[p, v, k_slots] = 1.0
        for r in range(k_slots):
            if r != u and r != v:
                perm_bank[p, r, r] = 1.0

    return perm_bank, pair_i, pair_j, pair_u, pair_v


class TwoLayerSwiGLUExpert(nn.Module):
    """Two-layer Pre-RMSNorm SwiGLU residual micro-expert with hidden dimension 1.5*D (Table 5.2)."""

    def __init__(self, d_model: int, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.w_gate1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down1 = nn.Linear(hidden_dim, d_model, bias=False)

        self.w_gate2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down2 = nn.Linear(hidden_dim, d_model, bias=False)

        # Residual output projections initialized small so 4-step pondering stays well-conditioned
        nn.init.normal_(self.w_down1.weight, std=0.02)
        nn.init.normal_(self.w_down2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is already unit-RMS normalized by SparseMicroMoE; pre-norm Layer 2 prevents x^4 SwiGLU blowup.
        h1 = F.silu(self.w_gate1(x)) * self.w_up1(x)
        x = x + self.w_down1(h1)
        x_n2 = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        h2 = F.silu(self.w_gate2(x_n2)) * self.w_up2(x_n2)
        return x + self.w_down2(h2)


class SparseMicroMoE(nn.Module):
    """Top-2 sparse micro-experts dispatched strictly on active high-surprisal samples."""

    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int = 2, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.eps = eps
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [TwoLayerSwiGLUExpert(d_model, hidden_dim, eps=eps) for _ in range(num_experts)]
        )

    def pack_weights(
        self, dtype: Optional[torch.dtype] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack all E expert weights into 4 contiguous batched-GEMM tensors (preserves state_dict 100%).

        Returns `(w_gu1 [E, D, 2*H], w_d1 [E, H, D], w_gu2 [E, D, 2*H], w_d2 [E, H, D])`.
        Calling once per `DCPAModel.forward` avoids E*6 sequential GEMMs AND redundant fp32->bf16 casts!
        """
        w_gu1 = torch.stack(
            [torch.cat([e.w_gate1.weight, e.w_up1.weight], dim=0) for e in self.experts], dim=0
        ).transpose(1, 2)
        w_d1 = torch.stack([e.w_down1.weight for e in self.experts], dim=0).transpose(1, 2)
        w_gu2 = torch.stack(
            [torch.cat([e.w_gate2.weight, e.w_up2.weight], dim=0) for e in self.experts], dim=0
        ).transpose(1, 2)
        w_d2 = torch.stack([e.w_down2.weight for e in self.experts], dim=0).transpose(1, 2)
        if dtype is not None:
            w_gu1 = w_gu1.to(dtype=dtype)
            w_d1 = w_d1.to(dtype=dtype)
            w_gu2 = w_gu2.to(dtype=dtype)
            w_d2 = w_d2.to(dtype=dtype)
        return w_gu1.contiguous(), w_d1.contiguous(), w_gu2.contiguous(), w_d2.contiguous()

    def forward(
        self,
        x: torch.Tensor,
        static_shape: bool = False,
        sample_mask: Optional[torch.Tensor] = None,
        packed_weights: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [N, D] -> (out [N, D], aux_bal_loss fp32 scalar = N_E * sum_e f_e * P_e).

        Uses 4 fused batched `torch.bmm` kernels in static/compiled mode (12x fewer CUDA GEMM launches
        than looping E*6 linear layers, zero GPU-CPU syncs, bit-equivalent to sequential experts).
        """
        if x.shape[0] == 0:
            return x, torch.zeros((), device=x.device, dtype=torch.float32)
        # Pre-RMSNorm input to router + Layer-1 SwiGLU so deep pondering recurrence cannot diverge
        x = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        logits = self.router(x).float()  # [N, E]
        router_probs = F.softmax(logits, dim=-1)  # [N, E] fp32
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [N, top_k]
        gates = F.softmax(top_vals, dim=-1).to(x.dtype)  # [N, top_k]

        # Standard Switch/Mixtral MoE load-balancing auxiliary term: N_E * sum_e f_e * P_e (branchless!)
        if sample_mask is None:
            mask_f = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float32)
        else:
            mask_f = sample_mask.float().view(x.shape[0], 1)
        m_sum = mask_f.sum()
        denom = m_sum.clamp_min(1.0)
        has_any = (m_sum > 0.0).float()
        p_e = (router_probs * mask_f).sum(dim=0) / denom  # [E]
        dispatch_oh = F.one_hot(top_idx, num_classes=self.num_experts).float().sum(dim=1) / float(self.top_k)
        f_e = (dispatch_oh * mask_f).sum(dim=0) / denom  # [E]
        aux_bal = has_any * float(self.num_experts) * (f_e.detach() * p_e).sum()

        # Fused 4-BMM branchless dispatch (CUDA torch.compile + MPS): zero syncs, 12x fewer GEMMs!
        if static_shape or x.device.type == "mps" or torch.compiler.is_compiling():
            gate_full = torch.zeros((x.shape[0], self.num_experts), device=x.device, dtype=x.dtype)
            gate_full = gate_full.scatter_add(1, top_idx, gates)
            if packed_weights is None:
                w_gu1, w_d1, w_gu2, w_d2 = self.pack_weights(dtype=x.dtype)
            else:
                w_gu1, w_d1, w_gu2, w_d2 = packed_weights
                if w_gu1.dtype != x.dtype:
                    w_gu1 = w_gu1.to(x.dtype)
                    w_d1 = w_d1.to(x.dtype)
                    w_gu2 = w_gu2.to(x.dtype)
                    w_d2 = w_d2.to(x.dtype)

            x_e = x.unsqueeze(0).expand(self.num_experts, -1, -1)  # [E, N, D]
            g1, u1 = torch.bmm(x_e, w_gu1).chunk(2, dim=-1)  # [E, N, H]
            x1 = x_e + torch.bmm(F.silu(g1) * u1, w_d1)  # [E, N, D]
            x_n2 = (x1.float() * torch.rsqrt(x1.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
            g2, u2 = torch.bmm(x_n2, w_gu2).chunk(2, dim=-1)  # [E, N, H]
            y_all = x1 + torch.bmm(F.silu(g2) * u2, w_d2)  # [E, N, D]
            out = torch.einsum("ne,end->nd", gate_full, y_all)
            return out, aux_bal

        out = torch.zeros_like(x)
        for e_id, expert in enumerate(self.experts):
            mask = top_idx == e_id  # [N, top_k]
            if not mask.any():
                continue
            sample_idx, k_pos = torch.nonzero(mask, as_tuple=True)
            y_e = expert(x[sample_idx])
            w_e = gates[sample_idx, k_pos].unsqueeze(-1)
            out.index_add_(0, sample_idx, (y_e * w_e).to(out.dtype))
        return out, aux_bal


class SlotCrossAttention(nn.Module):
    """Lightweight cross-attention from single concept query z_m^{(l)} to fixed K=8 slots S_bar."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

    def project_kv(self, kv_slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute slot K (fp32) & V once before L_max pondering steps (8x less KV work!)."""
        return self.w_k(kv_slots).float(), self.w_v(kv_slots)

    def forward_with_kv(self, q_vec: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.w_q(q_vec).float()  # [N, D]
        scores = torch.einsum("nd,nkd->nk", q, k) * self.scale  # [N, K] fp32
        attn = F.softmax(scores, dim=-1).to(v.dtype)
        ctx = torch.einsum("nk,nkd->nd", attn, v)
        return self.w_o(ctx)

    def forward(self, q_vec: torch.Tensor, kv_slots: torch.Tensor) -> torch.Tensor:
        """q_vec: [N, D], kv_slots: [N, K, D] -> [N, D]."""
        k, v = self.project_kv(kv_slots)
        return self.forward_with_kv(q_vec, k, v)


class BCGPC(nn.Module):
    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        K = cfg.k_slots
        D = cfg.d_concept
        d_r = cfg.d_r
        d_m = cfg.d_merge

        # Prebuilt constant permutation tensor [72, 8, 9] (Engineering Doc 2.3)
        perm_bank, pair_i, pair_j, pair_u, pair_v = build_perm_bank(K)
        self.register_buffer("perm_bank", perm_bank)
        self.register_buffer("pair_i", pair_i)
        self.register_buffer("pair_j", pair_j)
        self.register_buffer("pair_u", pair_u)
        self.register_buffer("pair_v", pair_v)
        # Prebuilt scatter matrix [72, K] mapping each pair p -> target slot u = min(i, j) (avoids MPS index_add fallback)
        self.register_buffer("pair_u_onehot", F.one_hot(pair_u, num_classes=K).float())
        # Static Slot-0 Thesis Pinning masks (Spec 2): 1.0 for the 16 pairs touching slot 0, 0.0 for the 56 free pairs
        self.register_buffer("pin_mask_72", ((pair_i == 0) | (pair_j == 0)).float(), persistent=False)
        self.register_buffer("slot0_onehot", F.one_hot(torch.tensor(0, dtype=torch.int64), num_classes=K).float(), persistent=False)

        # Learnable initial working memory graph G_0
        self.s0 = nn.Parameter(torch.randn(K, D) * 0.02)
        self.a0 = nn.Parameter(torch.zeros(K, K, d_r))
        self.omega0 = nn.Parameter(torch.zeros(K))

        # Neutral concept center c_0 (EMA updated during training, frozen during inference)
        self.register_buffer("c0", torch.zeros(D, dtype=torch.float32))

        # 1. Edge-conditioned graph message passing & top-down prior/precision
        self.w_e = nn.Linear(d_r, D, bias=False)
        self.f_pred_1 = nn.Linear(D, D, bias=True)
        self.f_pred_2 = nn.Linear(D, D, bias=True)
        self.f_prec_1 = nn.Linear(D, D, bias=True)
        self.f_prec_2 = nn.Linear(D, D, bias=True)

        # 2. Fast shortcut & event-driven pondering micro-experts
        self.w_fast = nn.Linear(D, D, bias=False)
        self.w_gen = nn.Linear(D, D, bias=False)
        self.cross_attn = SlotCrossAttention(D)
        self.moe = SparseMicroMoE(
            d_model=D,
            hidden_dim=cfg.expert_hidden,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k_experts,
            eps=cfg.eps,
        )
        self.norm_ponder = RMSNorm(D, eps=cfg.eps)

        # 3. Candidate slot expansion (K -> K+1) low-rank bilinear edge generator
        self.u_src = nn.Linear(D, d_r, bias=False)
        self.u_dst = nn.Linear(D, d_r, bias=False)

        # 4. Syntactic merge bottleneck (K+1 -> K)
        self.w_l = nn.Linear(D, d_m, bias=False)
        self.w_r = nn.Linear(D, d_m, bias=False)
        self.w_edge_score = nn.Linear(d_r, d_m, bias=False)
        self.w_merge = nn.Linear(d_m, 1, bias=False)

        self.w_g = nn.Linear(2 * D + d_r, D, bias=True)
        self.w_comp = nn.Linear(2 * D, D, bias=True)
        self.norm_merge = RMSNorm(D, eps=cfg.eps)

        # 5. Slot salience conservation projection
        self.w_omega = nn.Linear(D, 1, bias=False)
        self.kappa_omega = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self._static_kernel_mode: bool = False

        self._init_weights()

    def set_static_kernel_mode(self, enabled: bool) -> None:
        """Toggle branchless static-shape kernel mode for `torch.compile(..., dynamic=False)`."""
        self._static_kernel_mode = bool(enabled)

    def _init_weights(self) -> None:
        # Initialize f_prec bias so Softplus(bias) ~ 1.0 and w_gen near identity so pondering
        # residual (c_fused - W_gen z) measures true reconstruction error from Step 0.
        nn.init.constant_(self.f_prec_2.bias, 0.5413)
        nn.init.eye_(self.w_gen.weight)
        nn.init.xavier_uniform_(self.w_fast.weight)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> ConceptGraphState:
        dev = device if device is not None else self.s0.device
        dt = dtype if dtype is not None else self.s0.dtype
        s0_src = self.s0 if self.training else self.s0.detach()
        a0_src = self.a0 if self.training else self.a0.detach()
        w0_src = self.omega0 if self.training else self.omega0.detach()
        slots = s0_src.to(device=dev, dtype=dt).unsqueeze(0).expand(batch_size, -1, -1).clone()
        edges = a0_src.to(device=dev, dtype=dt).unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        salience = w0_src.float().to(device=dev).unsqueeze(0).expand(batch_size, -1).clone()
        slot0_pinned = torch.zeros((batch_size,), device=dev, dtype=torch.float32)
        return ConceptGraphState(slots=slots, edges=edges, salience=salience, slot0_pinned=slot0_pinned)

    def _message_pass_with_eproj(
        self, slots: torch.Tensor, e_proj: torch.Tensor, salience: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_attn = F.softmax(salience.float(), dim=-1)  # [B, K] fp32
        weighted_s = slots.to(e_proj.dtype) * w_attn.unsqueeze(-1).to(e_proj.dtype)  # [B, K, D]
        agg = torch.einsum("bijd,bjd->bid", e_proj, weighted_s)
        s_bar = slots + agg.to(slots.dtype)
        g_ctx = torch.einsum("bk,bkd->bd", w_attn.to(s_bar.dtype), s_bar)
        return s_bar, g_ctx

    def message_pass(self, slots: torch.Tensor, edges: torch.Tensor, salience: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute edge-aggregated node states S_bar [B, K, D] and context readout g [B, D]."""
        e_proj = torch.tanh(self.w_e(edges))  # [B, K, K, D], bounded in (-1, 1) for stable multi-step recurrence
        return self._message_pass_with_eproj(slots, e_proj, salience)

    def update_c0_from_slots(self, s_new: torch.Tensor, active_mask: torch.Tensor) -> None:
        """Update neutral concept center c_0 via EMA strictly over active docs (safe outside activation checkpointing)."""
        with torch.no_grad():
            B = s_new.shape[0]
            K = self.cfg.k_slots
            am_f = active_mask.float().view(B, 1, 1)
            denom_c0 = (am_f.sum() * float(K)).clamp_min(1.0)
            batch_mean = (s_new.detach().float() * am_f).sum(dim=(0, 1)) / denom_c0
            m = self.cfg.c0_ema_momentum
            self.c0.mul_(m).add_(batch_mean * (1.0 - m))

    @torch._dynamo.disable
    def _ponder_loop_sparse(
        self,
        z_cur: torch.Tensor,
        eps_cur: torch.Tensor,
        surp_cur: torch.Tensor,
        s_bar_prev: torch.Tensor,
        c_fused: torch.Tensor,
        pi_eff: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Event-driven sparse pondering + true Top-2 sparse MoE dispatch with exact closed-form early exit.

        Executes ONLY active high-surprisal samples through ONLY their Top-2 selected experts, and exits
        immediately once `active` empties while preserving bit-exact STE `ponder_steps` gradients.
        """
        B, D = c_fused.shape
        orig_dtype = c_fused.dtype
        active = active_mask.bool()
        am_f = active_mask.float()
        ponder_steps = torch.zeros((B,), device=c_fused.device, dtype=torch.float32)
        moe_bal_sum = torch.zeros((), device=c_fused.device, dtype=torch.float32)
        moe_bal_count = torch.zeros((), device=c_fused.device, dtype=torch.float32)

        for l_idx in range(self.cfg.l_max):
            thresh = self.cfg.tau_skip if l_idx == 0 else self.cfg.tau_halt
            prev_active_f = active.float()
            still_surprised = surp_cur > thresh
            active = active & still_surprised

            # Differentiable STE ponder counter gated strictly on samples still active entering step l_idx
            # (zero phantom gradient on already-skipped/halted samples; tau_ste=0.25 prevents sigmoid saturation).
            soft_act = torch.sigmoid((surp_cur - thresh) / 0.25) * prev_active_f
            hard_act = active.float()
            ste_act = (hard_act - soft_act).detach() + soft_act
            ponder_steps = ponder_steps + ste_act

            if not active.any():
                break

            act_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            z_a = z_cur[act_idx]
            eps_a = eps_cur[act_idx]
            if self.training and l_idx > 0:
                # Zero-ULP straight-through ponder carry damping (0.5): bit-identical forward while
                # bounding 8-step recurrent Jacobian amplification O(0.5^7) inside deep pondering.
                z_a = z_a.detach() + 0.5 * (z_a - z_a.detach())
                eps_a = eps_a.detach() + 0.5 * (eps_a - eps_a.detach())
            s_bar_a = s_bar_prev[act_idx]

            r_a = self.cross_attn(z_a, s_bar_a)
            delta_a, bal_l = self.moe(eps_a + r_a, static_shape=False)
            moe_bal_sum = moe_bal_sum + bal_l
            moe_bal_count = moe_bal_count + 1.0

            z_next_a = self.norm_ponder(z_a + self.cfg.ponder_eta * delta_a).to(z_cur.dtype)

            z_next = z_cur.clone()
            z_next[act_idx] = z_next_a
            z_cur = z_next

            eps_next_a = (
                pi_eff[act_idx] * (c_fused[act_idx].float() - self.w_gen(z_next_a).float())
            ).to(orig_dtype)
            eps_next = eps_cur.clone()
            eps_next[act_idx] = eps_next_a
            eps_cur = eps_next

            surp_next = surp_cur.clone()
            surp_next[act_idx] = torch.sqrt(eps_next_a.float().pow(2).mean(dim=-1) + 1e-8)
            surp_cur = surp_next

        moe_bal_loss = moe_bal_sum / moe_bal_count.clamp_min(1.0)
        return z_cur, ponder_steps, moe_bal_loss

    def forward(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        c_fused: torch.Tensor,
        prec_bu: torch.Tensor,
        ach: torch.Tensor,
        tau_m: torch.Tensor,
        active_mask: torch.Tensor,
        merge_uniform_mix: torch.Tensor,
        slot0_pinned: Optional[torch.Tensor] = None,
        pin_trigger: Optional[torch.Tensor] = None,
        use_gumbel: bool = True,
        update_c0_ema: bool = True,
        packed_moe_w_gu1: Optional[torch.Tensor] = None,
        packed_moe_w_d1: Optional[torch.Tensor] = None,
        packed_moe_w_gu2: Optional[torch.Tensor] = None,
        packed_moe_w_d2: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Hotspot A (`bcg_step_kernel`): single-step concept graph transition kernel.

        Designed for `torch.compile(..., dynamic=False)` on bucketed document batches `[V_pad, ...]`:
          - In `static_mode` (CUDA `torch.compile` & MPS): 100% zero-sync, zero-graph-break fused
            Inductor graph using 4 batched `torch.bmm` MoE kernels (12x fewer GEMM launches!) and
            precomputed `SlotCrossAttention` K/V (8x less KV projection work).
          - In eager CPU mode (`static_mode=False`): uses event-driven sparse pondering + Top-2 MoE.
        """
        B, D = c_fused.shape
        K = self.cfg.k_slots
        orig_dtype = c_fused.dtype
        static_mode = self._static_kernel_mode or c_fused.device.type == "mps" or torch.compiler.is_compiling()
        packed_weights = (
            (packed_moe_w_gu1, packed_moe_w_d1, packed_moe_w_gu2, packed_moe_w_d2)
            if packed_moe_w_gu1 is not None
            and packed_moe_w_d1 is not None
            and packed_moe_w_gu2 is not None
            and packed_moe_w_d2 is not None
            else None
        )

        pinned_cur = (
            torch.zeros((B,), device=c_fused.device, dtype=torch.float32)
            if slot0_pinned is None
            else slot0_pinned.float()
        )
        trig_cur = (
            torch.zeros((B,), device=c_fused.device, dtype=torch.float32)
            if pin_trigger is None
            else pin_trigger.float()
        )

        # 1. Top-down prior prediction & smoothly bounded precision-weighted residual
        s_bar_prev, g_ctx = self.message_pass(slots, edges, salience)
        c_hat = self.f_pred_2(F.silu(self.f_pred_1(g_ctx)))  # [B, D]
        pi_raw = F.softplus(self.f_prec_2(F.silu(self.f_prec_1(g_ctx))).float()) + 1e-4  # [B, D] fp32
        pi_m = 2.0 * torch.tanh(pi_raw * 0.5)

        ach_f32 = ach.float().clamp(0.0, 1.0)
        prec_bu_f32 = prec_bu.float().clamp_min(0.0)
        # Detach top-down prior precision pi_m inside pondering gate/fusion so L_ponder cannot
        # cheat pondering tax by collapsing pi_m -> 0 (pi_m is calibrated strictly via L_PC).
        pi_m_sg = pi_m.detach()
        pi_eff = pi_m_sg * (1.0 - ach_f32).unsqueeze(-1)

        eps_0_surp = (pi_eff * (c_fused.float() - c_hat.float())).to(orig_dtype)
        surprisal_0 = torch.sqrt(eps_0_surp.float().pow(2).mean(dim=-1) + 1e-8)  # [B] fp32

        # Bayesian precision fusion + fast shortcut projected onto unit-RMS concept manifold:
        #   - Detaching c_hat on the slot-write path (c_hat is supervised directly at every step by
        #     L_PC and L_lang) eliminates the unnormalized slots -> g_ctx -> f_pred -> w_fast -> z_0
        #     recurrent Jacobian feedback loop across long documents (512B..4096B).
        #   - Applying self.norm_ponder guarantees z_0 lives on the exact same unit-RMS sphere
        #     whether Pondering skips (L_m=0) or iterates (L_m=1..8), matching dcpa-core-rs.
        c_hat_z = c_hat.detach()
        eps_0 = (pi_eff * (c_fused.float() - c_hat_z.float())).to(orig_dtype)
        z_cur = (
            (pi_m_sg * c_hat_z.float() + prec_bu_f32 * c_fused.float())
            / (pi_m_sg + prec_bu_f32 + self.cfg.eps)
        ).to(orig_dtype)
        z_cur = self.norm_ponder(
            z_cur + (1.0 - ach_f32).unsqueeze(-1).to(orig_dtype) * self.w_fast(eps_0)
        )

        # 2. Event-Driven Pondering (L_max = 8) + Switch MoE load-balancing aux loss
        if static_mode:
            eps_cur = eps_0
            surp_cur = surprisal_0
            active = active_mask.bool()
            ponder_steps = torch.zeros((B,), device=c_fused.device, dtype=torch.float32)
            moe_bal_sum = torch.zeros((), device=c_fused.device, dtype=torch.float32)
            moe_bal_count = torch.zeros((), device=c_fused.device, dtype=torch.float32)

            # Precompute SlotCrossAttention K/V & packed MoE weights ONCE across all L_max steps!
            k_prev, v_prev = self.cross_attn.project_kv(s_bar_prev)
            eff_packed_w = packed_weights if packed_weights is not None else self.moe.pack_weights(dtype=orig_dtype)

            for l_idx in range(self.cfg.l_max):
                thresh = self.cfg.tau_skip if l_idx == 0 else self.cfg.tau_halt
                prev_active_f = active.float()
                still_surprised = surp_cur > thresh
                active = active & still_surprised

                # Differentiable STE ponder counter gated strictly on samples still active entering step l_idx
                soft_act = torch.sigmoid((surp_cur - thresh) / 0.25) * prev_active_f
                hard_act = active.float()
                ste_act = (hard_act - soft_act).detach() + soft_act
                ponder_steps = ponder_steps + ste_act

                if self.training and l_idx > 0:
                    z_in = z_cur.detach() + 0.5 * (z_cur - z_cur.detach())
                    eps_in = eps_cur.detach() + 0.5 * (eps_cur - eps_cur.detach())
                else:
                    z_in = z_cur
                    eps_in = eps_cur

                # Zero-sync fused 4-BMM pondering (zero .any() / nonzero() graph breaks!)
                r_all = self.cross_attn.forward_with_kv(z_in, k_prev, v_prev)
                delta_all, bal_l = self.moe(
                    eps_in + r_all,
                    static_shape=True,
                    sample_mask=active,
                    packed_weights=eff_packed_w,
                )
                moe_bal_sum = moe_bal_sum + bal_l
                moe_bal_count = moe_bal_count + (active.float().sum() > 0.0).float()

                z_cand = self.norm_ponder(z_in + self.cfg.ponder_eta * delta_all).to(z_cur.dtype)
                act_col = active.unsqueeze(-1)
                z_cur = torch.where(act_col, z_cand, z_cur)

                eps_cand = (
                    pi_eff * (c_fused.float() - self.w_gen(z_cand).float())
                ).to(orig_dtype)
                eps_cur = torch.where(act_col, eps_cand, eps_cur)
                surp_cand = torch.sqrt(eps_cand.float().pow(2).mean(dim=-1) + 1e-8)
                surp_cur = torch.where(active, surp_cand, surp_cur)

            moe_bal_loss = moe_bal_sum / moe_bal_count.clamp_min(1.0)
            z_star = z_cur  # [B, D]
        else:
            z_star, ponder_steps, moe_bal_loss = self._ponder_loop_sparse(
                z_cur, eps_0, surprisal_0, s_bar_prev, c_fused, pi_eff, active_mask
            )

        # 3. Candidate slot expansion (K -> K+1)
        s_tilde = torch.cat([slots, z_star.unsqueeze(1)], dim=1)  # [B, K+1, D]
        src_all = self.u_src(s_tilde)  # [B, K+1, d_r]
        dst_all = self.u_dst(s_tilde)  # [B, K+1, d_r]

        new_row = torch.tanh(src_all[:, K : K + 1, :] * dst_all)  # [B, K+1, d_r]
        new_col = torch.tanh(src_all[:, :K, :] * dst_all[:, K : K + 1, :])  # [B, K, d_r]
        top_part = torch.cat([edges, new_col.unsqueeze(2)], dim=2)  # [B, K, K+1, d_r]
        a_tilde = torch.cat([top_part, new_row.unsqueeze(1)], dim=1)  # [B, K+1, K+1, d_r]

        # 4. K+1 -> K Syntactic Merge Bottleneck over all P = K*(K+1) = 72 ordered pairs
        pi_idx = self.pair_i  # [72]
        pj_idx = self.pair_j  # [72]
        pu_idx = self.pair_u  # [72]

        a_ij = a_tilde[:, pi_idx, pj_idx, :]  # [B, 72, d_r]

        # Decouple discrete 72-pair merge router inputs from the recurrent slot/edge carrier Jacobian:
        # router weights (w_l, w_r, w_edge_score, w_merge) receive exact Gumbel-STE gradients without
        # injecting an O(||S||_2 / tau_m) discrete slot-swap Jacobian back into slots_{m-1}.
        s_tilde_score = s_tilde.detach()
        a_ij_score = a_ij.detach()
        wl_all = self.w_l(s_tilde_score)  # [B, K+1, d_m]
        wr_all = self.w_r(s_tilde_score)  # [B, K+1, d_m]
        h_merge = torch.tanh(wl_all[:, pi_idx, :] + wr_all[:, pj_idx, :] + self.w_edge_score(a_ij_score))
        m_score = self.w_merge(h_merge).squeeze(-1).float()  # [B, 72] fp32

        # Precision Island #3 (Red Line #1) + Branchless Slot-0 Pinning Mask (Spec 2 / Red Line #3)
        with fp32_island(c_fused.device):
            pin_penalty = 1e4 * pinned_cur.unsqueeze(-1) * self.pin_mask_72  # [B, 72] fp32
            m_score_f = m_score.float() - pin_penalty
            tau_eff = tau_m.float().clamp_min(1e-4)
            scaled_logits = m_score_f / tau_eff
            base_probs = F.softmax(scaled_logits, dim=-1)  # [B, 72] fp32
            eps_u = merge_uniform_mix.float().clamp(0.0, 0.99)
            valid_pair_mask = 1.0 - pinned_cur.unsqueeze(-1) * self.pin_mask_72  # [B, 72]
            uniform_p = valid_pair_mask / valid_pair_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            merge_probs = (1.0 - eps_u) * base_probs + eps_u * uniform_p

            if self.training:
                if use_gumbel:
                    u_rand = torch.rand_like(m_score_f)
                    gumbel = -torch.log(-torch.log(u_rand + 1e-7) + 1e-7)
                    unscaled_logits = torch.where(
                        eps_u > 0.0,
                        torch.log(merge_probs.clamp_min(1e-8)) - pin_penalty,
                        m_score_f,
                    )
                    y_soft = F.softmax((unscaled_logits + gumbel) / tau_eff, dim=-1)
                else:
                    y_soft = merge_probs
                win_idx = y_soft.argmax(dim=-1)  # [B]
                y_hard = F.one_hot(win_idx, num_classes=m_score_f.shape[-1]).float()
                alpha = (y_hard - y_soft).detach() + y_soft  # Gumbel-STE [B, 72] fp32
            else:
                win_idx = m_score_f.argmax(dim=-1)  # [B]
                alpha = F.one_hot(win_idx, num_classes=m_score_f.shape[-1]).float()

            # Single einsum contraction with prebuilt [72, K, K+1] perm_bank strictly in fp32
            p_mat = torch.einsum("bp,pka->bka", alpha, self.perm_bank.float())  # [B, K, K+1] fp32

        win_i = pi_idx[win_idx]  # [B]
        win_j = pj_idx[win_idx]  # [B]

        p_mat_s = p_mat.to(s_tilde.dtype)
        p_mat_a = p_mat.to(a_tilde.dtype)

        b_arange = torch.arange(B, device=c_fused.device)
        s_i_win = s_tilde[b_arange, win_i]  # [B, D]
        s_j_win = s_tilde[b_arange, win_j]  # [B, D]
        a_ij_win = a_tilde[b_arange, win_i, win_j]  # [B, d_r]
        g_win = torch.sigmoid(self.w_g(torch.cat([s_i_win, s_j_win, a_ij_win], dim=-1)))  # [B, D]
        comp_win = self.w_comp(torch.cat([s_i_win, s_j_win], dim=-1))  # [B, D]
        s_merged_win = self.norm_merge(g_win * comp_win + (1.0 - g_win) * s_i_win)  # [B, D]

        if self.training:
            alpha_win = alpha[b_arange, win_idx].to(s_tilde.dtype)  # [B], forward == 1.0
            corr_win = alpha_win.unsqueeze(-1) * (s_merged_win.to(s_tilde.dtype) - 0.5 * (s_i_win + s_j_win))
            win_u_oh = self.pair_u_onehot[win_idx].to(s_tilde.dtype)  # [B, K]
            slot_corr = win_u_oh.unsqueeze(-1) * corr_win.unsqueeze(1)  # [B, K, D]
            s_cand_new = (torch.einsum("bka,bad->bkd", p_mat_s, s_tilde) + slot_corr).to(slots.dtype)
        else:
            # 100% branchless, zero-ULP exact eval slot placement (compile-friendly!)
            win_u = pu_idx[win_idx]  # [B]
            win_v = self.pair_v[win_idx]  # [B]
            k_ids = torch.arange(K, device=c_fused.device).unsqueeze(0)  # [1, K]
            is_u = (k_ids == win_u.unsqueeze(1)).unsqueeze(-1)  # [B, K, 1]
            is_v = (k_ids == win_v.unsqueeze(1)).unsqueeze(-1)  # [B, K, 1] (False if win_v == K)
            s_cand_new = torch.where(is_u, s_merged_win.to(slots.dtype).unsqueeze(1), slots)
            s_cand_new = torch.where(is_v, z_star.to(slots.dtype).unsqueeze(1), s_cand_new)

        # Factored two-stage edge tensor contraction: P_m @ A_tilde @ P_m^T
        # `.contiguous()` preserves exact canonical [B, K, K, d_r] strides across steps (0 Dynamo recompiles!)
        a_tmp = torch.einsum("bka,bacr->bkcr", p_mat_a, a_tilde)
        a_cand_new = torch.einsum("blc,bkcr->bklr", p_mat_a, a_tmp).to(edges.dtype).contiguous()

        # 5. Exact LogSumExp Slot Salience Conservation Equation strictly in float32 (Precision Island #3)
        slot0_bool = self.slot0_onehot > 0.5  # [K]
        already_pinned_b = pinned_cur > 0.5  # [B]
        with fp32_island(c_fused.device):
            omega_old_decayed = salience.float() - self.cfg.delta_decay  # [B, K]
            omega_cand = (
                F.linear(z_star.float(), self.w_omega.weight.float()).squeeze(-1)
                + self.kappa_omega.float() * surprisal_0.float()
            )
            omega_tilde = torch.cat([omega_old_decayed, omega_cand.unsqueeze(-1)], dim=-1)  # [B, K+1]

            w_i = omega_tilde[:, pi_idx]  # [B, 72]
            w_j = omega_tilde[:, pj_idx]  # [B, 72]
            w_merged = torch.logaddexp(w_i, w_j)  # [B, 72]
            w_base = 0.5 * (w_i + w_j)
            w_corr_pairs = alpha * (w_merged - w_base)  # [B, 72]
            w_corr = torch.einsum("pk,bp->bk", self.pair_u_onehot.float(), w_corr_pairs)
            omega_cand_new = torch.einsum("bka,ba->bk", p_mat, omega_tilde) + w_corr

            omega_cand_new = torch.where(
                already_pinned_b.view(B, 1) & slot0_bool.view(1, K),
                torch.maximum(omega_cand_new, torch.full_like(omega_cand_new, 2.0)),
                omega_cand_new,
            )

        # Bit-exact static-shape Slot-0 invariance when already pinned (0.0 drift even in bf16!)
        s_cand_new = torch.where(already_pinned_b.view(B, 1, 1) & slot0_bool.view(1, K, 1), slots, s_cand_new)

        # Mask state updates strictly by active_mask (shields finished docs & padded virtual bucket slots!)
        am_s = active_mask.view(B, 1, 1)
        am_e = active_mask.view(B, 1, 1, 1)
        am_w = active_mask.view(B, 1)
        s_new = torch.where(am_s, s_cand_new, slots)
        a_new = torch.where(am_e, a_cand_new, edges)
        omega_new = torch.where(am_w, omega_cand_new, salience.float())

        # 6. Branchless Slot-0 Thesis Latch on SCRATCH_END (`\x1dE\n`) or first `\x1fA\n` (Spec 2)
        # S_{m,0} <- RMSNorm(g_m), omega_{m,0} <- max(omega_{m,0}, 2.0), slot0_pinned <- 1.0
        # Reuse `e_proj_post = tanh(W_e(a_new))` once across pre-latch g_post and post-latch s_bar_post!
        e_proj_post = torch.tanh(self.w_e(a_new))  # [B, K, K, D]
        _, g_post = self._message_pass_with_eproj(s_new, e_proj_post, omega_new)
        do_pin = active_mask.bool() & (trig_cur > 0.5) & (pinned_cur < 0.5)  # [B]
        g_post_f = g_post.float()
        thesis_vec = (
            g_post_f * torch.rsqrt(g_post_f.pow(2).mean(dim=-1, keepdim=True) + self.cfg.eps)
        ).to(s_new.dtype)  # [B, D]
        do_pin_s = do_pin.view(B, 1, 1) & slot0_bool.view(1, K, 1)
        do_pin_w = do_pin.view(B, 1) & slot0_bool.view(1, K)
        s_new = torch.where(do_pin_s, thesis_vec.unsqueeze(1).expand(-1, K, -1), s_new)
        omega_new = torch.where(do_pin_w, torch.maximum(omega_new, torch.full_like(omega_new, 2.0)), omega_new)
        slot0_pinned_next = torch.where(do_pin, torch.ones_like(pinned_cur), pinned_cur)

        # Update neutral concept center c_0 via EMA strictly over active docs
        if self.training and update_c0_ema:
            self.update_c0_from_slots(s_new, active_mask)

        s_bar_post, _ = self._message_pass_with_eproj(s_new, e_proj_post, omega_new)

        return (
            z_star,
            g_ctx,
            c_hat,
            pi_m,
            surprisal_0,
            ponder_steps,
            s_new,
            a_new,
            omega_new,
            s_bar_post,
            win_idx,
            win_i,
            win_j,
            merge_probs,
            alpha,
            slot0_pinned_next,
            moe_bal_loss,
        )

    def step(
        self,
        c_fused: torch.Tensor,
        prec_bu: torch.Tensor,
        ach: torch.Tensor,
        state: ConceptGraphState,
        tau_m: float = 1.0,
        use_gumbel: bool = True,
        update_c0_ema: bool = True,
        merge_uniform_mix: float = 0.0,
        active_mask: Optional[torch.Tensor] = None,
        pin_trigger: Optional[torch.Tensor] = None,
    ) -> BCGPCStepOutput:
        """Convenience dataclass wrapper around Hotspot A (`BCGPC.forward`)."""
        B = c_fused.shape[0]
        dev = c_fused.device
        am = (
            torch.ones((B,), device=dev, dtype=torch.bool)
            if active_mask is None
            else active_mask.bool()
        )
        pinned_in = (
            torch.zeros((B,), device=dev, dtype=torch.float32)
            if state.slot0_pinned is None
            else state.slot0_pinned.to(device=dev, dtype=torch.float32)
        )
        trig_in = (
            torch.zeros((B,), device=dev, dtype=torch.float32)
            if pin_trigger is None
            else pin_trigger.to(device=dev, dtype=torch.float32)
        )
        tau_t = torch.tensor(float(tau_m), device=dev, dtype=torch.float32)
        mix_t = torch.tensor(float(merge_uniform_mix), device=dev, dtype=torch.float32)
        (
            z_star,
            g_ctx,
            c_hat,
            pi_m,
            surprisal_0,
            ponder_steps,
            s_new,
            a_new,
            omega_new,
            s_bar_post,
            win_idx,
            win_i,
            win_j,
            merge_probs,
            alpha,
            slot0_pinned_next,
            moe_bal_loss,
        ) = self.forward(
            slots=state.slots,
            edges=state.edges,
            salience=state.salience,
            c_fused=c_fused,
            prec_bu=prec_bu,
            ach=ach,
            tau_m=tau_t,
            active_mask=am,
            merge_uniform_mix=mix_t,
            slot0_pinned=pinned_in,
            pin_trigger=trig_in,
            use_gumbel=use_gumbel,
            update_c0_ema=update_c0_ema,
        )
        return BCGPCStepOutput(
            z_star=z_star,
            g_ctx=g_ctx,
            c_hat=c_hat,
            pi_m=pi_m,
            surprisal=surprisal_0,
            ponder_steps=ponder_steps,
            state=ConceptGraphState(
                slots=s_new,
                edges=a_new,
                salience=omega_new,
                slot0_pinned=slot0_pinned_next,
            ),
            s_bar_post=s_bar_post,
            win_pair_idx=win_idx,
            win_i=win_i,
            win_j=win_j,
            merge_probs=merge_probs,
            merge_alpha=alpha,
            moe_bal_loss=moe_bal_loss,
        )
