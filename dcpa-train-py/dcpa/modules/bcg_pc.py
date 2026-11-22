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


@dataclass
class BCGPCStepOutput:
    z_star: torch.Tensor            # [B, D] disambiguated concept
    g_ctx: torch.Tensor             # [B, D] pre-step graph context readout
    c_hat: torch.Tensor             # [B, D] top-down prior prediction
    pi_m: torch.Tensor              # [B, D] float32 diagonal precision
    surprisal: torch.Tensor         # [B] float32 normalized residual E_m
    ponder_steps: torch.Tensor      # [B] float32 executed pondering steps (with STE surrogate)
    state: ConceptGraphState        # updated (S_m, A_m, omega_m)
    s_bar_post: torch.Tensor        # [B, K, D] post-merge message-passed slots (for Hippocampus q_{m,i})
    win_pair_idx: torch.Tensor      # [B] int64 winning ordered pair index in 0..71
    win_i: torch.Tensor             # [B] int64 i* of winning pair
    win_j: torch.Tensor             # [B] int64 j* of winning pair
    merge_probs: torch.Tensor       # [B, 72] float32 pre-hard softmax probabilities (for entropy monitor/KL)
    merge_alpha: torch.Tensor       # [B, 72] float32 hard one-hot forward STE merge selection


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N_active, D] -> [N_active, D] with zero GPU-CPU sync stalls on MPS/small batches."""
        if x.shape[0] == 0:
            return x
        # Pre-RMSNorm input to router + Layer-1 SwiGLU so deep pondering recurrence cannot diverge
        x = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        logits = self.router(x).float()  # [N, E]
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [N, top_k]
        gates = F.softmax(top_vals, dim=-1).to(x.dtype)  # [N, top_k]

        # On Apple MPS, avoid nonzero()/.any()/index_add_ GPU-CPU sync stalls by masking top-k
        # router weights directly over fixed-shape expert GEMMs (100% mathematically identical!).
        if x.device.type == "mps":
            # Build exact [N, E] sparse top-k gate matrix (strictly 0 for non-top-k experts!)
            gate_full = torch.zeros((x.shape[0], self.num_experts), device=x.device, dtype=x.dtype)
            gate_full.scatter_add_(1, top_idx, gates)
            out = torch.zeros_like(x)
            for e_id, expert in enumerate(self.experts):
                w_e = gate_full[:, e_id : e_id + 1]
                out = out + w_e * expert(x)
            return out

        out = torch.zeros_like(x)
        for e_id, expert in enumerate(self.experts):
            mask = top_idx == e_id  # [N, top_k]
            if not mask.any():
                continue
            sample_idx, k_pos = torch.nonzero(mask, as_tuple=True)
            y_e = expert(x[sample_idx])
            w_e = gates[sample_idx, k_pos].unsqueeze(-1)
            out.index_add_(0, sample_idx, y_e * w_e)
        return out


class SlotCrossAttention(nn.Module):
    """Lightweight cross-attention from single concept query z_m^{(l)} to fixed K=8 slots S_bar."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, q_vec: torch.Tensor, kv_slots: torch.Tensor) -> torch.Tensor:
        """q_vec: [N, D], kv_slots: [N, K, D] -> [N, D]."""
        q = self.w_q(q_vec).float()  # [N, D]
        k = self.w_k(kv_slots).float()  # [N, K, D]
        v = self.w_v(kv_slots)  # [N, K, D]
        scores = torch.einsum("nd,nkd->nk", q, k) * self.scale  # [N, K] fp32
        attn = F.softmax(scores, dim=-1).to(v.dtype)
        ctx = torch.einsum("nk,nkd->nd", attn, v)
        return self.w_o(ctx)


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

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize f_prec bias so Softplus(bias) ~ 1.0 and w_gen near identity so pondering
        # residual (c_fused - W_gen z) measures true reconstruction error from Step 0.
        nn.init.constant_(self.f_prec_2.bias, 0.5413)
        nn.init.eye_(self.w_gen.weight)
        nn.init.xavier_uniform_(self.w_fast.weight)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> ConceptGraphState:
        dev = device if device is not None else self.s0.device
        dt = dtype if dtype is not None else self.s0.dtype
        slots = self.s0.to(device=dev, dtype=dt).unsqueeze(0).expand(batch_size, -1, -1).clone()
        edges = self.a0.to(device=dev, dtype=dt).unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        salience = self.omega0.float().to(device=dev).unsqueeze(0).expand(batch_size, -1).clone()
        return ConceptGraphState(slots=slots, edges=edges, salience=salience)

    def message_pass(self, slots: torch.Tensor, edges: torch.Tensor, salience: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute edge-aggregated node states S_bar [B, K, D] and context readout g [B, D]."""
        w_attn = F.softmax(salience.float(), dim=-1)  # [B, K] fp32
        e_proj = torch.tanh(self.w_e(edges))  # [B, K, K, D], bounded in (-1, 1) for stable multi-step recurrence
        # S_bar_i = S_i + sum_j w_attn[j] * (tanh(W_e A_{i,j}) * S_j)
        weighted_s = slots * w_attn.unsqueeze(-1).to(slots.dtype)  # [B, K, D]
        agg = torch.einsum("bijd,bjd->bid", e_proj, weighted_s)
        s_bar = slots + agg
        g_ctx = torch.einsum("bk,bkd->bd", w_attn.to(s_bar.dtype), s_bar)
        return s_bar, g_ctx

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
    ) -> BCGPCStepOutput:
        """Execute one concept-step closed loop (Prior -> Masked Pondering -> K+1 -> K Merge).

        Args:
          c_fused: [B, D] observation concept (cortical/hippocampal fused).
          prec_bu: [B, D] float32 bottom-up sensory precision pi_m^{bu}.
          ach:     [B] float32 acetylcholine novelty signal in [0, 1].
          state:   ConceptGraphState (S_{m-1}, A_{m-1}, omega_{m-1}).
          tau_m:   Gumbel-Softmax temperature (annealed 1.2 -> 0.25 in Stage B).
          use_gumbel: add Gumbel noise during training.
        """
        B, D = c_fused.shape
        K = self.cfg.k_slots
        d_r = self.cfg.d_r
        orig_dtype = c_fused.dtype

        # 1. Top-down prior prediction & smoothly bounded precision-weighted residual
        s_bar_prev, g_ctx = self.message_pass(state.slots, state.edges, state.salience)
        c_hat = self.f_pred_2(F.silu(self.f_pred_1(g_ctx)))  # [B, D]
        pi_raw = F.softplus(self.f_prec_2(F.silu(self.f_prec_1(g_ctx))).float()) + 1e-4  # [B, D] fp32
        pi_m = 2.0 * torch.tanh(pi_raw * 0.5)  # smooth precision cap in (1e-4, 2.0) prevents prior domination/noise blowup

        ach_f32 = ach.float().clamp(0.0, 1.0)
        prec_bu_f32 = prec_bu.float().clamp_min(0.0)
        pi_eff = pi_m * (1.0 - ach_f32).unsqueeze(-1)  # suppress noise residual on first-seen novel words

        eps_0 = (pi_eff * (c_fused.float() - c_hat.float())).to(orig_dtype)
        surprisal_0 = eps_0.float().norm(dim=-1) / math.sqrt(D)  # [B] fp32

        # Bayesian precision fusion (Theory Doc Page 11) + fast shortcut
        z_cur = (
            (pi_m * c_hat.float() + prec_bu_f32 * c_fused.float())
            / (pi_m + prec_bu_f32 + self.cfg.eps)
        ).to(orig_dtype)
        z_cur = z_cur + (1.0 - ach_f32).unsqueeze(-1).to(orig_dtype) * self.w_fast(eps_0)

        # 2. Masked Unrolled Pondering (L_max = 4, zero per-sample dynamic branching)
        eps_cur = eps_0
        surp_cur = surprisal_0
        active = torch.ones((B,), device=c_fused.device, dtype=torch.bool)
        ponder_steps = torch.zeros((B,), device=c_fused.device, dtype=torch.float32)

        for l_idx in range(self.cfg.l_max):
            thresh = self.cfg.tau_skip if l_idx == 0 else self.cfg.tau_halt
            still_surprised = surp_cur > thresh
            active = active & still_surprised

            # Differentiable STE ponder counter so L_ponder pushes gradients to reduce iterations
            soft_act = torch.sigmoid((surp_cur - thresh) / 0.1)
            hard_act = active.float()
            ste_act = (hard_act - soft_act).detach() + soft_act
            ponder_steps = ponder_steps + ste_act

            if active.any():
                if c_fused.device.type == "mps":
                    # Avoid MPS index_put_ backward CPU fallback while still skipping when ~active.any()!
                    r_all = self.cross_attn(z_cur, s_bar_prev)
                    delta_all = self.moe(eps_cur + r_all)
                    z_cand = self.norm_ponder(z_cur + self.cfg.ponder_eta * delta_all)
                    act_col = active.unsqueeze(-1)
                    z_cur = torch.where(act_col, z_cand, z_cur)

                    eps_cand = (
                        pi_eff * (c_fused.float() - self.w_gen(z_cand).float())
                    ).to(orig_dtype)
                    eps_cur = torch.where(act_col, eps_cand, eps_cur)
                    surp_cand = eps_cand.float().norm(dim=-1) / math.sqrt(D)
                    surp_cur = torch.where(active, surp_cand, surp_cur)
                else:
                    act_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
                    z_a = z_cur[act_idx]
                    eps_a = eps_cur[act_idx]
                    s_bar_a = s_bar_prev[act_idx]

                    r_a = self.cross_attn(z_a, s_bar_a)
                    delta_a = self.moe(eps_a + r_a)
                    z_next_a = self.norm_ponder(z_a + self.cfg.ponder_eta * delta_a)

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
                    surp_next[act_idx] = eps_next_a.float().norm(dim=-1) / math.sqrt(D)
                    surp_cur = surp_next

        z_star = z_cur  # [B, D]

        # 3. Candidate slot expansion (K -> K+1)
        s_tilde = torch.cat([state.slots, z_star.unsqueeze(1)], dim=1)  # [B, K+1, D]
        src_all = self.u_src(s_tilde)  # [B, K+1, d_r]
        dst_all = self.u_dst(s_tilde)  # [B, K+1, d_r]

        # New edges for i == K or k == K; preserve existing K x K edges verbatim from state.edges.
        # Bounded via tanh so bilinear products src * dst stay in (-1, 1) and never diverge across steps.
        new_row = torch.tanh(src_all[:, K : K + 1, :] * dst_all)  # [B, K+1, d_r] (from K to 0..K)
        new_col = torch.tanh(src_all[:, :K, :] * dst_all[:, K : K + 1, :])  # [B, K, d_r] (from 0..K-1 to K)
        top_part = torch.cat([state.edges, new_col.unsqueeze(2)], dim=2)  # [B, K, K+1, d_r]
        a_tilde = torch.cat([top_part, new_row.unsqueeze(1)], dim=1)  # [B, K+1, K+1, d_r]

        # 4. K+1 -> K Syntactic Merge Bottleneck over all P = K*(K+1) = 72 ordered pairs
        pi_idx = self.pair_i  # [72]
        pj_idx = self.pair_j  # [72]
        pu_idx = self.pair_u  # [72]

        a_ij = a_tilde[:, pi_idx, pj_idx, :]  # [B, 72, d_r]

        # Exact 9-slot factorization of W_L s_i + W_R s_j (8x fewer FLOPs than 72-pair matmul!)
        wl_all = self.w_l(s_tilde)  # [B, K+1, d_m]
        wr_all = self.w_r(s_tilde)  # [B, K+1, d_m]
        h_merge = torch.tanh(wl_all[:, pi_idx, :] + wr_all[:, pj_idx, :] + self.w_edge_score(a_ij))
        m_score = self.w_merge(h_merge).squeeze(-1).float()  # [B, 72] fp32

        # Precision Island #3 (Red Line #1): Gumbel-Softmax -log(-log(U + 1e-7) + 1e-7)
        # and exact merge probability / exploration mixing strictly in float32!
        with fp32_island(c_fused.device):
            m_score_f = m_score.float()
            merge_probs = F.softmax(m_score_f / max(tau_m, 1e-4), dim=-1)  # [B, 72] fp32
            if merge_uniform_mix > 0.0:
                eps_u = min(max(float(merge_uniform_mix), 0.0), 0.99)
                uniform_p = 1.0 / float(m_score_f.shape[-1])
                merge_probs = (1.0 - eps_u) * merge_probs + eps_u * uniform_p

            if self.training:
                if use_gumbel:
                    # Temperature-annealed Gumbel-Max with explicit Red Line #1 numerical guard:
                    # -log(-log(U + 1e-7) + 1e-7) strictly in float32.
                    u_rand = torch.rand_like(m_score_f)
                    gumbel = -torch.log(-torch.log(u_rand + 1e-7) + 1e-7)
                    if merge_uniform_mix > 0.0:
                        logits_eff = torch.log(merge_probs.clamp_min(1e-8))
                    else:
                        logits_eff = m_score_f / max(tau_m, 1e-4)
                    y_soft = F.softmax(logits_eff + gumbel, dim=-1)
                else:
                    y_soft = merge_probs
                win_idx = y_soft.argmax(dim=-1)  # [B]
                y_hard = F.one_hot(win_idx, num_classes=m_score_f.shape[-1]).float()
                alpha = (y_hard - y_soft).detach() + y_soft  # Gumbel-STE [B, 72] fp32
            else:
                win_idx = m_score_f.argmax(dim=-1)  # [B]
                alpha = F.one_hot(win_idx, num_classes=m_score_f.shape[-1]).float()

        win_i = pi_idx[win_idx]  # [B]
        win_j = pj_idx[win_idx]  # [B]

        # Single einsum contraction with prebuilt [72, K, K+1] perm_bank (zero loops)
        p_mat = torch.einsum("bp,pka->bka", alpha, self.perm_bank)  # [B, K, K+1] fp32
        p_mat_dt = p_mat.to(orig_dtype)

        # Parent node composition & exact in-place slot update (O(B*D) winning-pair composition +
        # full 72-way Gumbel-STE P_m contraction: bit-exact forward, bit-exact W_g/W_comp gradients,
        # and 72x lower autograd activation memory vs materializing [B, 72, D]!)
        b_arange = torch.arange(B, device=c_fused.device)
        s_i_win = s_tilde[b_arange, win_i]  # [B, D]
        s_j_win = s_tilde[b_arange, win_j]  # [B, D]
        a_ij_win = a_tilde[b_arange, win_i, win_j]  # [B, d_r]
        g_win = torch.sigmoid(self.w_g(torch.cat([s_i_win, s_j_win, a_ij_win], dim=-1)))  # [B, D]
        comp_win = self.w_comp(torch.cat([s_i_win, s_j_win], dim=-1))  # [B, D]
        s_merged_win = self.norm_merge(g_win * comp_win + (1.0 - g_win) * s_i_win)  # [B, D]

        if self.training:
            # (P^{(i*,j*)} S_tilde)_{u*} == 0.5 * (s_i_win + s_j_win); add STE-weighted nonlinear residual
            # exclusively at u* = min(i*, j*) while P_m passes full 72-way Gumbel-STE gradients to M_score.
            alpha_win = alpha[b_arange, win_idx].to(orig_dtype)  # [B], forward value == 1.0
            corr_win = alpha_win.unsqueeze(-1) * (s_merged_win - 0.5 * (s_i_win + s_j_win))  # [B, D]
            win_u_oh = self.pair_u_onehot[win_idx].to(orig_dtype)  # [B, K]
            slot_corr = win_u_oh.unsqueeze(-1) * corr_win.unsqueeze(1)  # [B, K, D]
            s_new = torch.einsum("bka,bad->bkd", p_mat_dt, s_tilde) + slot_corr
        else:
            win_u = pu_idx[win_idx]
            win_v = self.pair_v[win_idx]
            s_new = state.slots.clone()
            s_new[b_arange, win_u] = s_merged_win
            case_b = win_v < K
            if case_b.any():
                b_cb = b_arange[case_b]
                s_new[b_cb, win_v[case_b]] = z_star[b_cb]

        # Factored two-stage edge tensor contraction: P_m @ A_tilde @ P_m^T (9x smaller intermediates!)
        a_tmp = torch.einsum("bka,bacr->bkcr", p_mat_dt, a_tilde)
        a_new = torch.einsum("blc,bkcr->bklr", p_mat_dt, a_tmp)

        # 5. Exact LogSumExp Slot Salience Conservation Equation in float32 (Engineering Doc 2.3)
        omega_old_decayed = state.salience.float() - self.cfg.delta_decay  # [B, K]
        omega_cand = self.w_omega(z_star).squeeze(-1).float() + self.kappa_omega.float() * surprisal_0  # [B]
        omega_tilde = torch.cat([omega_old_decayed, omega_cand.unsqueeze(-1)], dim=-1)  # [B, K+1]

        w_i = omega_tilde[:, pi_idx]  # [B, 72]
        w_j = omega_tilde[:, pj_idx]  # [B, 72]
        w_merged = torch.logaddexp(w_i, w_j)  # [B, 72]
        w_base = 0.5 * (w_i + w_j)
        w_corr_pairs = alpha * (w_merged - w_base)  # [B, 72]
        w_corr = torch.einsum("pk,bp->bk", self.pair_u_onehot, w_corr_pairs)
        omega_new = torch.einsum("bka,ba->bk", p_mat, omega_tilde) + w_corr

        # Update neutral concept center c_0 via EMA during training
        if self.training and update_c0_ema:
            with torch.no_grad():
                batch_mean = s_new.detach().float().mean(dim=(0, 1))
                m = self.cfg.c0_ema_momentum
                self.c0.mul_(m).add_(batch_mean * (1.0 - m))

        new_state = ConceptGraphState(slots=s_new, edges=a_new, salience=omega_new)
        s_bar_post, _ = self.message_pass(s_new, a_new, omega_new)

        return BCGPCStepOutput(
            z_star=z_star,
            g_ctx=g_ctx,
            c_hat=c_hat,
            pi_m=pi_m,
            surprisal=surprisal_0,
            ponder_steps=ponder_steps,
            state=new_state,
            s_bar_post=s_bar_post,
            win_pair_idx=win_idx,
            win_i=win_i,
            win_j=win_j,
            merge_probs=merge_probs,
            merge_alpha=alpha,
        )
