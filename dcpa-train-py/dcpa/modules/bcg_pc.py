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
from typing import Any, Callable, Optional

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
    phase: Optional[torch.Tensor] = None         # [B, K] float32 Slot Temporal Phase / logical age


@dataclass
class ClauseAccumulatorState:
    """Strictly O(1) online stabilized surprisal-weighted proposition accumulator + argmax-ACh tag."""

    num: torch.Tensor       # [B, D] float32 weighted sum of z_0 * exp(surp - max_s)
    den: torch.Tensor       # [B] float32 sum of exp(surp - max_s)
    max_surp: torch.Tensor  # [B] float32 running max surprisal stabilizer
    sum_surp: torch.Tensor  # [B] float32 running sum of surprisal (for mean clause surprisal)
    count: torch.Tensor     # [B] int32 number of concepts accumulated in current clause
    best_ach: torch.Tensor  # [B] float32 max ACh seen in current clause (Commandment #8)
    best_idx: torch.Tensor  # [B, s] int64 sparse DG key of highest-ACh concept in clause
    best_val: torch.Tensor  # [B, s] float32 unit-norm sparse values of highest-ACh concept
    best_f: torch.Tensor    # [B, d_s] stem vector of highest-ACh concept

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        d_concept: int,
        s_active: int,
        d_s: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "ClauseAccumulatorState":
        return cls(
            num=torch.zeros((batch_size, d_concept), device=device, dtype=torch.float32),
            den=torch.zeros((batch_size,), device=device, dtype=torch.float32),
            max_surp=torch.full((batch_size,), -1.0e9, device=device, dtype=torch.float32),
            sum_surp=torch.zeros((batch_size,), device=device, dtype=torch.float32),
            count=torch.zeros((batch_size,), device=device, dtype=torch.int32),
            best_ach=torch.full((batch_size,), -1.0, device=device, dtype=torch.float32),
            best_idx=torch.zeros((batch_size, s_active), device=device, dtype=torch.int64),
            best_val=torch.zeros((batch_size, s_active), device=device, dtype=torch.float32),
            best_f=torch.zeros((batch_size, d_s), device=device, dtype=dtype),
        )


class _DampedBoundedCarry(torch.autograd.Function):
    """Exact bit-identical forward carry with spectral-safe per-sample backward RMS clamping.

    Forward: exact identity `return x` (0.0 ULP drift, preserves Slot-0 pinning & Rust parity).
    Backward: scales incoming recurrent carry gradient by `gamma` and clamps per-sample RMS to
    `max_rms` in float32 — mathematically preventing exponential Jacobian explosion (grad_norm=inf)
    across arbitrary document lengths (512B -> 4096B / 50 -> 430 concept steps)!
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: float, max_rms: float) -> torch.Tensor:
        ctx.gamma = float(gamma)
        ctx.max_rms = float(max_rms)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        g = grad_out.float() * ctx.gamma
        if g.ndim > 1:
            reduce_dims = tuple(range(1, g.ndim))
            g_max = g.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
            g_scaled = g / g_max
            rms = g_max * torch.sqrt(g_scaled.pow(2).mean(dim=reduce_dims, keepdim=True) + 1e-12)
        else:
            rms = g.abs().clamp_min(1e-12)
        scale = torch.clamp(ctx.max_rms / rms.clamp_min(1e-12), max=1.0)
        return (g * scale).to(grad_out.dtype), None, None


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


def _coarse_bucket_count(n: int, bucket: int = 16) -> int:
    """Round active sample/expert count `n >= 1` up to a bounded set of Tensor-Core-aligned buckets."""
    b = max(1, int(bucket))
    n_pos = max(1, int(n))
    if n_pos <= b:
        return b
    if n_pos <= b * 2:
        return b * 2
    if n_pos <= b * 16:
        step = b * 2
        return ((n_pos + step - 1) // step) * step
    if n_pos <= b * 32:
        step = b * 4
        return ((n_pos + step - 1) // step) * step
    step = b * 8
    return ((n_pos + step - 1) // step) * step


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


_CUDA_SM80_CACHE: dict[int, bool] = {}


def _device_supports_grouped_mm(device: torch.device, allow_cpu_bf16: bool = False) -> bool:
    """Return True if `F.grouped_mm` BF16 grouped GEMM is supported on `device` (SM >= 80 on CUDA)."""
    if not hasattr(F, "grouped_mm"):
        return False
    if device.type == "cuda":
        idx = 0 if device.index is None else int(device.index)
        cached = _CUDA_SM80_CACHE.get(idx)
        if cached is None:
            try:
                major, _ = torch.cuda.get_device_capability(idx)
                cached = bool(major >= 8)
            except Exception:
                cached = False
            _CUDA_SM80_CACHE[idx] = cached
        return cached
    if device.type == "cpu" and allow_cpu_bf16:
        return True
    return False


class SparseMicroMoE(nn.Module):
    """Top-2 sparse micro-experts dispatched strictly on active high-surprisal samples.

    Stores all `E` 2-layer SwiGLU experts natively as 4 contiguous 3D parameter tensors
    (`w_gu1 [E, 2*H, D]`, `w_d1 [E, D, H]`, `w_gu2 [E, 2*H, D]`, `w_d2 [E, D, H]`) so
    `pack_weights()` returns zero-copy transposed views for `F.grouped_mm` / `torch.bmm`
    with zero per-batch stack/cat allocations. Seamlessly loads legacy per-expert
    `experts.{e}.w_*` state_dicts via `_load_from_state_dict`.
    """

    def __init__(self, d_model: int, hidden_dim: int, num_experts: int, top_k: int = 2, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.allow_cpu_grouped_mm: bool = False
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.w_gu1 = nn.Parameter(torch.empty((num_experts, 2 * hidden_dim, d_model)))
        self.w_d1 = nn.Parameter(torch.empty((num_experts, d_model, hidden_dim)))
        self.w_gu2 = nn.Parameter(torch.empty((num_experts, 2 * hidden_dim, d_model)))
        self.w_d2 = nn.Parameter(torch.empty((num_experts, d_model, hidden_dim)))
        self._init_expert_weights()

    def _init_expert_weights(self) -> None:
        h = self.hidden_dim
        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w_gu1.data[e, :h], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_gu1.data[e, h:], a=math.sqrt(5))
            nn.init.normal_(self.w_d1.data[e], std=0.02)
            nn.init.kaiming_uniform_(self.w_gu2.data[e, :h], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_gu2.data[e, h:], a=math.sqrt(5))
            nn.init.normal_(self.w_d2.data[e], std=0.02)

    def expert_parameters(self) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        return (self.w_gu1, self.w_d1, self.w_gu2, self.w_d2)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Migrate legacy per-expert `experts.{e}.w_*` keys transparently into native 3D tensors."""
        gu1_k = f"{prefix}w_gu1"
        legacy_probe = f"{prefix}experts.0.w_gate1.weight"
        if gu1_k not in state_dict and legacy_probe in state_dict:
            g1_list: list[torch.Tensor] = []
            u1_list: list[torch.Tensor] = []
            d1_list: list[torch.Tensor] = []
            g2_list: list[torch.Tensor] = []
            u2_list: list[torch.Tensor] = []
            d2_list: list[torch.Tensor] = []
            ok = True
            for e in range(self.num_experts):
                ep = f"{prefix}experts.{e}."
                ks = (
                    f"{ep}w_gate1.weight",
                    f"{ep}w_up1.weight",
                    f"{ep}w_down1.weight",
                    f"{ep}w_gate2.weight",
                    f"{ep}w_up2.weight",
                    f"{ep}w_down2.weight",
                )
                if not all(k in state_dict for k in ks):
                    ok = False
                    break
                g1_list.append(state_dict.pop(ks[0]))
                u1_list.append(state_dict.pop(ks[1]))
                d1_list.append(state_dict.pop(ks[2]))
                g2_list.append(state_dict.pop(ks[3]))
                u2_list.append(state_dict.pop(ks[4]))
                d2_list.append(state_dict.pop(ks[5]))
            if ok:
                state_dict[f"{prefix}w_gu1"] = torch.stack(
                    [torch.cat([g, u], dim=0) for g, u in zip(g1_list, u1_list)], dim=0
                ).contiguous()
                state_dict[f"{prefix}w_d1"] = torch.stack(d1_list, dim=0).contiguous()
                state_dict[f"{prefix}w_gu2"] = torch.stack(
                    [torch.cat([g, u], dim=0) for g, u in zip(g2_list, u2_list)], dim=0
                ).contiguous()
                state_dict[f"{prefix}w_d2"] = torch.stack(d2_list, dim=0).contiguous()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def pack_weights(
        self, dtype: Optional[torch.dtype] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return column-major transposed 3D views `(w_gu1 [E, D, 2*H], w_d1 [E, H, D], w_gu2, w_d2)`.

        Zero-copy when `dtype` matches master weights; single contiguous cast under BF16 autocast.
        Directly compatible with BOTH `F.grouped_mm` (SM >= 80 Tensor Cores) AND `torch.bmm`.
        """
        if dtype is not None and self.w_gu1.dtype != dtype:
            return (
                self.w_gu1.to(dtype=dtype).transpose(1, 2),
                self.w_d1.to(dtype=dtype).transpose(1, 2),
                self.w_gu2.to(dtype=dtype).transpose(1, 2),
                self.w_d2.to(dtype=dtype).transpose(1, 2),
            )
        return (
            self.w_gu1.transpose(1, 2),
            self.w_d1.transpose(1, 2),
            self.w_gu2.transpose(1, 2),
            self.w_d2.transpose(1, 2),
        )

    def _experts_4grouped_mm_impl(
        self,
        x_sorted: torch.Tensor,
        offs: torch.Tensor,
        w_gu1: torch.Tensor,
        w_d1: torch.Tensor,
        w_gu2: torch.Tensor,
        w_d2: torch.Tensor,
        active_row_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Zero-sync, zero-drop jagged Top-K SwiGLU expert kernel via `F.grouped_mm` on `[N*top_k, D]`.

        Executes ONLY active `n_active * top_k` rows (skipping skipped/halted/padded tokens and
        0-token experts completely!) with 0 CPU syncs. When `active_row_mask` (`[M, 1]` bool) is
        provided, cleanly masks trailing inactive rows past `offs[-1]` in BOTH forward and backward
        so uninitialized tail buffers in `F.grouped_mm` (especially when `offs[-1] == 0`) can NEVER
        leak NaN or garbage values!
        """
        def _gmm(inp: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
            if active_row_mask is not None:
                inp_m = torch.where(active_row_mask, inp, torch.zeros_like(inp))
                out_raw = F.grouped_mm(inp_m, wt, offs=offs)
                return torch.where(active_row_mask, out_raw, torch.zeros_like(out_raw))
            return F.grouped_mm(inp, wt, offs=offs)

        gu1 = _gmm(x_sorted, w_gu1)
        g1, u1 = gu1.chunk(2, dim=-1)
        act1 = (F.silu(g1) * u1).contiguous()
        x1 = x_sorted + _gmm(act1, w_d1)
        x_n2 = (
            x1.float() * torch.rsqrt(x1.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        ).to(x_sorted.dtype).contiguous()
        gu2 = _gmm(x_n2, w_gu2)
        g2, u2 = gu2.chunk(2, dim=-1)
        act2 = (F.silu(g2) * u2).contiguous()
        return x1 + _gmm(act2, w_d2)

    def _experts_4bmm_impl(
        self,
        x_e: torch.Tensor,
        w_gu1: torch.Tensor,
        w_d1: torch.Tensor,
        w_gu2: torch.Tensor,
        w_d2: torch.Tensor,
    ) -> torch.Tensor:
        """Pure static-shape 4-BMM 2-layer Pre-RMSNorm SwiGLU expert kernel on `[E, C, D]`."""
        g1, u1 = torch.bmm(x_e, w_gu1).chunk(2, dim=-1)  # [E, C, H]
        x1 = x_e + torch.bmm(F.silu(g1) * u1, w_d1)  # [E, C, D]
        x_n2 = (x1.float() * torch.rsqrt(x1.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x_e.dtype)
        g2, u2 = torch.bmm(x_n2, w_gu2).chunk(2, dim=-1)  # [E, C, H]
        return x1 + torch.bmm(F.silu(g2) * u2, w_d2)  # [E, C, D]

    def _resolve_packed_weights(
        self,
        dtype: torch.dtype,
        packed_weights: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if packed_weights is None:
            return self.pack_weights(dtype=dtype)
        w_gu1, w_d1, w_gu2, w_d2 = packed_weights
        if w_gu1.dtype != dtype:
            return (
                w_gu1.transpose(1, 2).to(dtype).contiguous().transpose(1, 2),
                w_d1.transpose(1, 2).to(dtype).contiguous().transpose(1, 2),
                w_gu2.transpose(1, 2).to(dtype).contiguous().transpose(1, 2),
                w_d2.transpose(1, 2).to(dtype).contiguous().transpose(1, 2),
            )
        return w_gu1, w_d1, w_gu2, w_d2

    def forward_compacted_bmm(
        self,
        x: torch.Tensor,
        n_valid: int,
        sample_mask: Optional[torch.Tensor] = None,
        packed_weights: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        compiled_experts_fn: Optional[Callable[..., torch.Tensor]] = None,
        compiled_grouped_mm_fn: Optional[Callable[..., torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lossless capacity-compacted Top-2 MoE dispatch on bucketed active tokens `[N_pad, D]`.

        - On SM >= 80 CUDA in BF16 (A100/H100/Blackwell): dispatches zero-sync jagged `F.grouped_mm`
          on ONLY `n_valid * top_k` rows (zero padding waste, zero `.item()` GPU->CPU syncs!).
        - Fallback (FP32 / MPS / older GPUs): shrinks `[E, N_pad, D] -> [E, C_cap, D]` via 4-BMM.
        """
        N, D = x.shape
        if N == 0 or n_valid <= 0:
            return torch.zeros_like(x), torch.zeros((), device=x.device, dtype=torch.float32)

        E = self.num_experts
        top_k = self.top_k
        x_norm = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        logits = self.router(x_norm).float()  # [N, E]
        router_probs = F.softmax(logits, dim=-1)  # [N, E] fp32
        top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)  # [N, top_k]
        gates_f32 = F.softmax(top_vals, dim=-1)  # [N, top_k] fp32
        gates = gates_f32.to(x.dtype)  # [N, top_k]

        if sample_mask is None:
            mask_f = torch.ones((N, 1), device=x.device, dtype=torch.float32)
        else:
            mask_f = sample_mask.float().view(N, 1)
        m_sum = mask_f.sum()
        denom = m_sum.clamp_min(1.0)
        has_any = (m_sum > 0.0).float()
        p_e = (router_probs * mask_f).sum(dim=0) / denom  # [E]
        dispatch_oh = F.one_hot(top_idx, num_classes=E).float().sum(dim=1) / float(top_k)
        f_e = (dispatch_oh * mask_f).sum(dim=0) / denom  # [E]
        aux_bal = has_any * float(E) * (f_e.detach() * p_e).sum()

        w_gu1, w_d1, w_gu2, w_d2 = self._resolve_packed_weights(x.dtype, packed_weights)

        if (
            x.dtype == torch.bfloat16
            and E >= 4
            and E > top_k
            and _device_supports_grouped_mm(x.device, allow_cpu_bf16=self.allow_cpu_grouped_mm)
        ):
            valid_top_idx = top_idx[:n_valid]
            flat_e = valid_top_idx.reshape(-1)
            perm = torch.argsort(flat_e, stable=True)
            e_sorted = flat_e[perm]
            tok_sorted = torch.div(perm, top_k, rounding_mode="floor")
            x_sorted = x_norm[:n_valid][tok_sorted].contiguous()
            counts = torch.zeros((E,), device=x.device, dtype=torch.int32).scatter_add(
                0, e_sorted, torch.ones_like(e_sorted, dtype=torch.int32)
            )
            offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
            gmm_fn = compiled_grouped_mm_fn if compiled_grouped_mm_fn is not None else self._experts_4grouped_mm_impl
            y_sorted = gmm_fn(x_sorted, offs, w_gu1, w_d1, w_gu2, w_d2)
            gate_sorted = gates_f32[:n_valid].reshape(-1)[perm].unsqueeze(-1)
            y_weighted = y_sorted.float() * gate_sorted
            out_valid = (
                torch.zeros((n_valid, D), device=x.device, dtype=torch.float32)
                .index_add(0, tok_sorted, y_weighted)
                .to(x.dtype)
            )
            if n_valid < N:
                return torch.cat([out_valid, torch.zeros((N - n_valid, D), device=x.device, dtype=x.dtype)], dim=0), aux_bal
            return out_valid, aux_bal

        experts_fn = compiled_experts_fn if compiled_experts_fn is not None else self._experts_4bmm_impl

        # Small active batches on <=8 experts fit in L2 cache directly without bincount/argsort overhead
        if N <= 48 and E <= 8:
            x_e = x_norm[:n_valid].unsqueeze(0).expand(E, -1, -1)
            gate_full = torch.zeros((n_valid, E), device=x.device, dtype=torch.float32).scatter_add(
                1, top_idx[:n_valid], gates_f32[:n_valid]
            )
            y_all = experts_fn(x_e, w_gu1, w_d1, w_gu2, w_d2)
            out_valid = torch.einsum("ne,end->nd", gate_full, y_all.float()).to(x.dtype)
            if n_valid < N:
                return torch.cat([out_valid, torch.zeros((N - n_valid, D), device=x.device, dtype=x.dtype)], dim=0), aux_bal
            return out_valid, aux_bal

        valid_top_idx = top_idx[:n_valid]  # [n_valid, top_k]
        counts = torch.bincount(valid_top_idx.reshape(-1), minlength=E)  # [E]
        c_cap = _coarse_bucket_count(max(1, int(counts.max().item())), bucket=16)

        if c_cap >= n_valid:
            x_e = x_norm[:n_valid].unsqueeze(0).expand(E, -1, -1)
            gate_full = torch.zeros((n_valid, E), device=x.device, dtype=torch.float32).scatter_add(
                1, valid_top_idx, gates_f32[:n_valid]
            )
            y_all = experts_fn(x_e, w_gu1, w_d1, w_gu2, w_d2)
            out_valid = torch.einsum("ne,end->nd", gate_full, y_all.float()).to(x.dtype)
            if n_valid < N:
                return torch.cat([out_valid, torch.zeros((N - n_valid, D), device=x.device, dtype=x.dtype)], dim=0), aux_bal
            return out_valid, aux_bal

        M = n_valid * top_k
        flat_e = valid_top_idx.reshape(-1)
        flat_tok = torch.arange(n_valid, device=x.device, dtype=torch.int64).unsqueeze(1).expand(n_valid, top_k).reshape(-1)
        flat_gate = gates_f32[:n_valid].reshape(-1)
        perm = torch.argsort(flat_e, stable=True)
        e_sorted = flat_e[perm]
        tok_sorted = flat_tok[perm]
        gate_sorted = flat_gate[perm]

        cu_e = torch.zeros((E + 1,), device=x.device, dtype=torch.int64)
        cu_e[1:] = torch.cumsum(counts, dim=0)
        slot_in_e = torch.arange(M, device=x.device, dtype=torch.int64) - cu_e[e_sorted]
        linear_pos = e_sorted * c_cap + slot_in_e

        x_pack = torch.zeros((E * c_cap, D), device=x.device, dtype=x.dtype)
        x_pack[linear_pos] = x_norm[tok_sorted]
        y_e = experts_fn(x_pack.view(E, c_cap, D), w_gu1, w_d1, w_gu2, w_d2)

        y_sorted = y_e.reshape(E * c_cap, D)[linear_pos].float() * gate_sorted.unsqueeze(-1)
        out_valid = (
            torch.zeros((n_valid, D), device=x.device, dtype=torch.float32)
            .index_add(0, tok_sorted, y_sorted)
            .to(x.dtype)
        )
        if n_valid < N:
            return torch.cat([out_valid, torch.zeros((N - n_valid, D), device=x.device, dtype=x.dtype)], dim=0), aux_bal
        return out_valid, aux_bal

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
        if not self.training and x.dtype == torch.float32 and not static_shape:
            logits = torch.bmm(
                x.unsqueeze(1), self.router.weight.float().t().unsqueeze(0).expand(x.shape[0], -1, -1)
            ).squeeze(1)
        else:
            logits = self.router(x).float()  # [N, E]
        router_probs = F.softmax(logits, dim=-1)  # [N, E] fp32
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [N, top_k]
        gates_f32 = F.softmax(top_vals, dim=-1)  # [N, top_k] fp32
        gates = gates_f32.to(x.dtype)  # [N, top_k]

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

        # Fused zero-sync dispatch (CUDA torch.compile + MPS):
        #   - On SM >= 80 CUDA in BF16 (A100/H100/Blackwell): uses lossless jagged `F.grouped_mm`
        #     executing ONLY `N * top_k` rows (16x less compute on 32 experts, 32x less on 64 experts!).
        #   - On FP32 / MPS / older CUDA: uses static 4-BMM kernel.
        if static_shape or x.device.type == "mps" or torch.compiler.is_compiling():
            w_gu1, w_d1, w_gu2, w_d2 = self._resolve_packed_weights(x.dtype, packed_weights)
            if (
                x.dtype == torch.bfloat16
                and self.num_experts >= 4
                and self.num_experts > self.top_k
                and _device_supports_grouped_mm(x.device, allow_cpu_bf16=self.allow_cpu_grouped_mm)
            ):
                N, D = x.shape
                E = self.num_experts
                flat_e_raw = top_idx.reshape(-1)
                if sample_mask is not None:
                    # Zero-sync Sentinel-E Masked Routing: push all skipped/halted/padded tokens to
                    # sentinel expert ID `E` at the tail so `offs[:E]` excludes them completely!
                    # When all samples halt (`l_idx = 2..7`), `offs` becomes `[0, ..., 0]` and
                    # `F.grouped_mm` skips 100% of expert HBM weight reads with 0 CPU syncs!
                    pair_active = sample_mask.bool().view(N, 1).expand(N, self.top_k).reshape(-1)
                    sentinel = torch.full_like(flat_e_raw, E)
                    flat_e = torch.where(pair_active, flat_e_raw, sentinel)
                    perm = torch.argsort(flat_e, stable=True)
                    e_sorted = flat_e[perm]
                    tok_sorted = torch.div(perm, self.top_k, rounding_mode="floor")
                    x_sorted = x[tok_sorted].contiguous()
                    counts_ext = torch.zeros((E + 1,), device=x.device, dtype=torch.int32).scatter_add(
                        0, e_sorted, torch.ones_like(e_sorted, dtype=torch.int32)
                    )
                    offs = torch.cumsum(counts_ext[:E], dim=0, dtype=torch.int32)
                    active_row_mask = (e_sorted < E).unsqueeze(-1)
                    y_sorted = self._experts_4grouped_mm_impl(
                        x_sorted, offs, w_gu1, w_d1, w_gu2, w_d2, active_row_mask=active_row_mask
                    )
                    gate_sorted = torch.where(
                        active_row_mask,
                        gates_f32.reshape(-1)[perm].unsqueeze(-1),
                        torch.zeros((N * self.top_k, 1), device=x.device, dtype=torch.float32),
                    )
                    y_weighted = torch.where(
                        active_row_mask,
                        y_sorted.float() * gate_sorted,
                        torch.zeros((N * self.top_k, D), device=x.device, dtype=torch.float32),
                    )
                else:
                    perm = torch.argsort(flat_e_raw, stable=True)
                    e_sorted = flat_e_raw[perm]
                    tok_sorted = torch.div(perm, self.top_k, rounding_mode="floor")
                    x_sorted = x[tok_sorted].contiguous()
                    counts = torch.zeros((E,), device=x.device, dtype=torch.int32).scatter_add(
                        0, e_sorted, torch.ones_like(e_sorted, dtype=torch.int32)
                    )
                    offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
                    y_sorted = self._experts_4grouped_mm_impl(x_sorted, offs, w_gu1, w_d1, w_gu2, w_d2)
                    gate_sorted = gates_f32.reshape(-1)[perm].unsqueeze(-1)
                    y_weighted = y_sorted.float() * gate_sorted
                out = (
                    torch.zeros((N, D), device=x.device, dtype=torch.float32)
                    .index_add(0, tok_sorted, y_weighted)
                    .to(x.dtype)
                )
                return out, aux_bal

            gate_full = torch.zeros((x.shape[0], self.num_experts), device=x.device, dtype=torch.float32)
            gate_full = gate_full.scatter_add(1, top_idx, gates_f32)
            x_e = x.unsqueeze(0).expand(self.num_experts, -1, -1)  # [E, N, D]
            y_all = self._experts_4bmm_impl(x_e, w_gu1, w_d1, w_gu2, w_d2)
            out = torch.einsum("ne,end->nd", gate_full, y_all.float()).to(x.dtype)
            return out, aux_bal

        if not self.training and x.dtype == torch.float32:
            N, D = x.shape
            out_f = torch.zeros((N, D), device=x.device, dtype=torch.float32)
            w_gu1_t, w_d1_t, w_gu2_t, w_d2_t = self._resolve_packed_weights(torch.float32, packed_weights)
            x_b1 = x.float().unsqueeze(1)  # [N, 1, D] — strictly row-independent GEMV per token!
            for k_i in range(self.top_k):
                e_ids = top_idx[:, k_i]
                g1, u1 = torch.bmm(x_b1, w_gu1_t[e_ids]).squeeze(1).chunk(2, dim=-1)
                x1 = x + torch.bmm((F.silu(g1) * u1).unsqueeze(1), w_d1_t[e_ids]).squeeze(1)
                x_n2 = x1 * torch.rsqrt(x1.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                g2, u2 = torch.bmm(x_n2.unsqueeze(1), w_gu2_t[e_ids]).squeeze(1).chunk(2, dim=-1)
                yk = x1 + torch.bmm((F.silu(g2) * u2).unsqueeze(1), w_d2_t[e_ids]).squeeze(1)
                out_f = out_f + gates_f32[:, k_i : k_i + 1] * yk
            return out_f.to(x.dtype), aux_bal

        out = torch.zeros_like(x)
        for e_id in range(self.num_experts):
            mask = top_idx == e_id  # [N, top_k]
            if not mask.any():
                continue
            sample_idx, k_pos = torch.nonzero(mask, as_tuple=True)
            xe = x[sample_idx]
            g1, u1 = F.linear(xe, self.w_gu1[e_id].to(dtype=xe.dtype)).chunk(2, dim=-1)
            x1 = xe + F.linear(F.silu(g1) * u1, self.w_d1[e_id].to(dtype=xe.dtype))
            x_n2 = (x1.float() * torch.rsqrt(x1.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(xe.dtype)
            g2, u2 = F.linear(x_n2, self.w_gu2[e_id].to(dtype=xe.dtype)).chunk(2, dim=-1)
            y_e = x1 + F.linear(F.silu(g2) * u2, self.w_d2[e_id].to(dtype=xe.dtype))
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

        # 5. Slot salience conservation + Log-Space Contractive Gate + Slot Temporal Phase (DCPA-Turbo)
        self.w_omega = nn.Linear(D, 1, bias=False)
        self.kappa_omega = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.w_lambda = nn.Linear(D, K, bias=True)
        self.w_slot_phase = nn.Linear(16, D, bias=False)
        self.norm_clause = RMSNorm(D, eps=cfg.eps)
        self._static_kernel_mode: bool = False
        self._compiled_pre_ponder: Optional[Callable[..., Any]] = None
        self._compiled_ponder_router: Optional[Callable[..., Any]] = None
        self._compiled_experts_bmm: Optional[Callable[..., Any]] = None
        self._compiled_experts_grouped_mm: Optional[Callable[..., Any]] = None
        self._compiled_ponder_update: Optional[Callable[..., Any]] = None
        self._compiled_post_merge: Optional[Callable[..., Any]] = None
        self._compiled_chunk_fn: Optional[Callable[..., Any]] = None
        self._compiled_clause_chunk_fn: Optional[Callable[..., Any]] = None

        self._init_weights()

    def set_static_kernel_mode(self, enabled: bool) -> None:
        """Toggle branchless static-shape kernel mode for `torch.compile(..., dynamic=False)`."""
        self._static_kernel_mode = bool(enabled)

    def enable_compiled_chunk_kernel(self, **compile_kwargs: Any) -> None:
        """Compile multi-step `K_chunk` Turbo clause-chunk kernel into a static Inductor graph."""
        compiled_fn = torch.compile(self._forward_clause_chunk_impl, **compile_kwargs)
        self._compiled_chunk_fn = compiled_fn
        self._compiled_clause_chunk_fn = compiled_fn

    def enable_compiled_subkernels(self, **compile_kwargs: Any) -> None:
        """Compile static-shape pre-ponder, active-compacted ponder/MoE, and post-merge sub-kernels."""
        self._compiled_pre_ponder = torch.compile(self._pre_ponder_impl, **compile_kwargs)
        self._compiled_ponder_router = torch.compile(self._ponder_router_impl, **compile_kwargs)
        self._compiled_experts_bmm = torch.compile(self.moe._experts_4bmm_impl, **compile_kwargs)
        if hasattr(F, "grouped_mm"):
            self._compiled_experts_grouped_mm = torch.compile(self.moe._experts_4grouped_mm_impl, **compile_kwargs)
        self._compiled_ponder_update = torch.compile(self._ponder_update_impl, **compile_kwargs)
        self._compiled_post_merge = torch.compile(self._post_ponder_merge_impl, **compile_kwargs)

    def disable_compiled_subkernels(self) -> None:
        self._compiled_pre_ponder = None
        self._compiled_ponder_router = None
        self._compiled_experts_bmm = None
        self._compiled_experts_grouped_mm = None
        self._compiled_ponder_update = None
        self._compiled_post_merge = None
        self._compiled_chunk_fn = None
        self._compiled_clause_chunk_fn = None

    def _init_weights(self) -> None:
        # Initialize f_prec bias so Softplus(bias) ~ 1.0 and w_gen near identity so pondering
        # residual (c_fused - W_gen z) measures true reconstruction error from Step 0.
        nn.init.constant_(self.f_prec_2.bias, 0.5413)
        nn.init.eye_(self.w_gen.weight)
        nn.init.xavier_uniform_(self.w_fast.weight)
        nn.init.normal_(self.w_lambda.weight, std=0.02)
        nn.init.constant_(self.w_lambda.bias, 2.0)  # initial Lambda ~ 0.988 in [0.92, 0.998]
        nn.init.normal_(self.w_slot_phase.weight, std=0.02)

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
        phase = torch.zeros((batch_size, self.cfg.k_slots), device=dev, dtype=torch.float32)
        return ConceptGraphState(
            slots=slots,
            edges=edges,
            salience=salience,
            slot0_pinned=slot0_pinned,
            phase=phase,
        )

    def _slot_phase_embed(self, phase: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Encode per-slot logical age `phase [B, K]` into `D`-dim temporal bias `[B, K, D]`."""
        with fp32_island(phase.device):
            idx = torch.arange(8, device=phase.device, dtype=torch.float32)
            freqs = torch.pow(128.0, -idx / 8.0)
            ang = phase.float().unsqueeze(-1) * freqs
            feat16 = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, K, 16]
        return self.w_slot_phase(feat16.to(dtype))

    def _message_pass_with_eproj(
        self,
        slots: torch.Tensor,
        e_proj: torch.Tensor,
        salience: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_attn = F.softmax(salience.float(), dim=-1)  # [B, K] fp32
        if phase is not None:
            s_phased = slots + self._slot_phase_embed(phase, slots.dtype)
        else:
            s_phased = slots
        weighted_s = s_phased.to(e_proj.dtype) * w_attn.unsqueeze(-1).to(e_proj.dtype)  # [B, K, D]
        agg = torch.einsum("bijd,bjd->bid", e_proj, weighted_s)
        s_bar = s_phased + agg.to(slots.dtype)
        g_ctx = torch.einsum("bk,bkd->bd", w_attn.to(s_bar.dtype), s_bar)
        return s_bar, g_ctx

    def message_pass(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute edge-aggregated node states S_bar [B, K, D] and context readout g [B, D]."""
        e_proj = torch.tanh(self.w_e(edges))  # [B, K, K, D], bounded in (-1, 1) for stable multi-step recurrence
        return self._message_pass_with_eproj(slots, e_proj, salience, phase=phase)

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

            if not self.training and z_a.dtype == torch.float32:
                na = int(z_a.shape[0])
                ka = int(s_bar_a.shape[1])
                d_m = int(z_a.shape[1])
                q_a = torch.bmm(
                    z_a.unsqueeze(1), self.cross_attn.w_q.weight.float().t().unsqueeze(0).expand(na, -1, -1)
                ).squeeze(1)
                sb_flat = s_bar_a.reshape(na * ka, 1, d_m)
                k_a = torch.bmm(
                    sb_flat, self.cross_attn.w_k.weight.float().t().unsqueeze(0).expand(na * ka, -1, -1)
                ).reshape(na, ka, d_m)
                v_a = torch.bmm(
                    sb_flat, self.cross_attn.w_v.weight.float().t().unsqueeze(0).expand(na * ka, -1, -1)
                ).reshape(na, ka, d_m)
                sc_a = torch.bmm(q_a.unsqueeze(1), k_a.transpose(1, 2)).squeeze(1) * self.cross_attn.scale
                at_a = F.softmax(sc_a, dim=-1)
                ctx_a = torch.bmm(at_a.unsqueeze(1), v_a).squeeze(1)
                r_a = torch.bmm(
                    ctx_a.unsqueeze(1), self.cross_attn.w_o.weight.float().t().unsqueeze(0).expand(na, -1, -1)
                ).squeeze(1)
            else:
                r_a = self.cross_attn(z_a, s_bar_a)
            delta_a, bal_l = self.moe(eps_a + r_a, static_shape=False)
            moe_bal_sum = moe_bal_sum + bal_l
            moe_bal_count = moe_bal_count + 1.0

            z_next_a = self.norm_ponder(z_a + self.cfg.ponder_eta * delta_a).to(z_cur.dtype)

            z_next = z_cur.clone()
            z_next[act_idx] = z_next_a
            z_cur = z_next

            if not self.training and z_next_a.dtype == torch.float32:
                gen_a = torch.bmm(
                    z_next_a.unsqueeze(1),
                    self.w_gen.weight.float().t().unsqueeze(0).expand(z_next_a.shape[0], -1, -1),
                ).squeeze(1)
            else:
                gen_a = self.w_gen(z_next_a).float()
            eps_next_a = (
                pi_eff[act_idx] * (c_fused[act_idx].float() - gen_a)
            ).to(orig_dtype)
            eps_next = eps_cur.clone()
            eps_next[act_idx] = eps_next_a
            eps_cur = eps_next

            surp_next = surp_cur.clone()
            surp_next[act_idx] = torch.sqrt(eps_next_a.float().pow(2).mean(dim=-1) + 1e-8)
            surp_cur = surp_next

        moe_bal_loss = moe_bal_sum / moe_bal_count.clamp_min(1.0)
        return z_cur, ponder_steps, moe_bal_loss

    def _pre_ponder_impl(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        c_fused: torch.Tensor,
        prec_bu: torch.Tensor,
        ach: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
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
    ]:
        """Static-shape pre-ponder graph readout, prior/precision prediction, and Bayesian fusion."""
        orig_dtype = c_fused.dtype
        s_bar_prev, g_ctx = self.message_pass(slots, edges, salience, phase=phase)
        c_hat = self.f_pred_2(F.silu(self.f_pred_1(g_ctx)))  # [B, D]
        pi_raw = F.softplus(self.f_prec_2(F.silu(self.f_prec_1(g_ctx))).float()) + 1e-4  # [B, D] fp32
        pi_m = 2.0 * torch.tanh(pi_raw * 0.5)

        ach_f32 = ach.float().clamp(0.0, 1.0)
        prec_bu_f32 = prec_bu.float().clamp_min(0.0)
        # Detach top-down prior precision pi_m inside pondering gate/fusion so L_ponder cannot
        # cheat pondering tax by collapsing pi_m -> 0 (pi_m is calibrated strictly via L_PC).
        pi_m_sg = pi_m.detach()
        pi_eff = pi_m_sg * (1.0 - ach_f32).unsqueeze(-1)

        eps_0 = (pi_eff * (c_fused.float() - c_hat.float())).to(orig_dtype)
        surprisal_0 = torch.sqrt(eps_0.float().pow(2).mean(dim=-1) + 1e-8)  # [B] fp32

        # End-to-end differentiable Bayesian precision fusion + fast shortcut onto unit-RMS sphere
        # (Log-Space Contractive Gate Lambda_m in [0.92, 0.998] bounds recurrent slot Jacobian cleanly).
        z_0 = (
            (pi_m_sg * c_hat.float() + prec_bu_f32 * c_fused.float())
            / (pi_m_sg + prec_bu_f32 + self.cfg.eps)
        ).to(orig_dtype)
        z_0 = self.norm_ponder(
            z_0 + (1.0 - ach_f32).unsqueeze(-1).to(orig_dtype) * self.w_fast(eps_0)
        )
        k_prev, v_prev = self.cross_attn.project_kv(s_bar_prev)
        return s_bar_prev, g_ctx, c_hat, pi_m, pi_eff, surprisal_0, eps_0, z_0, k_prev, v_prev

    def ponder_parallel(
        self,
        z_0: torch.Tensor,
        eps_0: torch.Tensor,
        surprisal_0: torch.Tensor,
        s_bar_prev: torch.Tensor,
        c_fused: torch.Tensor,
        pi_eff: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-batch parallel System-2 Pondering across all `[M_total, D]` segments simultaneously!

        Compacts ONLY surprised concepts (`surp_cur > tau_skip/tau_halt`) into a single dense
        `[M_surprised, D]` Tensor-Core matrix per step (`<= L_max` steps, exiting immediately once
        `M_surprised == 0`), turning 3,400+ tiny sequential MoE calls into 2-4 large batched GEMMs!
        """
        M, D = z_0.shape
        dev = z_0.device
        orig_dtype = z_0.dtype
        if M == 0:
            return (
                z_0,
                torch.zeros((0,), device=dev, dtype=torch.float32),
                torch.zeros((), device=dev, dtype=torch.float32),
            )

        am = torch.ones((M,), device=dev, dtype=torch.bool) if active_mask is None else active_mask.bool()
        if not self.training and dev.type == "cpu":
            return self._ponder_loop_sparse(z_0, eps_0, surprisal_0, s_bar_prev, c_fused, pi_eff, am)

        packed_w = self.moe.pack_weights(dtype=orig_dtype)
        k_prev, v_prev = self.cross_attn.project_kv(s_bar_prev)
        router_fn = self._compiled_ponder_router if self._compiled_ponder_router is not None else self._ponder_router_impl
        update_fn = self._compiled_ponder_update if self._compiled_ponder_update is not None else self._ponder_update_impl

        z_cur = z_0
        eps_cur = eps_0
        surp_cur = surprisal_0
        active = am
        ponder_steps = torch.zeros((M,), device=dev, dtype=torch.float32)
        moe_bal_sum = torch.zeros((), device=dev, dtype=torch.float32)
        moe_bal_count = torch.zeros((), device=dev, dtype=torch.float32)

        for l_idx in range(self.cfg.l_max):
            thresh = self.cfg.tau_skip if l_idx == 0 else self.cfg.tau_halt
            prev_active_f = active.float()
            still_surprised = surp_cur > thresh
            active = active & still_surprised

            soft_act = torch.sigmoid((surp_cur - thresh) / 0.25) * prev_active_f
            hard_act = active.float()
            ste_act = (hard_act - soft_act).detach() + soft_act
            ponder_steps = ponder_steps + ste_act

            act_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            n_act = int(act_idx.numel())
            if n_act == 0:
                break

            z_act = z_cur[act_idx]
            eps_act = eps_cur[act_idx]
            router_in = router_fn(z_act, eps_act, k_prev[act_idx], v_prev[act_idx])
            delta_act, bal_l = self.moe.forward_compacted_bmm(
                router_in,
                n_valid=n_act,
                sample_mask=None,
                packed_weights=packed_w,
                compiled_experts_fn=self._compiled_experts_bmm,
                compiled_grouped_mm_fn=self._compiled_experts_grouped_mm,
            )
            moe_bal_sum = moe_bal_sum + bal_l
            moe_bal_count = moe_bal_count + 1.0

            z_next_a, eps_next_a, surp_next_a = update_fn(
                z_act, delta_act, c_fused[act_idx], pi_eff[act_idx]
            )
            z_next = z_cur.clone()
            z_next[act_idx] = z_next_a
            z_cur = z_next

            eps_next = eps_cur.clone()
            eps_next[act_idx] = eps_next_a
            eps_cur = eps_next

            surp_next = surp_cur.clone()
            surp_next[act_idx] = surp_next_a
            surp_cur = surp_next

        moe_bal_loss = moe_bal_sum / moe_bal_count.clamp_min(1.0)
        return z_cur, ponder_steps, moe_bal_loss

    def _ponder_router_impl(
        self,
        z_in: torch.Tensor,
        eps_in: torch.Tensor,
        k_act: torch.Tensor,
        v_act: torch.Tensor,
    ) -> torch.Tensor:
        """Static-shape SlotCrossAttention query over compacted active surprised concepts `[N_pad, D]`."""
        return eps_in + self.cross_attn.forward_with_kv(z_in, k_act, v_act)

    def _ponder_update_impl(
        self,
        z_in: torch.Tensor,
        delta_act: torch.Tensor,
        c_fused_act: torch.Tensor,
        pi_eff_act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Static-shape Pondering state & residual update on compacted active concepts `[N_pad, D]`."""
        orig_dtype = c_fused_act.dtype
        z_next = self.norm_ponder(z_in + self.cfg.ponder_eta * delta_act).to(orig_dtype)
        eps_next = (pi_eff_act * (c_fused_act.float() - self.w_gen(z_next).float())).to(orig_dtype)
        surp_next = torch.sqrt(eps_next.float().pow(2).mean(dim=-1) + 1e-8)
        return z_next, eps_next, surp_next

    def _post_ponder_merge_with_phase(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        z_star: torch.Tensor,
        surprisal_0: torch.Tensor,
        tau_m: torch.Tensor,
        active_mask: torch.Tensor,
        merge_uniform_mix: torch.Tensor,
        pinned_cur: torch.Tensor,
        trig_cur: torch.Tensor,
        use_gumbel: bool,
        gumbel_u: Optional[torch.Tensor] = None,
        compute_s_bar_post: bool = True,
        check_pin: bool = True,
        phase: Optional[torch.Tensor] = None,
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
        """Log-Space Contractive Gate `Lambda_m in [0.92, 0.998]`, 72-pair Gumbel-STE merge, phase age, & Slot-0 latch."""
        B = z_star.shape[0]
        K = self.cfg.k_slots
        slot0_bool = self.slot0_onehot > 0.5  # [K]
        already_pinned_b = pinned_cur > 0.5  # [B]

        # Log-Space Contractive Gate Lambda_m in [0.920, 0.998] (replaces artificial 0.5/.detach() BPTT hacks!)
        with fp32_island(z_star.device):
            gamma_m = 0.002 + 0.081 * torch.sigmoid(-self.w_lambda(z_star.float()))  # [B, K]
            lam_slots = torch.exp(-gamma_m)                                          # [B, K]
            lam_slots = torch.where(
                already_pinned_b.view(B, 1) & slot0_bool.view(1, K),
                torch.ones_like(lam_slots),
                lam_slots,
            )
            slots_contracted = (slots.float() * lam_slots.unsqueeze(-1)).to(slots.dtype)

        # 3. Candidate slot expansion (K -> K+1)
        s_tilde = torch.cat([slots_contracted, z_star.unsqueeze(1)], dim=1)  # [B, K+1, D]
        src_all = self.u_src(s_tilde)  # [B, K+1, d_r]
        dst_all = self.u_dst(s_tilde)  # [B, K+1, d_r]

        new_row = torch.tanh(src_all[:, K : K + 1, :] * dst_all)  # [B, K+1, d_r]
        new_col = torch.tanh(src_all[:, :K, :] * dst_all[:, K : K + 1, :])  # [B, K, d_r]
        top_part = torch.cat([edges, new_col.unsqueeze(2)], dim=2)  # [B, K, K+1, d_r]
        a_tilde = torch.cat([top_part, new_row.unsqueeze(1)], dim=1)  # [B, K+1, K+1, d_r]

        # 4. End-to-end differentiable K+1 -> K Syntactic Merge Bottleneck over P = 72 ordered pairs
        pi_idx = self.pair_i  # [72]
        pj_idx = self.pair_j  # [72]
        pu_idx = self.pair_u  # [72]

        a_ij = a_tilde[:, pi_idx, pj_idx, :]  # [B, 72, d_r]
        wl_all = self.w_l(s_tilde)            # [B, K+1, d_m]
        wr_all = self.w_r(s_tilde)            # [B, K+1, d_m]
        h_merge = torch.tanh(wl_all[:, pi_idx, :] + wr_all[:, pj_idx, :] + self.w_edge_score(a_ij))
        m_score = self.w_merge(h_merge).squeeze(-1).float()  # [B, 72] fp32

        # Precision Island #3 (Red Line #1) + Branchless Slot-0 Pinning Mask (Spec 2 / Red Line #3)
        with fp32_island(z_star.device):
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
                    u_rand = torch.rand_like(m_score_f) if gumbel_u is None else gumbel_u.float()
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

            p_mat = torch.einsum("bp,pka->bka", alpha, self.perm_bank.float())  # [B, K, K+1] fp32

        win_i = pi_idx[win_idx]  # [B]
        win_j = pj_idx[win_idx]  # [B]
        win_u = pu_idx[win_idx]  # [B]
        win_v = self.pair_v[win_idx]  # [B]

        p_mat_s = p_mat.to(s_tilde.dtype)
        p_mat_a = p_mat.to(a_tilde.dtype)

        b_arange = torch.arange(B, device=z_star.device)
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
            k_ids = torch.arange(K, device=z_star.device).unsqueeze(0)  # [1, K]
            is_u = (k_ids == win_u.unsqueeze(1)).unsqueeze(-1)  # [B, K, 1]
            is_v = (k_ids == win_v.unsqueeze(1)).unsqueeze(-1)  # [B, K, 1]
            s_cand_new = torch.where(is_u, s_merged_win.to(slots.dtype).unsqueeze(1), slots_contracted)
            s_cand_new = torch.where(is_v, z_star.to(slots.dtype).unsqueeze(1), s_cand_new)

        a_tmp = torch.einsum("bka,bacr->bkcr", p_mat_a, a_tilde)
        a_cand_new = torch.einsum("blc,bkcr->bklr", p_mat_a, a_tmp).to(edges.dtype).contiguous()

        phase_in = torch.zeros((B, K), device=z_star.device, dtype=torch.float32) if phase is None else phase.float()
        with fp32_island(z_star.device):
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

            # Slot Temporal Phase update (`phase [B, K]`)
            phase_old = phase_in + 1.0
            phase_tilde = torch.cat([phase_old, torch.zeros((B, 1), device=z_star.device, dtype=torch.float32)], dim=1)
            k_ids_p = torch.arange(K, device=z_star.device).unsqueeze(0)
            p_merged = torch.minimum(phase_tilde[b_arange, win_i], phase_tilde[b_arange, win_j])
            phase_cand_new = torch.where(k_ids_p == win_u.unsqueeze(1), p_merged.unsqueeze(1), phase_old)
            phase_cand_new = torch.where(
                k_ids_p == win_v.unsqueeze(1), torch.zeros_like(phase_cand_new), phase_cand_new
            )
            phase_cand_new = torch.where(
                already_pinned_b.view(B, 1) & slot0_bool.view(1, K), phase_in, phase_cand_new
            )

        # Bit-exact 0.0 ULP preservation of pinned Slot-0!
        s_cand_new = torch.where(already_pinned_b.view(B, 1, 1) & slot0_bool.view(1, K, 1), slots, s_cand_new)

        am_s = active_mask.view(B, 1, 1)
        am_e = active_mask.view(B, 1, 1, 1)
        am_w = active_mask.view(B, 1)
        s_new = torch.where(am_s, s_cand_new, slots)
        a_new = torch.where(am_e, a_cand_new, edges)
        omega_new = torch.where(am_w, omega_cand_new, salience.float())
        phase_new = torch.where(am_w, phase_cand_new, phase_in)

        if check_pin:
            e_proj_post = torch.tanh(self.w_e(a_new))  # [B, K, K, D]
            _, g_post = self._message_pass_with_eproj(s_new, e_proj_post, omega_new, phase=phase_new)
            do_pin = active_mask.bool() & (trig_cur > 0.5) & (pinned_cur < 0.5)  # [B]
            g_post_f = g_post.float()
            thesis_vec = (
                g_post_f * torch.rsqrt(g_post_f.pow(2).mean(dim=-1, keepdim=True) + self.cfg.eps)
            ).to(s_new.dtype)  # [B, D]
            do_pin_s = do_pin.view(B, 1, 1) & slot0_bool.view(1, K, 1)
            do_pin_w = do_pin.view(B, 1) & slot0_bool.view(1, K)
            s_new = torch.where(do_pin_s, thesis_vec.unsqueeze(1).expand(-1, K, -1), s_new)
            omega_new = torch.where(do_pin_w, torch.maximum(omega_new, torch.full_like(omega_new, 2.0)), omega_new)
            phase_new = torch.where(do_pin_w, torch.zeros_like(phase_new), phase_new)
            slot0_pinned_next = torch.where(do_pin, torch.ones_like(pinned_cur), pinned_cur)

            if compute_s_bar_post:
                s_bar_post, _ = self._message_pass_with_eproj(s_new, e_proj_post, omega_new, phase=phase_new)
            else:
                s_bar_post = s_new
        else:
            slot0_pinned_next = pinned_cur
            if compute_s_bar_post:
                e_proj_post = torch.tanh(self.w_e(a_new))  # [B, K, K, D]
                s_bar_post, _ = self._message_pass_with_eproj(s_new, e_proj_post, omega_new, phase=phase_new)
            else:
                s_bar_post = s_new
        return (
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
            phase_new,
        )

    def _post_ponder_merge_impl(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        z_star: torch.Tensor,
        surprisal_0: torch.Tensor,
        tau_m: torch.Tensor,
        active_mask: torch.Tensor,
        merge_uniform_mix: torch.Tensor,
        pinned_cur: torch.Tensor,
        trig_cur: torch.Tensor,
        use_gumbel: bool,
        gumbel_u: Optional[torch.Tensor] = None,
        compute_s_bar_post: bool = True,
        check_pin: bool = True,
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
    ]:
        """10-tuple wrapper around `_post_ponder_merge_with_phase` preserving compiled kernel signatures."""
        return self._post_ponder_merge_with_phase(
            slots,
            edges,
            salience,
            z_star,
            surprisal_0,
            tau_m,
            active_mask,
            merge_uniform_mix,
            pinned_cur,
            trig_cur,
            use_gumbel,
            gumbel_u=gumbel_u,
            compute_s_bar_post=compute_s_bar_post,
            check_pin=check_pin,
            phase=None,
        )[:10]

    def forward_graph_step_no_ponder(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        phase: torch.Tensor,
        c_fused: torch.Tensor,
        prec_bu: torch.Tensor,
        ach: torch.Tensor,
        tau_m: torch.Tensor,
        active_mask: torch.Tensor,
        merge_uniform_mix: torch.Tensor,
        slot0_pinned: torch.Tensor,
        pin_trigger: torch.Tensor,
        use_gumbel: bool = True,
        gumbel_u: Optional[torch.Tensor] = None,
        compute_s_bar_post: bool = True,
        check_pin: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Zero-MoE lightweight contractive graph step (`0.36 GFLOPs/step`) for DCPA-Turbo Step 3!

        Computes prior prediction `(c_hat, pi_m)`, fast Bayesian fusion `z_0`, and `72`-pair
        Gumbel-STE + Log-Space Contractive Gate `Lambda_m` + `phase` update, while deferring
        System-2 Pondering MoE to `ponder_parallel` (`[M_total, D]`)!
        """
        s_bar_prev, g_ctx, c_hat, pi_m, pi_eff, surprisal_0, eps_0, z_0, _, _ = self._pre_ponder_impl(
            slots, edges, salience, c_fused, prec_bu, ach, phase=phase
        )
        (
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
            phase_new,
        ) = self._post_ponder_merge_with_phase(
            slots,
            edges,
            salience,
            z_0,
            surprisal_0,
            tau_m,
            active_mask,
            merge_uniform_mix,
            slot0_pinned.float(),
            pin_trigger.float(),
            use_gumbel,
            gumbel_u=gumbel_u,
            compute_s_bar_post=compute_s_bar_post,
            check_pin=check_pin,
            phase=phase,
        )
        return (
            z_0,
            g_ctx,
            c_hat,
            pi_m,
            pi_eff,
            surprisal_0,
            eps_0,
            s_bar_prev,
            s_new,
            a_new,
            omega_new,
            phase_new,
            s_bar_post,
            win_idx,
            win_i,
            win_j,
            merge_probs,
            alpha,
            slot0_pinned_next,
        )

    def _forward_clause_chunk_impl(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        phase: torch.Tensor,
        slot0_pinned: torch.Tensor,
        c_fused_seq: torch.Tensor,
        prec_bu_seq: torch.Tensor,
        ach_seq: torch.Tensor,
        active_mask_seq: torch.Tensor,
        trig_seq: torch.Tensor,
        unpin_seq: torch.Tensor,
        tau_m: torch.Tensor,
        mix_t: torch.Tensor,
        gumbel_u_seq: Optional[torch.Tensor] = None,
        use_gumbel: bool = True,
        check_pin: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Multi-step (`S = clause_chunk_size`, e.g. 16) zero-MoE contractive graph chunk kernel."""
        S = c_fused_seq.shape[0]
        B = slots.shape[0]
        K = self.cfg.k_slots

        cur_slots = slots
        cur_edges = edges
        cur_sal = salience
        cur_phase = phase
        cur_pin = slot0_pinned.float()

        z0_l, g_l, chat_l, pi_l, pieff_l, surp_l, eps_l, sbar_prev_l = [], [], [], [], [], [], [], []
        mp_l, alpha_l, c0_l = [], [], []
        pre_s_l, pre_e_l, pre_w_l, pre_p_l = [], [], [], []

        for k in range(S):
            eff_pin = torch.where(unpin_seq[k], torch.zeros_like(cur_pin), cur_pin)
            eff_phase = torch.where(unpin_seq[k].view(B, 1), torch.zeros_like(cur_phase), cur_phase)

            pre_s_l.append(cur_slots)
            pre_e_l.append(cur_edges)
            pre_w_l.append(cur_sal)
            pre_p_l.append(eff_phase)

            am_k = active_mask_seq[k]
            gu_k = gumbel_u_seq[k] if gumbel_u_seq is not None else None
            (
                z_0,
                g_ctx,
                c_hat,
                pi_m,
                pi_eff,
                surp_0,
                eps_0,
                s_bar_prev,
                s_next,
                a_next,
                omega_next,
                phase_next,
                _s_bar_post,
                _win_idx,
                _win_i,
                _win_j,
                mp_k,
                alpha_k,
                pin_next,
            ) = self.forward_graph_step_no_ponder(
                cur_slots,
                cur_edges,
                cur_sal,
                eff_phase,
                c_fused_seq[k],
                prec_bu_seq[k],
                ach_seq[k],
                tau_m,
                am_k,
                mix_t,
                eff_pin,
                trig_seq[k],
                use_gumbel=use_gumbel,
                gumbel_u=gu_k,
                compute_s_bar_post=False,
                check_pin=check_pin,
            )
            cur_slots = s_next
            cur_edges = a_next
            cur_sal = omega_next
            cur_phase = phase_next
            cur_pin = pin_next

            am_f = am_k.float().view(B, 1, 1)
            denom_c0 = (am_f.sum() * float(K)).clamp_min(1.0)
            c0_l.append((s_next.detach().float() * am_f).sum(dim=(0, 1)) / denom_c0)

            z0_l.append(z_0)
            g_l.append(g_ctx)
            chat_l.append(c_hat)
            pi_l.append(pi_m)
            pieff_l.append(pi_eff)
            surp_l.append(surp_0)
            eps_l.append(eps_0)
            sbar_prev_l.append(s_bar_prev)
            mp_l.append(mp_k)
            alpha_l.append(alpha_k)

        return (
            torch.stack(z0_l, dim=0),
            torch.stack(g_l, dim=0),
            torch.stack(chat_l, dim=0),
            torch.stack(pi_l, dim=0),
            torch.stack(pieff_l, dim=0),
            torch.stack(surp_l, dim=0),
            torch.stack(eps_l, dim=0),
            torch.stack(sbar_prev_l, dim=0),
            torch.stack(mp_l, dim=0),
            torch.stack(alpha_l, dim=0),
            cur_slots,
            cur_edges,
            cur_sal,
            cur_phase,
            cur_pin,
            torch.stack(c0_l, dim=0),
            torch.stack(pre_s_l, dim=0),
            torch.stack(pre_e_l, dim=0),
            torch.stack(pre_w_l, dim=0),
            torch.stack(pre_p_l, dim=0),
        )

    def forward_clause_chunk(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        phase: torch.Tensor,
        slot0_pinned: torch.Tensor,
        c_fused_seq: torch.Tensor,
        prec_bu_seq: torch.Tensor,
        ach_seq: torch.Tensor,
        active_mask_seq: torch.Tensor,
        trig_seq: torch.Tensor,
        unpin_seq: torch.Tensor,
        tau_m: torch.Tensor,
        mix_t: torch.Tensor,
        gumbel_u_seq: Optional[torch.Tensor] = None,
        use_gumbel: bool = True,
        check_pin: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        fn = (
            self._compiled_chunk_fn
            if self._compiled_chunk_fn is not None
            else (
                self._compiled_clause_chunk_fn
                if self._compiled_clause_chunk_fn is not None
                else self._forward_clause_chunk_impl
            )
        )
        return fn(
            slots,
            edges,
            salience,
            phase,
            slot0_pinned,
            c_fused_seq,
            prec_bu_seq,
            ach_seq,
            active_mask_seq,
            trig_seq,
            unpin_seq,
            tau_m,
            mix_t,
            gumbel_u_seq,
            use_gumbel,
            check_pin,
        )

    def forward_fast_dispatch(
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
        gumbel_u: Optional[torch.Tensor] = None,
        compute_s_bar_post: bool = True,
        check_pin: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Fast event-driven BCG-PC dispatcher with early-exit Pondering & lossless compacted Top-2 MoE.

        Skips dead Pondering iterations (`l_idx = 2..7` once `active` empties) and compacts active
        surprised rows + per-expert capacities into fixed Tensor-Core buckets while running all heavy
        math inside static-shape `torch.compile(..., dynamic=False)` sub-kernels!
        """
        if self._compiled_pre_ponder is None and c_fused.device.type == "cpu":
            return self.forward(
                slots,
                edges,
                salience,
                c_fused,
                prec_bu,
                ach,
                tau_m,
                active_mask,
                merge_uniform_mix,
                slot0_pinned,
                pin_trigger,
                use_gumbel=use_gumbel,
                update_c0_ema=update_c0_ema,
                packed_moe_w_gu1=packed_moe_w_gu1,
                packed_moe_w_d1=packed_moe_w_d1,
                packed_moe_w_gu2=packed_moe_w_gu2,
                packed_moe_w_d2=packed_moe_w_d2,
                gumbel_u=gumbel_u,
                compute_s_bar_post=compute_s_bar_post,
                check_pin=check_pin,
            )

        B, _ = c_fused.shape
        dev = c_fused.device
        orig_dtype = c_fused.dtype
        packed_weights = (
            (packed_moe_w_gu1, packed_moe_w_d1, packed_moe_w_gu2, packed_moe_w_d2)
            if packed_moe_w_gu1 is not None
            and packed_moe_w_d1 is not None
            and packed_moe_w_gu2 is not None
            and packed_moe_w_d2 is not None
            else None
        )
        eff_packed_w = packed_weights if packed_weights is not None else self.moe.pack_weights(dtype=orig_dtype)

        pinned_cur = (
            torch.zeros((B,), device=dev, dtype=torch.float32)
            if slot0_pinned is None
            else slot0_pinned.float()
        )
        trig_cur = (
            torch.zeros((B,), device=dev, dtype=torch.float32)
            if pin_trigger is None
            else pin_trigger.float()
        )

        pre_fn = self._compiled_pre_ponder if self._compiled_pre_ponder is not None else self._pre_ponder_impl
        router_fn = self._compiled_ponder_router if self._compiled_ponder_router is not None else self._ponder_router_impl
        update_fn = self._compiled_ponder_update if self._compiled_ponder_update is not None else self._ponder_update_impl
        post_fn = self._compiled_post_merge if self._compiled_post_merge is not None else self._post_ponder_merge_impl

        _s_bar_prev, g_ctx, c_hat, pi_m, pi_eff, surprisal_0, eps_cur, z_cur, k_prev, v_prev = pre_fn(
            slots, edges, salience, c_fused, prec_bu, ach
        )

        surp_cur = surprisal_0
        active = active_mask.bool()
        ponder_steps = torch.zeros((B,), device=dev, dtype=torch.float32)
        moe_bal_sum = torch.zeros((), device=dev, dtype=torch.float32)
        moe_bal_count = torch.zeros((), device=dev, dtype=torch.float32)

        for l_idx in range(self.cfg.l_max):
            thresh = self.cfg.tau_skip if l_idx == 0 else self.cfg.tau_halt
            prev_active_f = active.float()
            still_surprised = surp_cur > thresh
            active = active & still_surprised

            # Exact STE ponder counter before early-exit check (0.0 ULP drift vs full L_max loop!)
            soft_act = torch.sigmoid((surp_cur - thresh) / 0.25) * prev_active_f
            hard_act = active.float()
            ste_act = (hard_act - soft_act).detach() + soft_act
            ponder_steps = ponder_steps + ste_act

            act_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            n_act = int(act_idx.numel())
            if n_act == 0:
                break

            if n_act == B:
                z_act = z_cur
                eps_act = eps_cur
                if self.training and l_idx > 0:
                    z_act = z_act.detach() + 0.5 * (z_act - z_act.detach())
                    eps_act = eps_act.detach() + 0.5 * (eps_act - eps_act.detach())
                router_in = router_fn(z_act, eps_act, k_prev, v_prev)
                delta_act, bal_l = self.moe.forward_compacted_bmm(
                    router_in,
                    n_valid=B,
                    sample_mask=None,
                    packed_weights=eff_packed_w,
                    compiled_experts_fn=self._compiled_experts_bmm,
                    compiled_grouped_mm_fn=self._compiled_experts_grouped_mm,
                )
                moe_bal_sum = moe_bal_sum + bal_l
                moe_bal_count = moe_bal_count + 1.0
                z_cur, eps_cur, surp_cur = update_fn(z_act, delta_act, c_fused, pi_eff)
            else:
                n_pad = _coarse_bucket_count(n_act, bucket=16) if self._compiled_pre_ponder is not None else n_act
                pad_n = n_pad - n_act
                if pad_n > 0:
                    safe_idx = torch.cat([act_idx, torch.zeros((pad_n,), device=dev, dtype=torch.long)], dim=0)
                    pad_mask: Optional[torch.Tensor] = torch.arange(n_pad, device=dev) < n_act
                else:
                    safe_idx = act_idx
                    pad_mask = None

                z_act = z_cur[safe_idx]
                eps_act = eps_cur[safe_idx]
                if self.training and l_idx > 0:
                    z_act = z_act.detach() + 0.5 * (z_act - z_act.detach())
                    eps_act = eps_act.detach() + 0.5 * (eps_act - eps_act.detach())

                router_in = router_fn(z_act, eps_act, k_prev[safe_idx], v_prev[safe_idx])
                delta_act, bal_l = self.moe.forward_compacted_bmm(
                    router_in,
                    n_valid=n_act,
                    sample_mask=pad_mask,
                    packed_weights=eff_packed_w,
                    compiled_experts_fn=self._compiled_experts_bmm,
                    compiled_grouped_mm_fn=self._compiled_experts_grouped_mm,
                )
                moe_bal_sum = moe_bal_sum + bal_l
                moe_bal_count = moe_bal_count + 1.0

                z_next_p, eps_next_p, surp_next_p = update_fn(
                    z_act, delta_act, c_fused[safe_idx], pi_eff[safe_idx]
                )
                z_next = z_cur.clone()
                z_next[act_idx] = z_next_p[:n_act]
                z_cur = z_next

                eps_next = eps_cur.clone()
                eps_next[act_idx] = eps_next_p[:n_act]
                eps_cur = eps_next

                surp_next = surp_cur.clone()
                surp_next[act_idx] = surp_next_p[:n_act]
                surp_cur = surp_next

        moe_bal_loss = moe_bal_sum / moe_bal_count.clamp_min(1.0)
        z_star = z_cur

        (
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
        ) = post_fn(
            slots,
            edges,
            salience,
            z_star,
            surprisal_0,
            tau_m,
            active_mask,
            merge_uniform_mix,
            pinned_cur,
            trig_cur,
            use_gumbel,
            gumbel_u,
            compute_s_bar_post,
            check_pin,
        )

        if self.training and update_c0_ema:
            self.update_c0_from_slots(s_new, active_mask)

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
        gumbel_u: Optional[torch.Tensor] = None,
        compute_s_bar_post: bool = True,
        check_pin: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Hotspot A (`bcg_step_kernel`): single-step concept graph transition kernel.

        Designed for `torch.compile(..., dynamic=False)` on bucketed document batches `[V_pad, ...]`:
          - In `static_mode` (CUDA `torch.compile` & MPS): 100% zero-sync, zero-graph-break fused
            Inductor graph using 4 batched `torch.bmm` MoE kernels (12x fewer GEMM launches!) and
            precomputed `SlotCrossAttention` K/V (8x less KV projection work).
          - In eager CPU mode (`static_mode=False`): uses event-driven sparse pondering + Top-2 MoE.
        """
        B, _ = c_fused.shape
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

        s_bar_prev, g_ctx, c_hat, pi_m, pi_eff, surprisal_0, eps_0, z_cur, k_prev, v_prev = self._pre_ponder_impl(
            slots, edges, salience, c_fused, prec_bu, ach
        )

        # 2. Event-Driven Pondering (L_max = 8) + Switch MoE load-balancing aux loss
        if static_mode:
            eps_cur = eps_0
            surp_cur = surprisal_0
            active = active_mask.bool()
            ponder_steps = torch.zeros((B,), device=c_fused.device, dtype=torch.float32)
            moe_bal_sum = torch.zeros((), device=c_fused.device, dtype=torch.float32)
            moe_bal_count = torch.zeros((), device=c_fused.device, dtype=torch.float32)

            eff_packed_w = packed_weights if packed_weights is not None else self.moe.pack_weights(dtype=orig_dtype)

            for l_idx in range(self.cfg.l_max):
                thresh = self.cfg.tau_skip if l_idx == 0 else self.cfg.tau_halt
                prev_active_f = active.float()
                still_surprised = surp_cur > thresh
                active = active & still_surprised

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

                r_all = self.cross_attn.forward_with_kv(z_in, k_prev, v_prev)
                delta_all, bal_l = self.moe(
                    eps_in + r_all,
                    static_shape=True,
                    sample_mask=active,
                    packed_weights=eff_packed_w,
                )
                moe_bal_sum = moe_bal_sum + bal_l
                moe_bal_count = moe_bal_count + (active.float().sum() > 0.0).float()

                z_cand, eps_cand, surp_cand = self._ponder_update_impl(z_in, delta_all, c_fused, pi_eff)
                act_col = active.unsqueeze(-1)
                z_cur = torch.where(act_col, z_cand, z_cur)
                eps_cur = torch.where(act_col, eps_cand, eps_cur)
                surp_cur = torch.where(active, surp_cand, surp_cur)

            moe_bal_loss = moe_bal_sum / moe_bal_count.clamp_min(1.0)
            z_star = z_cur  # [B, D]
        else:
            z_star, ponder_steps, moe_bal_loss = self._ponder_loop_sparse(
                z_cur, eps_0, surprisal_0, s_bar_prev, c_fused, pi_eff, active_mask
            )

        (
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
        ) = self._post_ponder_merge_impl(
            slots,
            edges,
            salience,
            z_star,
            surprisal_0,
            tau_m,
            active_mask,
            merge_uniform_mix,
            pinned_cur,
            trig_cur,
            use_gumbel,
            gumbel_u,
            compute_s_bar_post,
            check_pin,
        )

        # Update neutral concept center c_0 via EMA strictly over active docs
        if self.training and update_c0_ema:
            self.update_c0_from_slots(s_new, active_mask)

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

    def _forward_chunk_impl(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        slot0_pinned: torch.Tensor,
        c_fused_seq: torch.Tensor,
        prec_bu_seq: torch.Tensor,
        ach_seq: torch.Tensor,
        active_mask_seq: torch.Tensor,
        trig_seq: torch.Tensor,
        unpin_seq: torch.Tensor,
        tau_m: torch.Tensor,
        mix_t: torch.Tensor,
        packed_moe_w_gu1: torch.Tensor,
        packed_moe_w_d1: torch.Tensor,
        packed_moe_w_gu2: torch.Tensor,
        packed_moe_w_d2: torch.Tensor,
        gumbel_u_seq: Optional[torch.Tensor] = None,
        carry_gamma: float = 0.5,
        carry_max_rms: float = 0.25,
        use_gumbel: bool = True,
        check_pin: bool = True,
        return_pre_states: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Multi-step static-shape BCG-PC kernel executing `S = c_fused_seq.shape[0]` concept steps
        inside ONE fused `torch.compile(..., dynamic=False)` graph (cutting Python/Dynamo/checkpoint
        boundary crossings by `S`x, e.g., 8x -> eliminates 100% single-core CPU dispatch bottleneck!).

        Caller applies `_DampedBoundedCarry` (or TBPTT `.detach()`) on the incoming `(slots, edges,
        salience)` before step 0; intra-chunk steps `k = 1..S-1` apply `_DampedBoundedCarry` identically.
        Also returns detached per-step `c0_means_seq [S, D]` so the caller can apply bit-exact single
        EMA updates to `self.c0` outside activation checkpointing.
        """
        S = c_fused_seq.shape[0]
        B = slots.shape[0]
        K = self.cfg.k_slots

        cur_slots = slots
        cur_edges = edges
        cur_sal = salience
        cur_pin = slot0_pinned.float()

        z_list: list[torch.Tensor] = []
        g_list: list[torch.Tensor] = []
        c_hat_list: list[torch.Tensor] = []
        pi_list: list[torch.Tensor] = []
        ponder_list: list[torch.Tensor] = []
        mp_list: list[torch.Tensor] = []
        bal_list: list[torch.Tensor] = []
        c0_mean_list: list[torch.Tensor] = []
        pre_slots_list: list[torch.Tensor] = []
        pre_edges_list: list[torch.Tensor] = []
        pre_sal_list: list[torch.Tensor] = []

        for k in range(S):
            if self.training and k > 0:
                in_s = _DampedBoundedCarry.apply(cur_slots, carry_gamma, carry_max_rms)
                in_e = _DampedBoundedCarry.apply(cur_edges, carry_gamma, carry_max_rms)
                in_w = _DampedBoundedCarry.apply(cur_sal, carry_gamma, carry_max_rms)
            else:
                in_s = cur_slots
                in_e = cur_edges
                in_w = cur_sal

            eff_pin = torch.where(unpin_seq[k], torch.zeros_like(cur_pin), cur_pin)
            if return_pre_states:
                pre_slots_list.append(in_s)
                pre_edges_list.append(in_e)
                pre_sal_list.append(in_w)

            am_k = active_mask_seq[k]
            gu_k = gumbel_u_seq[k] if gumbel_u_seq is not None else None
            (
                z_k,
                g_k,
                c_hat_k,
                pi_k,
                _surp_k,
                ponder_k,
                s_next,
                a_next,
                omega_next,
                _s_bar_post_k,
                _win_idx_k,
                _wi_k,
                _wj_k,
                mp_k,
                _alpha_k,
                pin_next,
                bal_k,
            ) = self.forward(
                in_s,
                in_e,
                in_w,
                c_fused_seq[k],
                prec_bu_seq[k],
                ach_seq[k],
                tau_m,
                am_k,
                mix_t,
                eff_pin,
                trig_seq[k],
                use_gumbel=use_gumbel,
                update_c0_ema=False,
                packed_moe_w_gu1=packed_moe_w_gu1,
                packed_moe_w_d1=packed_moe_w_d1,
                packed_moe_w_gu2=packed_moe_w_gu2,
                packed_moe_w_d2=packed_moe_w_d2,
                gumbel_u=gu_k,
                compute_s_bar_post=False,
                check_pin=check_pin,
            )
            cur_slots = s_next
            cur_edges = a_next
            cur_sal = omega_next
            cur_pin = pin_next

            am_f = am_k.float().view(B, 1, 1)
            denom_c0 = (am_f.sum() * float(K)).clamp_min(1.0)
            c0_mean_k = (s_next.detach().float() * am_f).sum(dim=(0, 1)) / denom_c0

            z_list.append(z_k)
            g_list.append(g_k)
            c_hat_list.append(c_hat_k)
            pi_list.append(pi_k)
            ponder_list.append(ponder_k)
            mp_list.append(mp_k)
            bal_list.append(bal_k)
            c0_mean_list.append(c0_mean_k)

        base_out = (
            torch.stack(z_list, dim=0),
            torch.stack(g_list, dim=0),
            torch.stack(c_hat_list, dim=0),
            torch.stack(pi_list, dim=0),
            torch.stack(ponder_list, dim=0),
            torch.stack(mp_list, dim=0),
            torch.stack(bal_list, dim=0),
            cur_slots,
            cur_edges,
            cur_sal,
            cur_pin,
            torch.stack(c0_mean_list, dim=0),
        )
        if return_pre_states:
            return base_out + (
                torch.stack(pre_slots_list, dim=0),
                torch.stack(pre_edges_list, dim=0),
                torch.stack(pre_sal_list, dim=0),
            )
        return base_out

    def forward_chunk(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        slot0_pinned: torch.Tensor,
        c_fused_seq: torch.Tensor,
        prec_bu_seq: torch.Tensor,
        ach_seq: torch.Tensor,
        active_mask_seq: torch.Tensor,
        trig_seq: torch.Tensor,
        unpin_seq: torch.Tensor,
        tau_m: torch.Tensor,
        mix_t: torch.Tensor,
        packed_moe_w_gu1: torch.Tensor,
        packed_moe_w_d1: torch.Tensor,
        packed_moe_w_gu2: torch.Tensor,
        packed_moe_w_d2: torch.Tensor,
        gumbel_u_seq: Optional[torch.Tensor] = None,
        carry_gamma: float = 0.5,
        carry_max_rms: float = 0.25,
        use_gumbel: bool = True,
        check_pin: bool = True,
        return_pre_states: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Dispatch multi-step chunk kernel via compiled Inductor graph when available."""
        fn = self._compiled_chunk_fn if self._compiled_chunk_fn is not None else self._forward_chunk_impl
        return fn(
            slots,
            edges,
            salience,
            slot0_pinned,
            c_fused_seq,
            prec_bu_seq,
            ach_seq,
            active_mask_seq,
            trig_seq,
            unpin_seq,
            tau_m,
            mix_t,
            packed_moe_w_gu1,
            packed_moe_w_d1,
            packed_moe_w_gu2,
            packed_moe_w_d2,
            gumbel_u_seq,
            carry_gamma,
            carry_max_rms,
            use_gumbel,
            check_pin,
            return_pre_states,
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
        phase_in = (
            torch.zeros((B, self.cfg.k_slots), device=dev, dtype=torch.float32)
            if state.phase is None
            else state.phase.to(device=dev, dtype=torch.float32)
        )
        s_bar_prev, g_ctx, c_hat, pi_m, pi_eff, surprisal_0, eps_0, z_cur, _, _ = self._pre_ponder_impl(
            state.slots, state.edges, state.salience, c_fused, prec_bu, ach, phase=phase_in
        )
        z_star, ponder_steps, moe_bal_loss = self.ponder_parallel(
            z_cur, eps_0, surprisal_0, s_bar_prev, c_fused, pi_eff, active_mask=am
        )
        (
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
            phase_new,
        ) = self._post_ponder_merge_with_phase(
            state.slots,
            state.edges,
            state.salience,
            z_star,
            surprisal_0,
            tau_t,
            am,
            mix_t,
            pinned_in,
            trig_in,
            use_gumbel,
            gumbel_u=None,
            compute_s_bar_post=True,
            check_pin=(pin_trigger is not None),
            phase=phase_in,
        )
        if self.training and update_c0_ema:
            self.update_c0_from_slots(s_new, am)
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
                phase=phase_new,
            ),
            s_bar_post=s_bar_post,
            win_pair_idx=win_idx,
            win_i=win_i,
            win_j=win_j,
            merge_probs=merge_probs,
            merge_alpha=alpha,
            moe_bal_loss=moe_bal_loss,
        )
