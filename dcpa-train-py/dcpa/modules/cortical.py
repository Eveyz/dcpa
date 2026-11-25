"""Parallel Cortical Sandwich Backbone (Concept-RoPE + Scope-Depth + Shared 48KB MLA Ring + Parallel MoE).

Implements DCPA-Turbo Architecture:
  1. Shared O(1) MLA-compressed local morpheme/concept ring `c_kv [W, d_s]` (~48 KB in BF16 at
     `W=128, d_s=192`), unifying local multi-head syntax attention and exact pointer-copy lexicon.
  2. Dual-coordinate `Concept-RoPE` (`3/4` relative concept order + `1/4` AST bracket/scope depth)
     computed via tiled overlapping `[W, 2W]` blocks (strictly O(M * W) linear time & memory,
     zero cross-document bleed, zero O(M^2) mask allocation).
  3. Full-batch `[M_total, D]` `Shared + Top-K Routed SwiGLU MoE` (`F.grouped_mm` on SM80+ Tensor Cores),
     lifting 90% of model FLOPs out of the recurrent graph loop.
  4. Unified causal `Graph (K=8 Slots + Directed Edges + Temporal Phase) + Strict-Past RoPE Window`
     single-softmax cross-attention in the Upper Cortical Decoder, plus position-aware pointer-copy
     stem residual feeding `PMD`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.config import DCPAConfig
from dcpa.modules.bcg_pc import _device_supports_grouped_mm
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.rmsnorm import RMSNorm
from dcpa.ops.rope import apply_concept_rope_2d


@dataclass
class CorticalCarryState:
    """Strictly O(1) constant-size streaming state for the Shared MLA-RoPE Window & Causal Shift."""

    ring_kv: torch.Tensor       # [B, W, d_s] circular/FIFO latent KV buffer
    ring_pos: torch.Tensor      # [B, W] int64 concept index in current document
    ring_depth: torch.Tensor    # [B, W] int32 causal scope depth at segment end
    ring_valid: torch.Tensor    # [B, W] bool valid slot mask
    doc_pos: torch.Tensor       # [B] int64 total concepts emitted in current document
    ast_depth: torch.Tensor     # [B] int32 running causal bracket/frame nesting depth
    prev_z_ref: torch.Tensor    # [B, D] previous refined concept z_refined[m-1]
    has_prev_seg: torch.Tensor  # [B] bool False before segment 0 of document


class SharedAndRoutedMoE(nn.Module):
    """Single-block Pre-RMSNorm SwiGLU MoE combining 1 Always-On Shared Expert + E Top-K Routed Experts.

    Executes across all `[N, D]` tokens in parallel (`N = M_total ~ 13,800` per batch), saturating
    GPU Tensor Cores via `F.grouped_mm` (SM >= 80 BF16) or batched `torch.bmm`.
    """

    def __init__(
        self,
        d_model: int,
        routed_hidden: int,
        shared_hidden: int,
        num_experts: int,
        top_k: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.routed_hidden = int(routed_hidden)
        self.shared_hidden = int(shared_hidden)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.eps = float(eps)
        self.allow_cpu_grouped_mm: bool = False

        self.norm = RMSNorm(self.d_model, eps=self.eps)
        # Always-on shared expert (captures cross-domain syntax, rhetoric, and code idioms)
        self.shared_w_gu = nn.Linear(self.d_model, 2 * self.shared_hidden, bias=False)
        self.shared_w_d = nn.Linear(self.shared_hidden, self.d_model, bias=False)

        # Top-K routed specialist experts stored as contiguous 3D tensors
        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.w_gu = nn.Parameter(torch.empty((self.num_experts, 2 * self.routed_hidden, self.d_model)))
        self.w_d = nn.Parameter(torch.empty((self.num_experts, self.d_model, self.routed_hidden)))
        self._init_weights()

    def _init_weights(self) -> None:
        hs = self.shared_hidden
        nn.init.kaiming_uniform_(self.shared_w_gu.weight[:hs], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_w_gu.weight[hs:], a=math.sqrt(5))
        nn.init.normal_(self.shared_w_d.weight, std=0.02)

        nn.init.normal_(self.router.weight, std=0.02)
        hr = self.routed_hidden
        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w_gu.data[e, :hr], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_gu.data[e, hr:], a=math.sqrt(5))
            nn.init.normal_(self.w_d.data[e], std=0.02)

    def expert_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return (self.w_gu, self.w_d)

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
        """Transparently load flat `experts.{e}.w_gate/w_up/w_down.weight` keys from `.safetensors`."""
        gu_k = f"{prefix}w_gu"
        probe = f"{prefix}experts.0.w_gate.weight"
        if gu_k not in state_dict and probe in state_dict:
            g_l: list[torch.Tensor] = []
            u_l: list[torch.Tensor] = []
            d_l: list[torch.Tensor] = []
            ok = True
            for e in range(self.num_experts):
                ep = f"{prefix}experts.{e}."
                ks = (f"{ep}w_gate.weight", f"{ep}w_up.weight", f"{ep}w_down.weight")
                if not all(k in state_dict for k in ks):
                    ok = False
                    break
                g_l.append(state_dict.pop(ks[0]))
                u_l.append(state_dict.pop(ks[1]))
                d_l.append(state_dict.pop(ks[2]))
            if ok:
                state_dict[f"{prefix}w_gu"] = torch.stack(
                    [torch.cat([g, u], dim=0) for g, u in zip(g_l, u_l)], dim=0
                ).contiguous()
                state_dict[f"{prefix}w_d"] = torch.stack(d_l, dim=0).contiguous()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: `[N, D]` -> `(y [N, D], aux_bal_loss scalar fp32)`."""
        N, D = x.shape
        if N == 0:
            return x, torch.zeros((), device=x.device, dtype=torch.float32)

        x_n = self.norm(x)
        # 1. Shared expert branch
        s_g, s_u = self.shared_w_gu(x_n).chunk(2, dim=-1)
        y_shared = self.shared_w_d(F.silu(s_g) * s_u)

        # 2. Router in float32 precision island
        E = self.num_experts
        top_k = self.top_k
        with fp32_island(x.device):
            logits = F.linear(x_n.float(), self.router.weight.float())  # [N, E]
            probs = F.softmax(logits, dim=-1)
            top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
            gates_f32 = F.softmax(top_vals, dim=-1)
            p_e = probs.mean(dim=0)
            f_e = F.one_hot(top_idx, num_classes=E).float().sum(dim=1).mean(dim=0) / float(top_k)
            aux_bal = float(E) * (f_e.detach() * p_e).sum()

        # 3. Routed specialist dispatch
        w_gu_t = self.w_gu.to(dtype=x_n.dtype).transpose(1, 2)  # [E, D, 2*H]
        w_d_t = self.w_d.to(dtype=x_n.dtype).transpose(1, 2)    # [E, H, D]

        if (
            x_n.dtype == torch.bfloat16
            and E >= 4
            and E > top_k
            and _device_supports_grouped_mm(x_n.device, allow_cpu_bf16=self.allow_cpu_grouped_mm)
        ):
            flat_e = top_idx.reshape(-1)
            perm = torch.argsort(flat_e, stable=True)
            e_sorted = flat_e[perm]
            tok_sorted = torch.div(perm, top_k, rounding_mode="floor")
            x_sorted = x_n[tok_sorted].contiguous()
            counts = torch.zeros((E,), device=x.device, dtype=torch.int32).scatter_add(
                0, e_sorted, torch.ones_like(e_sorted, dtype=torch.int32)
            )
            offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
            gu_out = F.grouped_mm(x_sorted, w_gu_t, offs=offs)
            g_r, u_r = gu_out.chunk(2, dim=-1)
            act_r = (F.silu(g_r) * u_r).contiguous()
            y_sorted = F.grouped_mm(act_r, w_d_t, offs=offs)
            gate_sorted = gates_f32.reshape(-1)[perm].unsqueeze(-1)
            y_routed = (
                torch.zeros((N, D), device=x.device, dtype=torch.float32)
                .index_add(0, tok_sorted, y_sorted.float() * gate_sorted)
                .to(x.dtype)
            )
            return x + y_shared + y_routed, aux_bal

        # Exact row-independent fp32 path when deterministic eval, else batched 2-BMM
        if not self.training and x_n.dtype == torch.float32:
            y_routed_f = torch.zeros((N, D), device=x.device, dtype=torch.float32)
            w_gu_f = self.w_gu.float()
            w_d_f = self.w_d.float()
            x_f = x_n.float()
            for k_i in range(top_k):
                e_ids = top_idx[:, k_i]  # [N]
                gu_k = torch.bmm(x_f.unsqueeze(1), w_gu_f[e_ids].transpose(1, 2)).squeeze(1)
                gk, uk = gu_k.chunk(2, dim=-1)
                act_k = F.silu(gk) * uk
                yk = torch.bmm(act_k.unsqueeze(1), w_d_f[e_ids].transpose(1, 2)).squeeze(1)
                y_routed_f = y_routed_f + gates_f32[:, k_i : k_i + 1] * yk
            return x + y_shared + y_routed_f.to(x.dtype), aux_bal

        gate_full = torch.zeros((N, E), device=x.device, dtype=torch.float32).scatter_add(1, top_idx, gates_f32)
        x_e = x_n.unsqueeze(0).expand(E, -1, -1)  # [E, N, D]
        g_all, u_all = torch.bmm(x_e, w_gu_t).chunk(2, dim=-1)
        y_all = torch.bmm(F.silu(g_all) * u_all, w_d_t)  # [E, N, D]
        y_routed = torch.einsum("ne,end->nd", gate_full, y_all.float()).to(x.dtype)
        return x + y_shared + y_routed, aux_bal


def build_slot_phase_features(phase: torch.Tensor, dim: int) -> torch.Tensor:
    """Map non-negative slot logical age `phase [..., K]` into bounded sinusoidal features `[..., K, dim]`."""
    half = dim // 2
    with fp32_island(phase.device):
        idx = torch.arange(half, device=phase.device, dtype=torch.float32)
        freqs = torch.pow(256.0, -idx / float(max(half, 1)))
        ang = phase.float().unsqueeze(-1) * freqs
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return feat


class CorticalEncoderLayer(nn.Module):
    """Parallel Lower Cortical Layer: Tiled `[W, 2W]` `Concept-RoPE` SWA + `Shared+Routed MoE`."""

    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_concept
        d_s = cfg.d_s
        self.n_heads = cfg.n_heads
        self.d_head = D // cfg.n_heads
        self.window_size = cfg.window_size

        self.norm_attn = RMSNorm(D, eps=cfg.eps)
        self.w_q = nn.Linear(D, D, bias=False)
        self.w_uk = nn.Linear(d_s, D, bias=False)
        self.w_uv = nn.Linear(d_s, D, bias=False)
        self.w_o = nn.Linear(D, D, bias=False)
        nn.init.normal_(self.w_o.weight, std=0.02)

        self.moe = SharedAndRoutedMoE(
            d_model=D,
            routed_hidden=cfg.cort_expert_hidden,
            shared_hidden=cfg.shared_expert_hidden,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k_experts,
            eps=cfg.eps,
        )

    def forward_parallel(
        self,
        x: torch.Tensor,
        c_kv: torch.Tensor,
        doc_ids: torch.Tensor,
        pos_in_doc: torch.Tensor,
        scope_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tiled O(M * W) causal sliding-window attention (`0 <= m - n < W`) across `[M_total, D]`."""
        M, D = x.shape
        if M == 0:
            return x, torch.zeros((), device=x.device, dtype=torch.float32)

        W = self.window_size
        H = self.n_heads
        Dh = self.d_head
        device = x.device
        orig_dtype = x.dtype

        x_n = self.norm_attn(x)
        q = self.w_q(x_n).view(M, H, Dh)
        k = self.w_uk(c_kv).view(M, H, Dh)
        v = self.w_uv(c_kv).view(M, H, Dh)

        q_rot = apply_concept_rope_2d(
            q, pos_in_doc, scope_depth, theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period
        )
        k_rot = apply_concept_rope_2d(
            k, pos_in_doc, scope_depth, theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period
        )

        # Pad M up to a multiple of W for zero-loop `[N_tiles, H, W, 2W]` batched attention
        n_tiles = (M + W - 1) // W
        M_pad = n_tiles * W
        pad = M_pad - M
        if pad > 0:
            z_q = torch.zeros((pad, H, Dh), device=device, dtype=q_rot.dtype)
            q_p = torch.cat([q_rot, z_q], dim=0)
            k_p = torch.cat([k_rot, z_q], dim=0)
            v_p = torch.cat([v, torch.zeros((pad, H, Dh), device=device, dtype=v.dtype)], dim=0)
            doc_p = torch.cat([doc_ids, torch.full((pad,), -2, device=device, dtype=torch.int64)], dim=0)
        else:
            q_p, k_p, v_p, doc_p = q_rot, k_rot, v, doc_ids

        q_tiles = q_p.view(n_tiles, W, H, Dh).permute(0, 2, 1, 3)  # [Nt, H, W, Dh]
        k_tiles = k_p.view(n_tiles, W, H, Dh).permute(0, 2, 1, 3)
        v_tiles = v_p.view(n_tiles, W, H, Dh).permute(0, 2, 1, 3)
        doc_tiles = doc_p.view(n_tiles, W)

        # Prepend 1 dummy tile (`doc_id = -1`) to form overlapping `[Nt, ..., 2W]` key/value windows
        zero_kv_tile = torch.zeros((1, H, W, Dh), device=device, dtype=k_tiles.dtype)
        k_prev = torch.cat([zero_kv_tile, k_tiles[:-1]], dim=0)
        v_prev = torch.cat([zero_kv_tile.to(v_tiles.dtype), v_tiles[:-1]], dim=0)
        k_2w = torch.cat([k_prev, k_tiles], dim=2)  # [Nt, H, 2W, Dh]
        v_2w = torch.cat([v_prev, v_tiles], dim=2)  # [Nt, H, 2W, Dh]

        doc_prev = torch.cat([torch.full((1, W), -1, device=device, dtype=torch.int64), doc_tiles[:-1]], dim=0)
        doc_2w = torch.cat([doc_prev, doc_tiles], dim=1)  # [Nt, 2W]

        q_idx_local = torch.arange(W, device=device, dtype=torch.int64) + W  # [W] in W..2W-1
        k_idx_local = torch.arange(2 * W, device=device, dtype=torch.int64)  # [2W] in 0..2W-1
        diff = q_idx_local.unsqueeze(1) - k_idx_local.unsqueeze(0)           # [W, 2W]
        in_win = (diff >= 0) & (diff < W)                                    # [W, 2W]
        same_doc = (doc_tiles.unsqueeze(2) == doc_2w.unsqueeze(1)) & (doc_tiles.unsqueeze(2) >= 0)
        valid = same_doc & in_win.unsqueeze(0)                               # [Nt, W, 2W]
        # Ensure padded tail queries (`doc < 0`) have self-diagonal valid to prevent 0/0 NaN
        diag_fallback = (diff == 0).unsqueeze(0) & (doc_tiles.unsqueeze(2) < 0)
        valid_safe = (valid | diag_fallback).unsqueeze(1)                    # [Nt, 1, W, 2W]

        scale = 1.0 / math.sqrt(float(Dh))
        with fp32_island(device):
            scores = torch.matmul(q_tiles.float(), k_2w.float().transpose(-2, -1)) * scale
            scores = torch.where(valid_safe, scores, torch.full_like(scores, -1.0e9))
            attn = F.softmax(scores, dim=-1)
            ctx_tiles = torch.matmul(attn, v_2w.float())  # [Nt, H, W, Dh]

        ctx = ctx_tiles.permute(0, 2, 1, 3).reshape(M_pad, D)[:M].to(orig_dtype)
        x = x + self.w_o(ctx)
        return self.moe(x)

    def step_streaming(
        self,
        x_cur: torch.Tensor,
        ring_kv: torch.Tensor,
        ring_pos: torch.Tensor,
        ring_depth: torch.Tensor,
        ring_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Single-concept O(1) streaming encoder step (`x_cur [B, D]`; `ring_*` ALREADY includes current `m`)."""
        B, D = x_cur.shape
        H = self.n_heads
        Dh = self.d_head
        W = ring_kv.shape[1]
        device = x_cur.device
        orig_dtype = x_cur.dtype

        x_n = self.norm_attn(x_cur)
        q = self.w_q(x_n).view(B, 1, H, Dh)
        k = self.w_uk(ring_kv).view(B, W, H, Dh)
        v = self.w_uv(ring_kv).view(B, W, H, Dh)

        # Latest entry in FIFO ring is at index W - 1
        cur_pos = ring_pos[:, -1:]
        cur_dep = ring_depth[:, -1:]
        q_rot = apply_concept_rope_2d(
            q, cur_pos, cur_dep, theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period
        ).permute(0, 2, 1, 3)  # [B, H, 1, Dh]
        k_rot = apply_concept_rope_2d(
            k, ring_pos, ring_depth, theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period
        ).permute(0, 2, 1, 3)  # [B, H, W, Dh]
        v_h = v.permute(0, 2, 1, 3)  # [B, H, W, Dh]

        valid = ring_valid.view(B, 1, 1, W)
        scale = 1.0 / math.sqrt(float(Dh))
        with fp32_island(device):
            scores = torch.matmul(q_rot.float(), k_rot.float().transpose(-2, -1)) * scale  # [B, H, 1, W]
            scores = torch.where(valid, scores, torch.full_like(scores, -1.0e9))
            attn = F.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v_h.float()).squeeze(2).reshape(B, D)  # [B, D]

        x_mid = x_cur + self.w_o(ctx.to(orig_dtype))
        y, _ = self.moe(x_mid)
        return y


class CorticalDecoderLayer(nn.Module):
    """Parallel Upper Cortical Layer: Unified `(Strict-Past RoPE Window ∪ K=8 Graph Slots)` + `MoE`."""

    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_concept
        d_s = cfg.d_s
        d_r = cfg.d_r
        self.n_heads = cfg.n_heads
        self.d_head = D // cfg.n_heads
        self.window_size = cfg.window_size

        self.norm_attn = RMSNorm(D, eps=cfg.eps)
        self.w_q = nn.Linear(D, D, bias=False)
        self.w_uk = nn.Linear(d_s, D, bias=False)
        self.w_uv = nn.Linear(d_s, D, bias=False)

        # Graph-slot K/V projections conditioned on directed incoming edge summary + temporal phase
        self.w_edge_in = nn.Linear(d_r, D, bias=False)
        self.w_phase = nn.Linear(32, D, bias=False)
        self.w_gk = nn.Linear(D, D, bias=False)
        self.w_gv = nn.Linear(D, D, bias=False)

        self.w_o = nn.Linear(D, D, bias=False)
        nn.init.normal_(self.w_o.weight, std=0.02)

        self.moe = SharedAndRoutedMoE(
            d_model=D,
            routed_hidden=cfg.cort_expert_hidden,
            shared_hidden=cfg.shared_expert_hidden,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k_experts,
            eps=cfg.eps,
        )

    def _enrich_graph_slots(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enrich `slots [N, K, D]` with incoming relation edges & temporal phase -> `(gk, gv, sal_bias)`."""
        N, K, D = slots.shape
        H = self.n_heads
        Dh = self.d_head
        with fp32_island(slots.device):
            w_sal = F.softmax(salience.float(), dim=-1)  # [N, K]
            # Weighted incoming edge summary per target slot j: sum_i w_sal[i] * A[i, j]
            edge_in = torch.einsum("ni,nijr->njr", w_sal, edges.float())  # [N, K, d_r]
            phase_feat = build_slot_phase_features(phase, 32)             # [N, K, 32]
            sal_log_bias = F.logsigmoid(salience.float())                 # [N, K] <= 0

        s_enriched = (
            slots
            + torch.tanh(self.w_edge_in(edge_in.to(slots.dtype)))
            + self.w_phase(phase_feat.to(slots.dtype))
        )
        gk = self.w_gk(s_enriched).view(N, K, H, Dh).permute(0, 2, 1, 3)  # [N, H, K, Dh]
        gv = self.w_gv(s_enriched).view(N, K, H, Dh).permute(0, 2, 1, 3)  # [N, H, K, Dh]
        return gk, gv, sal_log_bias

    def forward_parallel(
        self,
        x: torch.Tensor,
        c_kv: torch.Tensor,
        doc_ids: torch.Tensor,
        pos_in_doc: torch.Tensor,
        seg_start_depth: torch.Tensor,
        seg_end_depth: torch.Tensor,
        pre_slots: torch.Tensor,
        pre_edges: torch.Tensor,
        pre_salience: torch.Tensor,
        pre_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strictly `< m` causal attention over `(W=128 RoPE Window ∪ K=8 Graph Slots)`.

        Returns:
          `(x_out [M, D], f_copy [M, d_s], aux_bal_loss scalar fp32)`
        """
        M, D = x.shape
        d_s = c_kv.shape[-1]
        if M == 0:
            return (
                x,
                torch.zeros((0, d_s), device=x.device, dtype=x.dtype),
                torch.zeros((), device=x.device, dtype=torch.float32),
            )

        W = self.window_size
        H = self.n_heads
        Dh = self.d_head
        K = self.cfg.k_slots
        device = x.device
        orig_dtype = x.dtype

        x_n = self.norm_attn(x)
        q_raw = self.w_q(x_n).view(M, H, Dh)
        k_raw = self.w_uk(c_kv).view(M, H, Dh)
        v_win = self.w_uv(c_kv).view(M, H, Dh)

        # Query at segment m uses strictly causal `seg_start_depth[m]`; completed keys n < m use `seg_end_depth[n]`!
        q_rot = apply_concept_rope_2d(
            q_raw, pos_in_doc, seg_start_depth, theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period
        )
        k_rot = apply_concept_rope_2d(
            k_raw, pos_in_doc, seg_end_depth, theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period
        )
        gk, gv, sal_bias = self._enrich_graph_slots(pre_slots, pre_edges, pre_salience, pre_phase)  # [M, H, K, Dh]

        n_tiles = (M + W - 1) // W
        M_pad = n_tiles * W
        pad = M_pad - M
        if pad > 0:
            z_h = torch.zeros((pad, H, Dh), device=device, dtype=q_rot.dtype)
            q_rot_p = torch.cat([q_rot, z_h], dim=0)
            q_raw_p = torch.cat([q_raw, torch.zeros((pad, H, Dh), device=device, dtype=q_raw.dtype)], dim=0)
            k_rot_p = torch.cat([k_rot, z_h], dim=0)
            v_win_p = torch.cat([v_win, torch.zeros((pad, H, Dh), device=device, dtype=v_win.dtype)], dim=0)
            ckv_p = torch.cat([c_kv, torch.zeros((pad, d_s), device=device, dtype=c_kv.dtype)], dim=0)
            gk_p = torch.cat([gk, torch.zeros((pad, H, K, Dh), device=device, dtype=gk.dtype)], dim=0)
            gv_p = torch.cat([gv, torch.zeros((pad, H, K, Dh), device=device, dtype=gv.dtype)], dim=0)
            sb_p = torch.cat([sal_bias, torch.zeros((pad, K), device=device, dtype=torch.float32)], dim=0)
            doc_p = torch.cat([doc_ids, torch.full((pad,), -2, device=device, dtype=torch.int64)], dim=0)
        else:
            q_rot_p, q_raw_p, k_rot_p, v_win_p, ckv_p = q_rot, q_raw, k_rot, v_win, c_kv
            gk_p, gv_p, sb_p, doc_p = gk, gv, sal_bias, doc_ids

        q_tiles = q_rot_p.view(n_tiles, W, H, Dh).permute(0, 2, 1, 3)      # [Nt, H, W, Dh]
        k_tiles = k_rot_p.view(n_tiles, W, H, Dh).permute(0, 2, 1, 3)
        v_tiles = v_win_p.view(n_tiles, W, H, Dh).permute(0, 2, 1, 3)
        ckv_tiles = ckv_p.view(n_tiles, W, d_s)
        doc_tiles = doc_p.view(n_tiles, W)

        zero_kv_tile = torch.zeros((1, H, W, Dh), device=device, dtype=k_tiles.dtype)
        k_2w = torch.cat([torch.cat([zero_kv_tile, k_tiles[:-1]], dim=0), k_tiles], dim=2)  # [Nt, H, 2W, Dh]
        v_2w = torch.cat([torch.cat([zero_kv_tile.to(v_tiles.dtype), v_tiles[:-1]], dim=0), v_tiles], dim=2)
        ckv_prev = torch.zeros((1, W, d_s), device=device, dtype=ckv_tiles.dtype)
        ckv_2w = torch.cat([torch.cat([ckv_prev, ckv_tiles[:-1]], dim=0), ckv_tiles], dim=1)  # [Nt, 2W, d_s]
        doc_prev = torch.full((1, W), -1, device=device, dtype=torch.int64)
        doc_2w = torch.cat([torch.cat([doc_prev, doc_tiles[:-1]], dim=0), doc_tiles], dim=1)  # [Nt, 2W]

        q_idx_local = torch.arange(W, device=device, dtype=torch.int64) + W
        k_idx_local = torch.arange(2 * W, device=device, dtype=torch.int64)
        diff = q_idx_local.unsqueeze(1) - k_idx_local.unsqueeze(0)          # [W, 2W]
        # STRICT PAST (`1 <= m - n <= W`): segment `m` NEVER sees `c_kv[m]`!
        in_strict_past = (diff >= 1) & (diff <= W)
        valid_win = (
            (doc_tiles.unsqueeze(2) == doc_2w.unsqueeze(1))
            & (doc_tiles.unsqueeze(2) >= 0)
            & in_strict_past.unsqueeze(0)
        ).unsqueeze(1)  # [Nt, 1, W, 2W]

        scale = 1.0 / math.sqrt(float(Dh))
        gk_t = gk_p.view(n_tiles, W, H, K, Dh).permute(0, 2, 1, 3, 4)       # [Nt, H, W, K, Dh]
        gv_t = gv_p.view(n_tiles, W, H, K, Dh).permute(0, 2, 1, 3, 4)
        q_unrot_t = q_raw_p.view(n_tiles, W, H, Dh).permute(0, 2, 1, 3)     # [Nt, H, W, Dh]
        sb_t = sb_p.view(n_tiles, 1, W, K)                                  # [Nt, 1, W, K]

        with fp32_island(device):
            scores_win = torch.matmul(q_tiles.float(), k_2w.float().transpose(-2, -1)) * scale  # [Nt, H, W, 2W]
            scores_win = torch.where(valid_win, scores_win, torch.full_like(scores_win, -1.0e9))
            scores_graph = (
                torch.einsum("nhwd,nhwkd->nhwk", q_unrot_t.float(), gk_t.float()) * scale + sb_t
            )  # [Nt, H, W, K] — always valid!
            all_scores = torch.cat([scores_win, scores_graph], dim=-1)      # [Nt, H, W, 2W + K]
            all_attn = F.softmax(all_scores, dim=-1)
            attn_win = all_attn[..., : 2 * W]                               # [Nt, H, W, 2W]
            attn_graph = all_attn[..., 2 * W :]                             # [Nt, H, W, K]

            ctx_win = torch.matmul(attn_win, v_2w.float())                  # [Nt, H, W, Dh]
            ctx_graph = torch.einsum("nhwk,nhwkd->nhwd", attn_graph, gv_t.float())
            ctx_tot = ctx_win + ctx_graph

            # Head-averaged pointer-copy weights over shared `c_kv` latent stems `[Nt, W, d_s]`
            copy_weights = attn_win.mean(dim=1)                             # [Nt, W, 2W]
            f_copy_tiles = torch.matmul(copy_weights, ckv_2w.float())       # [Nt, W, d_s]

        ctx_flat = ctx_tot.permute(0, 2, 1, 3).reshape(M_pad, D)[:M].to(orig_dtype)
        f_copy = f_copy_tiles.reshape(M_pad, d_s)[:M].to(orig_dtype)
        x_mid = x + self.w_o(ctx_flat)
        x_out, aux_bal = self.moe(x_mid)
        return x_out, f_copy, aux_bal

    def step_streaming(
        self,
        x_cur: torch.Tensor,
        ring_kv: torch.Tensor,
        ring_pos: torch.Tensor,
        ring_depth: torch.Tensor,
        ring_valid: torch.Tensor,
        cur_pos: torch.Tensor,
        cur_start_depth: torch.Tensor,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-concept O(1) streaming decoder step (`ring_*` contains ONLY strict-past `< m` stems!)."""
        B, D = x_cur.shape
        W, d_s = ring_kv.shape[1], ring_kv.shape[2]
        H = self.n_heads
        Dh = self.d_head
        device = x_cur.device
        orig_dtype = x_cur.dtype

        x_n = self.norm_attn(x_cur)
        q_raw = self.w_q(x_n).view(B, 1, H, Dh)
        k_raw = self.w_uk(ring_kv).view(B, W, H, Dh)
        v_win = self.w_uv(ring_kv).view(B, W, H, Dh).permute(0, 2, 1, 3)  # [B, H, W, Dh]

        q_rot = apply_concept_rope_2d(
            q_raw,
            cur_pos.view(B, 1),
            cur_start_depth.view(B, 1),
            theta_base=self.cfg.rope_theta_concept,
            scope_period=self.cfg.rope_period,
        ).permute(0, 2, 1, 3)  # [B, H, 1, Dh]
        k_rot = apply_concept_rope_2d(
            k_raw,
            ring_pos,
            ring_depth,
            theta_base=self.cfg.rope_theta_concept,
            scope_period=self.cfg.rope_period,
        ).permute(0, 2, 1, 3)  # [B, H, W, Dh]
        q_unrot = q_raw.permute(0, 2, 1, 3)  # [B, H, 1, Dh]

        gk, gv, sal_bias = self._enrich_graph_slots(slots, edges, salience, phase)  # [B, H, K, Dh]
        # Restrict ring entries strictly within `[cur_pos - W, cur_pos - 1]`
        diff = cur_pos.view(B, 1) - ring_pos  # [B, W]
        valid_w = (ring_valid & (diff >= 1) & (diff <= W)).view(B, 1, 1, W)

        scale = 1.0 / math.sqrt(float(Dh))
        with fp32_island(device):
            scores_w = torch.matmul(q_rot.float(), k_rot.float().transpose(-2, -1)) * scale  # [B, H, 1, W]
            scores_w = torch.where(valid_w, scores_w, torch.full_like(scores_w, -1.0e9))
            scores_g = torch.matmul(q_unrot.float(), gk.float().transpose(-2, -1)) * scale + sal_bias.view(
                B, 1, 1, self.cfg.k_slots
            )  # [B, H, 1, K]
            all_attn = F.softmax(torch.cat([scores_w, scores_g], dim=-1), dim=-1)
            attn_w = all_attn[..., :W]   # [B, H, 1, W]
            attn_g = all_attn[..., W:]   # [B, H, 1, K]
            ctx = (torch.matmul(attn_w, v_win.float()) + torch.matmul(attn_g, gv.float())).squeeze(2).reshape(B, D)
            copy_w = attn_w.mean(dim=1)  # [B, 1, W]
            f_copy = torch.matmul(copy_w, ring_kv.float()).squeeze(1)  # [B, d_s]

        x_mid = x_cur + self.w_o(ctx.to(orig_dtype))
        y, _ = self.moe(x_mid)
        return y, f_copy.to(orig_dtype)


class CorticalSandwich(nn.Module):
    """Unified Lower Cortical Encoder + Shared 48KB MLA Ring + Upper Cortical Decoder."""

    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_concept
        d_s = cfg.d_s
        d_l = cfg.d_l

        # Shared 48KB MLA-latent KV compression (`f_stem + w_kv_c(c_slow) -> norm_kv -> [M, d_s]`)
        self.w_kv_c = nn.Linear(D, d_s, bias=False)
        self.norm_kv = RMSNorm(d_s, eps=cfg.eps)
        nn.init.normal_(self.w_kv_c.weight, std=0.02)

        # Lower Cortical Encoder stack (`n_enc_layers`)
        self.enc_layers = nn.ModuleList([CorticalEncoderLayer(cfg) for _ in range(cfg.n_enc_layers)])
        self.norm_enc_out = RMSNorm(D, eps=cfg.eps)

        # Learnable BOS concept vector for predicting segment 0 of each document
        self.bos_concept = nn.Parameter(torch.randn(D) * 0.02)
        self.w_lang_dec = nn.Linear(d_l, D, bias=False)
        nn.init.normal_(self.w_lang_dec.weight, std=0.02)

        # Upper Cortical Decoder stack (`n_dec_layers`)
        self.dec_layers = nn.ModuleList([CorticalDecoderLayer(cfg) for _ in range(cfg.n_dec_layers)])
        self.norm_dec_out = RMSNorm(D, eps=cfg.eps)

        # Position-aware pointer-copy stem residual (initialized to 0 so Stage 1->2 transition is smooth!)
        self.w_copy_out = nn.Linear(d_s, d_s, bias=False)
        nn.init.zeros_(self.w_copy_out.weight)

    def init_carry_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> CorticalCarryState:
        dev = device if device is not None else self.bos_concept.device
        W = self.cfg.window_size
        d_s = self.cfg.d_s
        D = self.cfg.d_concept
        return CorticalCarryState(
            ring_kv=torch.zeros((batch_size, W, d_s), device=dev, dtype=dtype),
            ring_pos=torch.zeros((batch_size, W), device=dev, dtype=torch.int64),
            ring_depth=torch.zeros((batch_size, W), device=dev, dtype=torch.int32),
            ring_valid=torch.zeros((batch_size, W), device=dev, dtype=torch.bool),
            doc_pos=torch.zeros((batch_size,), device=dev, dtype=torch.int64),
            ast_depth=torch.zeros((batch_size,), device=dev, dtype=torch.int32),
            prev_z_ref=self.bos_concept.detach().to(device=dev, dtype=dtype).unsqueeze(0).expand(batch_size, D).clone(),
            has_prev_seg=torch.zeros((batch_size,), device=dev, dtype=torch.bool),
        )

    def build_shared_kv(self, f_stem: torch.Tensor, c_slow: torch.Tensor) -> torch.Tensor:
        """Build `O(1)` shared MLA-latent `c_kv [..., d_s]` fusing spelling stem and slow semantics."""
        return self.norm_kv(f_stem + self.w_kv_c(c_slow.to(f_stem.dtype)))

    def encode_parallel(
        self,
        c_slow: torch.Tensor,
        f_stem: torch.Tensor,
        doc_ids: torch.Tensor,
        pos_in_doc: torch.Tensor,
        seg_end_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Shared KV compression + Lower Cortical Encoder across all `[M_total, D]` segments."""
        c_kv = self.build_shared_kv(f_stem, c_slow)
        x = c_slow
        bal_sum = torch.zeros((), device=c_slow.device, dtype=torch.float32)
        for layer in self.enc_layers:
            x, bal_l = layer.forward_parallel(x, c_kv, doc_ids, pos_in_doc, seg_end_depth)
            bal_sum = bal_sum + bal_l
        h_enc = self.norm_enc_out(x)
        bal_mean = bal_sum / float(max(len(self.enc_layers), 1))
        return h_enc, c_kv, bal_mean

    def decode_parallel(
        self,
        z_ref_prev: torch.Tensor,
        l_reg_seg: torch.Tensor,
        c_kv: torch.Tensor,
        doc_ids: torch.Tensor,
        pos_in_doc: torch.Tensor,
        seg_start_depth: torch.Tensor,
        seg_end_depth: torch.Tensor,
        pre_slots: torch.Tensor,
        pre_edges: torch.Tensor,
        pre_salience: torch.Tensor,
        pre_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Upper Cortical Decoder across all `[M_total, D]` segments strictly from `< m` inputs."""
        x = z_ref_prev + self.w_lang_dec(l_reg_seg.to(z_ref_prev.dtype))
        f_copy_last = torch.zeros((x.shape[0], self.cfg.d_s), device=x.device, dtype=x.dtype)
        bal_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        for layer in self.dec_layers:
            x, f_copy_last, bal_l = layer.forward_parallel(
                x=x,
                c_kv=c_kv,
                doc_ids=doc_ids,
                pos_in_doc=pos_in_doc,
                seg_start_depth=seg_start_depth,
                seg_end_depth=seg_end_depth,
                pre_slots=pre_slots,
                pre_edges=pre_edges,
                pre_salience=pre_salience,
                pre_phase=pre_phase,
            )
            bal_sum = bal_sum + bal_l
        z_dec = self.norm_dec_out(x)
        copy_stem = self.w_copy_out(f_copy_last)
        bal_mean = bal_sum / float(max(len(self.dec_layers), 1))
        return z_dec, copy_stem, bal_mean

    def push_ring_step(
        self,
        carry: CorticalCarryState,
        c_kv_step: torch.Tensor,
        seg_end_depth: torch.Tensor,
        unpin_reset: Optional[torch.Tensor] = None,
    ) -> CorticalCarryState:
        """O(1) FIFO shift appending 1 completed concept into `CorticalCarryState`."""
        cur_pos = carry.doc_pos
        new_kv = torch.cat([carry.ring_kv[:, 1:, :], c_kv_step.to(carry.ring_kv.dtype).unsqueeze(1)], dim=1)
        new_pos = torch.cat([carry.ring_pos[:, 1:], cur_pos.unsqueeze(1)], dim=1)
        new_dep = torch.cat([carry.ring_depth[:, 1:], seg_end_depth.to(torch.int32).unsqueeze(1)], dim=1)
        ones_b = torch.ones((carry.ring_valid.shape[0], 1), device=carry.ring_valid.device, dtype=torch.bool)
        new_val = torch.cat([carry.ring_valid[:, 1:], ones_b], dim=1)
        next_doc_pos = cur_pos + 1

        next_ast = seg_end_depth.to(torch.int32)
        if unpin_reset is not None:
            next_ast = torch.where(unpin_reset.bool(), torch.zeros_like(next_ast), next_ast)

        return CorticalCarryState(
            ring_kv=new_kv,
            ring_pos=new_pos,
            ring_depth=new_dep,
            ring_valid=new_val,
            doc_pos=next_doc_pos,
            ast_depth=next_ast,
            prev_z_ref=carry.prev_z_ref,
            has_prev_seg=carry.has_prev_seg,
        )

    def encode_step_streaming(
        self,
        c_slow_step: torch.Tensor,
        f_stem_step: torch.Tensor,
        seg_end_depth: torch.Tensor,
        carry: CorticalCarryState,
        unpin_reset: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, CorticalCarryState]:
        """Ingest 1 concept into O(1) ring and compute `h_enc [B, D]` matching `encode_parallel` 1-to-1."""
        c_kv_step = self.build_shared_kv(f_stem_step, c_slow_step)
        carry_next = self.push_ring_step(carry, c_kv_step, seg_end_depth, unpin_reset=unpin_reset)
        x = c_slow_step
        for layer in self.enc_layers:
            x = layer.step_streaming(
                x, carry_next.ring_kv, carry_next.ring_pos, carry_next.ring_depth, carry_next.ring_valid
            )
        return self.norm_enc_out(x), carry_next

    def decode_step_streaming(
        self,
        l_reg: torch.Tensor,
        carry: CorticalCarryState,
        slots: torch.Tensor,
        edges: torch.Tensor,
        salience: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next concept `m` strictly from `< m` `carry` + `graph` matching `decode_parallel` 1-to-1."""
        B = l_reg.shape[0]
        bos = self.bos_concept.to(device=l_reg.device, dtype=l_reg.dtype).unsqueeze(0).expand(B, -1)
        z_in = torch.where(carry.has_prev_seg.view(B, 1), carry.prev_z_ref.to(l_reg.dtype), bos)
        x = z_in + self.w_lang_dec(l_reg)
        f_copy_last = torch.zeros((B, self.cfg.d_s), device=l_reg.device, dtype=l_reg.dtype)
        for layer in self.dec_layers:
            x, f_copy_last = layer.step_streaming(
                x_cur=x,
                ring_kv=carry.ring_kv.to(l_reg.dtype),
                ring_pos=carry.ring_pos,
                ring_depth=carry.ring_depth,
                ring_valid=carry.ring_valid,
                cur_pos=carry.doc_pos,
                cur_start_depth=carry.ast_depth,
                slots=slots.to(l_reg.dtype),
                edges=edges.to(l_reg.dtype),
                salience=salience,
                phase=phase,
            )
        return self.norm_dec_out(x), self.w_copy_out(f_copy_last)
