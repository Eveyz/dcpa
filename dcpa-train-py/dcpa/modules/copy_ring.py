"""Holographic CopyRing for DCPA-Flat v3.3 (WS-2).

Dedicated pure `f_stem` local pointer-copy ring (`[W=128, d_s=192]`, 48 KB in BF16):
  - Strictly operates on pure `f_stem` (ZERO `c_kv_sem` semantic contamination) to preserve
    orthogonal holographic RoPE spelling phase for 1-shot OOV identifier copy.
  - Partitions static `[P, S, ...]` grids into `N_c = S // W` chunks of `W = cfg.window_size`
    and builds left-zero-padded double-width blocks `[P * N_c, 2*W, ...]` per lane (zero
    cross-lane or cross-document leakage).
  - Applies precomputed constant banded strict-past mask `1 <= Δs <= W` intersected with
    same-document mask `(q_doc == kv_doc) & (q_doc >= 0)`.
  - Appends a `(2*W + 1)`-th always-open learnable sink column `b_sink` (value vector = 0)
    so first-segment-of-document queries NEVER produce softmax NaNs in forward or backward.
  - Provides `step_streaming(...)` operating on `[B, W, d_s]` FIFO ring buffer with `< 1e-5`
    parity against batched `forward(...)`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.interfaces import DCPAFlatConfig
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.rmsnorm import RMSNorm
from dcpa.ops.rope import apply_concept_rope_2d


class HolographicCopyRing(nn.Module):
    """Pure `f_stem` chunked local pointer-copy ring with always-open sink column."""

    def __init__(self, cfg: DCPAFlatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_concept = int(cfg.d_concept)
        self.d_s = int(cfg.d_s)
        self.window_size = int(cfg.window_size)
        self.n_heads = int(cfg.n_heads)

        if self.d_s % self.n_heads == 0 and (self.d_s // self.n_heads) >= 8 and (self.d_s // self.n_heads) % 2 == 0:
            self.d_k = self.d_s // self.n_heads
        else:
            self.d_k = int(cfg.d_head)
        self.proj_dim = self.n_heads * self.d_k
        self.scale = 1.0 / math.sqrt(float(self.d_k))

        self.norm_q = RMSNorm(self.d_concept, eps=cfg.eps)
        self.w_q = nn.Linear(self.d_concept, self.proj_dim, bias=False)
        self.w_k = nn.Linear(self.d_s, self.proj_dim, bias=False)
        # Always-open (2*W + 1)-th sink column bias (corresponding value vector is strictly 0)
        self.b_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))

        # Precomputed constant strict-past banded mask `1 <= Δs <= W` of shape [1, 1, W, 2*W]
        W = self.window_size
        q_idx = torch.arange(W, dtype=torch.int64) + W
        k_idx = torch.arange(2 * W, dtype=torch.int64)
        diff = q_idx.unsqueeze(1) - k_idx.unsqueeze(0)  # [W, 2*W]
        band_mask = ((diff >= 1) & (diff <= W)).view(1, 1, W, 2 * W)
        self.register_buffer("band_mask", band_mask, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.w_q.weight, std=1.0 / math.sqrt(self.d_concept))
        nn.init.normal_(self.w_k.weight, std=1.0 / math.sqrt(self.d_s))
        nn.init.zeros_(self.b_sink)

    def forward(
        self,
        q_state: torch.Tensor,
        f_stem: torch.Tensor,
        seg_doc_id: torch.Tensor,
        seg_start_depth: Optional[torch.Tensor] = None,
        seg_end_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute chunked pure-`f_stem` pointer copy over static `[P, S, ...]` grid.

        Args:
          q_state:         `[P, S, D]` cortical query hidden states (e.g. L8 output).
          f_stem:          `[P, S, d_s]` PURE holographic stems (zero `c_kv_sem`).
          seg_doc_id:      `[P, S]` int32 (`>= 0` valid, unique negative sentinel invalid).
          seg_start_depth: optional `[P, S]` int32 causal AST depth at segment start (for queries).
          seg_end_depth:   optional `[P, S]` int32 causal AST depth at segment end (for past keys).

        Returns:
          f_copy: `[P, S, d_s]` pure copied holographic stem residual (same dtype as `f_stem`).
        """
        P, S, D = q_state.shape
        W = self.window_size
        if S % W != 0:
            raise ValueError(f"S ({S}) must be divisible by window_size ({W})")
        if f_stem.shape != (P, S, self.d_s):
            raise ValueError(f"f_stem shape {tuple(f_stem.shape)} != ({P}, {S}, {self.d_s})")

        N_c = S // W
        H = self.n_heads
        d_k = self.d_k
        d_s = self.d_s
        device = q_state.device
        orig_dtype = f_stem.dtype

        pos = torch.arange(S, device=device, dtype=torch.int32).unsqueeze(0).expand(P, S)
        q_depth = (
            torch.zeros((P, S), device=device, dtype=torch.int32)
            if seg_start_depth is None
            else seg_start_depth.to(torch.int32)
        )
        k_depth = (
            torch.zeros((P, S), device=device, dtype=torch.int32)
            if seg_end_depth is None
            else seg_end_depth.to(torch.int32)
        )

        q_raw = self.w_q(self.norm_q(q_state)).view(P, S, H, d_k)
        k_raw = self.w_k(f_stem).view(P, S, H, d_k)

        q_rot = apply_concept_rope_2d(
            q_raw,
            pos,
            q_depth,
            theta_base=self.cfg.rope_theta_concept,
            scope_period=self.cfg.rope_period,
        )
        k_rot = apply_concept_rope_2d(
            k_raw,
            pos,
            k_depth,
            theta_base=self.cfg.rope_theta_concept,
            scope_period=self.cfg.rope_period,
        )

        # Partition strictly within each lane [P, N_c, W, ...] before flattening P * N_c
        q_ch = q_rot.view(P, N_c, W, H, d_k)
        k_ch = k_rot.view(P, N_c, W, H, d_k)
        v_ch = f_stem.view(P, N_c, W, d_s)
        doc_ch = seg_doc_id.view(P, N_c, W)

        # Left-zero-padded double-width key/value/doc blocks [P * N_c, 2*W, ...]
        k_zero = torch.zeros((P, 1, W, H, d_k), device=device, dtype=k_ch.dtype)
        v_zero = torch.zeros((P, 1, W, d_s), device=device, dtype=v_ch.dtype)
        doc_pad = torch.full((P, 1, W), -1, device=device, dtype=doc_ch.dtype)

        if N_c > 1:
            k_prev = torch.cat([k_zero, k_ch[:, :-1]], dim=1)
            v_prev = torch.cat([v_zero, v_ch[:, :-1]], dim=1)
            doc_prev = torch.cat([doc_pad, doc_ch[:, :-1]], dim=1)
        else:
            k_prev = k_zero
            v_prev = v_zero
            doc_prev = doc_pad

        B_blk = P * N_c
        q_blk = q_ch.reshape(B_blk, W, H, d_k).permute(0, 2, 1, 3)                  # [P*N_c, H, W, d_k]
        k_2w = torch.cat([k_prev, k_ch], dim=2).reshape(B_blk, 2 * W, H, d_k).permute(0, 2, 1, 3)  # [P*N_c, H, 2W, d_k]
        v_2w = torch.cat([v_prev, v_ch], dim=2).reshape(B_blk, 2 * W, d_s)          # [P*N_c, 2W, d_s]
        doc_q = doc_ch.reshape(B_blk, W)                                            # [P*N_c, W]
        doc_2w = torch.cat([doc_prev, doc_ch], dim=2).reshape(B_blk, 2 * W)         # [P*N_c, 2W]

        same_doc = (doc_q.unsqueeze(2) == doc_2w.unsqueeze(1)) & (doc_q.unsqueeze(2) >= 0)  # [P*N_c, W, 2W]
        valid_win = same_doc.unsqueeze(1) & self.band_mask                          # [P*N_c, 1, W, 2W]

        with fp32_island(device):
            scores_win = torch.matmul(q_blk.float(), k_2w.float().transpose(-2, -1)) * self.scale
            scores_win = torch.where(valid_win, scores_win, torch.full_like(scores_win, float("-inf")))
            sink_col = self.b_sink.float().view(1, H, 1, 1).expand(B_blk, H, W, 1)
            all_scores = torch.cat([scores_win, sink_col], dim=-1)                  # [P*N_c, H, W, 2W + 1]
            all_attn = F.softmax(all_scores, dim=-1)                                # guaranteed NaN-free
            attn_win = all_attn[..., : 2 * W]                                       # sink value == 0
            copy_weights = attn_win.mean(dim=1)                                     # [P*N_c, W, 2W]
            f_copy_blk = torch.matmul(copy_weights, v_2w.float())                   # [P*N_c, W, d_s]

        return f_copy_blk.view(P, S, d_s).to(orig_dtype)

    def init_ring_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate empty O(1) FIFO ring buffers `(ring_stem, ring_pos, ring_depth, ring_valid)`."""
        dev = self.b_sink.device if device is None else device
        W = self.window_size
        return (
            torch.zeros((batch_size, W, self.d_s), device=dev, dtype=dtype),
            torch.zeros((batch_size, W), device=dev, dtype=torch.int32),
            torch.zeros((batch_size, W), device=dev, dtype=torch.int32),
            torch.zeros((batch_size, W), device=dev, dtype=torch.bool),
        )

    @staticmethod
    def push_ring(
        ring_stem: torch.Tensor,
        ring_pos: torch.Tensor,
        ring_depth: torch.Tensor,
        ring_valid: torch.Tensor,
        new_stem: torch.Tensor,
        new_pos: torch.Tensor,
        new_depth: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """O(1) FIFO shift appending 1 completed pure `f_stem [B, d_s]` into the `[B, W, d_s]` ring."""
        B = ring_stem.shape[0]
        dev = ring_stem.device
        dep = (
            torch.zeros((B,), device=dev, dtype=torch.int32)
            if new_depth is None
            else new_depth.to(device=dev, dtype=torch.int32).view(B)
        )
        new_ring_stem = torch.cat([ring_stem[:, 1:, :], new_stem.to(ring_stem.dtype).unsqueeze(1)], dim=1)
        new_ring_pos = torch.cat([ring_pos[:, 1:], new_pos.to(ring_pos.dtype).view(B, 1)], dim=1)
        new_ring_depth = torch.cat([ring_depth[:, 1:], dep.unsqueeze(1)], dim=1)
        ones_b = torch.ones((B, 1), device=dev, dtype=torch.bool)
        new_ring_valid = torch.cat([ring_valid[:, 1:], ones_b], dim=1)
        return new_ring_stem, new_ring_pos, new_ring_depth, new_ring_valid

    def step_streaming(
        self,
        q_state: torch.Tensor,
        ring_stem: torch.Tensor,
        ring_pos: torch.Tensor,
        ring_valid: torch.Tensor,
        cur_pos: torch.Tensor,
        ring_depth: Optional[torch.Tensor] = None,
        cur_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single-segment O(1) streaming pointer copy operating on `[B, W, d_s]` pure `f_stem` ring.

        Matches batched `forward(...)` to `< 1e-5` across arbitrary document lengths.
        """
        B, _ = q_state.shape
        W = ring_stem.shape[1]
        H = self.n_heads
        d_k = self.d_k
        device = q_state.device
        orig_dtype = ring_stem.dtype

        q_depth = (
            torch.zeros((B, 1), device=device, dtype=torch.int32)
            if cur_depth is None
            else cur_depth.to(torch.int32).view(B, 1)
        )
        k_depth = (
            torch.zeros((B, W), device=device, dtype=torch.int32)
            if ring_depth is None
            else ring_depth.to(torch.int32)
        )

        q_raw = self.w_q(self.norm_q(q_state)).view(B, 1, H, d_k)
        k_raw = self.w_k(ring_stem).view(B, W, H, d_k)

        q_rot = apply_concept_rope_2d(
            q_raw,
            cur_pos.to(torch.int32).view(B, 1),
            q_depth,
            theta_base=self.cfg.rope_theta_concept,
            scope_period=self.cfg.rope_period,
        ).permute(0, 2, 1, 3)  # [B, H, 1, d_k]
        k_rot = apply_concept_rope_2d(
            k_raw,
            ring_pos.to(torch.int32),
            k_depth,
            theta_base=self.cfg.rope_theta_concept,
            scope_period=self.cfg.rope_period,
        ).permute(0, 2, 1, 3)  # [B, H, W, d_k]

        diff = cur_pos.to(torch.int64).view(B, 1) - ring_pos.to(torch.int64)  # [B, W]
        valid_w = (ring_valid.bool() & (diff >= 1) & (diff <= self.window_size)).view(B, 1, 1, W)

        with fp32_island(device):
            scores_w = torch.matmul(q_rot.float(), k_rot.float().transpose(-2, -1)) * self.scale  # [B, H, 1, W]
            scores_w = torch.where(valid_w, scores_w, torch.full_like(scores_w, float("-inf")))
            sink_col = self.b_sink.float().view(1, H, 1, 1).expand(B, H, 1, 1)
            all_attn = F.softmax(torch.cat([scores_w, sink_col], dim=-1), dim=-1)  # [B, H, 1, W + 1]
            attn_w = all_attn[..., :W]
            copy_w = attn_w.mean(dim=1)  # [B, 1, W]
            f_copy = torch.matmul(copy_w, ring_stem.float()).squeeze(1)  # [B, d_s]

        return f_copy.to(orig_dtype)


__all__ = ["HolographicCopyRing"]
