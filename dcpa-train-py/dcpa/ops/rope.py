"""Orthogonal Holographic RoPE phase binding R_Theta(tau) and inverse R_Theta(-tau).

Uses harmonic frequencies theta_k = 2*pi*((k % H) + 1) / P (H=16, P=33) so that:
  - R_Theta(tau) in SO(d_s) is strictly orthogonal,
  - R_Theta(-tau) @ R_Theta(tau) == I identically,
  - For every |tau' - tau| in {1..32}, average cross-position cosine correlation is -1/32 (-0.03125),
    enabling near-lossless holographic superposition and single-vector spelling demodulation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from dcpa.ops.fp32 import fp32_island

_ROPE_FREQ_CACHE: dict[tuple[int, float, int, str], torch.Tensor] = {}
_ROPE_TABLE_CACHE: dict[tuple[int, float, int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
_CONCEPT_ROPE_CACHE: dict[tuple[int, float, str], torch.Tensor] = {}


def build_concept_rope_freqs(
    dim: int,
    theta_base: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build standard geometric rotary inverse-frequency bank `theta_base^(-2i / dim)` in float32."""
    if dim % 2 != 0:
        raise ValueError(f"Concept-RoPE dimension must be even, got {dim}")
    dev = torch.device("cpu") if device is None else torch.device(device)
    key = (int(dim), float(theta_base), str(dev))
    freqs = _CONCEPT_ROPE_CACHE.get(key)
    if freqs is None:
        half = dim // 2
        with torch.inference_mode(False), torch.no_grad():
            idx = torch.arange(half, device=dev, dtype=torch.float32)
            freqs = torch.pow(float(theta_base), -2.0 * idx / float(max(dim, 2)))
        if not torch.compiler.is_compiling():
            _CONCEPT_ROPE_CACHE[key] = freqs
    return freqs


def apply_concept_rope_2d(
    x: torch.Tensor,
    concept_pos: torch.Tensor,
    scope_depth: torch.Tensor,
    theta_base: float = 10000.0,
    scope_period: float = 33.0,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply dual-coordinate Concept-RoPE (`3/4` relative concept order + `1/4` AST scope depth).

    Args:
      x:           `[..., H, D_h]` or `[..., D_h]` where `D_h` is a multiple of 8.
      concept_pos: `[...]` float/int concept index within document (broadcastable across `H`).
      scope_depth: `[...]` float/int structural bracket/scope depth in `[0, 31]`.
      inverse:     if True, applies exact transpose `R(-pos, -depth)` (used in O(1) ring relative query).
    """
    orig_dtype = x.dtype
    d_head = x.shape[-1]
    if d_head % 2 != 0:
        raise ValueError(f"apply_concept_rope_2d requires even head dim, got {d_head}")
    half = d_head // 2
    scope_pairs = max(1, half // 4)
    pos_pairs = half - scope_pairs

    with fp32_island(x.device):
        pos_freqs = build_concept_rope_freqs(pos_pairs * 2, theta_base=theta_base, device=x.device)
        scope_idx = torch.arange(scope_pairs, device=x.device, dtype=torch.float32) + 1.0
        scope_freqs = (2.0 * math.pi / float(scope_period)) * scope_idx

        c_pos = concept_pos.to(dtype=torch.float32)
        s_dep = scope_depth.to(dtype=torch.float32)
        if inverse:
            c_pos = -c_pos
            s_dep = -s_dep

        # Align broadcasting when x has an explicit head axis `[..., H, D_h]`
        while c_pos.ndim < x.ndim - 1:
            c_pos = c_pos.unsqueeze(-1)
        while s_dep.ndim < x.ndim - 1:
            s_dep = s_dep.unsqueeze(-1)

        angles_pos = c_pos.unsqueeze(-1) * pos_freqs
        angles_scope = s_dep.unsqueeze(-1) * scope_freqs
        angles = torch.cat([angles_pos, angles_scope], dim=-1)  # [..., half]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        x_f32 = x.float()
        x_even = x_f32[..., 0::2]
        x_odd = x_f32[..., 1::2]
        y_even = x_even * cos_a - x_odd * sin_a
        y_odd = x_even * sin_a + x_odd * cos_a
        out = torch.stack((y_even, y_odd), dim=-1).flatten(-2)
    return out.to(orig_dtype)


def compute_causal_byte_scope_depth(
    raw_bytes: torch.Tensor,
    doc_reset_mask: Optional[torch.Tensor] = None,
    init_depth: Optional[torch.Tensor] = None,
    max_depth: int = 31,
    clamp_output: bool = True,
) -> torch.Tensor:
    """Strictly causal O(T) vectorized structural bracket/frame nesting depth `[B, T]` AFTER each byte `t`.

    Processes `raw_bytes[:, t]` causally (`+1` on `([{`, `-1` on `)]}` floored at `0`), resetting cleanly
    BEFORE byte `t` when `doc_reset_mask[:, t]` is True, and resetting to `0` AFTER byte `t` on control
    frame boundaries (`0x04`, `0x13`, `0x1E`, `\\x1dE\\n`). Matches `dcpa-core-rs` integer counter step-for-step!
    """
    B, T = raw_bytes.shape
    device = raw_bytes.device
    if T == 0:
        return torch.zeros((B, 0), device=device, dtype=torch.int32)

    rb = raw_bytes.to(dtype=torch.long)
    pre_reset = (
        doc_reset_mask.bool().clone()
        if doc_reset_mask is not None
        else torch.zeros((B, T), device=device, dtype=torch.bool)
    )

    post_reset = (rb == 0x04) | (rb == 0x13) | (rb == 0x1E)
    if T >= 3:
        z1 = torch.zeros((B, 1), device=device, dtype=torch.long)
        z2 = torch.zeros((B, 2), device=device, dtype=torch.long)
        rb_m1 = torch.cat([z1, rb[:, :-1]], dim=1)
        rb_m2 = torch.cat([z2, rb[:, :-2]], dim=1)
        scratch_end = (rb_m2 == 0x1D) & (rb_m1 == ord("E")) & (rb == 0x0A)
        if doc_reset_mask is not None:
            drm = doc_reset_mask.bool()
            drm_p1 = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.bool), drm[:, :-1]], dim=1)
            scratch_end = scratch_end & ~(drm | drm_p1)
        post_reset = post_reset | scratch_end

    is_open = (rb == 0x28) | (rb == 0x5B) | (rb == 0x7B)
    is_close = (rb == 0x29) | (rb == 0x5D) | (rb == 0x7D)
    delta = is_open.to(torch.int32) - is_close.to(torch.int32)
    delta = torch.where(post_reset, torch.zeros_like(delta), delta)

    reset_group = pre_reset | post_reset
    if init_depth is not None:
        d0 = init_depth.to(device=device, dtype=torch.int32).clamp_min(0)
        apply_d0 = ~reset_group[:, 0]
        delta[:, 0] = delta[:, 0] + torch.where(apply_d0, d0, torch.zeros_like(d0))
    reset_group[:, 0] = True

    grp_id = torch.cumsum(reset_group.to(torch.int64), dim=1)  # [B, T], >= 1
    cs = torch.cumsum(delta, dim=1)  # [B, T]
    flat_reset_pos = torch.where(
        reset_group,
        torch.arange(T, device=device, dtype=torch.int64).unsqueeze(0).expand(B, T),
        torch.zeros((B, T), device=device, dtype=torch.int64),
    )
    frame_start_t = torch.cummax(flat_reset_pos, dim=1).values
    cs_at_start = cs.gather(1, frame_start_t) - delta.gather(1, frame_start_t)
    rel_cs = cs - cs_at_start

    stride = max(T * 4, 1_048_576)
    deficit = (-rel_cs).clamp_min(0).to(torch.int64)
    packed = grp_id * stride + deficit
    running_max_deficit = (torch.cummax(packed, dim=1).values - grp_id * stride).to(torch.int32)
    raw_depth = (rel_cs + running_max_deficit).clamp_min(0)
    return raw_depth.clamp_max(int(max_depth)) if clamp_output else raw_depth


def compute_segment_scope_depths(
    raw_bytes: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seg_doc_start: torch.Tensor,
    doc_reset_mask: Optional[torch.Tensor] = None,
    init_depth: Optional[torch.Tensor] = None,
    max_depth: int = 31,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute strictly causal `(seg_start_depth [M], seg_end_depth [M], final_raw_depth [B])`.

    - `seg_start_depth[m]` depends ONLY on bytes `< cu_seqlens[m]` (zero intra-segment leakage!).
    - `seg_end_depth[m]` depends ONLY on bytes `< cu_seqlens[m + 1]`.
    - `final_raw_depth [B]` carries the exact unwrapped integer depth across streaming chunks.
    """
    B, T = raw_bytes.shape
    device = raw_bytes.device
    M = int(cu_seqlens.numel()) - 1
    d0 = (
        torch.zeros((B,), device=device, dtype=torch.int32)
        if init_depth is None
        else init_depth.to(device=device, dtype=torch.int32).clamp_min(0)
    )
    if M <= 0 or T <= 0:
        z_m = torch.zeros((max(M, 0),), device=device, dtype=torch.int32)
        return z_m, z_m.clone(), d0

    raw_after = compute_causal_byte_scope_depth(
        raw_bytes, doc_reset_mask=doc_reset_mask, init_depth=d0, max_depth=max_depth, clamp_output=False
    )  # [B, T]
    final_raw = raw_after[:, -1]
    raw_flat = raw_after.reshape(B * T)

    end_byte_idx = (cu_seqlens[1:].to(dtype=torch.int64) - 1).clamp(0, B * T - 1)
    seg_end_raw = raw_flat[end_byte_idx]  # [M]
    seg_end_depth = seg_end_raw.clamp(0, int(max_depth))

    # seg_start_depth[m] is depth BEFORE cu_seqlens[m] (i.e. seg_end_depth[m-1], or d0/0 at doc start!)
    seg_start_depth = torch.zeros_like(seg_end_depth)
    if M > 1:
        seg_start_depth[1:] = seg_end_depth[:-1]
    # For batch item b's very first segment (when init_depth is carried in streaming mode):
    if init_depth is not None and B == 1 and (doc_reset_mask is None or not bool(doc_reset_mask[0, 0].item())):
        seg_start_depth[0] = d0[0].clamp(0, int(max_depth))
        if M > 1:
            # Reset any subsequent document starts (`m > 0`) to 0
            subseq_starts = seg_doc_start.clone()
            subseq_starts[0] = False
            seg_start_depth = torch.where(subseq_starts, torch.zeros_like(seg_start_depth), seg_start_depth)
    else:
        seg_start_depth = torch.where(seg_doc_start, torch.zeros_like(seg_start_depth), seg_start_depth)

    return seg_start_depth, seg_end_depth, final_raw


def build_rope_freqs(
    dim: int,
    period: float = 33.0,
    num_harmonics: int = 16,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    dev = torch.device("cpu") if device is None else torch.device(device)
    key = (int(dim), float(period), int(num_harmonics), str(dev))
    freqs = _ROPE_FREQ_CACHE.get(key)
    if freqs is None:
        half = dim // 2
        with torch.inference_mode(False), torch.no_grad():
            idx = torch.arange(half, device=dev, dtype=torch.float32)
            harmonics = (idx % num_harmonics) + 1.0
            freqs = (2.0 * math.pi / period) * harmonics
        if not torch.compiler.is_compiling():
            _ROPE_FREQ_CACHE[key] = freqs
    return freqs if dtype == torch.float32 else freqs.to(dtype)


def _get_rope_tables(
    dim: int,
    period: float,
    num_harmonics: int,
    max_pos: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached `(cos_table, sin_table)` of shape `[max_pos + 1, dim // 2]` in float32."""
    key = (int(dim), float(period), int(num_harmonics), int(max_pos), str(device))
    cached = _ROPE_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    with torch.inference_mode(False), torch.no_grad():
        freqs = build_rope_freqs(dim, period=period, num_harmonics=num_harmonics, device=device)
        pos = torch.arange(max_pos + 1, device=device, dtype=torch.float32)
        angles = pos.unsqueeze(-1) * freqs
        tables = (torch.cos(angles).contiguous(), torch.sin(angles).contiguous())
    if not torch.compiler.is_compiling():
        _ROPE_TABLE_CACHE[key] = tables
    return tables


def apply_holographic_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    period: float = 33.0,
    num_harmonics: int = 16,
    inverse: bool = False,
    max_pos: Optional[int] = None,
) -> torch.Tensor:
    """Apply R_Theta(+positions) or R_Theta(-positions) to x[..., D].

    Pair convention (matched identically in dcpa-core-rs):
      Even-odd interleaved pairs (x[..., 0::2], x[..., 1::2]):
        y_even = x_even * cos(angle) - x_odd * sin(angle)
        y_odd  = x_even * sin(angle) + x_odd * cos(angle)
    Computed in float32 for phase precision, then cast back to x.dtype.
    When `max_pos` is provided for bounded integer positions (e.g. `tau in 1..max_seg_len+4`),
    uses a cached L1-resident cos/sin lookup table instead of recomputing millions of trig ops.
    """
    orig_dtype = x.dtype
    dim = x.shape[-1]
    with fp32_island(x.device):
        if max_pos is not None and not torch.is_floating_point(positions):
            cos_tbl, sin_tbl = _get_rope_tables(dim, period, num_harmonics, int(max_pos), x.device)
            idx = positions.to(dtype=torch.long).clamp(0, int(max_pos))
            cos_a = cos_tbl[idx]
            sin_a = -sin_tbl[idx] if inverse else sin_tbl[idx]
        else:
            freqs = build_rope_freqs(dim, period=period, num_harmonics=num_harmonics, device=x.device)
            pos = positions.to(dtype=torch.float32)
            if inverse:
                pos = -pos
            angles = pos.unsqueeze(-1) * freqs  # [..., dim // 2]
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)

        x_f32 = x.float()
        x_even = x_f32[..., 0::2]
        x_odd = x_f32[..., 1::2]
        y_even = x_even * cos_a - x_odd * sin_a
        y_odd = x_even * sin_a + x_odd * cos_a
        out = torch.stack((y_even, y_odd), dim=-1).flatten(-2)
    return out.to(orig_dtype)
