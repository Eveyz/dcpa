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
