"""Orthogonal Holographic RoPE phase binding R_Theta(tau) and inverse R_Theta(-tau).

Uses harmonic frequencies theta_k = 2*pi*((k % H) + 1) / P (H=16, P=33) so that:
  - R_Theta(tau) in SO(d_s) is strictly orthogonal,
  - R_Theta(-tau) @ R_Theta(tau) == I identically,
  - For every |tau' - tau| in {1..32}, average cross-position cosine correlation is -1/32 (-0.03125),
    enabling near-lossless holographic superposition and single-vector spelling demodulation.
"""

from __future__ import annotations

import math

import torch


def build_rope_freqs(
    dim: int,
    period: float = 33.0,
    num_harmonics: int = 16,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    half = dim // 2
    idx = torch.arange(half, device=device, dtype=torch.float32)
    harmonics = (idx % num_harmonics) + 1.0
    freqs = (2.0 * math.pi / period) * harmonics
    return freqs.to(dtype)


def apply_holographic_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    period: float = 33.0,
    num_harmonics: int = 16,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply R_Theta(+positions) or R_Theta(-positions) to x[..., D].

    Pair convention (matched identically in dcpa-core-rs):
      Even-odd interleaved pairs (x[..., 0::2], x[..., 1::2]):
        y_even = x_even * cos(angle) - x_odd * sin(angle)
        y_odd  = x_even * sin(angle) + x_odd * cos(angle)
    Computed in float32 for phase precision, then cast back to x.dtype.
    """
    orig_dtype = x.dtype
    dim = x.shape[-1]
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

    out = torch.empty_like(x_f32)
    out[..., 0::2] = y_even
    out[..., 1::2] = y_odd
    return out.to(orig_dtype)
