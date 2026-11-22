"""Zero-padding Ragged segment primitives driven exclusively by cu_seqlens (int32).

Enforces Commandment #4: NEVER constructs [M_total, max_seg_len, D] padded dense tensors.
All segment reductions, softmaxes, and intra-segment prefix sums operate directly on flat 1D buffers.
"""

from __future__ import annotations

import torch


def build_cu_seqlens(spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct int32 cu_seqlens [M_total + 1] and per-batch segment counts [B] from spikes [B, T].

    Each row spikes[b] MUST have spikes[b, T-1] == True (guaranteed by LIF scan).
    Returns:
      cu_seqlens: [M_total + 1] int32 cumulative byte offsets over flattened B*T stream.
      seg_counts: [B] int32 number of segments per batch item.
    """
    if spikes.ndim != 2:
        raise ValueError(f"Expected 2D spikes [B, T], got shape {tuple(spikes.shape)}")
    B, T = spikes.shape
    device = spikes.device
    spikes_bool = spikes.bool()

    seg_counts = spikes_bool.sum(dim=1, dtype=torch.int32)
    flat_spikes = spikes_bool.reshape(-1)
    # 1-based end offsets in the flattened [B*T] stream
    end_offsets = torch.nonzero(flat_spikes, as_tuple=False).squeeze(-1).to(torch.int32) + 1
    zero = torch.zeros(1, device=device, dtype=torch.int32)
    cu_seqlens = torch.cat([zero, end_offsets], dim=0)
    return cu_seqlens, seg_counts


def ragged_local_offsets(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-byte segment id, 1-based local offset tau, segment lengths, and end indices.

    Args:
      cu_seqlens: [M_total + 1] int32
      total_tokens: B * T

    Returns:
      seg_id: [N] int64 segment index in 0..M_total-1 for each flat byte.
      tau: [N] int64 1-based position within its segment (1..l_m).
      seg_lens: [M_total] int64 length l_m of each segment.
      seg_end_idx: [M_total] int64 index of the last byte t_m of each segment in 0..N-1.
    """
    device = cu_seqlens.device
    starts = cu_seqlens[:-1].to(torch.int64)
    ends = cu_seqlens[1:].to(torch.int64)
    seg_lens = ends - starts
    seg_end_idx = ends - 1

    m_total = seg_lens.numel()
    seg_id = torch.repeat_interleave(
        torch.arange(m_total, device=device, dtype=torch.int64),
        seg_lens,
    )
    global_idx = torch.arange(total_tokens, device=device, dtype=torch.int64)
    tau = (global_idx - starts[seg_id]) + 1
    return seg_id, tau, seg_lens, seg_end_idx


def ragged_segment_softmax(
    scores: torch.Tensor,
    seg_id: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Numerically stable 1D ragged segmented softmax strictly in float32."""
    scores_f32 = scores.float()
    max_per_seg = torch.full(
        (num_segments,),
        float("-inf"),
        device=scores.device,
        dtype=torch.float32,
    )
    max_per_seg.scatter_reduce_(0, seg_id, scores_f32, reduce="amax", include_self=False)
    shifted = scores_f32 - max_per_seg[seg_id]
    exp_s = torch.exp(shifted)
    sum_exp = torch.zeros((num_segments,), device=scores.device, dtype=torch.float32)
    sum_exp.scatter_add_(0, seg_id, exp_s)
    return exp_s / (sum_exp[seg_id] + 1e-12)


def ragged_segment_sum(
    values: torch.Tensor,
    seg_id: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Sum flat values [N, ...] into [num_segments, ...] with zero padding."""
    out_shape = (num_segments,) + values.shape[1:]
    out = torch.zeros(out_shape, device=values.device, dtype=values.dtype)
    out.index_add_(0, seg_id, values)
    return out


def ragged_segment_mean(
    values: torch.Tensor,
    seg_id: torch.Tensor,
    seg_lens: torch.Tensor,
) -> torch.Tensor:
    """Mean of flat float values [N] per segment [M_total] in float32."""
    sums = ragged_segment_sum(values.float(), seg_id, seg_lens.numel())
    return sums / seg_lens.float().clamp_min(1.0)


def ragged_segment_cumsum(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seg_id: torch.Tensor,
) -> torch.Tensor:
    """Parallel O(1)-depth intra-segment inclusive prefix sum via global cumsum subtraction.

    For flat 1D values [N], returns inclusive cumsum reset at each cu_seqlens boundary.
    Computed strictly in float32 (used for motor charge depletion E_{n,tau}).
    """
    v_f32 = values.float()
    global_cs = torch.cumsum(v_f32, dim=0)
    starts = cu_seqlens[:-1].to(torch.int64)
    # Value immediately preceding each segment start (0 for segment 0)
    prev_cs = torch.zeros_like(starts, dtype=torch.float32)
    if starts.numel() > 1:
        prev_cs[1:] = global_cs[starts[1:] - 1]
    return global_cs - prev_cs[seg_id]
