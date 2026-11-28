"""Static-shape prefix-sum segment grid primitives for DCPA-Flat v3.3 (WS-2).

Enforces DCPA-Flat v3.3 R4 & Global Compilation Red Lines:
  - ZERO `torch.nonzero`, ZERO `repeat_interleave`, ZERO dynamic boolean-mask slicing,
    ZERO `.item()`, `.cpu()`, `.tolist()`, and ZERO step-growing Python lists.
  - Allocates `S_alloc = S + 1` with dedicated overflow/pad trash slot `S`; all segment
    reductions execute on `[P, S + 1, ...]` and slice `[:, :S]` so slots `0..S-1` suffer
    ZERO cross-document or overflow contamination.
  - Enforces the Unique Negative Sentinel Isolation Law: every invalid segment slot `(p, s)`
    receives a globally distinct negative sentinel `-(p * S + s + 2)`.
  - 4-step parallel segmented Kogge-Stone scans (`max_span = 16`) for `intra_seg_pos`
    (`tau in 1..16`), differentiable LIF membrane potential `v_tilde`, and PMD motor charge.
  - Pure-tensor Pointer-Doubling Exact LIF Scan (`static_lif_scan`) with UTF-8 continuation
    guard, `max_seg_len = 16`, and `doc_end_byte` forced spikes — 100% compatible with
    `torch.compile(..., fullgraph=True, dynamic=False)` across CPU, MPS, and CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from dcpa.ops.fp32 import fp32_island


@dataclass(frozen=True)
class StaticSegmentMap:
    """Pure static-shape byte <-> segment routing map with trash slot `S`."""

    byte_to_seg: torch.Tensor   # [P, T] int32 in [0, S] (S == overflow/pad trash slot)
    intra_seg_pos: torch.Tensor # [P, T] int32 in [1, max_seg_len]
    byte_overflow: torch.Tensor # [P, T] bool  True for valid bytes with s_raw >= S
    seg_counts: torch.Tensor    # [P]    int32 valid segment count per lane clamped to [0, S]
    seg_valid: torch.Tensor     # [P, S] bool  valid segment mask V_{p,s}
    seg_doc_id: torch.Tensor    # [P, S] int32 >=0 for valid; -(p*S + s + 2) for invalid
    seg_reset: torch.Tensor     # [P, S] bool  R_{p,0}=True, R_{p,s}=(D_{p,s} != D_{p,s-1})
    seg_lens: torch.Tensor      # [P, S] int32 length of each valid segment (0 for invalid)
    is_seg_start: torch.Tensor  # [P, T] bool  True on first byte of every valid segment
    is_seg_end: torch.Tensor    # [P, T] bool  True on last byte of every valid segment
    byte_valid: torch.Tensor    # [P, T] bool  True where doc_id_byte >= 0


def compute_effective_doc_boundaries(
    doc_id_byte: torch.Tensor,
    doc_start_byte: torch.Tensor,
    doc_end_byte: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute defensive `(byte_valid, eff_doc_start, eff_doc_end)` masks on `[P, T]`.

    Guarantees every contiguous run of valid bytes with constant `doc_id_byte >= 0` has
    exact start/end flags even across lane edges or pad transitions, with zero syncs.
    """
    P, T = doc_id_byte.shape
    device = doc_id_byte.device
    byte_valid = doc_id_byte >= 0

    if T == 0:
        empty = torch.zeros((P, 0), device=device, dtype=torch.bool)
        return byte_valid, empty, empty

    ds_in = doc_start_byte.bool()
    de_in = doc_end_byte.bool()

    if T == 1:
        return byte_valid, byte_valid, byte_valid

    doc_change = doc_id_byte[:, 1:] != doc_id_byte[:, :-1]
    start_tail = (ds_in[:, 1:] | doc_change) & byte_valid[:, 1:]
    eff_doc_start = torch.cat([byte_valid[:, 0:1], start_tail], dim=1)

    end_head = (de_in[:, :-1] | start_tail | (~byte_valid[:, 1:])) & byte_valid[:, :-1]
    eff_doc_end = torch.cat([end_head, byte_valid[:, -1:]], dim=1)
    return byte_valid, eff_doc_start, eff_doc_end


def segmented_scan_4step_affine(
    a: torch.Tensor,
    b: torch.Tensor,
    max_span: int = 16,
) -> torch.Tensor:
    """Parallel segmented affine scan `x_t = a_t * x_{t-1} + b_t` with span `<= max_span`.

    For `max_span = 16`, runs at most 4 Kogge-Stone doubling steps (`stride = 1, 2, 4, 8`)
    with zero syncs and zero graph breaks.
    """
    T = int(a.shape[1])
    if T <= 1 or max_span <= 1:
        return b
    levels = (int(max_span) - 1).bit_length()
    cur_a = a
    cur_b = b
    for level in range(levels):
        stride = 1 << level
        if stride >= T:
            break
        a_prev = cur_a[:, :-stride]
        b_prev = cur_b[:, :-stride]
        a_curr = cur_a[:, stride:]
        b_curr = cur_b[:, stride:]
        new_a = a_curr * a_prev
        new_b = a_curr * b_prev + b_curr
        cur_a = torch.cat([cur_a[:, :stride], new_a], dim=1)
        cur_b = torch.cat([cur_b[:, :stride], new_b], dim=1)
    return cur_b


def compute_intra_seg_pos(
    is_seg_start: torch.Tensor,
    byte_valid: Optional[torch.Tensor] = None,
    max_span: int = 16,
) -> torch.Tensor:
    """4-step parallel segmented scan computing 1-indexed `intra_seg_pos` (`tau_{p,t} in 1..max_span`)."""
    P, T = is_seg_start.shape
    device = is_seg_start.device
    if T == 0:
        return torch.zeros((P, 0), device=device, dtype=torch.int32)

    reset = is_seg_start.bool()
    if byte_valid is not None:
        reset = reset | (~byte_valid.bool())

    a = (~reset).to(torch.int32)
    b = torch.ones((P, T), device=device, dtype=torch.int32)
    tau = segmented_scan_4step_affine(a, b, max_span=max_span)
    return tau.clamp(1, int(max_span))


def static_segment_scatter_add(
    values: torch.Tensor,
    byte_to_seg: torch.Tensor,
    grid_segs: int,
) -> torch.Tensor:
    """Scatter-add `values [P, T, ...]` into `[P, S + 1, ...]` and slice `[:, :S]` to drop trash slot `S`."""
    P = values.shape[0]
    S = int(grid_segs)
    tail_shape = values.shape[2:]
    out = torch.zeros((P, S + 1, *tail_shape), device=values.device, dtype=values.dtype)
    idx = byte_to_seg.to(torch.long)
    while idx.ndim < values.ndim:
        idx = idx.unsqueeze(-1)
    if tail_shape:
        idx = idx.expand(values.shape)
    out = out.scatter_add(1, idx, values)
    return out[:, :S]


def static_segment_scatter_at_mask(
    values: torch.Tensor,
    mask: torch.Tensor,
    byte_to_seg: torch.Tensor,
    grid_segs: int,
) -> torch.Tensor:
    """Scatter exactly-one-per-segment masked byte values (e.g. `is_seg_start` or `is_seg_end`) into `[P, S, ...]`.

    Bytes where `mask` is False are routed to trash slot `S` and zeroed, then sliced off.
    """
    S = int(grid_segs)
    m = mask.bool()
    m_val = m
    while m_val.ndim < values.ndim:
        m_val = m_val.unsqueeze(-1)
    masked_vals = torch.where(m_val, values, torch.zeros_like(values))
    idx_routed = torch.where(m, byte_to_seg.to(torch.int32), torch.full_like(byte_to_seg, S, dtype=torch.int32))
    return static_segment_scatter_add(masked_vals, idx_routed, S)


def static_segment_softmax(
    scores: torch.Tensor,
    byte_to_seg: torch.Tensor,
    grid_segs: int,
) -> torch.Tensor:
    """Max-stabilized static-shape `[P, S + 1]` segmented softmax strictly in float32.

    Computes per-segment max on `[P, S + 1]` (using detached scores for gradient-invariant
    numerical stability), normalizes strictly within each segment including trash slot `S`,
    and returns byte-level weights `[P, T]` float32 with zero host-device syncs.
    """
    P, _ = scores.shape
    S = int(grid_segs)
    device = scores.device
    idx = byte_to_seg.to(torch.long)

    with fp32_island(device):
        scores_f = scores.float()
        init_max = torch.full((P, S + 1), float("-inf"), device=device, dtype=torch.float32)
        seg_max = init_max.scatter_reduce(1, idx, scores_f.detach(), reduce="amax", include_self=True)
        seg_max_safe = torch.where(seg_max > -1.0e30, seg_max, torch.zeros_like(seg_max))
        shifted = scores_f - seg_max_safe.gather(1, idx)
        exp_s = torch.exp(shifted)
        denom = torch.zeros((P, S + 1), device=device, dtype=torch.float32).scatter_add(1, idx, exp_s)
        weights = exp_s / denom.gather(1, idx).clamp_min(1.0e-12)
    return weights


def static_segment_mean(
    values: torch.Tensor,
    byte_to_seg: torch.Tensor,
    seg_lens: torch.Tensor,
    grid_segs: int,
) -> torch.Tensor:
    """Mean of byte-level values `[P, T, ...]` per valid segment `[P, S, ...]` in float32."""
    with fp32_island(values.device):
        sums = static_segment_scatter_add(values.float(), byte_to_seg, grid_segs)
        denom = seg_lens.float().clamp_min(1.0)
        while denom.ndim < sums.ndim:
            denom = denom.unsqueeze(-1)
        valid = seg_lens > 0
        while valid.ndim < sums.ndim:
            valid = valid.unsqueeze(-1)
        mean = sums / denom
        return torch.where(valid, mean, torch.zeros_like(mean))


def gather_seg_to_byte(
    seg_tensor: torch.Tensor,
    byte_to_seg: torch.Tensor,
) -> torch.Tensor:
    """In-place-equivalent static gather `[P, S, ...] -> [P, T, ...]` with zero trash slot at index `S`.

    Appends a zero slice at index `S` along dim 1 before gathering, ensuring overflow and pad
    bytes (`byte_to_seg == S`) cleanly read `0` with zero dynamic masking or syncs.
    """
    P = seg_tensor.shape[0]
    tail_shape = seg_tensor.shape[2:]
    zero_trash = torch.zeros((P, 1, *tail_shape), device=seg_tensor.device, dtype=seg_tensor.dtype)
    padded = torch.cat([seg_tensor, zero_trash], dim=1)  # [P, S + 1, ...]
    idx = byte_to_seg.to(torch.long)
    while idx.ndim < padded.ndim:
        idx = idx.unsqueeze(-1)
    if tail_shape:
        idx = idx.expand((P, byte_to_seg.shape[1], *tail_shape))
    return padded.gather(1, idx)


def build_static_segment_map(
    spikes: torch.Tensor,
    doc_id_byte: torch.Tensor,
    doc_start_byte: torch.Tensor,
    doc_end_byte: torch.Tensor,
    grid_segs: int,
    max_seg_len: int = 16,
) -> StaticSegmentMap:
    """Pure prefix-sum static segment mapping with `S_alloc = S + 1` overflow trash slot (R4).

    Guarantees:
      1. `sp = (spikes | doc_end_byte) & byte_valid`.
      2. Segment index increments by at most 1 at boundaries (ORing `sp[:, :-1]`, `doc_start_byte[:, 1:]`,
         and `doc_id_byte` change prevents double-incrementing when `doc_end` and `doc_start` coincide).
      3. Both pad bytes (`doc_id_byte < 0`) and overflow bytes (`s_raw >= S`) route to trash slot `S`,
         so slots `0..S-1` NEVER receive pad, overflow, or cross-document bytes.
      4. Invalid segment slots `(p, s)` get globally unique negative sentinels `-(p * S + s + 2)`.
      5. `intra_seg_pos` (`tau_{p,t} in 1..max_seg_len`) uses a 4-step parallel segmented scan.
    """
    P, T = doc_id_byte.shape
    S = int(grid_segs)
    device = doc_id_byte.device

    byte_valid, eff_doc_start, eff_doc_end = compute_effective_doc_boundaries(
        doc_id_byte, doc_start_byte, doc_end_byte
    )
    sp = (spikes.bool() | eff_doc_end) & byte_valid

    if T == 0:
        s_idx = torch.arange(S, device=device, dtype=torch.int32).unsqueeze(0)
        p_idx = torch.arange(P, device=device, dtype=torch.int32).unsqueeze(1)
        neg_sent = -(p_idx * S + s_idx + 2)
        empty_i32 = torch.zeros((P, 0), device=device, dtype=torch.int32)
        empty_b = torch.zeros((P, 0), device=device, dtype=torch.bool)
        return StaticSegmentMap(
            byte_to_seg=empty_i32,
            intra_seg_pos=empty_i32,
            byte_overflow=empty_b,
            seg_counts=torch.zeros((P,), device=device, dtype=torch.int32),
            seg_valid=torch.zeros((P, S), device=device, dtype=torch.bool),
            seg_doc_id=neg_sent,
            seg_reset=torch.ones((P, S), device=device, dtype=torch.bool),
            seg_lens=torch.zeros((P, S), device=device, dtype=torch.int32),
            is_seg_start=empty_b,
            is_seg_end=empty_b,
            byte_valid=empty_b,
        )

    if T == 1:
        is_seg_start = byte_valid
        is_seg_end = byte_valid
    else:
        # ORing sp[:, :-1] with eff_doc_start[:, 1:] guarantees increment by exactly 1 on doc boundaries
        is_seg_start_tail = (sp[:, :-1] | eff_doc_start[:, 1:]) & byte_valid[:, 1:]
        is_seg_start = torch.cat([byte_valid[:, 0:1], is_seg_start_tail], dim=1)
        is_seg_end_head = byte_valid[:, :-1] & ((~byte_valid[:, 1:]) | is_seg_start_tail)
        is_seg_end = torch.cat([is_seg_end_head, byte_valid[:, -1:]], dim=1)

    seg_1based = is_seg_start.to(torch.int32).cumsum(dim=1)  # [P, T] int32
    s_raw = seg_1based - 1  # [P, T] int32; valid bytes >= 0
    byte_overflow = byte_valid & (s_raw >= S)
    trash = torch.full_like(s_raw, S)
    byte_to_seg = torch.where((~byte_valid) | (s_raw >= S), trash, s_raw)  # [P, T] int32 in [0, S]

    seg_counts = seg_1based[:, -1].clamp(0, S)  # [P] int32
    intra_seg_pos = compute_intra_seg_pos(is_seg_start, byte_valid=byte_valid, max_span=max_seg_len)

    # Scatter segment lengths onto [P, S + 1] and slice [:, :S]
    seg_lens = static_segment_scatter_add(byte_valid.to(torch.int32), byte_to_seg, S)

    # Scatter doc_id from exact-once segment starts (+1 offset so empty slots decode to -1)
    doc_plus_one = torch.where(is_seg_start, doc_id_byte.to(torch.int32) + 1, torch.zeros_like(doc_id_byte, dtype=torch.int32))
    raw_seg_doc_id = static_segment_scatter_at_mask(doc_plus_one, is_seg_start, byte_to_seg, S) - 1

    # Unique Negative Sentinel Isolation Law: invalid slots get -(p * S + s + 2)
    s_idx = torch.arange(S, device=device, dtype=torch.int32).unsqueeze(0)  # [1, S]
    p_idx = torch.arange(P, device=device, dtype=torch.int32).unsqueeze(1)  # [P, 1]
    seg_valid = (s_idx < seg_counts.unsqueeze(1)) & (raw_seg_doc_id >= 0)
    neg_sentinel = -(p_idx * S + s_idx + 2)
    seg_doc_id = torch.where(seg_valid, raw_seg_doc_id, neg_sentinel)

    if S == 1:
        seg_reset = torch.ones((P, 1), device=device, dtype=torch.bool)
    else:
        seg_reset = torch.cat(
            [
                torch.ones((P, 1), device=device, dtype=torch.bool),
                seg_doc_id[:, 1:] != seg_doc_id[:, :-1],
            ],
            dim=1,
        )

    return StaticSegmentMap(
        byte_to_seg=byte_to_seg,
        intra_seg_pos=intra_seg_pos,
        byte_overflow=byte_overflow,
        seg_counts=seg_counts,
        seg_valid=seg_valid,
        seg_doc_id=seg_doc_id,
        seg_reset=seg_reset,
        seg_lens=seg_lens,
        is_seg_start=is_seg_start,
        is_seg_end=is_seg_end,
        byte_valid=byte_valid,
    )


def compute_utf8_can_spike(
    raw_bytes: torch.Tensor,
    eff_doc_start: torch.Tensor,
) -> torch.Tensor:
    """Exact O(1)-depth (3-step) UTF-8 multi-byte completion mask `can_spike [P, T]` bool.

    Matches the UTF-8 byte state machine bit-for-bit with zero loops over T and zero syncs:
      - ASCII (0x00..0x7F) or malformed lead -> rem = 0 (can spike)
      - 2-byte lead (0xC0..0xDF) -> rem = 1
      - 3-byte lead (0xE0..0xEF) -> rem = 2
      - 4-byte lead (0xF0..0xF7) -> rem = 3
      - Continuation (0x80..0xBF) -> rem = max(prev_rem - 1, 0), reset to 0 on doc start.
    """
    P, T = raw_bytes.shape
    device = raw_bytes.device
    if T == 0:
        return torch.zeros((P, 0), device=device, dtype=torch.bool)

    rb = raw_bytes.to(torch.int32)
    is_cont = (rb & 0xC0) == 0x80
    lead_rem = torch.where(
        (rb & 0xE0) == 0xC0,
        torch.ones_like(rb),
        torch.where(
            (rb & 0xF0) == 0xE0,
            torch.full_like(rb, 2),
            torch.where((rb & 0xF8) == 0xF0, torch.full_like(rb, 3), torch.zeros_like(rb)),
        ),
    )
    rem = torch.where(is_cont, torch.zeros_like(rb), lead_rem)
    if T > 1:
        zero_col = torch.zeros((P, 1), device=device, dtype=torch.int32)
        for _ in range(3):
            prev_rem = torch.cat([zero_col, rem[:, :-1]], dim=1)
            prev_rem = torch.where(eff_doc_start, torch.zeros_like(prev_rem), prev_rem)
            rem = torch.where(is_cont, (prev_rem - 1).clamp_min(0), lead_rem)
    return rem == 0


def static_lif_scan(
    info: torch.Tensor,
    delta_info: torch.Tensor,
    raw_bytes: torch.Tensor,
    doc_id_byte: torch.Tensor,
    doc_start_byte: torch.Tensor,
    doc_end_byte: torch.Tensor,
    lif_lambda: float,
    beta1: float,
    beta2: float,
    theta: torch.Tensor,
    max_seg_len: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-tensor Pointer-Doubling Exact LIF Membrane Scan for static `[P, T]` grids.

    Features:
      - Exact UTF-8 continuation guard on threshold firing (`can_spike & (v >= theta)`).
      - Hard segment ceiling `max_seg_len = 16` guaranteeing `tau in 1..16` and exact 4-step scans.
      - Forced spike on `doc_end_byte` (and zero leakage across docs or into pad bytes).
      - ZERO `.item()`, ZERO `.cpu()`, ZERO `torch.nonzero`, ZERO loops of length `T`
        (only `max_seg_len` local window steps + `ceil(log2 T)` pointer-doubling steps).
      - Recomputes differentiable `v_tilde` via 4-step segmented affine scan in float32.
      - Compiles cleanly under `torch.compile(..., fullgraph=True, dynamic=False)`.
    """
    P, T = info.shape
    device = info.device
    L = int(max_seg_len)

    with fp32_island(device):
        byte_valid, eff_doc_start, eff_doc_end = compute_effective_doc_boundaries(
            doc_id_byte, doc_start_byte, doc_end_byte
        )
        drive = beta1 * info.float() + beta2 * delta_info.float()
        drive = torch.where(byte_valid, drive, torch.zeros_like(drive))

        if T == 0:
            return drive, torch.zeros((P, 0), device=device, dtype=torch.bool)

        with torch.no_grad():
            drive_det = drive.detach()
            theta_f = theta.detach().float()
            can_spike = compute_utf8_can_spike(raw_bytes, eff_doc_start)

            # Step 1: Local O(max_seg_len) window finding exact first firing offset from every start `s`
            pad_w = L - 1
            if pad_w > 0:
                drive_pad = torch.cat([drive_det, torch.zeros((P, pad_w), device=device, dtype=torch.float32)], dim=1)
                can_pad = torch.cat([can_spike, torch.ones((P, pad_w), device=device, dtype=torch.bool)], dim=1)
                de_pad = torch.cat([eff_doc_end, torch.ones((P, pad_w), device=device, dtype=torch.bool)], dim=1)
            else:
                drive_pad = drive_det
                can_pad = can_spike
                de_pad = eff_doc_end

            v = torch.zeros((P, T), device=device, dtype=torch.float32)
            found = torch.zeros((P, T), device=device, dtype=torch.bool)
            end_offset = torch.zeros((P, T), device=device, dtype=torch.int64)

            for k in range(L):
                d_k = drive_pad[:, k : k + T]
                v = d_k if k == 0 else (lif_lambda * v + d_k)
                if k == L - 1:
                    fire_k = torch.ones((P, T), device=device, dtype=torch.bool)
                else:
                    cs_k = can_pad[:, k : k + T]
                    de_k = de_pad[:, k : k + T]
                    fire_k = de_k | (cs_k & (v >= theta_f))
                hit_now = (~found) & fire_k
                end_offset = torch.where(hit_now, torch.full_like(end_offset, k), end_offset)
                found = found | fire_k

            pos_idx = torch.arange(T, device=device, dtype=torch.int64).unsqueeze(0)
            end_pos = (pos_idx + end_offset).clamp_max(T - 1)  # [P, T]
            end_is_doc_end = eff_doc_end.gather(1, end_pos)

            # Sentinel sink at index T disconnects independent documents cleanly
            sentinel = torch.full((P, T), T, device=device, dtype=torch.int64)
            next_start = torch.where((~byte_valid) | end_is_doc_end, sentinel, (end_pos + 1).clamp_max(T))
            g = torch.cat([next_start, torch.full((P, 1), T, device=device, dtype=torch.int64)], dim=1)  # [P, T+1]

            # Step 2: O(log T) Pointer-Doubling marks exact reachable segment starts from every doc_start
            active = torch.cat(
                [eff_doc_start.to(torch.int32), torch.zeros((P, 1), device=device, dtype=torch.int32)],
                dim=1,
            )  # [P, T+1]
            n_double = (T - 1).bit_length()
            for _ in range(n_double):
                active = active.scatter_add(1, g, active).clamp_max(1)
                g = g.gather(1, g)

            seg_starts = (active[:, :T] > 0) & byte_valid
            if T > 1:
                next_is_start = torch.cat(
                    [seg_starts[:, 1:], torch.zeros((P, 1), device=device, dtype=torch.bool)],
                    dim=1,
                )
            else:
                next_is_start = torch.zeros((P, 1), device=device, dtype=torch.bool)
            spikes = (eff_doc_end | next_is_start) & byte_valid

        # Step 3: Differentiable v_tilde via 4-step segmented affine scan (exact since seg_len <= L)
        decay_a = torch.where(
            seg_starts,
            torch.zeros_like(drive),
            torch.full_like(drive, float(lif_lambda)),
        )
        v_tilde = segmented_scan_4step_affine(decay_a, drive, max_span=L)
        v_tilde = torch.where(byte_valid, v_tilde, torch.zeros_like(v_tilde))
    return v_tilde, spikes


def compute_static_seg_depths_and_triggers(
    x_bytes: torch.Tensor,
    doc_id_byte: torch.Tensor,
    doc_start_byte: torch.Tensor,
    seg_map: StaticSegmentMap,
    grid_segs: int,
    max_depth: int = 31,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute causal segment AST bracket depths and protocol Slot-0 pin/unpin triggers.

    Returns:
      seg_start_depth: [P, S] int32 causal depth strictly BEFORE first byte of segment (clamped [0, max_depth])
      seg_end_depth:   [P, S] int32 causal depth AFTER last byte of segment (clamped [0, max_depth])
      seg_pin_trig:    [P, S] bool  segment-level OR-scatter of `\\x1dE\\n` closing trigger (for Carrier G2)
      seg_unpin_trig:  [P, S] bool  segment-level OR-scatter of `0x04` (EOT) or `0x1E` (user turn)
    """
    P, T = x_bytes.shape
    S = int(grid_segs)
    device = x_bytes.device

    if T == 0:
        z_i32 = torch.zeros((P, S), device=device, dtype=torch.int32)
        z_b = torch.zeros((P, S), device=device, dtype=torch.bool)
        return z_i32, z_i32.clone(), z_b, z_b.clone()

    byte_valid = seg_map.byte_valid
    if T == 1:
        eff_doc_start = byte_valid
    else:
        doc_change = doc_id_byte[:, 1:] != doc_id_byte[:, :-1]
        eff_doc_start = torch.cat([byte_valid[:, 0:1], (doc_start_byte.bool()[:, 1:] | doc_change) & byte_valid[:, 1:]], dim=1)

    rb = x_bytes.to(torch.int32)

    # Detect \x1dE\n (strictly within same document)
    if T >= 3:
        z1 = torch.zeros((P, 1), device=device, dtype=torch.int32)
        z2 = torch.zeros((P, 2), device=device, dtype=torch.int32)
        rb_m1 = torch.cat([z1, rb[:, :-1]], dim=1)
        rb_m2 = torch.cat([z2, rb[:, :-2]], dim=1)
        eds_m1 = torch.cat([torch.zeros((P, 1), device=device, dtype=torch.bool), eff_doc_start[:, :-1]], dim=1)
        pin_hit_byte = (
            (rb_m2 == 0x1D)
            & (rb_m1 == ord("E"))
            & (rb == 0x0A)
            & byte_valid
            & (~eff_doc_start)
            & (~eds_m1)
        )
    else:
        pin_hit_byte = torch.zeros((P, T), device=device, dtype=torch.bool)

    unpin_hit_byte = ((rb == 0x04) | (rb == 0x1E)) & byte_valid

    # Causal AST bracket depth [P, T] respecting doc_start_byte & protocol frame resets
    post_reset = ((rb == 0x04) | (rb == 0x13) | (rb == 0x1E)) & byte_valid
    post_reset = post_reset | pin_hit_byte

    is_open = (rb == 0x28) | (rb == 0x5B) | (rb == 0x7B)   # ( [ {
    is_close = (rb == 0x29) | (rb == 0x5D) | (rb == 0x7D)  # ) ] }
    delta = is_open.to(torch.int32) - is_close.to(torch.int32)
    delta = torch.where(byte_valid & (~post_reset), delta, torch.zeros_like(delta))

    pre_reset = eff_doc_start | (~byte_valid)
    reset_group = pre_reset | post_reset
    if T > 1:
        reset_group = torch.cat([torch.ones((P, 1), device=device, dtype=torch.bool), reset_group[:, 1:]], dim=1)
    else:
        reset_group = torch.ones((P, 1), device=device, dtype=torch.bool)

    grp_id = torch.cumsum(reset_group.to(torch.int64), dim=1)  # [P, T]
    cs = torch.cumsum(delta.to(torch.int64), dim=1)            # [P, T]
    t_idx = torch.arange(T, device=device, dtype=torch.int64).unsqueeze(0).expand(P, T)
    flat_reset_pos = torch.where(reset_group, t_idx, torch.zeros_like(t_idx))
    frame_start_t = torch.cummax(flat_reset_pos, dim=1).values
    cs_at_start = cs.gather(1, frame_start_t) - delta.to(torch.int64).gather(1, frame_start_t)
    rel_cs = cs - cs_at_start

    stride = max(T * 4, 1_048_576)
    deficit = (-rel_cs).clamp_min(0)
    packed = grp_id * stride + deficit
    running_max_deficit = torch.cummax(packed, dim=1).values - grp_id * stride
    byte_depth_after = (rel_cs + running_max_deficit).clamp(0, int(max_depth)).to(torch.int32)
    byte_depth_after = torch.where(byte_valid, byte_depth_after, torch.zeros_like(byte_depth_after))

    if T > 1:
        byte_depth_before = torch.cat(
            [torch.zeros((P, 1), device=device, dtype=torch.int32), byte_depth_after[:, :-1]],
            dim=1,
        )
    else:
        byte_depth_before = torch.zeros((P, 1), device=device, dtype=torch.int32)
    byte_depth_before = torch.where(eff_doc_start | (~byte_valid), torch.zeros_like(byte_depth_before), byte_depth_before)

    seg_start_depth = static_segment_scatter_at_mask(byte_depth_before, seg_map.is_seg_start, seg_map.byte_to_seg, S)
    seg_end_depth = static_segment_scatter_at_mask(byte_depth_after, seg_map.is_seg_end, seg_map.byte_to_seg, S)
    seg_pin_trig = static_segment_scatter_add(pin_hit_byte.to(torch.int32), seg_map.byte_to_seg, S) > 0
    seg_unpin_trig = static_segment_scatter_add(unpin_hit_byte.to(torch.int32), seg_map.byte_to_seg, S) > 0

    return seg_start_depth, seg_end_depth, seg_pin_trig, seg_unpin_trig


__all__ = [
    "StaticSegmentMap",
    "compute_effective_doc_boundaries",
    "segmented_scan_4step_affine",
    "compute_intra_seg_pos",
    "static_segment_scatter_add",
    "static_segment_scatter_at_mask",
    "static_segment_softmax",
    "static_segment_mean",
    "gather_seg_to_byte",
    "build_static_segment_map",
    "compute_utf8_can_spike",
    "static_lif_scan",
    "compute_static_seg_depths_and_triggers",
]
