"""Parallel Associative Scan (Chunked C=256 O(log T)) + OpenAI Triton CUDA kernels + JIT LIF scan.

Enforces Commandment #3 & Performance Audit Task 1:
  - ZERO byte-level Python `for t in range(T):` loops in SGSC or PMD teacher-forced training!
  - Uses associative operator (M_right, B_right) (+) (M_left, B_left) = (M_right @ M_left, M_right @ B_left + B_right)
    strictly inside `fp32_island(device)`.
  - 128B Pushdown Stack Xi_t uses Chunked Parallel Associative Scan (C = 256, log2(256) = 8 batched
    `torch.bmm` intra-chunk passes + inter-chunk summary scan + single broadcast `torch.bmm`).
  - Diagonal recurrences (SGSC h_t, PMD m_t, LIF v_tilde) use fused Triton CUDA scan or
    Chunked C=256 O(log T) parallel associative scan.
  - Scalar LIF threshold firing uses Triton GPU kernel (including packed varlen `doc_cu_seqlens`)
    and a compiled TorchScript C++ kernel on CPU/MPS (zero Python byte-loop bottleneck).
"""

from __future__ import annotations

from typing import Optional

import torch

from dcpa.ops.fp32 import fp32_island

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


DEFAULT_SCAN_CHUNK_SIZE: int = 256


if HAS_TRITON:

    @triton.jit
    def _combine_affine(a1, b1, a2, b2):
        return a2 * a1, a2 * b1 + b2

    @triton.jit
    def _triton_diag_scan_fwd_kernel(
        A_ptr,
        B_ptr,
        H_ptr,
        stride_b: tl.constexpr,
        stride_t: tl.constexpr,
        stride_d: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Fused parallel associative scan along time for diagonal recurrence h_t = a_t * h_{t-1} + b_t."""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_t = tl.arange(0, BLOCK_T)
        mask = offs_t < T

        base = pid_b * stride_b + pid_d * stride_d
        ptrs = base + offs_t * stride_t

        a = tl.load(A_ptr + ptrs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

        _, h = tl.associative_scan((a, b), axis=0, combine_fn=_combine_affine)
        tl.store(H_ptr + ptrs, h, mask=mask)

    @triton.jit
    def _triton_diag_scan_bwd_kernel(
        A_ptr,
        H_ptr,
        GH_ptr,
        GA_ptr,
        GB_ptr,
        H0_ptr,
        stride_b: tl.constexpr,
        stride_t: tl.constexpr,
        stride_d: tl.constexpr,
        T: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Reverse parallel associative scan for exact analytical gradient of h_t = a_t * h_{t-1} + b_t."""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_t = tl.arange(0, BLOCK_T)
        rev_t = (T - 1) - offs_t
        mask = offs_t < T

        base = pid_b * stride_b + pid_d * stride_d
        gh_rev = tl.load(GH_ptr + base + rev_t * stride_t, mask=mask, other=0.0).to(tl.float32)
        next_t = rev_t + 1
        next_mask = mask & (next_t < T)
        a_next_rev = tl.load(A_ptr + base + next_t * stride_t, mask=next_mask, other=0.0).to(tl.float32)

        _, gb_rev = tl.associative_scan((a_next_rev, gh_rev), axis=0, combine_fn=_combine_affine)
        tl.store(GB_ptr + base + rev_t * stride_t, gb_rev, mask=mask)

        prev_t = rev_t - 1
        has_prev = mask & (prev_t >= 0)
        h0_val = tl.load(H0_ptr + pid_b * stride_d * 0 + pid_d).to(tl.float32)
        h_prev = tl.load(H_ptr + base + prev_t * stride_t, mask=has_prev, other=h0_val).to(tl.float32)
        ga_rev = gb_rev * h_prev
        tl.store(GA_ptr + base + rev_t * stride_t, ga_rev, mask=mask)

    @triton.jit
    def _triton_lif_scan_varlen_kernel(
        drive_ptr,
        raw_ptr,
        starts_ptr,
        ends_ptr,
        spikes_ptr,
        lam,
        theta,
        max_seg_len: tl.constexpr,
    ):
        """Per-document parallel register-level scalar LIF scan supporting packed varlen cu_seqlens."""
        pid_doc = tl.program_id(0)
        doc_start = tl.load(starts_ptr + pid_doc)
        doc_end = tl.load(ends_ptr + pid_doc)

        v = 0.0
        seg_len = 0
        utf8_rem = 0

        idx = doc_start
        while idx < doc_end:
            d_val = tl.load(drive_ptr + idx).to(tl.float32)
            byte_val = tl.load(raw_ptr + idx).to(tl.int32)

            if byte_val < 0x80:
                utf8_rem = 0
            elif (byte_val & 0xE0) == 0xC0:
                utf8_rem = 1
            elif (byte_val & 0xF0) == 0xE0:
                utf8_rem = 2
            elif (byte_val & 0xF8) == 0xF0:
                utf8_rem = 3
            elif (byte_val & 0xC0) == 0x80:
                utf8_rem = tl.maximum(utf8_rem - 1, 0)
            else:
                utf8_rem = 0

            v_tilde = lam * v + d_val
            seg_len += 1

            can_spike = utf8_rem == 0
            is_last = idx == (doc_end - 1)
            fire = is_last | (can_spike & ((v_tilde >= theta) | (seg_len >= max_seg_len)))

            tl.store(spikes_ptr + idx, fire)
            v = tl.where(fire, 0.0, v_tilde)
            seg_len = tl.where(fire, 0, seg_len)
            idx += 1


@torch.jit.script
def _lif_scan_torchscript(
    drive_cpu: torch.Tensor,
    raw_cpu: torch.Tensor,
    start_cpu: torch.Tensor,
    end_cpu: torch.Tensor,
    lif_lambda: float,
    theta_val: float,
    max_seg_len: int,
) -> torch.Tensor:
    """Compiled native C++ TorchScript scalar LIF threshold scan (zero Python bytecode loop!)."""
    B = drive_cpu.size(0)
    T = drive_cpu.size(1)
    spikes = torch.zeros((B, T), dtype=torch.bool)
    for b in range(B):
        v: float = 0.0
        seg_len: int = 0
        utf8_rem: int = 0
        for t in range(T):
            if bool(start_cpu[b, t].item()):
                v = 0.0
                seg_len = 0
                utf8_rem = 0
            bv: int = int(raw_cpu[b, t].item())
            if bv < 0x80:
                utf8_rem = 0
            elif (bv & 0xE0) == 0xC0:
                utf8_rem = 1
            elif (bv & 0xF0) == 0xE0:
                utf8_rem = 2
            elif (bv & 0xF8) == 0xF0:
                utf8_rem = 3
            elif (bv & 0xC0) == 0x80:
                if utf8_rem > 0:
                    utf8_rem -= 1
            else:
                utf8_rem = 0

            v_t: float = lif_lambda * v + float(drive_cpu[b, t].item())
            seg_len += 1
            can_spike: bool = utf8_rem == 0
            is_last: bool = bool(end_cpu[b, t].item())
            fire: bool = is_last or (can_spike and (v_t >= theta_val or seg_len >= max_seg_len))
            spikes[b, t] = fire
            if fire:
                v = 0.0
                seg_len = 0
            else:
                v = v_t
    return spikes


def _kogge_stone_diag(a: torch.Tensor, b: torch.Tensor, levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Kogge-Stone associative scan returning `(a_prefix, b_prefix)` in float32."""
    T = a.shape[1]
    cur_a = a
    cur_b = b
    for level in range(levels):
        stride = 1 << level
        if stride >= T:
            break
        a_prev = cur_a[:, :-stride, :]
        b_prev = cur_b[:, :-stride, :]
        a_curr = cur_a[:, stride:, :]
        b_curr = cur_b[:, stride:, :]

        new_a_tail = a_curr * a_prev
        new_b_tail = a_curr * b_prev + b_curr

        cur_a = torch.cat([cur_a[:, :stride, :], new_a_tail], dim=1)
        cur_b = torch.cat([cur_b[:, :stride, :], new_b_tail], dim=1)
    return cur_a, cur_b


def _blelloch_scan_pytorch(
    a: torch.Tensor,
    b: torch.Tensor,
    max_span: Optional[int] = None,
    chunk_size: int = DEFAULT_SCAN_CHUNK_SIZE,
) -> torch.Tensor:
    """Chunked O(log T) parallel associative scan (C=256) for h_t = a_t * h_{t-1} + b_t in float32.

    When `T > chunk_size` (and `max_span` > `chunk_size` or None), splits into `N_chunks = ceil(T/C)`
    chunks: `log2(C) = 8` intra-chunk levels + tiny inter-chunk summary scan + 1 broadcast pass,
    cutting FLOPs and autograd memory ~2x at long T (32K-131K) with exact mathematical equivalence.
    """
    B, T, D = a.shape
    if T <= 1:
        return b.float()

    span = T if max_span is None else min(T, max(1, int(max_span)))
    if span <= 1:
        return b.float()

    a_f = a.float()
    b_f = b.float()

    # Small sequence or tight segment span (e.g. max_seg_len=16): direct O(log span) scan
    if T <= chunk_size or span <= chunk_size:
        levels = (span - 1).bit_length()
        _, cur_b = _kogge_stone_diag(a_f, b_f, levels)
        return cur_b

    # Chunked parallel scan (C = 256): Stage 1 intra-chunk (8 steps)
    C = int(chunk_size)
    pad = (C - (T % C)) % C
    if pad > 0:
        a_pad = torch.cat([a_f, torch.zeros((B, pad, D), device=a.device, dtype=torch.float32)], dim=1)
        b_pad = torch.cat([b_f, torch.zeros((B, pad, D), device=a.device, dtype=torch.float32)], dim=1)
    else:
        a_pad = a_f
        b_pad = b_f

    T_pad = a_pad.shape[1]
    n_chunks = T_pad // C

    a_ch = a_pad.reshape(B * n_chunks, C, D)
    b_ch = b_pad.reshape(B * n_chunks, C, D)
    intra_levels = (C - 1).bit_length()
    a_intra, b_intra = _kogge_stone_diag(a_ch, b_ch, intra_levels)

    # Stage 2: Inter-chunk summary scan over N_chunks endpoints ONLY
    a_end = a_intra[:, -1, :].reshape(B, n_chunks, D)
    b_end = b_intra[:, -1, :].reshape(B, n_chunks, D)
    max_chunk_span = min(n_chunks, (span + C - 1) // C + 1)
    inter_levels = (max_chunk_span - 1).bit_length()
    _, b_prefix = _kogge_stone_diag(a_end, b_end, inter_levels)

    # Stage 3: Exclusive chunk carry broadcast back into chunks
    zero_carry = torch.zeros((B, 1, D), device=a.device, dtype=torch.float32)
    carry = torch.cat([zero_carry, b_prefix[:, :-1, :]], dim=1)  # [B, n_chunks, D]
    carry_flat = carry.reshape(B * n_chunks, 1, D)
    h_chunks = a_intra * carry_flat + b_intra
    return h_chunks.reshape(B, T_pad, D)[:, :T, :]


class _TritonDiagScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        orig_dtype = b.dtype
        B, T, D = a.shape
        a_f32 = a.float().contiguous()
        b_f32 = b.float().contiguous()
        h0_f32 = h0.float().contiguous()

        b_folded = b_f32.clone()
        b_folded[:, 0, :] = a_f32[:, 0, :] * h0_f32 + b_f32[:, 0, :]

        h_out = torch.empty_like(b_f32)
        block_t = 1 << (T - 1).bit_length()
        grid = (B, D)
        _triton_diag_scan_fwd_kernel[grid](
            a_f32,
            b_folded,
            h_out,
            a_f32.stride(0),
            a_f32.stride(1),
            a_f32.stride(2),
            T=T,
            D=D,
            BLOCK_T=block_t,
        )
        ctx.save_for_backward(a_f32, h_out, h0_f32)
        ctx.orig_dtype = orig_dtype
        return h_out.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor):
        a_f32, h_out, h0_f32 = ctx.saved_tensors
        B, T, D = a_f32.shape
        gh_f32 = grad_h.float().contiguous()
        ga = torch.empty_like(a_f32)
        gb = torch.empty_like(a_f32)
        block_t = 1 << (T - 1).bit_length()
        grid = (B, D)
        _triton_diag_scan_bwd_kernel[grid](
            a_f32,
            h_out,
            gh_f32,
            ga,
            gb,
            h0_f32,
            a_f32.stride(0),
            a_f32.stride(1),
            a_f32.stride(2),
            T=T,
            BLOCK_T=block_t,
        )
        gh0 = (gb[:, 0, :] * a_f32[:, 0, :]).sum(dim=0)
        return ga.to(ctx.orig_dtype), gb.to(ctx.orig_dtype), gh0.to(h0_f32.dtype)


def parallel_diag_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    reset_mask: Optional[torch.Tensor] = None,
    max_span: Optional[int] = None,
    chunk_size: int = DEFAULT_SCAN_CHUNK_SIZE,
) -> torch.Tensor:
    """Parallel associative scan for h_t = a_t * h_{t-1} + b_t with optional doc-start reset_mask.

    Executes strictly inside `fp32_island(a.device)` and is 100% compatible with `torch.compile`.
    """
    orig_dtype = b.dtype
    B, T, D = a.shape
    with fp32_island(a.device):
        if h0 is None:
            h0_vec = torch.zeros((D,), device=a.device, dtype=torch.float32)
            h0_b = h0_vec.unsqueeze(0)
        elif h0.ndim == 1:
            h0_vec = h0.float()
            h0_b = h0_vec.unsqueeze(0)
        else:
            h0_b = h0.float()
            h0_vec = h0_b

        a_f32 = a.float()
        b_f32 = b.float()

        if reset_mask is not None and T > 0:
            rm = reset_mask.bool().unsqueeze(-1)  # [B, T, 1]
            h0_exp = h0_b.unsqueeze(1)  # [1 or B, 1, D]
            b_f32 = torch.where(rm, a_f32 * h0_exp + b_f32, b_f32)
            a_f32 = torch.where(rm, torch.zeros_like(a_f32), a_f32)
            # Branchless fold at t=0 when reset_mask[:, 0] is False (avoids .item() graph breaks!)
            b0 = torch.where(
                rm[:, 0:1, :],
                b_f32[:, 0:1, :],
                a_f32[:, 0:1, :] * h0_exp + b_f32[:, 0:1, :],
            )
            b_f32 = torch.cat([b0, b_f32[:, 1:, :]], dim=1) if T > 1 else b0
            if HAS_TRITON and a.is_cuda and T <= 4096 and not torch.compiler.is_compiling():
                return _TritonDiagScanFn.apply(
                    a_f32, b_f32, torch.zeros((D,), device=a.device, dtype=torch.float32)
                ).to(orig_dtype)
            h_f32 = _blelloch_scan_pytorch(a_f32, b_f32, max_span=max_span, chunk_size=chunk_size)
            return h_f32.to(orig_dtype)

        if (
            HAS_TRITON
            and a.is_cuda
            and T <= 4096
            and (h0 is None or h0.ndim == 1)
            and not torch.compiler.is_compiling()
        ):
            return _TritonDiagScanFn.apply(a, b, h0_vec)

        if T > 0:
            b_first = a_f32[:, 0:1, :] * h0_b.unsqueeze(1) + b_f32[:, 0:1, :]
            if T > 1:
                b_folded = torch.cat([b_first, b_f32[:, 1:, :]], dim=1)
            else:
                b_folded = b_first
        else:
            b_folded = b_f32
        h_f32 = _blelloch_scan_pytorch(a_f32, b_folded, max_span=max_span, chunk_size=chunk_size)
        return h_f32.to(orig_dtype)


def sequential_diag_scan_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Strict sequential reference implementation ONLY for unit tests / Gate 1 equivalence checking."""
    B, T, D = a.shape
    a_f32 = a.float()
    b_f32 = b.float()
    if h0 is None:
        h = torch.zeros((B, D), device=a.device, dtype=torch.float32)
    elif h0.ndim == 1:
        h = h0.float().unsqueeze(0).expand(B, D).clone()
    else:
        h = h0.float().clone()
    outs = []
    for t in range(T):
        h = a_f32[:, t, :] * h + b_f32[:, t, :]
        outs.append(h)
    return torch.stack(outs, dim=1).to(b.dtype)


def lif_membrane_scan(
    info: torch.Tensor,
    delta_info: torch.Tensor,
    raw_bytes: torch.Tensor,
    lif_lambda: float,
    beta1: float,
    beta2: float,
    theta: torch.Tensor,
    max_seg_len: int = 16,
    doc_cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leaky Integrate-and-Fire membrane potential scan in float32 with UTF-8, maxSegLen & doc-boundary guard.

    Dispatches scalar spike extraction to:
      - Fused per-document parallel Triton kernel on CUDA (both rectangular & packed varlen `doc_cu_seqlens`).
      - Compiled native C++ TorchScript kernel `_lif_scan_torchscript` on CPU/MPS.
    Recomputes differentiable `v_tilde` via O(log T) `parallel_diag_scan`.
    """
    B, T = info.shape
    device = info.device
    with fp32_island(device):
        info_f32 = info.float()
        dinfo_f32 = delta_info.float()
        drive = beta1 * info_f32 + beta2 * dinfo_f32  # [B, T] float32, differentiable!

        theta_val = float(theta.detach().float().item())

        is_doc_start = torch.zeros((B, T), device=device, dtype=torch.bool)
        is_doc_end = torch.zeros((B, T), device=device, dtype=torch.bool)
        if T > 0:
            is_doc_start[:, 0] = True
            is_doc_end[:, T - 1] = True

        if doc_cu_seqlens is not None:
            if B != 1:
                raise ValueError(f"doc_cu_seqlens packed varlen mode expects B=1, got B={B}")
            starts = doc_cu_seqlens[:-1].to(dtype=torch.int64, device=device)
            ends = doc_cu_seqlens[1:].to(dtype=torch.int64, device=device)
            if starts.numel() > 0:
                is_doc_start[0, starts] = True
                is_doc_end[0, ends - 1] = True
        else:
            starts = torch.arange(B, device=device, dtype=torch.int64) * T
            ends = starts + T

        if HAS_TRITON and info.is_cuda and T > 0 and starts.numel() > 0:
            spikes_flat = torch.empty((B * T,), device=device, dtype=torch.bool)
            n_docs = int(starts.numel())
            _triton_lif_scan_varlen_kernel[(n_docs,)](
                drive.detach().reshape(-1).contiguous(),
                raw_bytes.to(torch.uint8).reshape(-1).contiguous(),
                starts.contiguous(),
                ends.contiguous(),
                spikes_flat,
                lam=lif_lambda,
                theta=theta_val,
                max_seg_len=max_seg_len,
            )
            spikes = spikes_flat.view(B, T)
        else:
            # Native C++ TorchScript scalar scan (zero Python byte-loop overhead!)
            drive_cpu = drive.detach().to("cpu", dtype=torch.float32).contiguous()
            raw_cpu = raw_bytes.detach().to("cpu", dtype=torch.uint8).contiguous()
            start_cpu = is_doc_start.detach().to("cpu", dtype=torch.bool).contiguous()
            end_cpu = is_doc_end.detach().to("cpu", dtype=torch.bool).contiguous()
            spikes = _lif_scan_torchscript(
                drive_cpu,
                raw_cpu,
                start_cpu,
                end_cpu,
                float(lif_lambda),
                float(theta_val),
                int(max_seg_len),
            ).to(device)

        # Recompute v_tilde differentiably via O(log T) parallel scan with exact segment & doc resets!
        if T > 1:
            prev_spikes = torch.cat(
                [torch.zeros((B, 1), device=device, dtype=torch.bool), spikes[:, :-1]],
                dim=1,
            )
        else:
            prev_spikes = torch.zeros((B, T), device=device, dtype=torch.bool)

        reset_prev = prev_spikes | is_doc_start
        decay_a = torch.where(
            reset_prev,
            torch.zeros((B, T), device=device, dtype=torch.float32),
            torch.full((B, T), lif_lambda, device=device, dtype=torch.float32),
        )
        v_tilde = parallel_diag_scan(
            decay_a.unsqueeze(-1), drive.unsqueeze(-1), max_span=max_seg_len + 4
        ).squeeze(-1)
    return v_tilde, spikes


def _kogge_stone_matrix_bmm(
    M: torch.Tensor,
    B_mat: torch.Tensor,
    levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associative matrix-affine scan via batched `torch.bmm`:
    `(M_right, B_right) (+) (M_left, B_left) = (M_right @ M_left, M_right @ B_left + B_right)`.
    """
    N, L_seq, L, _ = M.shape
    d_tag = B_mat.shape[-1]
    cur_M = M
    cur_B = B_mat
    for level in range(levels):
        stride = 1 << level
        if stride >= L_seq:
            break
        rem = L_seq - stride
        M_left = cur_M[:, :rem].reshape(N * rem, L, L)
        B_left = cur_B[:, :rem].reshape(N * rem, L, d_tag)
        M_right = cur_M[:, stride:].reshape(N * rem, L, L)

        new_M_tail = torch.bmm(M_right, M_left).reshape(N, rem, L, L)
        new_B_tail = torch.bmm(M_right, B_left).reshape(N, rem, L, d_tag) + cur_B[:, stride:]

        cur_M = torch.cat([cur_M[:, :stride], new_M_tail], dim=1)
        cur_B = torch.cat([cur_B[:, :stride], new_B_tail], dim=1)
    return cur_M, cur_B


def parallel_stack_scan(
    p_push: torch.Tensor,
    p_pop: torch.Tensor,
    p_hold: torch.Tensor,
    v_tag: torch.Tensor,
    xi_init: Optional[torch.Tensor] = None,
    reset_mask: Optional[torch.Tensor] = None,
    max_span: Optional[int] = None,
    chunk_size: int = DEFAULT_SCAN_CHUNK_SIZE,
) -> torch.Tensor:
    """Exact Chunked Parallel Associative Scan (C = 256) for the 128B differentiable pushdown stack Xi_t.

    Implements Performance Audit Task 1 strictly inside `fp32_island(device)`:
      - M_t = (1 - reset_t) * (p_hold * I_16 + p_push * S_down + p_pop * S_up) in R^{16x16}
      - B_t = p_push * (e_1 @ v_tag^T) in R^{16x4}
      - Chunk size C = 256:
          Stage 1: log2(256) = 8 intra-chunk batched `torch.bmm` scan steps.
          Stage 2: Inter-chunk summary scan over N_chunks = ceil(T / 256) chunk summaries.
          Stage 3: Single batched `torch.bmm` broadcast of exclusive chunk carries back into chunks.
      - ZERO byte-level Python `for t in range(T):` loops; 100% `torch.compile`-friendly!
    """
    B, T, d_tag = v_tag.shape
    L = 16
    device = v_tag.device

    with fp32_island(device):
        p_push_f = p_push.float()
        p_pop_f = p_pop.float()
        p_hold_f = p_hold.float()
        v_tag_f = v_tag.float()

        eye = torch.eye(L, device=device, dtype=torch.float32)
        shift_down = torch.diag(torch.ones(L - 1, device=device, dtype=torch.float32), diagonal=-1)
        shift_up = shift_down.transpose(0, 1)

        M_raw = (
            p_hold_f[..., None, None] * eye
            + p_push_f[..., None, None] * shift_down
            + p_pop_f[..., None, None] * shift_up
        )  # [B, T, 16, 16]

        # B_t = p_push * (e_1 @ v_tag^T): zero everywhere except row 0
        e1 = torch.zeros((L,), device=device, dtype=torch.float32)
        e1[0] = 1.0
        B_mat = (p_push_f.unsqueeze(-1) * v_tag_f).unsqueeze(-2) * e1.view(1, 1, L, 1)  # [B, T, 16, d_tag]

        if reset_mask is not None and T > 0:
            reset_f = reset_mask.to(dtype=torch.float32)  # [B, T]
            M = (1.0 - reset_f)[..., None, None] * M_raw
            if xi_init is not None:
                xi0 = xi_init.float()  # [B, 16, d_tag]
                B_init_inj = torch.matmul(M_raw, xi0.unsqueeze(1))
                B_mat = B_mat + reset_f[..., None, None] * B_init_inj
                init_carry = (1.0 - reset_f[:, 0])[..., None, None] * xi0
            else:
                init_carry = None
        else:
            M = M_raw
            init_carry = xi_init.float() if xi_init is not None else None

        if T == 0:
            return B_mat
        if T == 1:
            if init_carry is not None:
                return torch.bmm(M[:, 0], init_carry).unsqueeze(1) + B_mat
            return B_mat

        C = int(chunk_size)
        if T <= C:
            span = T if max_span is None else min(T, max(1, int(max_span)))
            levels = (span - 1).bit_length()
            M_scan, B_scan = _kogge_stone_matrix_bmm(M, B_mat, levels)
            if init_carry is not None:
                carry_exp = init_carry.unsqueeze(1).expand(B, T, L, d_tag).reshape(B * T, L, d_tag)
                return torch.bmm(M_scan.reshape(B * T, L, L), carry_exp).reshape(B, T, L, d_tag) + B_scan
            return B_scan

        # --- Chunked Parallel Scan (C = 256) ---
        pad = (C - (T % C)) % C
        if pad > 0:
            M_pad = torch.cat([M, torch.zeros((B, pad, L, L), device=device, dtype=torch.float32)], dim=1)
            B_pad = torch.cat([B_mat, torch.zeros((B, pad, L, d_tag), device=device, dtype=torch.float32)], dim=1)
        else:
            M_pad = M
            B_pad = B_mat

        T_pad = M_pad.shape[1]
        n_chunks = T_pad // C

        # Stage 1: Intra-chunk scan (log2(256) = 8 batched bmm steps)
        M_ch = M_pad.reshape(B * n_chunks, C, L, L)
        B_ch = B_pad.reshape(B * n_chunks, C, L, d_tag)
        intra_levels = (C - 1).bit_length()
        M_intra, B_intra = _kogge_stone_matrix_bmm(M_ch, B_ch, intra_levels)

        # Stage 2: Inter-chunk summary scan over N_chunks endpoints ONLY
        M_summary = M_intra[:, -1].reshape(B, n_chunks, L, L)
        B_summary = B_intra[:, -1].reshape(B, n_chunks, L, d_tag)
        if max_span is not None:
            max_chunk_span = min(n_chunks, (max(1, int(max_span)) + C - 1) // C + 1)
        else:
            max_chunk_span = n_chunks
        inter_levels = (max_chunk_span - 1).bit_length()
        M_pref, B_pref = _kogge_stone_matrix_bmm(M_summary, B_summary, inter_levels)

        # Build exclusive per-chunk incoming carry [B, n_chunks, 16, d_tag]
        if init_carry is not None:
            carry_tail = (
                torch.bmm(
                    M_pref[:, :-1].reshape(B * (n_chunks - 1), L, L),
                    init_carry.unsqueeze(1).expand(B, n_chunks - 1, L, d_tag).reshape(B * (n_chunks - 1), L, d_tag),
                ).reshape(B, n_chunks - 1, L, d_tag)
                + B_pref[:, :-1]
            )
            carry = torch.cat([init_carry.unsqueeze(1), carry_tail], dim=1)
        else:
            zero_carry = torch.zeros((B, 1, L, d_tag), device=device, dtype=torch.float32)
            carry = torch.cat([zero_carry, B_pref[:, :-1]], dim=1)

        # Stage 3: Single batched torch.bmm broadcasting carries back into all chunk slots
        carry_exp = carry.unsqueeze(2).expand(B, n_chunks, C, L, d_tag).reshape(B * T_pad, L, d_tag)
        Xi_pad = (
            torch.bmm(M_intra.reshape(B * T_pad, L, L), carry_exp).reshape(B, T_pad, L, d_tag)
            + B_intra.reshape(B, T_pad, L, d_tag)
        )
        return Xi_pad[:, :T]


def sequential_stack_scan_ref(
    p_push: torch.Tensor,
    p_pop: torch.Tensor,
    p_hold: torch.Tensor,
    v_tag: torch.Tensor,
    xi_init: Optional[torch.Tensor] = None,
    reset_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Strict O(T) byte-by-byte sequential pushdown stack reference ONLY for unit-test equivalence checking."""
    B, T, d_tag = v_tag.shape
    L = 16
    device = v_tag.device
    xi = (
        torch.zeros((B, L, d_tag), device=device, dtype=torch.float32)
        if xi_init is None
        else xi_init.float().clone()
    )
    zero_bot = torch.zeros((B, 1, d_tag), device=device, dtype=torch.float32)
    outs = []
    for t in range(T):
        if reset_mask is not None:
            rm = reset_mask[:, t].bool().view(B, 1, 1)
            base_xi = (
                torch.zeros_like(xi)
                if xi_init is None
                else xi_init.float()
            )
            xi = torch.where(rm, base_xi, xi)
        pp = p_push[:, t].float().view(B, 1, 1)
        po = p_pop[:, t].float().view(B, 1, 1)
        ph = p_hold[:, t].float().view(B, 1, 1)
        vt = v_tag[:, t].float().unsqueeze(1)  # [B, 1, d_tag]

        pushed = torch.cat([vt, xi[:, :-1, :]], dim=1)
        popped = torch.cat([xi[:, 1:, :], zero_bot], dim=1)
        xi = ph * xi + pp * pushed + po * popped
        outs.append(xi)
    return torch.stack(outs, dim=1)
