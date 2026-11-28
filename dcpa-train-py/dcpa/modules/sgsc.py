"""Module 1: Surprisal-Gated Spike Codec (SGSC).

Implements the unified forward contract (Engineering Blueprint Section 2.1, 2.2 & Chapter 3):
  1. Strictly causal byte recurrence via O(log T) Blelloch parallel associative scan / Triton kernel.
  2. Strictly causal prior p_t = Softmax(W_out @ h_{t-1}) in float32 (zero current-step leakage).
  3. Register-level LIF membrane potential scan with UTF-8 multi-byte completion guard & maxSegLen=16.
  4. Hard forward spikes + STE surrogate gradients flowing to threshold theta and sensory micro-core.
  5. Zero-padding Ragged cu_seqlens intra-segment RoPE cross-attention & phase-bound stem f_m.
  6. Cortical slow semantic projection c_m^slow = RMSNorm(W_res h_{t_m} + W_sem f_m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.config import DCPAConfig
from dcpa.interfaces import DCPAFlatConfig, ScatterBridgeOut, StaticGridBatch
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.ragged import (
    build_cu_seqlens,
    ragged_local_offsets,
    ragged_segment_mean,
    ragged_segment_softmax,
    ragged_segment_sum,
)
from dcpa.ops.rmsnorm import RMSNorm
from dcpa.ops.rope import apply_holographic_rope
from dcpa.ops.scan import HAS_TRITON, lif_membrane_scan, parallel_diag_scan
from dcpa.ops.static_grid import (
    build_static_segment_map,
    compute_effective_doc_boundaries,
    compute_static_seg_depths_and_triggers,
    gather_seg_to_byte,
    segmented_scan_4step_affine,
    static_lif_scan,
    static_segment_mean,
    static_segment_scatter_add,
    static_segment_scatter_at_mask,
    static_segment_softmax,
)
from dcpa.stream_carry import RowLayout, from_rows, pad_entries, to_rows


@dataclass
class SGSCOutput:
    f_stem: torch.Tensor       # [M_total, d_s]
    c_slow: torch.Tensor       # [M_total, D]
    sem_norm: torch.Tensor     # [M_total] float32 ||W_sem f_m||_2 for Module 3 cortical familiarity
    mean_info: torch.Tensor    # [M_total] float32 average surprisal I_bar_m
    cu_seqlens: torch.Tensor   # [M_total + 1] int32 cumulative offsets over flattened B*T
    seg_counts: torch.Tensor   # [B] int32 segments per batch element
    spikes: torch.Tensor       # [B, T] bool hard discrete spikes
    s_tilde: torch.Tensor      # [B, T] float32 smooth spike probabilities (for L_rate)
    s_ste: torch.Tensor        # [B, T] float32 STE spikes
    prior_probs: torch.Tensor  # [B, T, 256] float32 strictly causal prior p_t
    prior_logits: torch.Tensor # [B, T, 256] float32 strictly causal logits
    info: torch.Tensor         # [B, T] float32 self-information I_t (Nats)
    v_tilde: torch.Tensor      # [B, T] float32 pre-reset LIF membrane potential
    h_seq: torch.Tensor        # [B, T, d_s] sensory hidden states
    theta: torch.Tensor        # scalar float32 effective LIF threshold


class SGSC(nn.Module):
    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_s = cfg.d_s
        D = cfg.d_concept

        # Shared with PMD (bound by DCPAModel)
        self.embed = nn.Embedding(cfg.vocab_size, d_s)
        self.w_out = nn.Linear(d_s, cfg.vocab_size, bias=False)

        # Learnable initial hidden state h_0
        self.h0 = nn.Parameter(torch.zeros(d_s))

        # Causal diagonal sensory recurrence
        self.w_alpha = nn.Linear(d_s, d_s, bias=True)
        self.w_u = nn.Linear(d_s, d_s, bias=False)

        # Learnable LIF threshold theta >= lif_theta_min via scaled softplus parameterization
        target_delta = max(cfg.lif_theta_init - cfg.lif_theta_min, 1e-3)
        raw_init = math.log(math.expm1(target_delta)) / max(cfg.lif_theta_scale, 1e-6)
        self.theta_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # Unified phase-bound stem extraction (Engineering Doc 2.1)
        self.w_q = nn.Linear(d_s, d_s, bias=False)
        self.w_k = nn.Linear(d_s, d_s, bias=False)
        self.w_f = nn.Linear(d_s, d_s, bias=False)
        self.norm_f = RMSNorm(d_s, eps=cfg.eps)

        # Slow cortical semantic projection
        self.w_res = nn.Linear(d_s, D, bias=False)
        self.w_sem = nn.Linear(d_s, D, bias=False)
        self.norm_c = RMSNorm(D, eps=cfg.eps)

        self._compiled_byte_scan = None
        self._init_weights()

    def enable_compiled_byte_scan(self, **compile_kwargs) -> None:
        """Compile Hotspot B1: static-shape per-byte gated projections + chunked diag scan + prior."""
        self._compiled_byte_scan = torch.compile(self._byte_scan_impl, **compile_kwargs)

    def _byte_scan_impl(
        self,
        raw_bytes: torch.Tensor,
        h_start: torch.Tensor,
        doc_reset_mask: Optional[torch.Tensor] = None,
        max_span: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hotspot B1: Static-shape per-byte embedding, diagonal scan, and cosine causal prior."""
        B, T = raw_bytes.shape
        device = raw_bytes.device
        d_s = self.cfg.d_s

        byte_ids = raw_bytes.to(dtype=torch.long)
        u = self.embed(byte_ids)  # [B, T, d_s]
        alpha = torch.sigmoid(self.w_alpha(u))  # [B, T, d_s]
        b_inj = (1.0 - alpha) * F.silu(self.w_u(u))  # [B, T, d_s]

        # Precision Island #1 (Red Line #1): SGSC diagonal recurrence scan & causal byte prior in fp32
        with fp32_island(device):
            h_seq_f = parallel_diag_scan(
                alpha.float(),
                b_inj.float(),
                h0=h_start.float(),
                reset_mask=doc_reset_mask,
                max_span=max_span,
            )  # [B, T, d_s] fp32

            if h_start.ndim == 1:
                h0_col = h_start.float().view(1, 1, d_s).expand(B, 1, d_s)
            else:
                h0_col = h_start.float().unsqueeze(1)
            if T > 1:
                h_prev = torch.cat([h0_col, h_seq_f[:, :-1, :]], dim=1)
            else:
                h_prev = h0_col

            if doc_reset_mask is not None:
                h_prev = torch.where(doc_reset_mask.unsqueeze(-1), h0_col.expand(B, T, d_s), h_prev)

            h_prev_f = h_prev.float()
            h_prev_n = h_prev_f * torch.rsqrt(h_prev_f.pow(2).sum(dim=-1, keepdim=True) + 1e-2)
            w_out_f = self.w_out.weight.float()
            w_out_n = w_out_f * torch.rsqrt(w_out_f.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
            prior_logits = F.linear(h_prev_n, w_out_n) * 8.0  # [B, T, 256] fp32
            log_probs = F.log_softmax(prior_logits, dim=-1)
            prior_probs = log_probs.exp()
            info = -log_probs.gather(-1, byte_ids.unsqueeze(-1)).squeeze(-1)  # [B, T] fp32 >= 0

            if T > 1:
                dinfo_tail = F.relu(info[:, 1:] - info[:, :-1])
                delta_info = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.float32), dinfo_tail], dim=1)
            else:
                delta_info = torch.zeros((B, T), device=device, dtype=torch.float32)

            if doc_reset_mask is not None:
                delta_info = torch.where(doc_reset_mask, torch.zeros_like(delta_info), delta_info)

        return h_seq_f.to(u.dtype), prior_logits, prior_probs, info, delta_info

    def _init_weights(self) -> None:
        # Unit-scale byte embeddings align SGSC h_seq magnitude with RMSNormed PMD m_seq
        # on the shared weight-tied W_out matrix, preventing dead flat prior_logits at init.
        nn.init.normal_(self.embed.weight, std=1.0)
        nn.init.normal_(self.w_out.weight, std=1.0 / math.sqrt(self.cfg.d_s))
        nn.init.xavier_uniform_(self.w_alpha.weight)
        nn.init.constant_(self.w_alpha.bias, 1.0)  # bias toward retaining local context initially
        nn.init.xavier_uniform_(self.w_u.weight, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_f.weight)
        nn.init.xavier_uniform_(self.w_res.weight)
        nn.init.xavier_uniform_(self.w_sem.weight)

    @property
    def theta(self) -> torch.Tensor:
        return self.cfg.lif_theta_min + F.softplus(self.cfg.lif_theta_scale * self.theta_raw.float())

    def _byte_scan_dispatch(
        self,
        raw_bytes: torch.Tensor,
        h_start: torch.Tensor,
        doc_reset_mask: Optional[torch.Tensor],
        doc_max_span: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Hotspot B1 through the static-shape compiled kernel (when enabled) or eagerly."""
        B, T = raw_bytes.shape
        device = raw_bytes.device
        if self._compiled_byte_scan is not None and T > 0:
            # Align T up to 16KB for prod batches >=16KB (4KB for 4-16KB, 256B for tiny tests)
            # strictly inside Hotspot B1 so 100% of 64KB/128KB/256KB ragged batches hit ONE static
            # Inductor graph (<0.8% padding overhead, zero mid-run recompiles!).
            align_b = 16384 if T >= 16384 else (4096 if T >= 4096 else 256)
            pad = (align_b - (T % align_b)) % align_b
            if pad > 0:
                rb_in = torch.cat([raw_bytes, torch.zeros((B, pad), device=device, dtype=raw_bytes.dtype)], dim=1)
                if doc_reset_mask is not None:
                    rm_in = torch.cat([doc_reset_mask, torch.ones((B, pad), device=device, dtype=torch.bool)], dim=1)
                else:
                    rm_in = torch.cat(
                        [
                            torch.zeros((B, T), device=device, dtype=torch.bool),
                            torch.ones((B, pad), device=device, dtype=torch.bool),
                        ],
                        dim=1,
                    )
            else:
                rb_in = raw_bytes
                rm_in = (
                    doc_reset_mask
                    if doc_reset_mask is not None
                    else torch.zeros((B, T), device=device, dtype=torch.bool)
                )
            h_seq_p, prior_logits_p, prior_probs_p, info_p, delta_info_p = self._compiled_byte_scan(
                rb_in, h_start, rm_in, None
            )
            return (
                h_seq_p[:, :T, :],
                prior_logits_p[:, :T, :],
                prior_probs_p[:, :T, :],
                info_p[:, :T],
                delta_info_p[:, :T],
            )
        return self._byte_scan_impl(raw_bytes, h_start, doc_reset_mask, doc_max_span)

    def forward(
        self,
        raw_bytes: torch.Tensor,
        tau_s: float = 1.0,
        forced_spikes: Optional[torch.Tensor] = None,
        h_init: Optional[torch.Tensor] = None,
        doc_cu_seqlens: Optional[torch.Tensor] = None,
        row_layout: Optional[RowLayout] = None,
        entry_cont: Optional[torch.Tensor] = None,
        entry_last_info: Optional[torch.Tensor] = None,
    ) -> SGSCOutput:
        """Run SGSC on raw UTF-8 bytes [B, T] (uint8), with optional packed varlen doc_cu_seqlens.

        Stream-carry mode (`row_layout` given): `h_init [V, d_s]` is the per-entry initial hidden state and
        `entry_cont`/`entry_last_info [V]` restore the exact cross-chunk surprisal delta at continuations.
        """
        if raw_bytes.ndim != 2:
            raise ValueError(f"raw_bytes must be 2D [B, T], got {tuple(raw_bytes.shape)}")
        B, T = raw_bytes.shape
        device = raw_bytes.device
        d_s = self.cfg.d_s

        doc_reset_mask: Optional[torch.Tensor] = None
        doc_max_span: Optional[int] = None
        if doc_cu_seqlens is not None:
            if B != 1:
                raise ValueError(f"Packed varlen doc_cu_seqlens requires B=1, got B={B}")
            doc_reset_mask = torch.zeros((1, T), device=device, dtype=torch.bool)
            starts = doc_cu_seqlens[:-1].to(dtype=torch.int64, device=device)
            ends = doc_cu_seqlens[1:].to(dtype=torch.int64, device=device)
            if starts.numel() > 0:
                doc_reset_mask[0, starts] = True
                # Skip GPU->CPU sync when compiled byte scan, row_layout, or CUDA+Triton ignores max_span
                if (
                    self._compiled_byte_scan is None
                    and row_layout is None
                    and not (HAS_TRITON and device.type == "cuda")
                ):
                    doc_max_span = int((ends - starts).max().item())

        h_start = self.h0 if h_init is None else h_init
        # 1-3. Hotspot B1: per-byte projections + O(log T) diagonal scan + strictly causal prior
        if row_layout is not None:
            # Stream-carry mode: per-entry initial hidden states `h_init [V, d_s]` injected exactly via a
            # right-padded `[n_rows, row_len]` row layout (causal scans are exact under right padding).
            if doc_cu_seqlens is None or B != 1:
                raise ValueError("SGSC row_layout (stream carry) requires packed varlen doc_cu_seqlens with B=1")
            if h_start.ndim != 2 or int(h_start.shape[0]) != row_layout.n_entries:
                raise ValueError(
                    f"stream-carry h_init must be [V={row_layout.n_entries}, d_s], got {tuple(h_start.shape)}"
                )
            rb_rows = to_rows(raw_bytes[0], row_layout, 0)
            h_rows = pad_entries(h_start, row_layout.n_rows, 0.0)
            h_r, pl_r, pp_r, info_r, dinfo_r = self._byte_scan_dispatch(rb_rows, h_rows, None, row_layout.row_len)
            h_seq = from_rows(h_r, row_layout).unsqueeze(0)
            prior_logits = from_rows(pl_r, row_layout).unsqueeze(0)
            prior_probs = from_rows(pp_r, row_layout).unsqueeze(0)
            info = from_rows(info_r, row_layout).unsqueeze(0)
            delta_info = from_rows(dinfo_r, row_layout).unsqueeze(0)
            if entry_cont is not None and entry_last_info is not None and row_layout.n_entries > 0:
                # Exact cross-chunk `delta_info` at continuation starts (fresh entries keep 0 = doc reset).
                st = row_layout.starts
                with fp32_island(device):
                    d0 = F.relu(info[0, st].float() - entry_last_info.float())
                    d0 = torch.where(entry_cont.bool(), d0, torch.zeros_like(d0))
                delta_info = delta_info.index_put((torch.zeros_like(st), st), d0.to(delta_info.dtype))
        else:
            h_seq, prior_logits, prior_probs, info, delta_info = self._byte_scan_dispatch(
                raw_bytes, h_start, doc_reset_mask, doc_max_span
            )

        # 4. LIF membrane potential scan + UTF-8 & maxSegLen=16 & doc-boundary guard in float32
        with fp32_island(device):
            theta_eff = self.theta
            if forced_spikes is not None:
                spikes = forced_spikes.bool().to(device)
                drive = self.cfg.lif_beta1 * info + self.cfg.lif_beta2 * delta_info
                if T > 1:
                    prev_s = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.bool), spikes[:, :-1]], dim=1)
                else:
                    prev_s = torch.zeros((B, T), device=device, dtype=torch.bool)
                if doc_reset_mask is not None:
                    prev_s = prev_s | doc_reset_mask
                decay_a = torch.where(
                    prev_s,
                    torch.zeros((B, T), device=device, dtype=torch.float32),
                    torch.full((B, T), self.cfg.lif_lambda, device=device, dtype=torch.float32),
                )
                v_tilde = parallel_diag_scan(decay_a.unsqueeze(-1), drive.unsqueeze(-1)).squeeze(-1)
            else:
                v_tilde, spikes = lif_membrane_scan(
                    info=info,
                    delta_info=delta_info,
                    raw_bytes=raw_bytes,
                    lif_lambda=self.cfg.lif_lambda,
                    beta1=self.cfg.lif_beta1,
                    beta2=self.cfg.lif_beta2,
                    theta=theta_eff,
                    max_seg_len=self.cfg.max_seg_len,
                    doc_cu_seqlens=doc_cu_seqlens,
                )

            # STE surrogate spikes (Commandment #6)
            s_tilde = torch.sigmoid((v_tilde - theta_eff) / max(tau_s, 1e-4))
            s_ste = (spikes.float() - s_tilde).detach() + s_tilde

        # 5. Zero-padding Ragged cu_seqlens aggregation & Holographic RoPE phase binding
        cu_seqlens, seg_counts = build_cu_seqlens(spikes)
        m_total = cu_seqlens.numel() - 1
        seg_id, tau, seg_lens, seg_end_idx = ragged_local_offsets(cu_seqlens, B * T)

        h_flat = h_seq.reshape(B * T, d_s)
        h_end = h_flat[seg_end_idx]  # [M_total, d_s]

        rope_max_pos = (self.cfg.max_seg_len + 4) if forced_spikes is None else None
        q_seg = apply_holographic_rope(
            self.w_q(h_end),
            seg_lens,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=rope_max_pos,
        )  # [M_total, d_s]
        k_flat = apply_holographic_rope(
            self.w_k(h_flat),
            tau,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=rope_max_pos,
        )  # [B*T, d_s]

        # Intra-segment attention scores & softmax strictly in float32
        attn_scores = (k_flat.float() * q_seg.float()[seg_id]).sum(dim=-1) / math.sqrt(d_s)
        gamma = ragged_segment_softmax(attn_scores, seg_id, m_total)  # [B*T] float32

        v_rot = apply_holographic_rope(
            self.w_f(h_flat),
            tau,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=rope_max_pos,
        )  # [B*T, d_s]
        weighted_v = gamma.unsqueeze(-1).to(v_rot.dtype) * v_rot
        agg = ragged_segment_sum(weighted_v, seg_id, m_total)  # [M_total, d_s]

        s_ste_end = s_ste.reshape(-1)[seg_end_idx].unsqueeze(-1).to(agg.dtype)
        f_stem = self.norm_f(h_end + s_ste_end * agg)  # [M_total, d_s]

        # 6. Slow cortical semantic upscaling
        sem_proj = self.w_sem(f_stem)  # [M_total, D]
        sem_norm = sem_proj.float().norm(dim=-1)  # [M_total] fp32
        c_slow = self.norm_c(self.w_res(h_end) + sem_proj)  # [M_total, D]

        mean_info = ragged_segment_mean(info.reshape(-1), seg_id, seg_lens)  # [M_total] fp32

        return SGSCOutput(
            f_stem=f_stem,
            c_slow=c_slow,
            sem_norm=sem_norm,
            mean_info=mean_info,
            cu_seqlens=cu_seqlens,
            seg_counts=seg_counts,
            spikes=spikes,
            s_tilde=s_tilde,
            s_ste=s_ste,
            prior_probs=prior_probs,
            prior_logits=prior_logits,
            info=info,
            v_tilde=v_tilde,
            h_seq=h_seq,
            theta=theta_eff,
        )


@dataclass
class SGSCFlatStreamState:
    """O(1) single-stream runtime state for `SGSCFlat.step_streaming_byte` (parity with batched `SGSCFlat`)."""

    h: torch.Tensor               # [B, d_s]
    v: torch.Tensor               # [B] float32 LIF membrane potential
    prev_info: torch.Tensor       # [B] float32 previous byte self-information
    has_prev_byte: torch.Tensor   # [B] bool False on first byte of document
    seg_len: torch.Tensor         # [B] int32 bytes accumulated in current segment
    utf8_rem: torch.Tensor        # [B] int32 remaining UTF-8 continuation bytes
    ast_depth: torch.Tensor       # [B] int32 running causal AST bracket depth [0, 31]
    seg_start_depth: torch.Tensor # [B] int32 AST depth before current segment started
    prev2_bytes: torch.Tensor     # [B, 2] int32 previous 2 bytes for `\x1dE\n` detection
    seg_pin: torch.Tensor         # [B] bool accumulated `\x1dE\n` hit in current segment
    seg_unpin: torch.Tensor       # [B] bool accumulated `0x04`/`0x1E` hit in current segment
    seg_h_buf: torch.Tensor       # [B, max_seg_len, d_s] bounded segment hidden-state buffer
    seg_info_sum: torch.Tensor    # [B] float32 sum of surprisal in current segment


class SGSCFlat(nn.Module):
    """DCPA-Flat v3.3 Surprisal-Gated Spike Codec operating purely on static `[P, T] -> [P, S]` grids.

    Key v3.3 upgrades:
      - Row-L2-normalized shared output vocab `w_out` + independent learnable `log_temp_sgsc` (init `ln 8.0`).
      - Pure-tensor static-grid LIF scan (`static_lif_scan`) + `S_alloc = S + 1` overflow trash-slot scatter.
      - GPU-only `seg_l_reg [P, S, 64]` lookup from `batch.doc_meta` using `seg_doc_id.clamp_min(0)` (zero syncs).
      - Compiles cleanly under `torch.compile(..., fullgraph=True, dynamic=False)` with zero graph breaks.
    """

    def __init__(
        self,
        cfg: DCPAFlatConfig,
        shared_embed: Optional[nn.Embedding] = None,
        shared_w_out: Optional[nn.Linear] = None,
        lang_embed: Optional[nn.Embedding] = None,
        style_embed: Optional[nn.Embedding] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d_s = cfg.d_s
        D = cfg.d_concept
        d_l_half = cfg.d_l // 2

        # Shared byte vocabulary embeddings & row-L2-normalized output head (tied with PMDFlat)
        self.embed = shared_embed if shared_embed is not None else nn.Embedding(cfg.vocab_size, d_s)
        self.w_out = shared_w_out if shared_w_out is not None else nn.Linear(d_s, cfg.vocab_size, bias=False)
        self.log_temp_sgsc = nn.Parameter(torch.tensor(math.log(cfg.vocab_temp_init), dtype=torch.float32))

        # Canonical orthogonal 32D lang ⊕ 32D style embeddings (GPU-only lookup via batch.doc_meta)
        self.lang_embed = lang_embed if lang_embed is not None else nn.Embedding(cfg.num_lang_domains, d_l_half)
        self.style_embed = style_embed if style_embed is not None else nn.Embedding(cfg.num_style_domains, d_l_half)
        self.norm_lang = RMSNorm(d_l_half, eps=cfg.eps)
        self.norm_style = RMSNorm(d_l_half, eps=cfg.eps)

        # Learnable initial hidden state h_0 & causal diagonal sensory recurrence
        self.h0 = nn.Parameter(torch.zeros(d_s))
        self.w_alpha = nn.Linear(d_s, d_s, bias=True)
        self.w_u = nn.Linear(d_s, d_s, bias=False)

        # Learnable LIF threshold theta >= lif_theta_min
        target_delta = max(cfg.lif_theta_init - cfg.lif_theta_min, 1e-3)
        raw_init = math.log(math.expm1(target_delta)) / max(cfg.lif_theta_scale, 1e-6)
        self.theta_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # Doc-masked causal k=3 depthwise conv weights [3, d_s] (tap 0=cur, 1=t-1, 2=t-2)
        self.conv_w = nn.Parameter(torch.zeros((3, d_s), dtype=torch.float32))

        # Holographic RoPE phase-bound stem extractor
        self.w_q = nn.Linear(d_s, d_s, bias=False)
        self.w_k = nn.Linear(d_s, d_s, bias=False)
        self.w_f = nn.Linear(d_s, d_s, bias=False)
        self.norm_f = RMSNorm(d_s, eps=cfg.eps)

        # Slow cortical concept upscaling
        self.w_res = nn.Linear(d_s, D, bias=False)
        self.w_sem = nn.Linear(d_s, D, bias=False)
        self.norm_c = RMSNorm(D, eps=cfg.eps)

        self._init_weights(
            init_embed=(shared_embed is None),
            init_w_out=(shared_w_out is None),
            init_lang=(lang_embed is None),
            init_style=(style_embed is None),
        )

    def _init_weights(
        self,
        *,
        init_embed: bool = True,
        init_w_out: bool = True,
        init_lang: bool = True,
        init_style: bool = True,
    ) -> None:
        if init_embed:
            nn.init.normal_(self.embed.weight, std=1.0)
        if init_w_out:
            nn.init.normal_(self.w_out.weight, std=1.0 / math.sqrt(self.cfg.d_s))
        if init_lang:
            nn.init.normal_(self.lang_embed.weight, std=1.0)
        if init_style:
            nn.init.normal_(self.style_embed.weight, std=1.0)
        nn.init.xavier_uniform_(self.w_alpha.weight)
        nn.init.constant_(self.w_alpha.bias, 1.0)
        nn.init.xavier_uniform_(self.w_u.weight, gain=math.sqrt(2.0))
        with torch.no_grad():
            self.conv_w.zero_()
            self.conv_w[0].fill_(1.0)
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_f.weight)
        nn.init.xavier_uniform_(self.w_res.weight)
        nn.init.xavier_uniform_(self.w_sem.weight)

    @property
    def theta(self) -> torch.Tensor:
        return self.cfg.lif_theta_min + F.softplus(self.cfg.lif_theta_scale * self.theta_raw.float())

    @property
    def vocab_temp(self) -> torch.Tensor:
        return self.log_temp_sgsc.float().clamp(-2.0, 4.0).exp()

    def get_lang_vec(
        self,
        lang_id: torch.Tensor,
        style_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute canonical orthogonal 64D register vector `[RMSNorm_32(lang) || RMSNorm_32(style)]`."""
        lid = lang_id.to(torch.long).clamp(0, self.cfg.num_lang_domains - 1)
        if style_id is None:
            sid = torch.zeros_like(lid)
        else:
            sid = style_id.to(torch.long).clamp(0, self.cfg.num_style_domains - 1)
        lang_h = self.norm_lang(self.lang_embed(lid))
        style_h = self.norm_style(self.style_embed(sid))
        return torch.cat([lang_h, style_h], dim=-1)

    def _compute_prior_logits(self, h_prev: torch.Tensor) -> torch.Tensor:
        """Row-L2-normalized cosine prior logits scaled by learnable `exp(log_temp_sgsc)` in float32."""
        h_f = h_prev.float()
        h_n = h_f * torch.rsqrt(h_f.pow(2).sum(dim=-1, keepdim=True) + 1e-2)
        w_f = self.w_out.weight.float()
        w_n = w_f * torch.rsqrt(w_f.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        return F.linear(h_n, w_n) * self.vocab_temp

    def forward(
        self,
        batch: StaticGridBatch,
        tau_s: float = 1.0,
        forced_spikes: Optional[torch.Tensor] = None,
        grid_segs: Optional[int] = None,
    ) -> ScatterBridgeOut:
        """Run static-grid SGSC on `StaticGridBatch [P, T]` -> frozen `ScatterBridgeOut [P, S]`."""
        raw_bytes = batch.x_bytes
        doc_id_byte = batch.doc_id_byte
        doc_start_byte = batch.doc_start_byte
        doc_end_byte = batch.doc_end_byte

        P, T = raw_bytes.shape
        S = int(self.cfg.grid_segs if grid_segs is None else grid_segs)
        d_s = self.cfg.d_s
        device = raw_bytes.device

        byte_valid, eff_doc_start, eff_doc_end = compute_effective_doc_boundaries(
            doc_id_byte, doc_start_byte, doc_end_byte
        )
        # Diagonal scan resets at lane start t=0 and every document/pad boundary
        if T > 1:
            scan_reset = torch.cat(
                [
                    torch.ones((P, 1), device=device, dtype=torch.bool),
                    eff_doc_start[:, 1:] | (~byte_valid[:, 1:]),
                ],
                dim=1,
            )
        else:
            scan_reset = torch.ones((P, T), device=device, dtype=torch.bool)

        byte_ids = raw_bytes.to(torch.long)
        u0 = self.embed(byte_ids)  # [P, T, d_s]
        # Strictly document-isolated causal k=3 depthwise shift (never leaks across lane doc boundaries!)
        if T > 1:
            valid_m1 = (~scan_reset) & byte_valid
            u_m1 = torch.cat([torch.zeros_like(u0[:, :1, :]), u0[:, :-1, :]], dim=1) * valid_m1.unsqueeze(-1).to(u0.dtype)
        else:
            u_m1 = torch.zeros_like(u0)
        if T > 2:
            prev_not_rst = torch.cat([torch.zeros((P, 1), device=device, dtype=torch.bool), ~scan_reset[:, :-1]], dim=1)
            valid_m2 = (~scan_reset) & prev_not_rst & byte_valid
            u_m2 = torch.cat([torch.zeros((P, 2, d_s), device=device, dtype=u0.dtype), u0[:, :-2, :]], dim=1) * valid_m2.unsqueeze(-1).to(u0.dtype)
        else:
            u_m2 = torch.zeros_like(u0)
        cw = self.conv_w.to(dtype=u0.dtype)
        u = u0 * cw[0] + u_m1 * cw[1] + u_m2 * cw[2]
        alpha = torch.sigmoid(self.w_alpha(u))
        b_inj = (1.0 - alpha) * F.silu(self.w_u(u))

        # 1. Causal byte diagonal scan & strictly causal prior logits in float32
        with fp32_island(device):
            h_seq_f = parallel_diag_scan(
                alpha.float(),
                b_inj.float(),
                h0=self.h0.float(),
                reset_mask=scan_reset,
            )  # [P, T, d_s] fp32

            h0_col = self.h0.float().view(1, 1, d_s).expand(P, 1, d_s)
            if T > 1:
                h_prev = torch.cat([h0_col, h_seq_f[:, :-1, :]], dim=1)
            else:
                h_prev = h0_col
            h_prev = torch.where(scan_reset.unsqueeze(-1), h0_col.expand(P, T, d_s), h_prev)

            prior_logits = self._compute_prior_logits(h_prev)  # [P, T, 256] fp32
            log_probs = F.log_softmax(prior_logits, dim=-1)
            info = -log_probs.gather(-1, byte_ids.unsqueeze(-1)).squeeze(-1)  # [P, T] fp32

            if T > 1:
                dinfo_tail = F.relu(info[:, 1:] - info[:, :-1])
                delta_info = torch.cat([torch.zeros((P, 1), device=device, dtype=torch.float32), dinfo_tail], dim=1)
            else:
                delta_info = torch.zeros((P, T), device=device, dtype=torch.float32)

            delta_info = torch.where(scan_reset, torch.zeros_like(delta_info), delta_info)
            info = torch.where(byte_valid, info, torch.zeros_like(info))
            delta_info = torch.where(byte_valid, delta_info, torch.zeros_like(delta_info))

            # 2. Pure-tensor static-grid LIF scan (zero syncs, zero graph breaks)
            theta_eff = self.theta
            if forced_spikes is not None:
                spikes = (forced_spikes.bool() | eff_doc_end) & byte_valid
                drive = self.cfg.lif_beta1 * info + self.cfg.lif_beta2 * delta_info
                if T > 1:
                    prev_sp = torch.cat([torch.zeros((P, 1), device=device, dtype=torch.bool), spikes[:, :-1]], dim=1)
                else:
                    prev_sp = torch.zeros((P, T), device=device, dtype=torch.bool)
                seg_starts_f = (prev_sp | eff_doc_start) & byte_valid
                decay_a = torch.where(
                    seg_starts_f,
                    torch.zeros_like(drive),
                    torch.full_like(drive, float(self.cfg.lif_lambda)),
                )
                v_tilde = segmented_scan_4step_affine(decay_a, drive, max_span=self.cfg.max_seg_len)
                v_tilde = torch.where(byte_valid, v_tilde, torch.zeros_like(v_tilde))
            else:
                v_tilde, spikes = static_lif_scan(
                    info=info,
                    delta_info=delta_info,
                    raw_bytes=raw_bytes,
                    doc_id_byte=doc_id_byte,
                    doc_start_byte=doc_start_byte,
                    doc_end_byte=doc_end_byte,
                    lif_lambda=self.cfg.lif_lambda,
                    beta1=self.cfg.lif_beta1,
                    beta2=self.cfg.lif_beta2,
                    theta=theta_eff,
                    max_seg_len=self.cfg.max_seg_len,
                )

            s_tilde = torch.sigmoid((v_tilde - theta_eff) / max(float(tau_s), 1e-4))
            s_tilde = torch.where(byte_valid, s_tilde, torch.zeros_like(s_tilde))
            s_ste = (spikes.float() - s_tilde).detach() + s_tilde

        h_seq = h_seq_f.to(u.dtype)

        # 3. Pure prefix-sum static segment mapping with S_alloc = S + 1 trash slot
        seg_map = build_static_segment_map(
            spikes=spikes,
            doc_id_byte=doc_id_byte,
            doc_start_byte=doc_start_byte,
            doc_end_byte=doc_end_byte,
            grid_segs=S,
            max_seg_len=self.cfg.max_seg_len,
        )

        # 4. Holographic RoPE phase-bound stem extraction onto static [P, S, d_s]
        h_end = static_segment_scatter_at_mask(h_seq, seg_map.is_seg_end, seg_map.byte_to_seg, S)  # [P, S, d_s]
        s_ste_end = static_segment_scatter_at_mask(s_ste, seg_map.is_seg_end, seg_map.byte_to_seg, S)  # [P, S]

        q_seg = apply_holographic_rope(
            self.w_q(h_end),
            seg_map.seg_lens.clamp(1, self.cfg.max_seg_len),
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=self.cfg.max_seg_len,
        )  # [P, S, d_s]
        q_byte = gather_seg_to_byte(q_seg, seg_map.byte_to_seg)  # [P, T, d_s]
        k_byte = apply_holographic_rope(
            self.w_k(h_seq),
            seg_map.intra_seg_pos,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=self.cfg.max_seg_len,
        )  # [P, T, d_s]

        with fp32_island(device):
            attn_scores = (k_byte.float() * q_byte.float()).sum(dim=-1) / math.sqrt(float(d_s))  # [P, T]
            gamma = static_segment_softmax(attn_scores, seg_map.byte_to_seg, S)  # [P, T] fp32

        v_rot = apply_holographic_rope(
            self.w_f(h_seq),
            seg_map.intra_seg_pos,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=self.cfg.max_seg_len,
        )  # [P, T, d_s]
        weighted_v = gamma.unsqueeze(-1).to(v_rot.dtype) * v_rot
        agg = static_segment_scatter_add(weighted_v, seg_map.byte_to_seg, S)  # [P, S, d_s]

        # Detach s_ste_end (identically 1.0 on valid segment ends) so norm_f scale invariance never
        # injects spurious <grad, agg> residual-balance gradients into theta_raw / prior_logits!
        f_stem_raw = self.norm_f(h_end + s_ste_end.detach().unsqueeze(-1).to(agg.dtype) * agg)
        valid_3d = seg_map.seg_valid.unsqueeze(-1)
        f_stem = torch.where(valid_3d, f_stem_raw, torch.zeros_like(f_stem_raw))

        # 5. Slow cortical concept projection & auxiliary monitors
        sem_proj = self.w_sem(f_stem)  # [P, S, D]
        sem_norm = torch.where(
            seg_map.seg_valid,
            sem_proj.float().norm(dim=-1),
            torch.zeros((P, S), device=device, dtype=torch.float32),
        )
        x0_raw = self.norm_c(self.w_res(h_end) + sem_proj)
        x0 = torch.where(valid_3d, x0_raw, torch.zeros_like(x0_raw))
        mean_info = static_segment_mean(info, seg_map.byte_to_seg, seg_map.seg_lens, S)

        # 6. Causal AST bracket depths & G2 Slot-0 pin/unpin triggers
        seg_start_depth, seg_end_depth, seg_pin_trig, seg_unpin_trig = compute_static_seg_depths_and_triggers(
            x_bytes=raw_bytes,
            doc_id_byte=doc_id_byte,
            doc_start_byte=doc_start_byte,
            seg_map=seg_map,
            grid_segs=S,
            max_depth=31,
        )

        # 7. GPU-only seg_l_reg lookup from batch.doc_meta using seg_doc_id.clamp_min(0) (zero sync!)
        max_meta_idx = max(int(batch.doc_meta.shape[0]) - 1, 0)
        doc_idx_safe = seg_map.seg_doc_id.to(torch.long).clamp(0, max_meta_idx)
        meta_seg = batch.doc_meta[doc_idx_safe]  # [P, S, 3] int32
        seg_l_reg_raw = self.get_lang_vec(meta_seg[..., 0], meta_seg[..., 1]).to(f_stem.dtype)
        seg_l_reg = torch.where(valid_3d, seg_l_reg_raw, torch.zeros_like(seg_l_reg_raw))

        return ScatterBridgeOut(
            f_stem=f_stem,
            x0=x0,
            seg_valid=seg_map.seg_valid,
            seg_doc_id=seg_map.seg_doc_id,
            seg_reset=seg_map.seg_reset,
            seg_pin_trig=seg_pin_trig,
            seg_unpin_trig=seg_unpin_trig,
            seg_start_depth=seg_start_depth,
            seg_end_depth=seg_end_depth,
            seg_l_reg=seg_l_reg,
            byte_to_seg=seg_map.byte_to_seg,
            intra_seg_pos=seg_map.intra_seg_pos,
            byte_overflow=seg_map.byte_overflow,
            seg_counts=seg_map.seg_counts,
            prior_logits=prior_logits,
            spikes=spikes,
            s_tilde=s_tilde,
            mean_info=mean_info,
            sem_norm=sem_norm,
        )

    def init_stream_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> SGSCFlatStreamState:
        """Initialize O(1) single-stream state for `step_streaming_byte`."""
        dev = self.h0.device if device is None else device
        B = int(batch_size)
        return SGSCFlatStreamState(
            h=self.h0.detach().to(device=dev, dtype=dtype).unsqueeze(0).expand(B, self.cfg.d_s).clone(),
            v=torch.zeros((B,), device=dev, dtype=torch.float32),
            prev_info=torch.zeros((B,), device=dev, dtype=torch.float32),
            has_prev_byte=torch.zeros((B,), device=dev, dtype=torch.bool),
            seg_len=torch.zeros((B,), device=dev, dtype=torch.int32),
            utf8_rem=torch.zeros((B,), device=dev, dtype=torch.int32),
            ast_depth=torch.zeros((B,), device=dev, dtype=torch.int32),
            seg_start_depth=torch.zeros((B,), device=dev, dtype=torch.int32),
            prev2_bytes=torch.full((B, 2), -1, device=dev, dtype=torch.int32),
            seg_pin=torch.zeros((B,), device=dev, dtype=torch.bool),
            seg_unpin=torch.zeros((B,), device=dev, dtype=torch.bool),
            seg_h_buf=torch.zeros((B, self.cfg.max_seg_len, self.cfg.d_s), device=dev, dtype=dtype),
            seg_info_sum=torch.zeros((B,), device=dev, dtype=torch.float32),
        )

    def step_streaming_byte(
        self,
        byte_val: torch.Tensor,
        state: SGSCFlatStreamState,
        force_end: bool = False,
    ) -> tuple[dict[str, torch.Tensor], SGSCFlatStreamState]:
        """Single-byte O(1) streaming SGSC step matching batched `SGSCFlat.forward` 1-to-1.

        Args:
          byte_val: `[B]` uint8/int32/int64 current raw UTF-8 byte.
          state:    `SGSCFlatStreamState`.
          force_end: True on final byte of document (forces segment emission).

        Returns:
          `(step_out_dict, next_state)` where `step_out_dict` holds `prior_logits [B, 256]`,
          `spike [B] bool`, `f_stem [B, d_s]`, `x0 [B, D]`, `sem_norm [B]`, `mean_info [B]`,
          `seg_start_depth [B]`, `seg_end_depth [B]`, `seg_pin_trig [B]`, `seg_unpin_trig [B]`.
        """
        B = byte_val.shape[0]
        device = byte_val.device
        L_max = self.cfg.max_seg_len
        b_long = byte_val.to(torch.long).view(B)
        b_i32 = byte_val.to(torch.int32).view(B)

        u0 = self.embed(b_long)  # [B, d_s]
        b_m1 = state.prev2_bytes[:, 1]
        b_m2 = state.prev2_bytes[:, 0]
        u_m1 = self.embed(b_m1.clamp_min(0).to(torch.long)) * (b_m1 >= 0).unsqueeze(-1).to(u0.dtype)
        u_m2 = self.embed(b_m2.clamp_min(0).to(torch.long)) * (b_m2 >= 0).unsqueeze(-1).to(u0.dtype)
        cw = self.conv_w.to(dtype=u0.dtype)
        u = u0 * cw[0] + u_m1 * cw[1] + u_m2 * cw[2]
        alpha = torch.sigmoid(self.w_alpha(u))
        b_inj = (1.0 - alpha) * F.silu(self.w_u(u))

        with fp32_island(device):
            prior_logits = self._compute_prior_logits(state.h)  # [B, 256] fp32
            log_probs = F.log_softmax(prior_logits, dim=-1)
            info = -log_probs.gather(-1, b_long.unsqueeze(-1)).squeeze(-1)  # [B] fp32
            delta_info = torch.where(
                state.has_prev_byte,
                F.relu(info - state.prev_info),
                torch.zeros_like(info),
            )
            h_new_f = alpha.float() * state.h.float() + b_inj.float()
            drive = self.cfg.lif_beta1 * info + self.cfg.lif_beta2 * delta_info
            v_tilde = self.cfg.lif_lambda * state.v + drive

            # UTF-8 continuation state update
            is_cont = (b_i32 & 0xC0) == 0x80
            lead_rem = torch.where(
                (b_i32 & 0xE0) == 0xC0,
                torch.ones_like(b_i32),
                torch.where(
                    (b_i32 & 0xF0) == 0xE0,
                    torch.full_like(b_i32, 2),
                    torch.where((b_i32 & 0xF8) == 0xF0, torch.full_like(b_i32, 3), torch.zeros_like(b_i32)),
                ),
            )
            utf8_rem_new = torch.where(is_cont, (state.utf8_rem - 1).clamp_min(0), lead_rem)
            can_spike = utf8_rem_new == 0

            seg_len_new = state.seg_len + 1
            fire = (seg_len_new >= L_max) | (can_spike & (v_tilde >= self.theta))
            if force_end:
                fire = torch.ones_like(fire)

            # Protocol pin/unpin & AST bracket depth update
            pin_hit = (
                (state.prev2_bytes[:, 0] == 0x1D)
                & (state.prev2_bytes[:, 1] == ord("E"))
                & (b_i32 == 0x0A)
                & state.has_prev_byte
            )
            unpin_hit = (b_i32 == 0x04) | (b_i32 == 0x1E)
            post_reset = (b_i32 == 0x04) | (b_i32 == 0x13) | (b_i32 == 0x1E) | pin_hit
            is_open = (b_i32 == 0x28) | (b_i32 == 0x5B) | (b_i32 == 0x7B)
            is_close = (b_i32 == 0x29) | (b_i32 == 0x5D) | (b_i32 == 0x7D)
            delta_d = is_open.to(torch.int32) - is_close.to(torch.int32)
            depth_new = torch.where(
                post_reset,
                torch.zeros_like(state.ast_depth),
                (state.ast_depth + delta_d).clamp(0, 31),
            )

        h_new = h_new_f.to(u.dtype)
        # Write h_new into bounded [B, L_max, d_s] buffer at slot (seg_len_new - 1)
        slot_idx = (seg_len_new - 1).clamp(0, L_max - 1).to(torch.long)
        slot_mask = F.one_hot(slot_idx, num_classes=L_max).to(h_new.dtype).unsqueeze(-1)  # [B, L_max, 1]
        seg_h_buf_upd = state.seg_h_buf * (1.0 - slot_mask) + h_new.unsqueeze(1) * slot_mask
        seg_info_sum_upd = state.seg_info_sum + info
        seg_pin_upd = state.seg_pin | pin_hit
        seg_unpin_upd = state.seg_unpin | unpin_hit

        # Vectorized segment readout (exact match to batched SGSCFlat when fire == True)
        pos_tbl = torch.arange(1, L_max + 1, device=device, dtype=torch.int32).unsqueeze(0).expand(B, L_max)
        valid_pos = pos_tbl <= seg_len_new.unsqueeze(1)
        q_seg = apply_holographic_rope(
            self.w_q(h_new),
            seg_len_new.clamp(1, L_max),
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=L_max,
        )  # [B, d_s]
        k_buf = apply_holographic_rope(
            self.w_k(seg_h_buf_upd),
            pos_tbl,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=L_max,
        )  # [B, L_max, d_s]
        with fp32_island(device):
            scores = (k_buf.float() * q_seg.float().unsqueeze(1)).sum(dim=-1) / math.sqrt(float(self.cfg.d_s))
            scores = torch.where(valid_pos, scores, torch.full_like(scores, float("-inf")))
            gamma = F.softmax(scores, dim=-1)  # [B, L_max]
        v_buf = apply_holographic_rope(
            self.w_f(seg_h_buf_upd),
            pos_tbl,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
            max_pos=L_max,
        )
        agg = (gamma.unsqueeze(-1).to(v_buf.dtype) * v_buf).sum(dim=1)  # [B, d_s]
        f_stem = self.norm_f(h_new + agg)
        sem_proj = self.w_sem(f_stem)
        sem_norm = sem_proj.float().norm(dim=-1)
        x0 = self.norm_c(self.w_res(h_new) + sem_proj)
        mean_info = seg_info_sum_upd / seg_len_new.float().clamp_min(1.0)

        step_out = {
            "prior_logits": prior_logits,
            "spike": fire,
            "f_stem": f_stem,
            "x0": x0,
            "sem_norm": sem_norm,
            "mean_info": mean_info,
            "seg_start_depth": state.seg_start_depth,
            "seg_end_depth": depth_new,
            "seg_pin_trig": seg_pin_upd,
            "seg_unpin_trig": seg_unpin_upd,
            "intra_seg_pos": seg_len_new,
        }

        next_prev2 = torch.stack([state.prev2_bytes[:, 1], b_i32], dim=1)
        next_state = SGSCFlatStreamState(
            h=h_new,
            v=torch.where(fire, torch.zeros_like(v_tilde), v_tilde),
            prev_info=info,
            has_prev_byte=torch.ones((B,), device=device, dtype=torch.bool),
            seg_len=torch.where(fire, torch.zeros_like(seg_len_new), seg_len_new),
            utf8_rem=utf8_rem_new,
            ast_depth=depth_new,
            seg_start_depth=torch.where(fire, depth_new, state.seg_start_depth),
            prev2_bytes=next_prev2,
            seg_pin=torch.where(fire, torch.zeros_like(seg_pin_upd), seg_pin_upd),
            seg_unpin=torch.where(fire, torch.zeros_like(seg_unpin_upd), seg_unpin_upd),
            seg_h_buf=torch.where(fire.view(B, 1, 1), torch.zeros_like(seg_h_buf_upd), seg_h_buf_upd),
            seg_info_sum=torch.where(fire, torch.zeros_like(seg_info_sum_upd), seg_info_sum_upd),
        )
        return step_out, next_state

    @torch.no_grad()
    def encode_streaming_segments(
        self,
        raw_bytes_1d: torch.Tensor,
        lang_id: int = 0,
        style_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Convenience single-document streaming encoder (for inference & parity tests)."""
        device = raw_bytes_1d.device
        dtype = self.w_q.weight.dtype
        T_doc = int(raw_bytes_1d.shape[0])
        state = self.init_stream_state(batch_size=1, device=device, dtype=dtype)
        f_stems: list[torch.Tensor] = []
        x0s: list[torch.Tensor] = []
        sem_norms: list[torch.Tensor] = []
        mean_infos: list[torch.Tensor] = []
        start_depths: list[torch.Tensor] = []
        end_depths: list[torch.Tensor] = []
        pin_trigs: list[torch.Tensor] = []
        unpin_trigs: list[torch.Tensor] = []
        prior_logits_list: list[torch.Tensor] = []
        spikes_list: list[torch.Tensor] = []

        for t in range(T_doc):
            out, state = self.step_streaming_byte(
                raw_bytes_1d[t : t + 1],
                state,
                force_end=(t == T_doc - 1),
            )
            prior_logits_list.append(out["prior_logits"][0])
            spikes_list.append(out["spike"][0])
            if bool(out["spike"][0].item()):
                f_stems.append(out["f_stem"][0])
                x0s.append(out["x0"][0])
                sem_norms.append(out["sem_norm"][0])
                mean_infos.append(out["mean_info"][0])
                start_depths.append(out["seg_start_depth"][0])
                end_depths.append(out["seg_end_depth"][0])
                pin_trigs.append(out["seg_pin_trig"][0])
                unpin_trigs.append(out["seg_unpin_trig"][0])

        M = len(f_stems)
        l_vec = self.get_lang_vec(
            torch.tensor([lang_id], device=device, dtype=torch.int32),
            torch.tensor([style_id], device=device, dtype=torch.int32),
        ).to(dtype)
        return {
            "f_stem": torch.stack(f_stems, dim=0) if M > 0 else torch.zeros((0, self.cfg.d_s), device=device, dtype=dtype),
            "x0": torch.stack(x0s, dim=0) if M > 0 else torch.zeros((0, self.cfg.d_concept), device=device, dtype=dtype),
            "sem_norm": torch.stack(sem_norms, dim=0) if M > 0 else torch.zeros((0,), device=device, dtype=torch.float32),
            "mean_info": torch.stack(mean_infos, dim=0) if M > 0 else torch.zeros((0,), device=device, dtype=torch.float32),
            "seg_start_depth": torch.stack(start_depths, dim=0) if M > 0 else torch.zeros((0,), device=device, dtype=torch.int32),
            "seg_end_depth": torch.stack(end_depths, dim=0) if M > 0 else torch.zeros((0,), device=device, dtype=torch.int32),
            "seg_pin_trig": torch.stack(pin_trigs, dim=0) if M > 0 else torch.zeros((0,), device=device, dtype=torch.bool),
            "seg_unpin_trig": torch.stack(unpin_trigs, dim=0) if M > 0 else torch.zeros((0,), device=device, dtype=torch.bool),
            "seg_l_reg": l_vec.expand(M, -1).clone() if M > 0 else torch.zeros((0, self.cfg.d_l), device=device, dtype=dtype),
            "prior_logits": torch.stack(prior_logits_list, dim=0) if T_doc > 0 else torch.zeros((0, 256), device=device, dtype=torch.float32),
            "spikes": torch.stack(spikes_list, dim=0) if T_doc > 0 else torch.zeros((0,), device=device, dtype=torch.bool),
        }
