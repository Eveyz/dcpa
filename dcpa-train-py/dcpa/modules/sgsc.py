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
from dcpa.ops.scan import lif_membrane_scan, parallel_diag_scan


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

        self._init_weights()

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

    def forward(
        self,
        raw_bytes: torch.Tensor,
        tau_s: float = 1.0,
        forced_spikes: Optional[torch.Tensor] = None,
        h_init: Optional[torch.Tensor] = None,
        doc_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> SGSCOutput:
        """Run SGSC on raw UTF-8 bytes [B, T] (uint8), with optional packed varlen doc_cu_seqlens."""
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
                doc_max_span = int((ends - starts).max().item())

        # 1. Parallel byte embedding & diagonal recurrence inputs
        byte_ids = raw_bytes.to(dtype=torch.long)
        u = self.embed(byte_ids)  # [B, T, d_s]
        alpha = torch.sigmoid(self.w_alpha(u))  # [B, T, d_s]
        b_inj = (1.0 - alpha) * F.silu(self.w_u(u))  # [B, T, d_s]

        h_start = self.h0 if h_init is None else h_init
        # Precision Island #1 (Red Line #1): SGSC diagonal recurrence scan, causal byte prior
        # log_softmax self-information I_t, and LIF membrane potential integral v_tilde in pure fp32!
        with fp32_island(device):
            # 2. Parallel Associative Scan (O(log T) depth, strictly reset at every doc_cu_seqlens boundary)
            h_seq = parallel_diag_scan(
                alpha.float(),
                b_inj.float(),
                h0=h_start.float(),
                reset_mask=doc_reset_mask,
                max_span=doc_max_span,
            )  # [B, T, d_s] fp32

            # 3. Strictly causal shifted state [h_0, h_1, ..., h_{T-1}] -> prior p_t in float32
            if h_start.ndim == 1:
                h0_col = h_start.float().view(1, 1, d_s).expand(B, 1, d_s)
            else:
                h0_col = h_start.float().unsqueeze(1)
            if T > 1:
                h_prev = torch.cat([h0_col, h_seq[:, :-1, :]], dim=1)
            else:
                h_prev = h0_col

            if doc_reset_mask is not None:
                h_prev = torch.where(doc_reset_mask.unsqueeze(-1), h0_col.expand(B, T, d_s), h_prev)

            # Scale-invariant cosine prior on strictly weight-tied W_out: decouples PMD logit-norm
            # growth from SGSC byte surprisal I_t, eliminating late-stage spike boundary churn!
            # Smooth +1e-2 denominator on h_prev prevents gradient singularity at zero-initialized h_0.
            h_prev_f = h_prev.float()
            h_prev_n = h_prev_f * torch.rsqrt(h_prev_f.pow(2).sum(dim=-1, keepdim=True) + 1e-2)
            w_out_f = self.w_out.weight.float()
            w_out_n = w_out_f * torch.rsqrt(w_out_f.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
            prior_logits = F.linear(h_prev_n, w_out_n) * 8.0  # [B, T, 256] fp32
            log_probs = F.log_softmax(prior_logits, dim=-1)
            prior_probs = log_probs.exp()
            info = -log_probs.gather(-1, byte_ids.unsqueeze(-1)).squeeze(-1)  # [B, T] fp32 >= 0

            # Strictly causal surprisal jump: Delta_I_0 = 0 (and 0 at every doc_start!), Delta_I_t = max(0, I_t - I_{t-1})
            if T > 1:
                dinfo_tail = F.relu(info[:, 1:] - info[:, :-1])
                delta_info = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.float32), dinfo_tail], dim=1)
            else:
                delta_info = torch.zeros((B, T), device=device, dtype=torch.float32)

            if doc_reset_mask is not None:
                delta_info = torch.where(doc_reset_mask, torch.zeros_like(delta_info), delta_info)

            # 4. LIF membrane potential scan + UTF-8 & maxSegLen=16 & doc-boundary guard in float32
            theta_eff = self.theta
            if forced_spikes is not None:
                spikes = forced_spikes.bool().to(device)
                # Still compute differentiable v_tilde with resets at forced_spikes & doc starts
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

        h_seq = h_seq.to(u.dtype)

        # 5. Zero-padding Ragged cu_seqlens aggregation & Holographic RoPE phase binding
        cu_seqlens, seg_counts = build_cu_seqlens(spikes)
        m_total = cu_seqlens.numel() - 1
        seg_id, tau, seg_lens, seg_end_idx = ragged_local_offsets(cu_seqlens, B * T)

        h_flat = h_seq.reshape(B * T, d_s)
        h_end = h_flat[seg_end_idx]  # [M_total, d_s]

        q_seg = apply_holographic_rope(
            self.w_q(h_end),
            seg_lens,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
        )  # [M_total, d_s]
        k_flat = apply_holographic_rope(
            self.w_k(h_flat),
            tau,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=False,
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
