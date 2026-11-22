"""Module 4: Pragmatic Motor Decoder (PMD).

Implements Engineering Blueprint Section 2.5, Chapter 3 & Theory Doc Part 4:
  1. Broca Competitive Queuing (CQ) Linearizer + Inhibition of Return (IoR) with language-conditioned
     edge precedence operator W_ord [d_r, d_l] (2048 params) and STE winner selection.
  2. Dual-mode cooperative upper concept feed sharing one unified lower motor executor:
       - Mode 1 (Streaming Reconstruction, Stages A/B)
       - Mode 2 (Broca CQ Full-Graph Generation, Stages C/D & Inference)
  3. RoPE inverse holographic phase demodulation d_{n,tau}^{stem} = R_Theta(-tau)(U_demod f_n^{out})
     with stem-affix handoff gate chi_{n,tau}.
  4. 128-byte Differentiable Pushdown Structure Stack Xi_t in R^{16x4}, executed in O(log T)
     parallel associative scan during GPU training and O(1) shift during streaming inference.
  5. Weight-tied causal byte motor micro-core (strictly sharing Embed & W_out with SGSC!)
     executed via O(log T) Blelloch parallel associative scan.
  6. Dual Motor Charge Depletion Gate E_{n,tau} with parallel intra-segment prefix sum &
     zero-alignment L_halt BCE supervision against SGSC spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.config import DCPAConfig
from dcpa.modules.bcg_pc import ConceptGraphState
from dcpa.ops.ragged import ragged_local_offsets, ragged_segment_cumsum
from dcpa.ops.rmsnorm import RMSNorm
from dcpa.ops.rope import apply_holographic_rope
from dcpa.ops.scan import parallel_diag_scan, parallel_stack_scan


def _compute_multi_frame_stack_reset(
    target_bytes: torch.Tensor,
    prev_bytes: torch.Tensor,
    doc_reset_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized Spec 1 PMD 128B stack reset mask:
      reset_t = is_doc_start[t] | (x_{t-1} in {0x04, 0x13, 0x1E}) | (x_{t-3:t} == b"\\x1dE\\n")
    """
    B, T = target_bytes.shape
    device = target_bytes.device
    ctrl_single = (prev_bytes == 0x04) | (prev_bytes == 0x13) | (prev_bytes == 0x1E)
    if T >= 3:
        rb = target_bytes.to(dtype=torch.long)
        z2 = torch.zeros((B, 2), device=device, dtype=torch.long)
        z3 = torch.zeros((B, 3), device=device, dtype=torch.long)
        prev2 = torch.cat([z2, rb[:, :-2]], dim=1)
        prev3 = torch.cat([z3, rb[:, :-3]], dim=1)
        scratch_end_hit = (prev3 == 0x1D) & (prev2 == ord("E")) & (prev_bytes == 0x0A)
        if doc_reset_mask is not None:
            drm = doc_reset_mask.bool()
            drm_p1 = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.bool), drm[:, :-1]], dim=1)
            drm_p2 = torch.cat([torch.zeros((B, 2), device=device, dtype=torch.bool), drm[:, :-2]], dim=1)
            scratch_end_hit = scratch_end_hit & ~(drm | drm_p1 | drm_p2)
    else:
        scratch_end_hit = torch.zeros((B, T), device=device, dtype=torch.bool)
    frame_reset = ctrl_single | scratch_end_hit
    if doc_reset_mask is not None:
        return doc_reset_mask.bool() | frame_reset
    return frame_reset


@dataclass
class BrocaStepOutput:
    z_out: torch.Tensor       # [B, D]
    delta_syn: torch.Tensor   # [B, d_s]
    pi_sel: torch.Tensor      # [B, K] STE one-hot selection
    q_next: torch.Tensor      # [B, K] float32 updated IoR queue
    a_prev_next: torch.Tensor # [B, K, d_r] edge from selected slot to all slots
    done: torch.Tensor        # [B] bool max_i q_next < tau_done


@dataclass
class PMDDecodeOutput:
    byte_logits: torch.Tensor  # [B, T, 256] float32
    p_seg_end: torch.Tensor    # [B, T] float32 boundary termination probability
    charge_e: torch.Tensor     # [B, T] float32 motor charge E_{n,tau}
    stack_xi: torch.Tensor     # [B, T, 16, 4] float32 differentiable pushdown stack
    m_seq: torch.Tensor        # [B, T, d_s] motor micro-core hidden states


class PMD(nn.Module):
    def __init__(
        self,
        cfg: DCPAConfig,
        shared_embed: nn.Embedding,
        shared_w_out: nn.Linear,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d_s = cfg.d_s
        D = cfg.d_concept
        d_r = cfg.d_r
        d_l = cfg.d_l

        # Strictly weight-tied with SGSC (Chapter 3 contract!)
        self.embed = shared_embed
        self.w_out = shared_w_out

        # Orthogonal 32D Language ⊕ 32D Style control vector l_reg in R^64 (Spec 3):
        # l_reg = [RMSNorm_32(lang_embed(lang_id)) || RMSNorm_32(style_embed(style_id))]
        d_l_half = d_l // 2
        self.lang_embed = nn.Embedding(cfg.num_lang_domains, d_l_half)
        self.style_embed = nn.Embedding(cfg.num_style_domains, d_l_half)
        self.norm_lang = RMSNorm(d_l_half, eps=cfg.eps)
        self.norm_style = RMSNorm(d_l_half, eps=cfg.eps)

        # 1. Broca Competitive Queuing (CQ) linearizer + IoR
        self.w_ord = nn.Linear(d_l, d_r, bias=False)  # 64 x 32 = 2048 params
        self.w_coh = nn.Linear(d_l, d_r, bias=False)
        self.w_syn = nn.Linear(d_r + d_l, d_s, bias=True)
        self.u_syn = nn.Linear(D, d_s, bias=False)
        # Mode 1 streaming bridge projecting g_{m-1} into role space feeding the same w_syn/u_syn
        self.w_ctx_role = nn.Linear(D, d_r, bias=False)

        # 2. Holographic RoPE inverse demodulator & stem-affix handoff gate
        self.u_demod = nn.Linear(d_s, d_s, bias=False)
        self.w_chi = nn.Linear(3 * d_s, 1, bias=True)

        # 3. 128B Differentiable Pushdown Stack (16 x 4)
        self.w_act = nn.Linear(d_s + d_l, 3, bias=True)  # [push, pop, hold]
        self.w_tag = nn.Linear(d_s, cfg.stack_tag_dim, bias=True)
        self.w_st = nn.Linear(2 * cfg.stack_tag_dim, d_s, bias=False)  # top-2 stack slice R^8 -> R^{d_s}

        # 4. Causal Byte Motor Micro-Core (mirror of SGSC)
        self.m0 = nn.Parameter(torch.zeros(d_s))
        self.w_alpha_mot = nn.Linear(d_s, d_s, bias=True)
        self.w_m = nn.Linear(d_s, d_s, bias=False)

        # 5. Dual Motor Charge Depletion Gate (float32)
        self.w_e_init = nn.Linear(2 * d_s + d_l, 1, bias=True)
        self.w_drain = nn.Linear(2 * d_s, 1, bias=True)

        self._compiled_motor_scan = None
        self._init_weights()

    def enable_compiled_motor_scan(self, **compile_kwargs) -> None:
        """Compile Hotspot B2: 128B C=256 parallel stack scan + motor diagonal scan + byte logit head."""
        self._compiled_motor_scan = torch.compile(self._motor_scan_impl, **compile_kwargs)

    def _motor_scan_impl(
        self,
        u_prev: torch.Tensor,
        d_drive: torch.Tensor,
        l_reg_bt: torch.Tensor,
        m_start: torch.Tensor,
        xi_init: Optional[torch.Tensor] = None,
        doc_reset_mask: Optional[torch.Tensor] = None,
        stack_reset_mask: Optional[torch.Tensor] = None,
        max_span: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hotspot B2: Static-shape 128B chunked stack scan + O(log T) motor micro-core scan + logits."""
        B, T, _ = u_prev.shape
        # 2. 128B Differentiable Pushdown Stack via C=256 Chunked Parallel Associative Scan (fp32!)
        # Resets at both `is_doc_start` AND the tick immediately following ASCII EOT `0x04`.
        act_probs = F.softmax(self.w_act(torch.cat([u_prev, l_reg_bt], dim=-1)).float(), dim=-1)  # [B, T, 3]
        p_push = act_probs[..., 0]
        p_pop = act_probs[..., 1]
        p_hold = act_probs[..., 2]
        v_tag = torch.tanh(self.w_tag(u_prev).float())  # [B, T, 4]

        eff_stack_reset = doc_reset_mask if stack_reset_mask is None else stack_reset_mask
        stack_xi = parallel_stack_scan(
            p_push,
            p_pop,
            p_hold,
            v_tag,
            xi_init=xi_init,
            reset_mask=eff_stack_reset,
            max_span=max_span,
        )  # [B, T, 16, 4] fp32
        xi_top2 = stack_xi[:, :, :2, :].reshape(B, T, 2 * self.cfg.stack_tag_dim).to(u_prev.dtype)

        # 3. Weight-tied causal byte motor micro-core via O(log T) parallel associative scan
        x_in = u_prev + d_drive + self.w_st(xi_top2)  # [B, T, d_s]
        alpha_mot = torch.sigmoid(self.w_alpha_mot(x_in))
        b_mot = (1.0 - alpha_mot) * F.silu(self.w_m(x_in))
        m_seq = parallel_diag_scan(
            alpha_mot, b_mot, h0=m_start, reset_mask=doc_reset_mask, max_span=max_span
        )  # [B, T, d_s]

        # Shared output classification head (float32 logits)
        byte_logits = F.linear(m_seq.float(), self.w_out.weight.float())  # [B, T, 256] fp32
        return byte_logits, stack_xi, m_seq

    def _init_weights(self) -> None:
        nn.init.normal_(self.lang_embed.weight, std=1.0)
        nn.init.normal_(self.style_embed.weight, std=1.0)
        nn.init.xavier_uniform_(self.w_ord.weight)
        nn.init.xavier_uniform_(self.w_coh.weight)
        nn.init.xavier_uniform_(self.w_syn.weight)
        nn.init.xavier_uniform_(self.u_syn.weight)
        nn.init.xavier_uniform_(self.w_ctx_role.weight)
        nn.init.eye_(self.u_demod.weight)  # initialize demod close to exact RoPE inverse!
        nn.init.constant_(self.w_chi.bias, 1.0)
        # Bias stack initially toward hold (index 2)
        nn.init.zeros_(self.w_act.weight)
        self.w_act.bias.data.copy_(torch.tensor([-1.5, -1.5, 2.0]))
        nn.init.xavier_uniform_(self.w_alpha_mot.weight)
        nn.init.constant_(self.w_alpha_mot.bias, 0.5)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.constant_(self.w_e_init.bias, 2.5)
        nn.init.constant_(self.w_drain.bias, -0.5)

    def get_lang_vec(
        self,
        lang_id: torch.Tensor,
        style_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute orthogonal 64D register vector: [RMSNorm_32(lang) || RMSNorm_32(style)]."""
        lid = lang_id.long().clamp(0, self.cfg.num_lang_domains - 1)
        if style_id is None:
            sid = torch.zeros_like(lid)
        else:
            sid = style_id.long().clamp(0, self.cfg.num_style_domains - 1)
        lang_h = self.norm_lang(self.lang_embed(lid))
        style_h = self.norm_style(self.style_embed(sid))
        return torch.cat([lang_h, style_h], dim=-1)

    def compute_streaming_delta_syn(
        self,
        g_ctx: torch.Tensor,
        z_star: torch.Tensor,
        l_reg: torch.Tensor,
    ) -> torch.Tensor:
        """Mode 1 (Stages A/B): derive delta_syn directly from context summary g_{m-1} and z_m^*."""
        r_role = self.w_ctx_role(g_ctx)
        return F.silu(self.w_syn(torch.cat([r_role, l_reg], dim=-1)) + self.u_syn(z_star))

    def broca_cq_step(
        self,
        graph: ConceptGraphState,
        q_prev: torch.Tensor,
        q_init: torch.Tensor,
        a_prev: torch.Tensor,
        l_reg: torch.Tensor,
        tau_q: Optional[float] = None,
    ) -> BrocaStepOutput:
        """Execute one Broca Competitive Queuing (CQ) + Inhibition of Return (IoR) step.

        Args:
          graph:  ConceptGraphState (S: [B, K, D], A: [B, K, K, d_r], omega: [B, K])
          q_prev: [B, K] float32 remaining salience queue
          q_init: [B, K] float32 initial salience q_0 = sigmoid(omega)
          a_prev: [B, K, d_r] edge vector from previous selected slot to each slot i
          l_reg:  [B, d_l] language/register control vector
        """
        B, K, _ = graph.slots.shape
        tau = self.cfg.tau_q if tau_q is None else tau_q

        # Edge precedence inhibition matrix P^{ord}_{i,j}(l_reg) = tanh((W_ord l_reg)^T A_{i,j})
        ord_vec = self.w_ord(l_reg).float()  # [B, d_r]
        p_ord = torch.tanh(torch.einsum("br,bijr->bij", ord_vec, graph.edges.float()))  # [B, K, K]

        # Lateral inhibition from unexpressed j != i
        eye_mask = 1.0 - torch.eye(K, device=q_prev.device, dtype=torch.float32).unsqueeze(0)
        inh = torch.einsum("bj,bij->bi", q_prev.float(), F.relu(p_ord) * eye_mask)  # [B, K]

        coh_vec = self.w_coh(l_reg).float()  # [B, d_r]
        coh = torch.einsum("br,bir->bi", coh_vec, a_prev.float())  # [B, K]

        u_ready = torch.log(q_prev.float().clamp_min(self.cfg.eps)) - self.cfg.lambda_inh * inh + coh  # [B, K] fp32

        # STE winner selection
        pi_soft = F.softmax(u_ready / max(tau, 1e-4), dim=-1)
        win = u_ready.argmax(dim=-1)
        pi_hard = F.one_hot(win, num_classes=K).float()
        pi_sel = (pi_hard - pi_soft).detach() + pi_soft  # [B, K] fp32

        pi_dt = pi_sel.to(graph.slots.dtype)
        z_out = torch.einsum("bk,bkd->bd", pi_dt, graph.slots)  # [B, D]
        r_role = torch.einsum("bi,bj,bjir->br", pi_dt, q_init.to(graph.edges.dtype), graph.edges)  # [B, d_r]
        delta_syn = F.silu(self.w_syn(torch.cat([r_role, l_reg], dim=-1)) + self.u_syn(z_out))  # [B, d_s]

        # Inhibition of Return (IoR)
        q_next = (q_prev.float() * (1.0 - pi_sel)).clamp(0.0, 1.0)
        a_prev_next = torch.einsum("bk,bkir->bir", pi_dt, graph.edges)  # [B, K, d_r]
        done = q_next.max(dim=-1).values < self.cfg.tau_done

        return BrocaStepOutput(
            z_out=z_out,
            delta_syn=delta_syn,
            pi_sel=pi_sel,
            q_next=q_next,
            a_prev_next=a_prev_next,
            done=done,
        )

    def decode_teacher_forced(
        self,
        target_bytes: torch.Tensor,
        cu_seqlens: torch.Tensor,
        f_out_seg: torch.Tensor,
        delta_syn_seg: torch.Tensor,
        l_reg: torch.Tensor,
        m_init: Optional[torch.Tensor] = None,
        xi_init: Optional[torch.Tensor] = None,
        doc_cu_seqlens: Optional[torch.Tensor] = None,
    ) -> PMDDecodeOutput:
        """Fully parallel O(log T) teacher-forced byte decoding across all segments in a batch!

        Zero Python loops over sequence length T (Commandment #3) and zero segment padding (Commandment #4).
        Supports optional packed varlen `doc_cu_seqlens` (resetting BOS byte, pushdown stack Xi, and
        motor micro-core hidden state m cleanly at every document boundary).
        """
        B, T = target_bytes.shape
        device = target_bytes.device
        d_s = self.cfg.d_s
        N = B * T

        doc_reset_mask: Optional[torch.Tensor] = None
        doc_max_span: Optional[int] = None
        if doc_cu_seqlens is not None:
            if B != 1:
                raise ValueError(f"Packed varlen doc_cu_seqlens requires B=1, got B={B}")
            doc_reset_mask = torch.zeros((1, T), device=device, dtype=torch.bool)
            doc_starts = doc_cu_seqlens[:-1].to(dtype=torch.int64, device=device)
            doc_ends = doc_cu_seqlens[1:].to(dtype=torch.int64, device=device)
            doc_lens = doc_ends - doc_starts
            num_docs = doc_lens.numel()
            if num_docs > 0:
                doc_reset_mask[0, doc_starts] = True
                doc_max_span = int(doc_lens.max().item())
            doc_id_flat = torch.repeat_interleave(
                torch.arange(num_docs, device=device, dtype=torch.int64),
                doc_lens,
            )  # [N]
        else:
            doc_id_flat = torch.arange(B, device=device, dtype=torch.int64).unsqueeze(1).expand(B, T).reshape(N)

        # Previous emitted byte y_{t-1} (BOS byte = 0x00 at t=0 and at every doc_start!)
        bos = torch.zeros((B, 1), device=device, dtype=torch.long)
        prev_bytes = torch.cat([bos, target_bytes[:, :-1].long()], dim=1)  # [B, T]
        if doc_reset_mask is not None:
            prev_bytes = torch.where(doc_reset_mask, torch.zeros_like(prev_bytes), prev_bytes)
        u_prev = self.embed(prev_bytes)  # [B, T, d_s]

        # Ragged local step tau in {1..l_m} & segment ID for every byte in [0..N-1]
        seg_id, tau, _, _ = ragged_local_offsets(cu_seqlens, N)

        # 1. Holographic RoPE inverse demodulation per byte step -tau
        demod_seg = self.u_demod(f_out_seg)  # [M_total, d_s]
        d_stem_flat = apply_holographic_rope(
            demod_seg[seg_id],
            tau,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=True,
        )  # [N, d_s]
        delta_syn_flat = delta_syn_seg[seg_id]  # [N, d_s]
        u_prev_flat = u_prev.reshape(N, d_s)

        # Stem-affix smooth handoff gate chi_{n, tau}
        chi = torch.sigmoid(
            self.w_chi(torch.cat([d_stem_flat, u_prev_flat, delta_syn_flat], dim=-1))
        )  # [N, 1]
        d_drive_flat = chi * d_stem_flat + (1.0 - chi) * delta_syn_flat  # [N, d_s]
        d_drive = d_drive_flat.view(B, T, d_s)

        # 2 & 3. Hotspot B2: 128B Chunked Parallel Stack Scan + Causal Byte Motor Micro-Core + Logits
        # Spec 1 multi-frame isolation: reset 128B pushdown stack Xi_t at doc starts,
        # after 0x04 / 0x13 / 0x1E, and immediately after `b"\x1dE\n"`.
        full_stack_reset = _compute_multi_frame_stack_reset(target_bytes, prev_bytes, doc_reset_mask)
        has_any_reset = bool(full_stack_reset.any().item()) if not torch.compiler.is_compiling() else True
        stack_reset_mask = full_stack_reset if has_any_reset else doc_reset_mask

        l_reg_bt = l_reg[doc_id_flat].view(B, T, self.cfg.d_l)
        m_start = self.m0 if m_init is None else m_init
        if self._compiled_motor_scan is not None and T > 0 and xi_init is None:
            pad = (256 - (T % 256)) % 256
            if pad > 0:
                u_in = torch.cat([u_prev, torch.zeros((B, pad, d_s), device=device, dtype=u_prev.dtype)], dim=1)
                d_in = torch.cat([d_drive, torch.zeros((B, pad, d_s), device=device, dtype=d_drive.dtype)], dim=1)
                l_in = torch.cat([l_reg_bt, torch.zeros((B, pad, self.cfg.d_l), device=device, dtype=l_reg_bt.dtype)], dim=1)
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
                if stack_reset_mask is not None:
                    srm_in = torch.cat([stack_reset_mask, torch.ones((B, pad), device=device, dtype=torch.bool)], dim=1)
                else:
                    srm_in = rm_in
            else:
                u_in = u_prev
                d_in = d_drive
                l_in = l_reg_bt
                rm_in = (
                    doc_reset_mask
                    if doc_reset_mask is not None
                    else torch.zeros((B, T), device=device, dtype=torch.bool)
                )
                srm_in = stack_reset_mask if stack_reset_mask is not None else rm_in
            logits_p, xi_p, m_p = self._compiled_motor_scan(u_in, d_in, l_in, m_start, None, rm_in, srm_in, None)
            byte_logits = logits_p[:, :T, :]
            stack_xi = xi_p[:, :T, :, :]
            m_seq = m_p[:, :T, :]
        else:
            byte_logits, stack_xi, m_seq = self._motor_scan_impl(
                u_prev, d_drive, l_reg_bt, m_start, xi_init, doc_reset_mask, stack_reset_mask, doc_max_span
            )

        # 4. Dual Motor Charge Depletion Gate E_{n,tau} via segmented associative scan (zero float32 bleed!)
        seg_start_idx = cu_seqlens[:-1].to(torch.int64)
        seg_doc_id = doc_id_flat[seg_start_idx]  # [M_total]
        l_reg_seg = l_reg[seg_doc_id]  # [M_total, d_l]

        e_in = torch.cat([f_out_seg, delta_syn_seg, l_reg_seg], dim=-1).float()
        e_init_seg = F.softplus(
            (e_in * self.w_e_init.weight.float()).sum(dim=-1) + self.w_e_init.bias.float()
        )  # [M_total] fp32 > 0 (row-independent dot product avoids BLAS sgemv row-offset drift)
        m_flat = m_seq.reshape(N, d_s).float()
        m_norm = m_flat * torch.rsqrt(m_flat.pow(2).mean(dim=-1, keepdim=True) + self.cfg.eps)
        drain_in = torch.cat([m_norm, d_drive_flat.float()], dim=-1)
        drain_flat = F.softplus(
            (drain_in * self.w_drain.weight.float()).sum(dim=-1) + self.w_drain.bias.float()
        )  # [N] fp32 > 0
        cum_drain = ragged_segment_cumsum(drain_flat, cu_seqlens, seg_id)  # [N] fp32
        charge_e = (e_init_seg[seg_id] - cum_drain).view(B, T)  # [B, T] fp32
        p_seg_end = torch.sigmoid(-charge_e / max(self.cfg.tau_e, 1e-4))  # [B, T] fp32

        return PMDDecodeOutput(
            byte_logits=byte_logits,
            p_seg_end=p_seg_end,
            charge_e=charge_e,
            stack_xi=stack_xi,
            m_seq=m_seq,
        )

    def step_byte_streaming(
        self,
        prev_byte: torch.Tensor,
        f_out: torch.Tensor,
        delta_syn: torch.Tensor,
        l_reg: torch.Tensor,
        tau_step: int,
        m_prev: torch.Tensor,
        xi_prev: torch.Tensor,
        charge_prev: torch.Tensor,
        extra_stack_reset: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-byte streaming motor step matching dcpa-core-rs 1-to-1!

        Args:
          prev_byte: [B] long previous byte
          f_out:     [B, d_s]
          delta_syn: [B, d_s]
          l_reg:     [B, d_l]
          tau_step:  1-based local byte offset within concept n
          m_prev:    [B, d_s]
          xi_prev:   [B, 16, 4] float32
          charge_prev: [B] float32 E_{n, tau-1}
          extra_stack_reset: optional [B] bool (e.g. set True immediately after `b"\\x1dE\\n"`)

        Returns:
          logits [B, 256] fp32, m_new [B, d_s], xi_new [B, 16, 4] fp32, charge_new [B] fp32, p_end [B] fp32
        """
        B = prev_byte.shape[0]
        device = prev_byte.device
        u_prev = self.embed(prev_byte.long())  # [B, d_s]

        tau_t = torch.full((B,), float(tau_step), device=device, dtype=torch.float32)
        d_stem = apply_holographic_rope(
            self.u_demod(f_out),
            tau_t,
            period=self.cfg.rope_period,
            num_harmonics=self.cfg.rope_num_harmonics,
            inverse=True,
        )
        chi = torch.sigmoid(self.w_chi(torch.cat([d_stem, u_prev, delta_syn], dim=-1)))
        d_drive = chi * d_stem + (1.0 - chi) * delta_syn

        # O(1) stack shift in float32
        act = F.softmax(self.w_act(torch.cat([u_prev, l_reg], dim=-1)).float(), dim=-1)  # [B, 3]
        p_push = act[:, 0:1].unsqueeze(-1)
        p_pop = act[:, 1:2].unsqueeze(-1)
        p_hold = act[:, 2:3].unsqueeze(-1)
        v_tag = torch.tanh(self.w_tag(u_prev).float()).unsqueeze(1)  # [B, 1, 4]

        xi_f = xi_prev.float()
        # Spec 1 multi-frame stack reset: 0x04 (EOT), 0x13 (tool suspend), 0x1E (user turn), or `\x1dE\n`
        pb_l = prev_byte.long()
        ctrl_reset = (pb_l == 0x04) | (pb_l == 0x13) | (pb_l == 0x1E)
        if extra_stack_reset is not None:
            ctrl_reset = ctrl_reset | extra_stack_reset.bool()
        xi_f = torch.where(ctrl_reset.view(B, 1, 1), torch.zeros_like(xi_f), xi_f)
        pushed = torch.cat([v_tag, xi_f[:, :-1, :]], dim=1)
        zero_bot = torch.zeros((B, 1, self.cfg.stack_tag_dim), device=device, dtype=torch.float32)
        popped = torch.cat([xi_f[:, 1:, :], zero_bot], dim=1)
        xi_new = p_hold * xi_f + p_push * pushed + p_pop * popped  # [B, 16, 4] fp32

        xi_top2 = xi_new[:, :2, :].reshape(B, 2 * self.cfg.stack_tag_dim).to(u_prev.dtype)
        x_in = u_prev + d_drive + self.w_st(xi_top2)
        alpha_mot = torch.sigmoid(self.w_alpha_mot(x_in))
        m_new = alpha_mot * m_prev + (1.0 - alpha_mot) * F.silu(self.w_m(x_in))

        logits = F.linear(m_new.float(), self.w_out.weight.float())
        m_f = m_new.float()
        m_norm = m_f * torch.rsqrt(m_f.pow(2).mean(dim=-1, keepdim=True) + self.cfg.eps)
        drain_in = torch.cat([m_norm, d_drive.float()], dim=-1)
        drain = F.softplus(
            (drain_in * self.w_drain.weight.float()).sum(dim=-1) + self.w_drain.bias.float()
        )
        charge_new = charge_prev.float() - drain
        p_end = torch.sigmoid(-charge_new / max(self.cfg.tau_e, 1e-4))
        return logits, m_new, xi_new, charge_new, p_end

    # Explicit O(1) autoregressive single-byte shift alias (Performance Audit Task 1)
    step_single_byte = step_byte_streaming

    def init_segment_charge(
        self,
        f_out: torch.Tensor,
        delta_syn: torch.Tensor,
        l_reg: torch.Tensor,
    ) -> torch.Tensor:
        e_in = torch.cat([f_out, delta_syn, l_reg], dim=-1).float()
        return F.softplus(
            (e_in * self.w_e_init.weight.float()).sum(dim=-1) + self.w_e_init.bias.float()
        )
