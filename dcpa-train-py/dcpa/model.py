"""Unified DCPA full-stack model wiring SGSC -> Hippocampus -> BCG-PC -> PMD.

Supports all 4 developmental curriculum stages:
  - Stage A: SGSC + PMD lower motor decoder (bypass BCG-PC & Hippocampus, reconstruct segments via c_slow/f_stem).
  - Stage B: Enable BCG-PC concept graph bottleneck, L_PC, variance hinge, pondering tax.
  - Stage C: Activate Broca CQ full-graph linearization (50% streaming + 50% CQ mode).
  - Stage D: Freeze sensory byte embed & LIF threshold, enable Hippocampus 3-factor Gated Delta
             dual memory for episodic fast-mapping meta-training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.config import DCPAConfig
from dcpa.modules.bcg_pc import BCGPC, ConceptGraphState
from dcpa.modules.hippocampus import (
    ColumnMajorFastWeights,
    EligibilityState,
    HippocampusDualMemory,
    HippocampusStateType,
)
from dcpa.modules.pmd import PMD, PMDDecodeOutput
from dcpa.modules.sgsc import SGSC, SGSCOutput
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.scan import parallel_diag_scan


@dataclass
class DCPACarryState:
    """Strictly O(1) constant-size runtime working memory carried across streaming chunks.

    Zero dynamic Python lists and zero sequence-length-dependent caches:
      - sgsc_h:      [B, d_s]
      - graph_state: S [B, K, D], A [B, K, K, d_r], omega [B, K]  (K == 8)
      - pmd_m:       [B, d_s]
      - pmd_xi:      [B, 16, 4]  (128B pushdown stack)
      - fast_state:  ColumnMajorFastWeights (fixed [d_h, D], [d_h, d_s], [d_h] arrays)
      - elig_state:  EligibilityState ([B, K, s], [B, K, d_s], [B, K])
    """

    sgsc_h: torch.Tensor
    graph_state: ConceptGraphState
    pmd_m: torch.Tensor
    pmd_xi: torch.Tensor
    fast_state: ColumnMajorFastWeights
    elig_state: EligibilityState


@dataclass
class DCPAForwardOutput:
    loss_total: torch.Tensor
    loss_lang: torch.Tensor
    loss_sgsc_prior: torch.Tensor
    loss_pc: torch.Tensor
    loss_var: torch.Tensor
    loss_rate: torch.Tensor
    loss_ponder: torch.Tensor
    loss_halt: torch.Tensor
    loss_merge_kl: torch.Tensor
    loss_ach_hint: torch.Tensor
    sgsc: SGSCOutput
    pmd: PMDDecodeOutput
    graph_state: Optional[ConceptGraphState]
    fast_state: Optional[HippocampusStateType]
    elig_state: Optional[EligibilityState]
    metrics: dict[str, float]
    s_trace: Optional[torch.Tensor] = None           # [M_total, K, D]
    a_trace: Optional[torch.Tensor] = None           # [M_total, K, K, d_r]
    merge_alpha_trace: Optional[torch.Tensor] = None # [M_total, 72] hard one-hot STE alpha
    win_pair_trace: Optional[torch.Tensor] = None    # [M_total, 3] int64 (i*, j*, u*)
    ach_trace: Optional[torch.Tensor] = None         # [M_total] float32 ACh_m per segment
    beta_max_trace: Optional[torch.Tensor] = None    # [M_total] float32 max_u beta_{m,u} per segment


class DCPAModel(nn.Module):
    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sgsc = SGSC(cfg)
        self.bcg_pc = BCGPC(cfg)
        self.hippocampus = HippocampusDualMemory(cfg)
        # Strict weight tying between SGSC and PMD for Embed [256, d_s] and W_out [256, d_s]
        self.pmd = PMD(cfg, shared_embed=self.sgsc.embed, shared_w_out=self.sgsc.w_out)

    def count_parameters(self) -> dict[str, int]:
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        sgsc_n = _count(self.sgsc)
        bcg_n = _count(self.bcg_pc)
        hip_n = _count(self.hippocampus)
        # Exclude tied embed & w_out from PMD unique count
        tied_ids = {id(self.sgsc.embed.weight), id(self.sgsc.w_out.weight)}
        pmd_unique = sum(p.numel() for p in self.pmd.parameters() if id(p) not in tied_ids)
        total = sum(p.numel() for p in self.parameters())
        return {
            "sgsc": sgsc_n,
            "bcg_pc": bcg_n,
            "hippocampus_slow": hip_n,
            "pmd_unique": pmd_unique,
            "total": total,
        }

    def parameter_groups(self) -> list[dict]:
        """Grouped parameter sets per Engineering Doc Section 5.2:
        1. Micro-cores (SGSC + PMD) -> lr_micro, weight_decay
        2. Concept core (BCG-PC + slow Hippocampus projections) -> lr_concept, weight_decay
        3. LIF threshold & scalar gating parameters -> lr_scalar_gate (1e-4), weight_decay=0.0
        """
        scalar_params = [self.sgsc.theta_raw, self.bcg_pc.kappa_omega] + self.hippocampus.scalar_gate_parameters()
        scalar_ids = {id(p) for p in scalar_params}

        micro_params: list[nn.Parameter] = []
        seen: set[int] = set(scalar_ids)
        for mod in (self.sgsc, self.pmd):
            for p in mod.parameters():
                if id(p) not in seen:
                    micro_params.append(p)
                    seen.add(id(p))

        concept_params: list[nn.Parameter] = []
        for mod in (self.bcg_pc, self.hippocampus):
            for p in mod.parameters():
                if id(p) not in seen:
                    concept_params.append(p)
                    seen.add(id(p))

        return [
            {"name": "micro_cores", "params": micro_params, "lr": self.cfg.lr_micro, "weight_decay": self.cfg.weight_decay},
            {"name": "concept_core", "params": concept_params, "lr": self.cfg.lr_concept, "weight_decay": self.cfg.weight_decay},
            {"name": "scalar_gates", "params": scalar_params, "lr": self.cfg.lr_scalar_gate, "weight_decay": 0.0},
        ]

    def set_stage_freeze(self, stage: str) -> None:
        """Apply developmental stage parameter freezing (Stage D freezes sensory embed & LIF threshold)."""
        for p in self.parameters():
            p.requires_grad_(True)
        if stage.upper() == "D":
            self.sgsc.embed.weight.requires_grad_(False)
            self.sgsc.theta_raw.requires_grad_(False)

    def init_carry_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> DCPACarryState:
        """Initialize strictly O(1) constant-size streaming working memory state."""
        if batch_size != 1:
            raise ValueError("Streaming O(1) carry state uses ColumnMajorFastWeights with batch_size=1.")
        dev = device if device is not None else self.sgsc.h0.device
        sgsc_h = self.sgsc.h0.detach().to(device=dev, dtype=dtype).unsqueeze(0).clone()
        graph_state = self.bcg_pc.init_state(1, device=dev, dtype=dtype)
        pmd_m = self.pmd.m0.detach().to(device=dev, dtype=dtype).unsqueeze(0).clone()
        pmd_xi = torch.zeros((1, self.cfg.stack_depth, self.cfg.stack_tag_dim), device=dev, dtype=torch.float32)
        fast_state = self.hippocampus.init_fast_state(1, mode="column_major", storage_dtype=torch.float32, device=dev)
        assert isinstance(fast_state, ColumnMajorFastWeights)
        elig_state = EligibilityState.zeros(
            1, self.cfg.k_slots, self.cfg.s_active, self.cfg.d_s, device=dev, dtype=dtype
        )
        return DCPACarryState(
            sgsc_h=sgsc_h,
            graph_state=graph_state,
            pmd_m=pmd_m,
            pmd_xi=pmd_xi,
            fast_state=fast_state,
            elig_state=elig_state,
        )

    def forward_streaming_chunk(
        self,
        raw_bytes: torch.Tensor,
        lang_ids: torch.Tensor,
        carry_state: Optional[DCPACarryState] = None,
        stage: str = "D",
        tau_s: float = 0.35,
        tau_m: float = 0.25,
    ) -> tuple[DCPAForwardOutput, DCPACarryState]:
        """Run one chunk through all 4 modules while carrying strictly O(1) constant state."""
        B = raw_bytes.shape[0]
        if carry_state is None:
            carry_state = self.init_carry_state(batch_size=B, device=raw_bytes.device, dtype=torch.float32)

        out = self.forward(
            raw_bytes=raw_bytes,
            lang_ids=lang_ids,
            stage=stage,
            tau_s=tau_s,
            tau_m=tau_m,
            use_broca_cq=False,
            fast_state=carry_state.fast_state,
            graph_state=carry_state.graph_state,
            elig_state=carry_state.elig_state,
            h_init=carry_state.sgsc_h,
            m_init=carry_state.pmd_m,
            xi_init=carry_state.pmd_xi,
        )
        assert out.graph_state is not None
        assert isinstance(out.fast_state, ColumnMajorFastWeights)
        assert out.elig_state is not None
        next_carry = DCPACarryState(
            sgsc_h=out.sgsc.h_seq[:, -1, :].detach().clone(),
            graph_state=ConceptGraphState(
                slots=out.graph_state.slots.detach().clone(),
                edges=out.graph_state.edges.detach().clone(),
                salience=out.graph_state.salience.detach().clone(),
            ),
            pmd_m=out.pmd.m_seq[:, -1, :].detach().clone(),
            pmd_xi=out.pmd.stack_xi[:, -1, :, :].detach().clone(),
            fast_state=out.fast_state,
            elig_state=EligibilityState(
                e_idx=out.elig_state.e_idx.detach().clone(),
                e_val=out.elig_state.e_val.detach().clone(),
                f_tilde=out.elig_state.f_tilde.detach().clone(),
                eta=out.elig_state.eta.detach().clone(),
            ),
        )
        return out, next_carry

    def forward(
        self,
        raw_bytes: torch.Tensor,
        lang_ids: torch.Tensor,
        stage: str = "B",
        tau_s: float = 0.5,
        tau_m: float = 0.8,
        use_broca_cq: bool = False,
        loss_weights: Optional[torch.Tensor] = None,
        ach_hint_targets: Optional[torch.Tensor] = None,
        ach_hint_weight: float = 0.0,
        lambda_rate_override: Optional[float] = None,
        dual_rate_lagrange: float = 0.0,
        lambda_var_override: Optional[float] = None,
        lambda_pc_override: Optional[float] = None,
        lambda_act_override: Optional[float] = None,
        lambda_halt_override: Optional[float] = None,
        target_spike_rate_override: Optional[float] = None,
        merge_kl_weight: float = 0.0,
        merge_uniform_mix: float = 0.0,
        familiar_ach_reg_weight: float = 0.0,
        familiar_doc_mask: Optional[torch.Tensor] = None,
        write_threshold_override: Optional[float] = None,
        fast_state: Optional[HippocampusStateType] = None,
        graph_state: Optional[ConceptGraphState] = None,
        elig_state: Optional[EligibilityState] = None,
        doc_cu_seqlens: Optional[torch.Tensor] = None,
        h_init: Optional[torch.Tensor] = None,
        m_init: Optional[torch.Tensor] = None,
        xi_init: Optional[torch.Tensor] = None,
    ) -> DCPAForwardOutput:
        """Execute full-stack forward pass (rectangular [B, T] or packed varlen doc_cu_seqlens)."""
        stage = stage.upper()
        B, T = raw_bytes.shape
        device = raw_bytes.device
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 1. Module 1: SGSC sensory causal scan + LIF spiking + Ragged RoPE phase-bound stems
        sgsc_out = self.sgsc(
            raw_bytes, tau_s=tau_s, h_init=h_init, doc_cu_seqlens=doc_cu_seqlens
        )
        m_total = sgsc_out.f_stem.shape[0]
        l_reg = self.pmd.get_lang_vec(lang_ids)  # [V, d_l] where V=D_docs if packed else B

        # Build per-document segment counts & offsets (V = num_docs in packed mode, else B)
        if doc_cu_seqlens is not None:
            if B != 1:
                raise ValueError(f"Packed varlen doc_cu_seqlens requires B=1, got B={B}")
            doc_starts = doc_cu_seqlens[:-1].to(dtype=torch.int64, device=device)
            doc_ends = doc_cu_seqlens[1:].to(dtype=torch.int64, device=device)
            V = int(doc_starts.numel())
            if l_reg.shape[0] != V:
                raise ValueError(f"lang_ids len ({l_reg.shape[0]}) must match num_docs ({V})")
            # Count spikes strictly inside each document span [doc_start, doc_end)
            spike_cs = torch.cumsum(sgsc_out.spikes[0].to(torch.int64), dim=0)
            end_counts = spike_cs[doc_ends - 1]
            start_counts = torch.zeros_like(end_counts)
            if V > 1:
                start_counts[1:] = spike_cs[doc_starts[1:] - 1]
            seg_counts = (end_counts - start_counts).to(torch.int32)  # [V]
        else:
            V = B
            seg_counts = sgsc_out.seg_counts  # [V]

        seg_offsets = torch.zeros_like(seg_counts)
        if V > 1:
            seg_offsets[1:] = torch.cumsum(seg_counts[:-1], dim=0)
        seg_doc_ids = torch.repeat_interleave(
            torch.arange(V, device=device, dtype=torch.int64),
            seg_counts.to(torch.int64),
        )  # [M_total]
        l_reg_seg = l_reg[seg_doc_ids]  # [M_total, d_l]

        # Rate anchor loss L_rate using STE spike rate + Huber (SmoothL1, beta=0.02):
        # bounds early |theta_raw.grad| so 1 scalar never hijacks global clip_grad_norm_(1.0)
        # while applying strong smooth quadratic restoring force tightly around target_spike_rate.
        with fp32_island(device):
            lam_rate = self.cfg.lambda_rate if lambda_rate_override is None else lambda_rate_override
            tgt_rate_val = (
                self.cfg.target_spike_rate
                if target_spike_rate_override is None
                else float(target_spike_rate_override)
            )
            mean_s_rate = sgsc_out.s_ste.float().mean()
            target_rate = torch.tensor(tgt_rate_val, device=device, dtype=torch.float32)
            rate_err = mean_s_rate - target_rate
            loss_rate = (0.5 * lam_rate) * F.smooth_l1_loss(mean_s_rate, target_rate, beta=0.02) + dual_rate_lagrange * rate_err

            # Sensory causal byte predictor loss (trains W_out @ h_{t-1} to produce accurate surprisal I_t)
            loss_sgsc_prior = F.cross_entropy(
                sgsc_out.prior_logits.float().reshape(-1, 256),
                raw_bytes.reshape(-1).long(),
            )

            # Precision Island #4 (Red Line #1): dimension-wise std hinge sqrt(Var(c_{m,d}) + 1e-4) in fp32
            lam_var = self.cfg.lambda_var if lambda_var_override is None else lambda_var_override
            if m_total > 1:
                c_std = torch.sqrt(sgsc_out.c_slow.float().var(dim=0, unbiased=False) + 1e-4)
                loss_var = lam_var * F.relu(1.0 - c_std).mean()
                c_slow_std_mean = float(c_std.mean().detach().item())
                c_slow_std_min = float(c_std.min().detach().item())
            else:
                loss_var = zero
                c_slow_std_mean = 1.0
                c_slow_std_min = 1.0

        loss_pc = zero
        loss_ponder = zero
        loss_merge_kl = zero
        loss_ach_hint = zero
        loss_ach_reg = zero
        avg_ponder = 0.0
        ponder_skip_rate = 1.0
        merge_entropy_val = 0.0
        mean_ach_val = 0.0
        mean_beta_val = 0.0
        s_trace: Optional[torch.Tensor] = None
        a_trace: Optional[torch.Tensor] = None
        merge_alpha_trace: Optional[torch.Tensor] = None
        win_pair_trace: Optional[torch.Tensor] = None
        ach_trace: Optional[torch.Tensor] = None
        beta_max_trace: Optional[torch.Tensor] = None

        if stage == "A":
            # Stage 1 (A): Bypass BCG-PC and Hippocampus; z_m^* = c_m^slow,
            # g_{m-1} = causal per-document EMA(c_{<m}^slow) feeding PMD directly!
            if m_total > 0:
                seg_doc_start = torch.zeros((m_total,), device=device, dtype=torch.bool)
                valid_docs = seg_counts > 0
                if valid_docs.any():
                    seg_doc_start[seg_offsets[valid_docs].to(torch.int64)] = True
                c_prev = torch.zeros_like(sgsc_out.c_slow)
                if m_total > 1:
                    c_prev[1:] = sgsc_out.c_slow[:-1]
                c_prev = torch.where(seg_doc_start.unsqueeze(-1), torch.zeros_like(c_prev), c_prev)
                with fp32_island(device):
                    ema_decay = 0.8
                    a_ema = torch.full((1, m_total, sgsc_out.c_slow.shape[-1]), ema_decay, device=device, dtype=torch.float32)
                    b_ema = (1.0 - ema_decay) * c_prev.float().unsqueeze(0)
                    max_seg_span = int(seg_counts.max().item()) if seg_counts.numel() > 0 else m_total
                    g_ema_prev = parallel_diag_scan(
                        a_ema,
                        b_ema,
                        reset_mask=seg_doc_start.unsqueeze(0),
                        max_span=max(max_seg_span, 1),
                    ).squeeze(0).to(sgsc_out.c_slow.dtype)
            else:
                g_ema_prev = sgsc_out.c_slow

            delta_syn_seg = self.pmd.compute_streaming_delta_syn(
                g_ctx=g_ema_prev,
                z_star=sgsc_out.c_slow,
                l_reg=l_reg_seg,
            )
            f_out_seg = sgsc_out.f_stem
            pmd_out = self.pmd.decode_teacher_forced(
                target_bytes=raw_bytes,
                cu_seqlens=sgsc_out.cu_seqlens,
                f_out_seg=f_out_seg,
                delta_syn_seg=delta_syn_seg,
                l_reg=l_reg,
                m_init=m_init,
                xi_init=xi_init,
                doc_cu_seqlens=doc_cu_seqlens,
            )
            out_graph = None
            out_fast = None
            out_elig = None
        else:
            # Stages B, C, D: Run BCG-PC concept graph (+ Hippocampus in Stage D) with virtual batch size V
            hip_enabled = stage == "D"
            eff_write_thresh: Optional[float] = write_threshold_override
            if eff_write_thresh is None and hip_enabled and self.training:
                # Stage 4 meta-training: allow all positive crystallization writes so gating scalars
                # (alpha_1, theta_i, alpha_q, theta_q, alpha_out) always receive clean Turn-2 gradients.
                eff_write_thresh = 0.0
            g_state = self.bcg_pc.init_state(V, device=device, dtype=sgsc_out.c_slow.dtype) if graph_state is None else graph_state
            f_state = (
                self.hippocampus.init_fast_state(V, mode="compact", device=device)
                if (hip_enabled and fast_state is None)
                else fast_state
            )
            e_state = (
                EligibilityState.zeros(V, self.cfg.k_slots, self.cfg.s_active, self.cfg.d_s, device, sgsc_out.f_stem.dtype)
                if (hip_enabled and elig_state is None)
                else elig_state
            )

            m_max = int(seg_counts.max().item())
            D = self.cfg.d_concept
            d_s = self.cfg.d_s
            K = self.cfg.k_slots
            d_r = self.cfg.d_r
            num_pairs = self.cfg.num_merge_pairs

            f_out_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
            delta_syn_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)

            s_trace_flat = torch.zeros((m_total, K, D), device=device, dtype=sgsc_out.c_slow.dtype)
            a_trace_flat = torch.zeros((m_total, K, K, d_r), device=device, dtype=sgsc_out.c_slow.dtype)
            alpha_trace_flat = torch.zeros((m_total, num_pairs), device=device, dtype=torch.float32)
            win_pair_flat = torch.zeros((m_total, 3), device=device, dtype=torch.int64)
            ach_trace_flat = torch.zeros((m_total,), device=device, dtype=torch.float32)
            beta_max_flat = torch.zeros((m_total,), device=device, dtype=torch.float32)

            pc_nll_sum = zero
            ponder_sum = zero
            ponder_skip_sum = zero
            merge_probs_sum = torch.zeros((num_pairs,), device=device, dtype=torch.float32)
            ach_list: list[torch.Tensor] = []
            familiar_ach_list: list[torch.Tensor] = []
            beta_sum = 0.0
            beta_count = 0
            total_valid_steps = 0

            if (doc_cu_seqlens is not None) and (not self.training) and (graph_state is None) and (V <= 8):
                # Strict batch-invariant per-document isolated stepping in packed varlen eval mode:
                # prevents Doc A's MoE routing or pondering halt count from changing BLAS GEMM/GEMV
                # batch size M seen by Doc B (guarantees bit-exact 0.0 cross-document isolation).
                last_g_slots = []
                last_g_edges = []
                last_g_sal = []
                for d_idx in range(V):
                    m_doc = int(seg_counts[d_idx].item())
                    d_off = int(seg_offsets[d_idx].item())
                    gd = self.bcg_pc.init_state(1, device=device, dtype=sgsc_out.c_slow.dtype)
                    fd = (
                        self.hippocampus.init_fast_state(1, mode="compact", device=device)
                        if hip_enabled
                        else None
                    )
                    ed = (
                        EligibilityState.zeros(
                            1, self.cfg.k_slots, self.cfg.s_active, self.cfg.d_s, device, sgsc_out.f_stem.dtype
                        )
                        if hip_enabled
                        else None
                    )
                    l_reg_d = l_reg[d_idx : d_idx + 1]
                    for m_step in range(m_doc):
                        f_idx = d_off + m_step
                        f_step = sgsc_out.f_stem[f_idx : f_idx + 1]
                        c_step = sgsc_out.c_slow[f_idx : f_idx + 1]
                        sn_step = sgsc_out.sem_norm[f_idx : f_idx + 1]
                        mi_step = sgsc_out.mean_info[f_idx : f_idx + 1]

                        hip_read = self.hippocampus.read_comprehension(
                            f_stem=f_step,
                            c_slow=c_step,
                            sem_norm=sn_step,
                            mean_info=mi_step,
                            fast_state=fd,
                            enabled=hip_enabled,
                        )
                        bcg_out = self.bcg_pc.step(
                            c_fused=hip_read.c_fused,
                            prec_bu=hip_read.prec_bu,
                            ach=hip_read.ach_signal,
                            state=gd,
                            tau_m=tau_m,
                            use_gumbel=False,
                            merge_uniform_mix=merge_uniform_mix,
                        )
                        gd = bcg_out.state
                        s_trace_flat[f_idx] = gd.slots[0].detach().to(s_trace_flat.dtype)
                        a_trace_flat[f_idx] = gd.edges[0].detach().to(a_trace_flat.dtype)
                        alpha_trace_flat[f_idx] = bcg_out.merge_alpha[0].detach().float()
                        wi = bcg_out.win_i[0].detach()
                        wj = bcg_out.win_j[0].detach()
                        win_pair_flat[f_idx, 0] = wi
                        win_pair_flat[f_idx, 1] = wj
                        win_pair_flat[f_idx, 2] = torch.minimum(wi, wj)

                        if hip_enabled and fd is not None and ed is not None:
                            ed = self.hippocampus.inherit_eligibility_on_merge(
                                elig=ed,
                                cand_idx=hip_read.psi_idx,
                                cand_val=hip_read.psi_val,
                                cand_f=f_step,
                                cand_ach=hip_read.ach_signal,
                                win_i=bcg_out.win_i,
                                win_j=bcg_out.win_j,
                            )
                            ed, _, beta_mat = self.hippocampus.three_factor_write(
                                s_bar_post=bcg_out.s_bar_post,
                                edges_post=gd.edges,
                                c0=self.bcg_pc.c0,
                                elig=ed,
                                fast_state=fd,
                                write_threshold=eff_write_thresh,
                            )
                            beta_sum += float(beta_mat.detach().mean().item())
                            beta_count += 1
                            ach_list.append(hip_read.ach_signal)
                            ach_trace_flat[f_idx] = hip_read.ach_signal[0].detach().float()
                            beta_max_flat[f_idx] = beta_mat[0].detach().float().max()
                            if familiar_doc_mask is not None and bool(familiar_doc_mask[d_idx].item()):
                                familiar_ach_list.append(hip_read.ach_signal)

                        # Precision Island #4 (Red Line #1): L_PC free-energy loss strictly in float32
                        with fp32_island(device):
                            c_target_sg = hip_read.c_fused.detach().float()
                            c_pred = bcg_out.c_hat.float()
                            pi_act = bcg_out.pi_m.float()
                            err_sq = (c_target_sg - c_pred).pow(2)
                            nll_step = 0.5 * ((pi_act * err_sq).mean(dim=-1) - torch.log(pi_act.clamp_min(1e-6)).mean(dim=-1))
                            pc_nll_sum = pc_nll_sum + nll_step.sum()
                            ponder_sum = ponder_sum + bcg_out.ponder_steps.float().sum()
                            ponder_skip_sum = ponder_skip_sum + (bcg_out.ponder_steps.detach().float() < 0.5).float().sum()
                            merge_probs_sum = merge_probs_sum + bcg_out.merge_probs.float().sum(dim=0)
                        total_valid_steps += 1

                        f_out_step, _ = self.hippocampus.read_expression(
                            z_out=bcg_out.z_star,
                            l_reg=l_reg_d,
                            fast_state=fd if hip_enabled else None,
                            enabled=hip_enabled,
                        )
                        delta_syn_step = self.pmd.compute_streaming_delta_syn(
                            g_ctx=bcg_out.g_ctx,
                            z_star=bcg_out.z_star,
                            l_reg=l_reg_d,
                        )
                        f_out_flat[f_idx] = f_out_step[0].to(f_out_flat.dtype)
                        delta_syn_flat[f_idx] = delta_syn_step[0].to(delta_syn_flat.dtype)

                    last_g_slots.append(gd.slots)
                    last_g_edges.append(gd.edges)
                    last_g_sal.append(gd.salience)

                g_state = ConceptGraphState(
                    slots=torch.cat(last_g_slots, dim=0),
                    edges=torch.cat(last_g_edges, dim=0),
                    salience=torch.cat(last_g_sal, dim=0),
                )
            else:
                # Zero-index_put gather + single end-of-loop cat/permute (eliminates hundreds of MPS index_put_ bwd stalls!)
                seg_offsets_i64 = seg_offsets.to(torch.int64)
                zero_idx = torch.zeros_like(seg_offsets_i64)
                f_out_chunks: list[torch.Tensor] = []
                delta_syn_chunks: list[torch.Tensor] = []
                flat_idx_chunks: list[torch.Tensor] = []

                for m_step in range(m_max):
                    active_mask = m_step < seg_counts  # [V] bool
                    active_f = active_mask.float()
                    active_b = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
                    if active_b.numel() == 0:
                        break
                    flat_idx = seg_offsets_i64[active_b] + m_step  # [V_act]
                    safe_idx = torch.where(active_mask, seg_offsets_i64 + m_step, zero_idx)

                    am_col_f = active_f.unsqueeze(-1).to(sgsc_out.f_stem.dtype)
                    am_col_c = active_f.unsqueeze(-1).to(sgsc_out.c_slow.dtype)
                    f_step = sgsc_out.f_stem[safe_idx] * am_col_f
                    c_step = sgsc_out.c_slow[safe_idx] * am_col_c
                    sn_step = sgsc_out.sem_norm[safe_idx] * active_f
                    mi_step = sgsc_out.mean_info[safe_idx] * active_f

                    # Module 3 Read Path
                    hip_read = self.hippocampus.read_comprehension(
                        f_stem=f_step,
                        c_slow=c_step,
                        sem_norm=sn_step,
                        mean_info=mi_step,
                        fast_state=f_state,
                        enabled=hip_enabled,
                    )

                    # Module 2 BCG-PC Step
                    if self.training and m_step > 0:
                        # Straight-through BPTT carry damping (gamma=0.5): bit-identical forward
                        # pass while bounding multi-step recurrent Jacobian O(sum 0.5^k = 2) so
                        # long concept chains cannot exponentially explode gradients into s0/a0.
                        g_in = ConceptGraphState(
                            slots=0.5 * g_state.slots + 0.5 * g_state.slots.detach(),
                            edges=0.5 * g_state.edges + 0.5 * g_state.edges.detach(),
                            salience=0.5 * g_state.salience + 0.5 * g_state.salience.detach(),
                        )
                    else:
                        g_in = g_state
                    bcg_out = self.bcg_pc.step(
                        c_fused=hip_read.c_fused,
                        prec_bu=hip_read.prec_bu,
                        ach=hip_read.ach_signal,
                        state=g_in,
                        tau_m=tau_m,
                        use_gumbel=self.training,
                        merge_uniform_mix=merge_uniform_mix,
                    )

                    # Freeze graph state on inactive (finished) documents
                    am_s = active_mask.view(V, 1, 1)
                    am_e = active_mask.view(V, 1, 1, 1)
                    am_w = active_mask.view(V, 1)
                    g_state = ConceptGraphState(
                        slots=torch.where(am_s, bcg_out.state.slots, g_state.slots),
                        edges=torch.where(am_e, bcg_out.state.edges, g_state.edges),
                        salience=torch.where(am_w, bcg_out.state.salience, g_state.salience),
                    )

                    # Record detached per-segment graph trajectory & forward discrete merge alpha
                    if not self.training:
                        s_trace_flat[flat_idx] = g_state.slots[active_b].detach().to(s_trace_flat.dtype)
                        a_trace_flat[flat_idx] = g_state.edges[active_b].detach().to(a_trace_flat.dtype)
                        wi_a = bcg_out.win_i[active_b].detach()
                        wj_a = bcg_out.win_j[active_b].detach()
                        win_pair_flat[flat_idx, 0] = wi_a
                        win_pair_flat[flat_idx, 1] = wj_a
                        win_pair_flat[flat_idx, 2] = torch.minimum(wi_a, wj_a)
                    alpha_trace_flat[flat_idx] = bcg_out.merge_alpha[active_b].detach().float()

                    # Module 3 Eligibility Inheritance & Three-Factor Gated Delta Write (Stage D)
                    if hip_enabled and f_state is not None and e_state is not None:
                        e_next = self.hippocampus.inherit_eligibility_on_merge(
                            elig=e_state,
                            cand_idx=hip_read.psi_idx,
                            cand_val=hip_read.psi_val,
                            cand_f=f_step,
                            cand_ach=hip_read.ach_signal * active_f,
                            win_i=bcg_out.win_i,
                            win_j=bcg_out.win_j,
                        )
                        e_next_masked = EligibilityState(
                            e_idx=torch.where(active_mask.view(V, 1, 1), e_next.e_idx, e_state.e_idx),
                            e_val=torch.where(active_mask.view(V, 1, 1), e_next.e_val, e_state.e_val),
                            f_tilde=torch.where(active_mask.view(V, 1, 1), e_next.f_tilde, e_state.f_tilde),
                            eta=torch.where(active_mask.view(V, 1), e_next.eta, torch.zeros_like(e_next.eta)),
                        )
                        e_state, _, beta_mat = self.hippocampus.three_factor_write(
                            s_bar_post=bcg_out.s_bar_post,
                            edges_post=g_state.edges,
                            c0=self.bcg_pc.c0,
                            elig=e_next_masked,
                            fast_state=f_state,
                            write_threshold=eff_write_thresh,
                        )
                        beta_sum += float(beta_mat[active_b].detach().mean().item())
                        beta_count += 1
                        ach_act = hip_read.ach_signal[active_b]
                        ach_list.append(ach_act)
                        if not self.training:
                            ach_trace_flat[flat_idx] = ach_act.detach().float()
                            beta_max_flat[flat_idx] = beta_mat[active_b].detach().float().max(dim=-1).values
                        if familiar_doc_mask is not None:
                            fam_active = familiar_doc_mask[active_b]
                            if fam_active.any():
                                familiar_ach_list.append(ach_act[fam_active])

                    # Precision Island #4 (Red Line #1): Accumulate L_PC strictly in float32
                    # with STRICT stop_gradient on c_fused!
                    with fp32_island(device):
                        c_target_sg = hip_read.c_fused[active_b].detach().float()
                        c_pred = bcg_out.c_hat[active_b].float()
                        pi_act = bcg_out.pi_m[active_b].float()
                        err_sq = (c_target_sg - c_pred).pow(2)
                        nll_step = 0.5 * ((pi_act * err_sq).mean(dim=-1) - torch.log(pi_act.clamp_min(1e-6)).mean(dim=-1))
                        pc_nll_sum = pc_nll_sum + nll_step.sum()
                        ponder_sum = ponder_sum + bcg_out.ponder_steps[active_b].float().sum()
                        ponder_skip_sum = ponder_skip_sum + (bcg_out.ponder_steps[active_b].detach().float() < 0.5).float().sum()
                        merge_probs_sum = merge_probs_sum + bcg_out.merge_probs[active_b].float().sum(dim=0)
                    total_valid_steps += int(active_b.numel())

                    # Expressive stem retrieval for active steps (align batch dim V with f_state, then slice active_b!)
                    f_out_all, _ = self.hippocampus.read_expression(
                        z_out=bcg_out.z_star,
                        l_reg=l_reg,
                        fast_state=f_state if hip_enabled else None,
                        enabled=hip_enabled,
                    )
                    f_out_step = f_out_all[active_b]
                    delta_syn_step = self.pmd.compute_streaming_delta_syn(
                        g_ctx=bcg_out.g_ctx[active_b],
                        z_star=bcg_out.z_star[active_b],
                        l_reg=l_reg[active_b],
                    )

                    f_out_chunks.append(f_out_step.to(sgsc_out.f_stem.dtype))
                    delta_syn_chunks.append(delta_syn_step.to(sgsc_out.f_stem.dtype))
                    flat_idx_chunks.append(flat_idx)

                if flat_idx_chunks:
                    order = torch.cat(flat_idx_chunks, dim=0)
                    inv_perm = torch.argsort(order)
                    f_out_flat = torch.cat(f_out_chunks, dim=0)[inv_perm]
                    delta_syn_flat = torch.cat(delta_syn_chunks, dim=0)[inv_perm]

            s_trace = s_trace_flat
            a_trace = a_trace_flat
            merge_alpha_trace = alpha_trace_flat
            win_pair_trace = win_pair_flat
            ach_trace = ach_trace_flat
            beta_max_trace = beta_max_flat

            denom = max(total_valid_steps, 1)
            lam_pc = 1.0 if lambda_pc_override is None else float(lambda_pc_override)
            lam_act = self.cfg.lambda_act if lambda_act_override is None else float(lambda_act_override)
            with fp32_island(device):
                loss_pc = lam_pc * (pc_nll_sum / denom) + loss_var
                loss_ponder = lam_act * (ponder_sum / denom)
                avg_ponder = float((ponder_sum / denom).detach().item())
                ponder_skip_rate = float((ponder_skip_sum / denom).detach().item())

                # Merge pair selection entropy & optional uniform KL rescue (Section 5.4)
                mean_merge_probs = (merge_probs_sum / denom).clamp_min(1e-8)
                merge_entropy = -(mean_merge_probs * torch.log(mean_merge_probs)).sum()
                merge_entropy_val = float(merge_entropy.detach().item())
                if merge_kl_weight > 0.0:
                    uniform = torch.full_like(mean_merge_probs, 1.0 / mean_merge_probs.numel())
                    loss_merge_kl = merge_kl_weight * F.kl_div(
                        torch.log(mean_merge_probs), uniform, reduction="sum"
                    )

                if ach_list:
                    all_ach = torch.cat(ach_list, dim=0)
                    mean_ach_val = float(all_ach.detach().mean().item())
                    if ach_hint_targets is not None and ach_hint_weight > 0.0 and ach_hint_targets.numel() == all_ach.numel():
                        loss_ach_hint = ach_hint_weight * F.binary_cross_entropy(
                            all_ach.clamp(1e-5, 1.0 - 1e-5),
                            ach_hint_targets.float(),
                        )
                if familiar_ach_reg_weight > 0.0 and familiar_ach_list:
                    all_fam_ach = torch.cat(familiar_ach_list, dim=0)
                    loss_ach_reg = float(familiar_ach_reg_weight) * all_fam_ach.mean()
            if beta_count > 0:
                mean_beta_val = beta_sum / beta_count

            # Mode 2 (Stage C/D Broca CQ graph linearization) vs Mode 1 (Streaming Reconstruction)
            if use_broca_cq and stage in ("C", "D"):
                q_init = torch.sigmoid(g_state.salience.float())
                q_cur = q_init
                a_prev = torch.zeros((V, self.cfg.k_slots, self.cfg.d_r), device=device, dtype=g_state.edges.dtype)
                cq_f_out = f_out_flat.clone()
                cq_delta_syn = delta_syn_flat.clone()
                for n_step in range(min(self.cfg.k_slots, m_max)):
                    cq_out = self.pmd.broca_cq_step(
                        graph=g_state,
                        q_prev=q_cur,
                        q_init=q_init,
                        a_prev=a_prev,
                        l_reg=l_reg,
                    )
                    q_cur = cq_out.q_next
                    a_prev = cq_out.a_prev_next
                    active_b = torch.nonzero(n_step < seg_counts, as_tuple=False).squeeze(-1)
                    if active_b.numel() > 0:
                        f_cq_all, _ = self.hippocampus.read_expression(
                            z_out=cq_out.z_out,
                            l_reg=l_reg,
                            fast_state=f_state if hip_enabled else None,
                            enabled=hip_enabled,
                        )
                        f_cq = f_cq_all[active_b]
                        flat_idx = seg_offsets[active_b].to(torch.int64) + n_step
                        cq_f_out[flat_idx] = f_cq.to(cq_f_out.dtype)
                        cq_delta_syn[flat_idx] = cq_out.delta_syn[active_b].to(cq_delta_syn.dtype)
                f_decode = cq_f_out
                d_decode = cq_delta_syn
            else:
                f_decode = f_out_flat
                d_decode = delta_syn_flat

            pmd_out = self.pmd.decode_teacher_forced(
                target_bytes=raw_bytes,
                cu_seqlens=sgsc_out.cu_seqlens,
                f_out_seg=f_decode,
                delta_syn_seg=d_decode,
                l_reg=l_reg,
                m_init=m_init,
                xi_init=xi_init,
                doc_cu_seqlens=doc_cu_seqlens,
            )
            out_graph = g_state
            out_fast = f_state
            out_elig = e_state

        # Pragmatic language reconstruction loss L_lang (with optional per-byte target emphasis)
        with fp32_island(device):
            ce_per_byte = F.cross_entropy(
                pmd_out.byte_logits.float().reshape(-1, 256),
                raw_bytes.reshape(-1).long(),
                reduction="none",
            ).view(B, T)
            if loss_weights is not None:
                w = loss_weights.float()
                loss_lang = (ce_per_byte * w).sum() / w.sum().clamp_min(1.0)
            else:
                loss_lang = ce_per_byte.mean()

            # Zero-alignment motor boundary depletion loss L_halt against SGSC spikes
            lam_halt = self.cfg.lambda_halt if lambda_halt_override is None else float(lambda_halt_override)
            if lam_halt > 0.0:
                bce_halt = F.binary_cross_entropy(
                    pmd_out.p_seg_end.float().clamp(1e-6, 1.0 - 1e-6),
                    sgsc_out.spikes.float(),
                    reduction="none",
                )
                if stage == "D" and loss_weights is not None:
                    h_mask = (loss_weights.float() > 0.0).float()
                    loss_halt = lam_halt * ((bce_halt * h_mask).sum() / h_mask.sum().clamp_min(1.0))
                else:
                    loss_halt = lam_halt * bce_halt.mean()
            else:
                loss_halt = zero

            # Combine joint developmental losses
            prior_w = 0.5 if stage == "A" else (0.05 if stage in ("B", "C") else 0.0)
            loss_total = (
                loss_lang
                + prior_w * loss_sgsc_prior
                + (loss_pc if stage != "A" else loss_var)
                + (loss_rate if stage != "D" else zero)
                + loss_ponder
                + loss_halt
                + loss_merge_kl
                + loss_ach_hint
                + loss_ach_reg
            )

        spike_rate = float(sgsc_out.spikes.float().mean().detach().item())
        compression_ratio = 1.0 / max(spike_rate, 1e-6)

        metrics = {
            "loss_total": float(loss_total.detach().item()),
            "loss_lang": float(loss_lang.detach().item()),
            "loss_sgsc_prior": float(loss_sgsc_prior.detach().item()),
            "loss_pc": float(loss_pc.detach().item()),
            "loss_var": float(loss_var.detach().item()),
            "loss_rate": float(loss_rate.detach().item()),
            "loss_ponder": float(loss_ponder.detach().item()),
            "loss_halt": float(loss_halt.detach().item()),
            "loss_ach_reg": float(loss_ach_reg.detach().item()),
            "spike_rate": spike_rate,
            "compression_ratio": compression_ratio,
            "theta": float(sgsc_out.theta.detach().item()),
            "c_slow_std_mean": c_slow_std_mean,
            "c_slow_std_min": c_slow_std_min,
            "avg_ponder_steps": avg_ponder,
            "ponder_skip_rate": ponder_skip_rate,
            "merge_entropy_nats": merge_entropy_val,
            "mean_ach": mean_ach_val,
            "mean_beta": mean_beta_val,
        }

        return DCPAForwardOutput(
            loss_total=loss_total,
            loss_lang=loss_lang,
            loss_sgsc_prior=loss_sgsc_prior,
            loss_pc=loss_pc,
            loss_var=loss_var,
            loss_rate=loss_rate,
            loss_ponder=loss_ponder,
            loss_halt=loss_halt,
            loss_merge_kl=loss_merge_kl,
            loss_ach_hint=loss_ach_hint,
            sgsc=sgsc_out,
            pmd=pmd_out,
            graph_state=out_graph,
            fast_state=out_fast,
            elig_state=out_elig,
            metrics=metrics,
            s_trace=s_trace,
            a_trace=a_trace,
            merge_alpha_trace=merge_alpha_trace,
            win_pair_trace=win_pair_trace,
            ach_trace=ach_trace,
            beta_max_trace=beta_max_trace,
        )
