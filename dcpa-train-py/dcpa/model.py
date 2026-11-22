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
    EligibilityState,
    HippocampusDualMemory,
    HippocampusStateType,
)
from dcpa.modules.pmd import PMD, PMDDecodeOutput
from dcpa.modules.sgsc import SGSC, SGSCOutput


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
        merge_kl_weight: float = 0.0,
        fast_state: Optional[HippocampusStateType] = None,
        graph_state: Optional[ConceptGraphState] = None,
        elig_state: Optional[EligibilityState] = None,
    ) -> DCPAForwardOutput:
        """Execute full-stack forward pass and compute joint developmental loss."""
        stage = stage.upper()
        B, T = raw_bytes.shape
        device = raw_bytes.device
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 1. Module 1: SGSC sensory causal scan + LIF spiking + Ragged RoPE phase-bound stems
        sgsc_out = self.sgsc(raw_bytes, tau_s=tau_s)
        m_total = sgsc_out.f_stem.shape[0]
        l_reg = self.pmd.get_lang_vec(lang_ids)  # [B, d_l]

        # Build per-segment batch mapping
        seg_counts = sgsc_out.seg_counts  # [B]
        seg_offsets = torch.zeros_like(seg_counts)
        if B > 1:
            seg_offsets[1:] = torch.cumsum(seg_counts[:-1], dim=0)
        seg_batch_ids = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.int64),
            seg_counts.to(torch.int64),
        )  # [M_total]
        l_reg_seg = l_reg[seg_batch_ids]  # [M_total, d_l]

        # Rate anchor loss L_rate (with optional dual Lagrangian rescue term)
        lam_rate = self.cfg.lambda_rate if lambda_rate_override is None else lambda_rate_override
        mean_s_tilde = sgsc_out.s_tilde.mean()
        rate_err = mean_s_tilde - self.cfg.target_spike_rate
        loss_rate = lam_rate * rate_err.pow(2) + dual_rate_lagrange * rate_err

        # Sensory causal byte predictor loss (trains W_out @ h_{t-1} to produce accurate surprisal I_t)
        loss_sgsc_prior = F.cross_entropy(
            sgsc_out.prior_logits.reshape(-1, 256),
            raw_bytes.reshape(-1).long(),
        )

        # Variance hinge anti-collapse loss on c_slow (forced lambda_var > 0!)
        lam_var = self.cfg.lambda_var if lambda_var_override is None else lambda_var_override
        if m_total > 1:
            c_std = torch.sqrt(sgsc_out.c_slow.float().var(dim=0, unbiased=False) + 1e-4)
            loss_var = lam_var * F.relu(1.0 - c_std).mean()
            c_slow_std_mean = float(c_std.mean().detach().item())
        else:
            loss_var = zero
            c_slow_std_mean = 1.0

        loss_pc = zero
        loss_ponder = zero
        loss_merge_kl = zero
        loss_ach_hint = zero
        avg_ponder = 0.0
        merge_entropy_val = 0.0
        mean_ach_val = 0.0
        mean_beta_val = 0.0

        if stage == "A":
            # Stage A: Bypass BCG-PC and Hippocampus; directly reconstruct segments via f_stem & c_slow
            delta_syn_seg = self.pmd.compute_streaming_delta_syn(
                g_ctx=sgsc_out.c_slow,
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
            )
            out_graph = None
            out_fast = None
            out_elig = None
        else:
            # Stages B, C, D: Run BCG-PC concept graph (+ Hippocampus in Stage D)
            hip_enabled = stage == "D"
            g_state = self.bcg_pc.init_state(B, device=device, dtype=sgsc_out.c_slow.dtype) if graph_state is None else graph_state
            f_state = (
                self.hippocampus.init_fast_state(B, mode="compact", device=device)
                if (hip_enabled and fast_state is None)
                else fast_state
            )
            e_state = (
                EligibilityState.zeros(B, self.cfg.k_slots, self.cfg.s_active, self.cfg.d_s, device, sgsc_out.f_stem.dtype)
                if (hip_enabled and elig_state is None)
                else elig_state
            )

            m_max = int(seg_counts.max().item())
            D = self.cfg.d_concept
            d_s = self.cfg.d_s

            z_star_flat = torch.zeros((m_total, D), device=device, dtype=sgsc_out.c_slow.dtype)
            g_ctx_flat = torch.zeros((m_total, D), device=device, dtype=sgsc_out.c_slow.dtype)
            f_out_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
            delta_syn_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)

            pc_nll_sum = zero
            ponder_sum = zero
            merge_probs_sum = torch.zeros((self.cfg.num_merge_pairs,), device=device, dtype=torch.float32)
            ach_list: list[torch.Tensor] = []
            beta_sum = 0.0
            beta_count = 0
            total_valid_steps = 0

            for m_step in range(m_max):
                active_mask = m_step < seg_counts  # [B] bool
                active_b = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
                if active_b.numel() == 0:
                    break
                flat_idx = seg_offsets[active_b].to(torch.int64) + m_step  # [B_act]

                # Gather [B, ...] with zero placeholder on finished sequences (strictly masked out!)
                f_step = torch.zeros((B, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
                c_step = torch.zeros((B, D), device=device, dtype=sgsc_out.c_slow.dtype)
                sn_step = torch.zeros((B,), device=device, dtype=torch.float32)
                mi_step = torch.zeros((B,), device=device, dtype=torch.float32)

                f_step[active_b] = sgsc_out.f_stem[flat_idx]
                c_step[active_b] = sgsc_out.c_slow[flat_idx]
                sn_step[active_b] = sgsc_out.sem_norm[flat_idx]
                mi_step[active_b] = sgsc_out.mean_info[flat_idx]

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
                bcg_out = self.bcg_pc.step(
                    c_fused=hip_read.c_fused,
                    prec_bu=hip_read.prec_bu,
                    ach=hip_read.ach_signal,
                    state=g_state,
                    tau_m=tau_m,
                    use_gumbel=self.training,
                )

                # Freeze graph state on inactive (finished) batch items
                am_s = active_mask.view(B, 1, 1)
                am_e = active_mask.view(B, 1, 1, 1)
                am_w = active_mask.view(B, 1)
                g_state = ConceptGraphState(
                    slots=torch.where(am_s, bcg_out.state.slots, g_state.slots),
                    edges=torch.where(am_e, bcg_out.state.edges, g_state.edges),
                    salience=torch.where(am_w, bcg_out.state.salience, g_state.salience),
                )

                # Module 3 Eligibility Inheritance & Three-Factor Gated Delta Write (Stage D)
                if hip_enabled and f_state is not None and e_state is not None:
                    e_next = self.hippocampus.inherit_eligibility_on_merge(
                        elig=e_state,
                        cand_idx=hip_read.psi_idx,
                        cand_val=hip_read.psi_val,
                        cand_f=f_step,
                        cand_ach=hip_read.ach_signal * active_mask.float(),
                        win_i=bcg_out.win_i,
                        win_j=bcg_out.win_j,
                    )
                    # Zero out eta on finished sequences so they do not write
                    e_next_masked = EligibilityState(
                        e_idx=torch.where(active_mask.view(B, 1, 1), e_next.e_idx, e_state.e_idx),
                        e_val=torch.where(active_mask.view(B, 1, 1), e_next.e_val, e_state.e_val),
                        f_tilde=torch.where(active_mask.view(B, 1, 1), e_next.f_tilde, e_state.f_tilde),
                        eta=torch.where(active_mask.view(B, 1), e_next.eta, torch.zeros_like(e_next.eta)),
                    )
                    e_state, _, beta_mat = self.hippocampus.three_factor_write(
                        s_bar_post=bcg_out.s_bar_post,
                        edges_post=g_state.edges,
                        c0=self.bcg_pc.c0,
                        elig=e_next_masked,
                        fast_state=f_state,
                    )
                    beta_sum += float(beta_mat[active_b].detach().mean().item())
                    beta_count += 1
                    ach_list.append(hip_read.ach_signal[active_b])

                # Accumulate free-energy predictive coding loss L_PC with STRICT stop_gradient on c_fused!
                c_target_sg = hip_read.c_fused[active_b].detach().float()
                c_pred = bcg_out.c_hat[active_b].float()
                pi_act = bcg_out.pi_m[active_b].float()
                err_sq = (c_target_sg - c_pred).pow(2)
                nll_step = 0.5 * ((pi_act * err_sq).mean(dim=-1) - torch.log(pi_act.clamp_min(1e-6)).mean(dim=-1))
                pc_nll_sum = pc_nll_sum + nll_step.sum()
                ponder_sum = ponder_sum + bcg_out.ponder_steps[active_b].sum()
                merge_probs_sum = merge_probs_sum + bcg_out.merge_probs[active_b].sum(dim=0)
                total_valid_steps += int(active_b.numel())

                # Expressive stem retrieval for active steps
                f_out_step, _ = self.hippocampus.read_expression(
                    z_out=bcg_out.z_star[active_b],
                    l_reg=l_reg[active_b],
                    fast_state=f_state if hip_enabled else None,
                    enabled=hip_enabled,
                )
                delta_syn_step = self.pmd.compute_streaming_delta_syn(
                    g_ctx=bcg_out.g_ctx[active_b],
                    z_star=bcg_out.z_star[active_b],
                    l_reg=l_reg[active_b],
                )

                z_star_flat[flat_idx] = bcg_out.z_star[active_b]
                g_ctx_flat[flat_idx] = bcg_out.g_ctx[active_b]
                f_out_flat[flat_idx] = f_out_step
                delta_syn_flat[flat_idx] = delta_syn_step

            denom = max(total_valid_steps, 1)
            loss_pc = (pc_nll_sum / denom) + loss_var
            loss_ponder = self.cfg.lambda_act * (ponder_sum / denom)
            avg_ponder = float((ponder_sum / denom).detach().item())

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
            if beta_count > 0:
                mean_beta_val = beta_sum / beta_count

            # Mode 2 (Stage C/D Broca CQ graph linearization) vs Mode 1 (Streaming Reconstruction)
            if use_broca_cq and stage in ("C", "D"):
                # Run Broca CQ steps from final concept graph to produce graph-linearized stems
                q_init = torch.sigmoid(g_state.salience.float())
                q_cur = q_init
                a_prev = torch.zeros((B, self.cfg.k_slots, self.cfg.d_r), device=device, dtype=g_state.edges.dtype)
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
                        f_cq, _ = self.hippocampus.read_expression(
                            z_out=cq_out.z_out[active_b],
                            l_reg=l_reg[active_b],
                            fast_state=f_state if hip_enabled else None,
                            enabled=hip_enabled,
                        )
                        flat_idx = seg_offsets[active_b].to(torch.int64) + n_step
                        cq_f_out[flat_idx] = f_cq
                        cq_delta_syn[flat_idx] = cq_out.delta_syn[active_b]
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
            )
            out_graph = g_state
            out_fast = f_state
            out_elig = e_state

        # Pragmatic language reconstruction loss L_lang (with optional per-byte target emphasis)
        ce_per_byte = F.cross_entropy(
            pmd_out.byte_logits.reshape(-1, 256),
            raw_bytes.reshape(-1).long(),
            reduction="none",
        ).view(B, T)
        if loss_weights is not None:
            w = loss_weights.float()
            loss_lang = (ce_per_byte * w).sum() / w.sum().clamp_min(1.0)
        else:
            loss_lang = ce_per_byte.mean()

        # Zero-alignment motor boundary depletion loss L_halt against SGSC spikes
        loss_halt = self.cfg.lambda_halt * F.binary_cross_entropy(
            pmd_out.p_seg_end.clamp(1e-6, 1.0 - 1e-6),
            sgsc_out.spikes.float(),
        )

        # Combine joint developmental losses
        prior_w = 0.5 if stage == "A" else (0.2 if stage in ("B", "C") else 0.0)
        loss_total = (
            loss_lang
            + prior_w * loss_sgsc_prior
            + (loss_pc if stage != "A" else loss_var * 0.1)
            + (loss_rate if stage != "D" else zero)
            + loss_ponder
            + loss_halt
            + loss_merge_kl
            + loss_ach_hint
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
            "spike_rate": spike_rate,
            "compression_ratio": compression_ratio,
            "theta": float(sgsc_out.theta.detach().item()),
            "c_slow_std_mean": c_slow_std_mean,
            "avg_ponder_steps": avg_ponder,
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
        )
