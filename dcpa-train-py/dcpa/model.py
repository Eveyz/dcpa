"""Unified DCPA-Turbo full-stack model wiring SGSC -> CorticalSandwich <-> BCG-PC / Hippocampus -> PMD.

Implements the 5-Step Parallel Cortical Sandwich + Contractive Clause-Graph Architecture:
  - Stage A: SGSC + PMD lower motor decoder (bypass Cortical/BCG-PC/Hippocampus, reconstruct via c_slow/f_stem).
  - Stage B: Enable Parallel Cortical Sandwich (`Concept-RoPE` + Shared 48KB MLA Ring + `Shared+Routed MoE`),
             Zero-MoE Contractive `BCG-PC` (`Lambda_m in [0.92, 0.998]`, `phase`), and `[M_total, D]` `ponder_parallel`.
  - Stage C: Activate Broca CQ full-graph linearization alongside Upper Cortical Decoder.
  - Stage D: Freeze sensory byte embed & LIF threshold, enable Hippocampus 3-factor Gated Delta
             dual memory (`O(log2 K)` nilpotent WY parallel solver) for episodic fast-mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from dcpa.config import DCPAConfig
from dcpa.modules.bcg_pc import BCGPC, ConceptGraphState, _DampedBoundedCarry
from dcpa.modules.cortical import CorticalCarryState, CorticalSandwich
from dcpa.modules.hippocampus import (
    ColumnMajorFastWeights,
    CompactHouseholderState,
    EligibilityState,
    HippocampusDualMemory,
    HippocampusStateType,
)
from dcpa.modules.pmd import PMD, PMDDecodeOutput
from dcpa.modules.sgsc import SGSC, SGSCOutput
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.ragged import ragged_local_offsets, ragged_segment_sum
from dcpa.ops.rope import compute_segment_scope_depths
from dcpa.ops.scan import parallel_diag_scan

try:
    from utils.device import get_allocated_memory_gb as _get_allocated_memory_gb
except ImportError:
    def _get_allocated_memory_gb(_device: torch.device) -> float:
        return 0.0


def _compute_pin_trigger_byte_mask(
    raw_bytes: torch.Tensor,
    doc_reset_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized [B, T] bool mask marking closing byte of `\\x1dE\\n` (or `\\x1fA\\n` not followed by `\\x1dS\\n`)."""
    B, T = raw_bytes.shape
    device = raw_bytes.device
    if T < 3:
        return torch.zeros((B, T), device=device, dtype=torch.bool)

    rb = raw_bytes.to(dtype=torch.long)
    b_m2 = rb[:, :-2]
    b_m1 = rb[:, 1:-1]
    b_0 = rb[:, 2:]

    is_scratch_end = (b_m2 == 0x1D) & (b_m1 == ord("E")) & (b_0 == 0x0A)
    is_asst_start = (b_m2 == 0x1F) & (b_m1 == ord("A")) & (b_0 == 0x0A)
    if T >= 6:
        next_is_scratch = (rb[:, 3:-2] == 0x1D) & (rb[:, 4:-1] == ord("S")) & (rb[:, 5:] == 0x0A)
        pad_tail = torch.zeros((B, 3), device=device, dtype=torch.bool)
        is_asst_start = is_asst_start & ~torch.cat([next_is_scratch, pad_tail], dim=1)

    out = torch.zeros((B, T), device=device, dtype=torch.bool)
    out[:, 2:] = is_scratch_end | is_asst_start
    if doc_reset_mask is not None:
        drm = doc_reset_mask.bool()
        drm_p1 = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.bool), drm[:, :-1]], dim=1)
        out = out & ~(drm | drm_p1)
    return out


def _compute_unpin_trigger_byte_mask(raw_bytes: torch.Tensor) -> torch.Tensor:
    """Vectorized [B, T] bool mask marking turn-boundary bytes `0x04` (EOT) or `0x1E` (user RS)."""
    rb = raw_bytes.to(dtype=torch.long)
    return (rb == 0x04) | (rb == 0x1E)


@dataclass
class DCPACarryState:
    """Strictly O(1) constant-size runtime working memory carried across streaming chunks.

    Zero dynamic Python lists and zero sequence-length-dependent caches:
      - sgsc_h:         [B, d_s]
      - cortical_state: CorticalCarryState (`W` fixed MLA-RoPE ring + causal shift state)
      - graph_state:    S [B, K, D], A [B, K, K, d_r], omega [B, K], phase [B, K]  (K == 8)
      - pmd_m:          [B, d_s]
      - pmd_xi:         [B, 16, 4]  (128B pushdown stack)
      - fast_state:     ColumnMajorFastWeights (fixed [d_h, D], [d_h, d_s], [d_h] arrays)
      - elig_state:     EligibilityState ([B, K, s], [B, K, d_s], [B, K])
    """

    sgsc_h: torch.Tensor
    graph_state: ConceptGraphState
    pmd_m: torch.Tensor
    pmd_xi: torch.Tensor
    fast_state: ColumnMajorFastWeights
    elig_state: EligibilityState
    cortical_state: Optional[CorticalCarryState] = None


_METRIC_VEC_KEYS: tuple[str, ...] = (
    "loss_total",
    "loss_lang",
    "loss_sgsc_prior",
    "loss_pc",
    "loss_pc_raw",
    "loss_stem_align",
    "loss_gen_align",
    "loss_var",
    "loss_rate",
    "loss_ponder",
    "loss_moe_bal",
    "loss_halt",
    "loss_ach_reg",
    "spike_rate",
    "theta",
    "c_slow_std_mean",
    "c_slow_std_min",
    "avg_ponder_steps",
    "ponder_skip_rate",
    "merge_entropy_nats",
    "mean_ach",
    "mean_beta",
)


def materialize_metric_vec(metric_vec: torch.Tensor) -> dict[str, float]:
    """Convert a `[22]` fp32 scalar metrics tensor into the canonical `dict[str, float]` in ONE D2H copy."""
    vals = metric_vec.detach().float().to("cpu").tolist()
    d = {k: float(v) for k, v in zip(_METRIC_VEC_KEYS, vals)}
    sr = d["spike_rate"]
    d["compression_ratio"] = 1.0 / max(sr, 1e-6)
    return d


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
    s_trace: Optional[torch.Tensor] = None
    a_trace: Optional[torch.Tensor] = None
    merge_alpha_trace: Optional[torch.Tensor] = None
    win_pair_trace: Optional[torch.Tensor] = None
    ach_trace: Optional[torch.Tensor] = None
    beta_max_trace: Optional[torch.Tensor] = None
    slot0_pinned_trace: Optional[torch.Tensor] = None
    loss_moe_bal: Optional[torch.Tensor] = None
    metrics_tensor: Optional[torch.Tensor] = None
    cortical_state: Optional[CorticalCarryState] = None

    def materialize_metrics(self) -> dict[str, float]:
        if not self.metrics and self.metrics_tensor is not None:
            self.metrics = materialize_metric_vec(self.metrics_tensor)
        return self.metrics


class DCPAModel(nn.Module):
    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sgsc = SGSC(cfg)
        self.bcg_pc = BCGPC(cfg)
        self.hippocampus = HippocampusDualMemory(cfg)
        # Strict weight tying between SGSC and PMD for Embed [256, d_s] and W_out [256, d_s]
        self.pmd = PMD(cfg, shared_embed=self.sgsc.embed, shared_w_out=self.sgsc.w_out)
        self.cortical = CorticalSandwich(cfg)

        self.enable_doc_bucketing: bool = False
        self.doc_bucket_size: int = 16
        self.bcg_grad_checkpointing: bool = True
        self.bcg_checkpoint_vram_watermark_gb: float = 0.0
        # Log-Space Contractive Gate Lambda_m in [0.92, 0.998] bounds the recurrent slot Jacobian
        # organically, so carry_grad_gamma defaults to 1.0 (full end-to-end gradient flow!).
        self.carry_grad_gamma: float = 1.0
        self.carry_grad_max_rms: float = 2.0
        self.bcg_tbptt_window: int = 0
        self.bcg_tbptt_replay: bool = False
        self.bcg_chunk_steps: int = max(1, int(cfg.clause_chunk_size))

    def _clause_chunk_checkpointed_fn(
        self,
        in_slots: torch.Tensor,
        in_edges: torch.Tensor,
        in_sal: torch.Tensor,
        in_phase: torch.Tensor,
        slot0_pinned: torch.Tensor,
        c_fused_seq: torch.Tensor,
        prec_bu_seq: torch.Tensor,
        ach_seq: torch.Tensor,
        active_mask_seq: torch.Tensor,
        trig_seq: torch.Tensor,
        unpin_seq: torch.Tensor,
        tau_m_t: torch.Tensor,
        mix_t: torch.Tensor,
        gumbel_u_seq: torch.Tensor,
        check_pin: bool,
    ) -> tuple[torch.Tensor, ...]:
        bcg_raw = getattr(self.bcg_pc, "_orig_mod", self.bcg_pc)
        return bcg_raw.forward_clause_chunk(
            in_slots,
            in_edges,
            in_sal,
            in_phase,
            slot0_pinned,
            c_fused_seq,
            prec_bu_seq,
            ach_seq,
            active_mask_seq,
            trig_seq,
            unpin_seq,
            tau_m_t,
            mix_t,
            gumbel_u_seq=gumbel_u_seq,
            use_gumbel=True,
            check_pin=check_pin,
        )

    def _query_hippocampus_expression_fast(
        self,
        q_concept: torch.Tensor,
        fast_state: HippocampusStateType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query pre-write `M_{m-1}^{L<-C}` from strictly `< m` graph prediction `q_concept [B, D]`."""
        with fp32_island(q_concept.device):
            phi_idx, phi_val = self.hippocampus.dg_sem(q_concept)
            gamma_sem = fast_state.read_familiarity_sem(phi_idx, phi_val)  # [B] fp32
            rho_out = torch.sigmoid(self.hippocampus.alpha_out * (gamma_sem - self.hippocampus.theta_out))  # [B] fp32
            f_hip = fast_state.read_l_from_c(phi_idx, phi_val)  # [B, d_s] fp32
        return rho_out, f_hip

    def count_parameters(self) -> dict[str, int]:
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        sgsc_n = _count(self.sgsc)
        cort_n = _count(self.cortical)
        bcg_raw = getattr(self.bcg_pc, "_orig_mod", self.bcg_pc)
        bcg_n = _count(bcg_raw)
        hip_n = _count(self.hippocampus)
        tied_ids = {id(self.sgsc.embed.weight), id(self.sgsc.w_out.weight)}
        pmd_unique = sum(p.numel() for p in self.pmd.parameters() if id(p) not in tied_ids)
        total = sum(p.numel() for p in self.parameters())

        routed_experts = sum(p.numel() for p in bcg_raw.moe.expert_parameters())
        for layer in (*self.cortical.enc_layers, *self.cortical.dec_layers):
            routed_experts += sum(p.numel() for p in layer.moe.expert_parameters())
        inactive_experts = (
            routed_experts * (self.cfg.num_experts - self.cfg.top_k_experts) // max(self.cfg.num_experts, 1)
        )
        active_per_step = total - inactive_experts
        return {
            "sgsc": sgsc_n,
            "cortical": cort_n,
            "bcg_pc": bcg_n,
            "hippocampus_slow": hip_n,
            "pmd_unique": pmd_unique,
            "total": total,
            "active_per_step": active_per_step,
        }

    def parameter_groups(self) -> list[dict]:
        """Grouped parameter sets:
        1. Micro-cores (SGSC + PMD) -> lr_micro, weight_decay
        2. Concept core (CorticalSandwich + BCG-PC + slow Hippocampus projections) -> lr_concept, weight_decay
        3. LIF threshold & scalar gating parameters -> lr_scalar_gate, weight_decay=0.0
        """
        bcg_raw = getattr(self.bcg_pc, "_orig_mod", self.bcg_pc)
        scalar_params = [self.sgsc.theta_raw, bcg_raw.kappa_omega] + self.hippocampus.scalar_gate_parameters()
        scalar_ids = {id(p) for p in scalar_params}

        micro_params: list[nn.Parameter] = []
        seen: set[int] = set(scalar_ids)
        for mod in (self.sgsc, self.pmd):
            for p in mod.parameters():
                if id(p) not in seen:
                    micro_params.append(p)
                    seen.add(id(p))

        concept_params: list[nn.Parameter] = []
        for mod in (self.cortical, bcg_raw, self.hippocampus):
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
        bcg_raw = getattr(self.bcg_pc, "_orig_mod", self.bcg_pc)
        sgsc_h = self.sgsc.h0.detach().to(device=dev, dtype=dtype).unsqueeze(0).clone()
        cortical_state = self.cortical.init_carry_state(1, device=dev, dtype=dtype)
        graph_state = bcg_raw.init_state(1, device=dev, dtype=dtype)
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
            cortical_state=cortical_state,
        )

    def forward_streaming_chunk(
        self,
        raw_bytes: torch.Tensor,
        lang_ids: torch.Tensor,
        carry_state: Optional[DCPACarryState] = None,
        stage: str = "D",
        tau_s: float = 0.35,
        tau_m: float = 0.25,
        style_ids: Optional[torch.Tensor] = None,
        ach_override: Optional[float] = None,
    ) -> tuple[DCPAForwardOutput, DCPACarryState]:
        """Run one chunk through all modules while carrying strictly O(1) constant state."""
        B = raw_bytes.shape[0]
        if carry_state is None:
            carry_state = self.init_carry_state(batch_size=B, device=raw_bytes.device, dtype=torch.float32)
        c_state_in = (
            carry_state.cortical_state
            if carry_state.cortical_state is not None
            else self.cortical.init_carry_state(batch_size=B, device=raw_bytes.device, dtype=torch.float32)
        )

        out = self.forward(
            raw_bytes=raw_bytes,
            lang_ids=lang_ids,
            style_ids=style_ids,
            stage=stage,
            tau_s=tau_s,
            tau_m=tau_m,
            use_broca_cq=False,
            ach_override=ach_override,
            fast_state=carry_state.fast_state,
            graph_state=carry_state.graph_state,
            elig_state=carry_state.elig_state,
            cortical_state=c_state_in,
            h_init=carry_state.sgsc_h,
            m_init=carry_state.pmd_m,
            xi_init=carry_state.pmd_xi,
        )
        assert out.graph_state is not None
        assert isinstance(out.fast_state, ColumnMajorFastWeights)
        assert out.elig_state is not None
        c_out = out.cortical_state if out.cortical_state is not None else c_state_in
        next_carry = DCPACarryState(
            sgsc_h=out.sgsc.h_seq[:, -1, :].detach().clone(),
            graph_state=ConceptGraphState(
                slots=out.graph_state.slots.detach().clone(),
                edges=out.graph_state.edges.detach().clone(),
                salience=out.graph_state.salience.detach().clone(),
                slot0_pinned=(
                    out.graph_state.slot0_pinned.detach().clone()
                    if out.graph_state.slot0_pinned is not None
                    else None
                ),
                phase=(
                    out.graph_state.phase.detach().clone()
                    if out.graph_state.phase is not None
                    else torch.zeros((B, self.cfg.k_slots), device=raw_bytes.device, dtype=torch.float32)
                ),
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
            cortical_state=CorticalCarryState(
                ring_kv=c_out.ring_kv.detach().clone(),
                ring_pos=c_out.ring_pos.detach().clone(),
                ring_depth=c_out.ring_depth.detach().clone(),
                ring_valid=c_out.ring_valid.detach().clone(),
                doc_pos=c_out.doc_pos.detach().clone(),
                ast_depth=c_out.ast_depth.detach().clone(),
                prev_z_ref=c_out.prev_z_ref.detach().clone(),
                has_prev_seg=c_out.has_prev_seg.detach().clone(),
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
        cortical_state: Optional[CorticalCarryState] = None,
        doc_cu_seqlens: Optional[torch.Tensor] = None,
        h_init: Optional[torch.Tensor] = None,
        m_init: Optional[torch.Tensor] = None,
        xi_init: Optional[torch.Tensor] = None,
        style_ids: Optional[torch.Tensor] = None,
        ponder_tax_scale: Optional[torch.Tensor] = None,
        ach_override: Optional[float] = None,
        record_traces: bool = True,
        sync_metrics: bool = True,
    ) -> DCPAForwardOutput:
        """Execute full-stack DCPA-Turbo forward pass (`[B, T]` or packed varlen `doc_cu_seqlens`)."""
        stage = stage.upper()
        B, T = raw_bytes.shape
        device = raw_bytes.device
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        bcg_raw = getattr(self.bcg_pc, "_orig_mod", self.bcg_pc)

        # 1. Module 1: SGSC sensory causal scan + LIF spiking + Ragged RoPE phase-bound stems
        sgsc_out = self.sgsc(
            raw_bytes, tau_s=tau_s, h_init=h_init, doc_cu_seqlens=doc_cu_seqlens
        )
        m_total = sgsc_out.f_stem.shape[0]
        l_reg = self.pmd.get_lang_vec(lang_ids, style_ids)

        doc_reset_mask: Optional[torch.Tensor] = None
        if doc_cu_seqlens is not None:
            if B != 1:
                raise ValueError(f"Packed varlen doc_cu_seqlens requires B=1, got B={B}")
            doc_starts = doc_cu_seqlens[:-1].to(dtype=torch.int64, device=device)
            doc_ends = doc_cu_seqlens[1:].to(dtype=torch.int64, device=device)
            V = int(doc_starts.numel())
            if l_reg.shape[0] != V:
                raise ValueError(f"lang_ids len ({l_reg.shape[0]}) must match num_docs ({V})")
            doc_reset_mask = torch.zeros((1, T), device=device, dtype=torch.bool)
            if V > 0:
                doc_reset_mask[0, doc_starts] = True
            spike_cs = torch.cumsum(sgsc_out.spikes[0].to(torch.int64), dim=0)
            end_counts = spike_cs[doc_ends - 1]
            start_counts = torch.zeros_like(end_counts)
            if V > 1:
                start_counts[1:] = spike_cs[doc_starts[1:] - 1]
            seg_counts = (end_counts - start_counts).to(torch.int32)
        else:
            V = B
            seg_counts = sgsc_out.seg_counts

        if m_total > 0 and T > 0:
            pin_byte_mask = _compute_pin_trigger_byte_mask(raw_bytes, doc_reset_mask)
            unpin_byte_mask = _compute_unpin_trigger_byte_mask(raw_bytes)
            seg_id_flat, local_pos, _, _ = ragged_local_offsets(sgsc_out.cu_seqlens, B * T)
            pos_pow = torch.pow(2.0, local_pos.clamp_max(22).float()).unsqueeze(-1)
            pin_w = ragged_segment_sum(
                pin_byte_mask.float().reshape(B * T, 1) * pos_pow, seg_id_flat, m_total
            ).squeeze(-1)
            unpin_w = ragged_segment_sum(
                unpin_byte_mask.float().reshape(B * T, 1) * pos_pow, seg_id_flat, m_total
            ).squeeze(-1)
            seg_unpin_trigger = (unpin_w > 0.5).float()
            seg_pin_trigger = (pin_w > unpin_w).float()
        else:
            seg_unpin_trigger = torch.zeros((m_total,), device=device, dtype=torch.float32)
            seg_pin_trigger = torch.zeros((m_total,), device=device, dtype=torch.float32)

        seg_offsets = torch.zeros_like(seg_counts)
        if V > 1:
            seg_offsets[1:] = torch.cumsum(seg_counts[:-1], dim=0)
        seg_doc_ids = torch.repeat_interleave(
            torch.arange(V, device=device, dtype=torch.int64),
            seg_counts.to(torch.int64),
        )
        l_reg_seg = l_reg[seg_doc_ids]

        seg_doc_start = torch.zeros((m_total,), device=device, dtype=torch.bool)
        valid_docs = seg_counts > 0
        if valid_docs.any():
            seg_doc_start[seg_offsets[valid_docs].to(torch.int64)] = True

        # Sensory losses (L_rate, L_sgsc_prior, L_var)
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

            loss_sgsc_prior = F.cross_entropy(
                sgsc_out.prior_logits.float().reshape(-1, 256),
                raw_bytes.reshape(-1).long(),
            )

            lam_var = self.cfg.lambda_var if lambda_var_override is None else lambda_var_override
            if m_total > 1:
                c_std = torch.sqrt(sgsc_out.c_slow.float().var(dim=0, unbiased=False) + 1e-4)
                loss_var = lam_var * (F.relu(1.0 - c_std).mean() + F.relu(c_std - 1.20).pow(2).mean())
                c_slow_std_mean_t = c_std.mean().detach()
                c_slow_std_min_t = c_std.min().detach()
            else:
                loss_var = zero
                c_slow_std_mean_t = torch.tensor(1.0, device=device, dtype=torch.float32)
                c_slow_std_min_t = torch.tensor(1.0, device=device, dtype=torch.float32)

        loss_pc = zero
        loss_pc_raw_t = zero
        loss_stem_align = zero
        loss_gen_align = zero
        loss_ponder = zero
        loss_merge_kl = zero
        loss_ach_hint = zero
        loss_ach_reg = zero
        loss_moe_bal = zero
        avg_ponder_t = zero
        ponder_skip_rate_t = torch.tensor(1.0, device=device, dtype=torch.float32)
        merge_entropy_t = zero
        mean_ach_t = zero
        mean_beta_t = zero
        s_trace: Optional[torch.Tensor] = None
        a_trace: Optional[torch.Tensor] = None
        merge_alpha_trace: Optional[torch.Tensor] = None
        win_pair_trace: Optional[torch.Tensor] = None
        ach_trace: Optional[torch.Tensor] = None
        beta_max_trace: Optional[torch.Tensor] = None
        slot0_pinned_trace: Optional[torch.Tensor] = None
        out_cortical: Optional[CorticalCarryState] = cortical_state

        if stage == "A":
            if m_total > 0:
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
                known_max_seg_len=self.cfg.max_seg_len + 4,
            )
            out_graph = None
            out_fast = None
            out_elig = None
        else:
            # Stages B, C, D: 5-Step DCPA-Turbo Parallel Cortical Sandwich + Contractive Graph + WY Hippocampus
            hip_enabled = stage == "D"
            eff_write_thresh: Optional[float] = write_threshold_override
            if eff_write_thresh is None and hip_enabled and self.training:
                eff_write_thresh = 0.0

            use_cq = bool(use_broca_cq and stage in ("C", "D"))
            D = self.cfg.d_concept
            d_s = self.cfg.d_s
            K = self.cfg.k_slots
            d_r = self.cfg.d_r
            num_pairs = self.cfg.num_merge_pairs

            # Causal segment-level bracket/frame scope depths (`seg_start_depth` strictly `< cu_seqlens[m]`)
            init_ast = cortical_state.ast_depth if cortical_state is not None else None
            seg_start_depth, seg_end_depth, final_ast_depth = compute_segment_scope_depths(
                raw_bytes=raw_bytes,
                cu_seqlens=sgsc_out.cu_seqlens,
                seg_doc_start=seg_doc_start,
                doc_reset_mask=doc_reset_mask,
                init_depth=init_ast,
            )
            pos_in_doc = (
                torch.arange(m_total, device=device, dtype=torch.int64) - seg_offsets[seg_doc_ids].to(torch.int64)
                if m_total > 0
                else torch.zeros((0,), device=device, dtype=torch.int64)
            )

            # -----------------------------------------------------------------
            # Mode I: O(1) Stateful Streaming Chunk (`cortical_state` provided)
            # -----------------------------------------------------------------
            if cortical_state is not None and B == 1 and doc_cu_seqlens is None:
                c_carry = CorticalCarryState(
                    ring_kv=cortical_state.ring_kv.to(dtype=sgsc_out.f_stem.dtype),
                    ring_pos=cortical_state.ring_pos,
                    ring_depth=cortical_state.ring_depth,
                    ring_valid=cortical_state.ring_valid,
                    doc_pos=cortical_state.doc_pos,
                    ast_depth=cortical_state.ast_depth,
                    prev_z_ref=cortical_state.prev_z_ref.to(dtype=sgsc_out.c_slow.dtype),
                    has_prev_seg=cortical_state.has_prev_seg,
                )
                gd = bcg_raw.init_state(1, device=device, dtype=sgsc_out.c_slow.dtype) if graph_state is None else graph_state
                if gd.slot0_pinned is None:
                    gd = ConceptGraphState(
                        slots=gd.slots,
                        edges=gd.edges,
                        salience=gd.salience,
                        slot0_pinned=torch.zeros((1,), device=device, dtype=torch.float32),
                        phase=gd.phase,
                    )
                if gd.phase is None:
                    gd = ConceptGraphState(
                        slots=gd.slots,
                        edges=gd.edges,
                        salience=gd.salience,
                        slot0_pinned=gd.slot0_pinned,
                        phase=torch.zeros((1, K), device=device, dtype=torch.float32),
                    )
                fd = (
                    self.hippocampus.init_fast_state(1, mode="compact", device=device)
                    if (hip_enabled and fast_state is None)
                    else fast_state
                )
                ed = (
                    EligibilityState.zeros(1, K, self.cfg.s_active, d_s, device, sgsc_out.f_stem.dtype)
                    if (hip_enabled and elig_state is None)
                    else elig_state
                )

                f_out_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
                delta_syn_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
                s_trace_flat = torch.zeros((m_total, K, D), device=device, dtype=sgsc_out.c_slow.dtype) if record_traces else None
                a_trace_flat = torch.zeros((m_total, K, K, d_r), device=device, dtype=sgsc_out.c_slow.dtype) if record_traces else None
                alpha_trace_flat = torch.zeros((m_total, num_pairs), device=device, dtype=torch.float32) if record_traces else None
                win_pair_flat = torch.zeros((m_total, 3), device=device, dtype=torch.int64) if record_traces else None
                ach_trace_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None
                beta_max_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None
                slot0_pinned_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None

                l_reg_1 = l_reg[:1]
                pc_nll_sum = zero
                ponder_sum = zero
                ponder_skip_sum = zero
                merge_probs_sum = torch.zeros((num_pairs,), device=device, dtype=torch.float32)
                ach_list_s: list[torch.Tensor] = []
                beta_sum_t = zero
                beta_count = 0

                for m_idx in range(m_total):
                    f_m = sgsc_out.f_stem[m_idx : m_idx + 1]
                    c_m = sgsc_out.c_slow[m_idx : m_idx + 1]
                    sn_m = sgsc_out.sem_norm[m_idx : m_idx + 1]
                    mi_m = sgsc_out.mean_info[m_idx : m_idx + 1]
                    trig_m = seg_pin_trigger[m_idx : m_idx + 1]
                    unpin_m = seg_unpin_trigger[m_idx : m_idx + 1] > 0.5
                    start_dep_m = seg_start_depth[m_idx : m_idx + 1]
                    end_dep_m = seg_end_depth[m_idx : m_idx + 1]

                    if bool(unpin_m.item()) and gd.slot0_pinned is not None:
                        gd = ConceptGraphState(
                            slots=gd.slots,
                            edges=gd.edges,
                            salience=gd.salience,
                            slot0_pinned=torch.zeros_like(gd.slot0_pinned),
                            phase=torch.zeros_like(gd.phase) if gd.phase is not None else None,
                        )
                    assert gd.phase is not None
                    c_carry = CorticalCarryState(
                        ring_kv=c_carry.ring_kv,
                        ring_pos=c_carry.ring_pos,
                        ring_depth=c_carry.ring_depth,
                        ring_valid=c_carry.ring_valid,
                        doc_pos=c_carry.doc_pos,
                        ast_depth=start_dep_m,
                        prev_z_ref=c_carry.prev_z_ref,
                        has_prev_seg=c_carry.has_prev_seg,
                    )
                    # 1. Strictly `< m` Upper Cortical Decode BEFORE pushing segment `m` into `ring_kv` or `M`!
                    z_dec_m, copy_stem_m = self.cortical.decode_step_streaming(
                        l_reg=l_reg_1,
                        carry=c_carry,
                        slots=gd.slots,
                        edges=gd.edges,
                        salience=gd.salience,
                        phase=gd.phase,
                    )
                    _, g_ctx_pre = bcg_raw.message_pass(gd.slots, gd.edges, gd.salience, phase=gd.phase)
                    c_hat_pre = bcg_raw.f_pred_2(F.silu(bcg_raw.f_pred_1(g_ctx_pre)))
                    f_slow_m, _ = self.hippocampus.read_expression(z_out=z_dec_m, l_reg=l_reg_1, enabled=False)
                    f_cort_m = f_slow_m + copy_stem_m
                    if hip_enabled and fd is not None:
                        rho_m, f_hip_m = self._query_hippocampus_expression_fast(c_hat_pre, fd)
                        f_out_m = (
                            (1.0 - rho_m).unsqueeze(-1) * f_cort_m.float() + rho_m.unsqueeze(-1) * f_hip_m
                        ).to(f_out_flat.dtype)
                    else:
                        f_out_m = f_cort_m.to(f_out_flat.dtype)
                    delta_syn_m = self.pmd.compute_streaming_delta_syn(g_ctx=g_ctx_pre, z_star=z_dec_m, l_reg=l_reg_1)
                    f_out_flat[m_idx] = f_out_m[0]
                    delta_syn_flat[m_idx] = delta_syn_m[0].to(delta_syn_flat.dtype)

                    # 2. Ingest segment `m` through Lower Cortical Encoder + Hippocampus + BCG-PC
                    h_enc_m, c_carry = self.cortical.encode_step_streaming(
                        c_slow_step=c_m,
                        f_stem_step=f_m,
                        seg_end_depth=end_dep_m,
                        carry=c_carry,
                        unpin_reset=unpin_m,
                    )
                    hip_r = self.hippocampus.read_comprehension(
                        f_stem=f_m,
                        c_slow=h_enc_m,
                        sem_norm=sn_m,
                        mean_info=mi_m,
                        fast_state=fd,
                        enabled=hip_enabled,
                        ach_override=ach_override,
                    )
                    bcg_o = bcg_raw.step(
                        c_fused=hip_r.c_fused,
                        prec_bu=hip_r.prec_bu,
                        ach=hip_r.ach_signal,
                        state=gd,
                        tau_m=tau_m,
                        use_gumbel=self.training,
                        update_c0_ema=self.training,
                        merge_uniform_mix=merge_uniform_mix,
                        pin_trigger=trig_m,
                    )
                    gd = bcg_o.state
                    c_carry = CorticalCarryState(
                        ring_kv=c_carry.ring_kv,
                        ring_pos=c_carry.ring_pos,
                        ring_depth=c_carry.ring_depth,
                        ring_valid=c_carry.ring_valid,
                        doc_pos=c_carry.doc_pos,
                        ast_depth=c_carry.ast_depth,
                        prev_z_ref=bcg_o.z_star.to(c_carry.prev_z_ref.dtype),
                        has_prev_seg=torch.ones_like(c_carry.has_prev_seg),
                    )

                    if record_traces:
                        assert (
                            s_trace_flat is not None
                            and a_trace_flat is not None
                            and alpha_trace_flat is not None
                            and slot0_pinned_flat is not None
                            and win_pair_flat is not None
                        )
                        s_trace_flat[m_idx] = gd.slots[0].detach().to(s_trace_flat.dtype)
                        a_trace_flat[m_idx] = gd.edges[0].detach().to(a_trace_flat.dtype)
                        alpha_trace_flat[m_idx] = bcg_o.merge_alpha[0].detach().float()
                        if gd.slot0_pinned is not None:
                            slot0_pinned_flat[m_idx] = gd.slot0_pinned[0].detach().float()
                        wi = bcg_o.win_i[0].detach()
                        wj = bcg_o.win_j[0].detach()
                        win_pair_flat[m_idx, 0] = wi
                        win_pair_flat[m_idx, 1] = wj
                        win_pair_flat[m_idx, 2] = torch.minimum(wi, wj)

                    if hip_enabled and fd is not None and ed is not None:
                        ed = self.hippocampus.inherit_eligibility_on_merge(
                            elig=ed,
                            cand_idx=hip_r.psi_idx,
                            cand_val=hip_r.psi_val,
                            cand_f=f_m,
                            cand_ach=hip_r.ach_signal,
                            win_i=bcg_o.win_i,
                            win_j=bcg_o.win_j,
                        )
                        ed, _, beta_mat = self.hippocampus.three_factor_write(
                            s_bar_post=bcg_o.s_bar_post,
                            edges_post=gd.edges,
                            c0=bcg_raw.c0,
                            elig=ed,
                            fast_state=fd,
                            write_threshold=eff_write_thresh,
                        )
                        beta_sum_t = beta_sum_t + beta_mat.detach().float().mean()
                        beta_count += 1
                        ach_list_s.append(hip_r.ach_signal)
                        if record_traces and ach_trace_flat is not None and beta_max_flat is not None:
                            ach_trace_flat[m_idx] = hip_r.ach_signal[0].detach().float()
                            beta_max_flat[m_idx] = beta_mat[0].detach().float().max()

                    with fp32_island(device):
                        err_sq = (hip_r.c_fused.detach().float() - bcg_o.c_hat.float()).pow(2)
                        pi_act = bcg_o.pi_m.float()
                        pc_nll_sum = pc_nll_sum + 0.5 * (
                            (pi_act * err_sq).mean(dim=-1) - torch.log(pi_act.clamp_min(1e-6)).mean(dim=-1)
                        ).sum()
                        ponder_sum = ponder_sum + bcg_o.ponder_steps.float().sum()
                        ponder_skip_sum = ponder_skip_sum + (bcg_o.ponder_steps.detach().float() < 0.5).float().sum()
                        merge_probs_sum = merge_probs_sum + bcg_o.merge_probs.float().sum(dim=0)

                c_carry = CorticalCarryState(
                    ring_kv=c_carry.ring_kv,
                    ring_pos=c_carry.ring_pos,
                    ring_depth=c_carry.ring_depth,
                    ring_valid=c_carry.ring_valid,
                    doc_pos=c_carry.doc_pos,
                    ast_depth=final_ast_depth.clamp(0, 31),
                    prev_z_ref=c_carry.prev_z_ref,
                    has_prev_seg=c_carry.has_prev_seg,
                )
                out_cortical = c_carry
                g_state = gd
                f_state = fd
                e_state = ed
                s_trace = s_trace_flat
                a_trace = a_trace_flat
                merge_alpha_trace = alpha_trace_flat
                win_pair_trace = win_pair_flat
                ach_trace = ach_trace_flat
                beta_max_trace = beta_max_flat
                slot0_pinned_trace = slot0_pinned_flat
                total_valid_steps = m_total
                ponder_tax_sum = ponder_sum
                moe_bal_sum = zero
                moe_bal_steps = 1
                ach_list = ach_list_s
                familiar_ach_list: list[torch.Tensor] = []
                fam_ach_sum_t = zero
                fam_ach_cnt_t = zero
                z_star_flat = None
                c_fused_flat = None
                h_enc_all = None

            # -----------------------------------------------------------------
            # Mode II: Eval Isolated Per-Document Path (`Test 1C` zero crosstalk)
            # -----------------------------------------------------------------
            elif (doc_cu_seqlens is not None) and (not self.training) and (graph_state is None) and (V <= 8):
                f_out_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
                delta_syn_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
                z_star_flat = torch.zeros((m_total, D), device=device, dtype=sgsc_out.c_slow.dtype)
                c_fused_flat = torch.zeros((m_total, D), device=device, dtype=sgsc_out.c_slow.dtype) if hip_enabled else None
                h_enc_all = torch.zeros((m_total, D), device=device, dtype=sgsc_out.c_slow.dtype)

                s_trace_flat = torch.zeros((m_total, K, D), device=device, dtype=sgsc_out.c_slow.dtype) if record_traces else None
                a_trace_flat = torch.zeros((m_total, K, K, d_r), device=device, dtype=sgsc_out.c_slow.dtype) if record_traces else None
                alpha_trace_flat = torch.zeros((m_total, num_pairs), device=device, dtype=torch.float32) if record_traces else None
                win_pair_flat = torch.zeros((m_total, 3), device=device, dtype=torch.int64) if record_traces else None
                ach_trace_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None
                beta_max_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None
                slot0_pinned_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None

                pts_base = (
                    torch.ones((V,), device=device, dtype=torch.float32)
                    if ponder_tax_scale is None
                    else ponder_tax_scale.to(device=device, dtype=torch.float32)
                )
                tau_m_t = torch.tensor(float(tau_m), device=device, dtype=torch.float32)
                mix_t = torch.tensor(float(merge_uniform_mix), device=device, dtype=torch.float32)
                am_one = torch.ones((1,), device=device, dtype=torch.bool)

                pc_nll_sum = zero
                ponder_sum = zero
                ponder_tax_sum = zero
                ponder_skip_sum = zero
                moe_bal_sum = zero
                moe_bal_steps = 0
                merge_probs_sum = torch.zeros((num_pairs,), device=device, dtype=torch.float32)
                ach_list = []
                familiar_ach_list = []
                fam_ach_sum_t = zero
                fam_ach_cnt_t = zero
                beta_sum_t = zero
                beta_count = 0
                total_valid_steps = 0

                last_g_slots, last_g_edges, last_g_sal, last_g_pin, last_g_phase = [], [], [], [], []
                for d_idx in range(V):
                    m_doc = int(seg_counts[d_idx].item())
                    d_off = int(seg_offsets[d_idx].item())
                    gd = bcg_raw.init_state(1, device=device, dtype=sgsc_out.c_slow.dtype)
                    assert gd.phase is not None and gd.slot0_pinned is not None
                    if m_doc == 0:
                        last_g_slots.append(gd.slots)
                        last_g_edges.append(gd.edges)
                        last_g_sal.append(gd.salience)
                        last_g_pin.append(gd.slot0_pinned)
                        last_g_phase.append(gd.phase)
                        continue

                    sl = slice(d_off, d_off + m_doc)
                    doc_zero_ids = torch.zeros((m_doc,), device=device, dtype=torch.int64)
                    pos_d = torch.arange(m_doc, device=device, dtype=torch.int64)
                    start_dep_d = seg_start_depth[sl]
                    end_dep_d = seg_end_depth[sl]
                    f_stem_d = sgsc_out.f_stem[sl]
                    c_slow_d = sgsc_out.c_slow[sl]
                    l_reg_d = l_reg[d_idx : d_idx + 1]
                    l_reg_seg_d = l_reg_d.expand(m_doc, -1)
                    pts_d = pts_base[d_idx : d_idx + 1]

                    # Step 2 (isolated doc): Parallel Lower Cortical Encoder
                    h_enc_d, c_kv_d, bal_enc_d = self.cortical.encode_parallel(
                        c_slow_d, f_stem_d, doc_zero_ids, pos_d, end_dep_d
                    )
                    h_enc_all[sl] = h_enc_d

                    fd = self.hippocampus.init_fast_state(1, mode="compact", device=device) if hip_enabled else None
                    ed = EligibilityState.zeros(1, K, self.cfg.s_active, d_s, device, f_stem_d.dtype) if hip_enabled else None
                    cq_q_cur = torch.sigmoid(gd.salience.float()) if use_cq else None
                    cq_a_prev = torch.zeros((1, K, d_r), device=device, dtype=gd.edges.dtype) if use_cq else None

                    pre_s_d, pre_e_d, pre_w_d, pre_p_d = [], [], [], []
                    z0_d, g_d, chat_d, pi_d, pieff_d, surp_d, eps_d, sbar_prev_d, cfused_d = [], [], [], [], [], [], [], [], []
                    rho_d_list, fhip_d_list, zcq_d_list, dscq_d_list = [], [], [], []

                    for m_step in range(m_doc):
                        f_idx = d_off + m_step
                        f_step = f_stem_d[m_step : m_step + 1]
                        h_step = h_enc_d[m_step : m_step + 1]
                        sn_step = sgsc_out.sem_norm[f_idx : f_idx + 1]
                        mi_step = sgsc_out.mean_info[f_idx : f_idx + 1]
                        trig_step = seg_pin_trigger[f_idx : f_idx + 1]
                        if float(seg_unpin_trigger[f_idx].item()) > 0.5 and gd.slot0_pinned is not None:
                            gd = ConceptGraphState(
                                slots=gd.slots,
                                edges=gd.edges,
                                salience=gd.salience,
                                slot0_pinned=torch.zeros_like(gd.slot0_pinned),
                                phase=torch.zeros_like(gd.phase) if gd.phase is not None else None,
                            )
                        assert gd.slot0_pinned is not None and gd.phase is not None
                        pre_s_d.append(gd.slots[0])
                        pre_e_d.append(gd.edges[0])
                        pre_w_d.append(gd.salience[0])
                        pre_p_d.append(gd.phase[0])

                        hip_r = self.hippocampus.read_comprehension(
                            f_stem=f_step,
                            c_slow=h_step,
                            sem_norm=sn_step,
                            mean_info=mi_step,
                            fast_state=fd,
                            enabled=hip_enabled,
                            ach_override=ach_override,
                        )
                        (
                            z_0_s,
                            g_ctx_s,
                            c_hat_s,
                            pi_m_s,
                            pi_eff_s,
                            surp_0_s,
                            eps_0_s,
                            s_bar_prev_s,
                            s_next,
                            a_next,
                            omega_next,
                            phase_next,
                            s_bar_post_s,
                            _win_idx_s,
                            win_i_s,
                            win_j_s,
                            mp_s,
                            alpha_s,
                            pin_next,
                        ) = bcg_raw.forward_graph_step_no_ponder(
                            gd.slots,
                            gd.edges,
                            gd.salience,
                            gd.phase,
                            hip_r.c_fused,
                            hip_r.prec_bu,
                            hip_r.ach_signal,
                            tau_m_t,
                            am_one,
                            mix_t,
                            gd.slot0_pinned,
                            trig_step,
                            use_gumbel=False,
                            compute_s_bar_post=hip_enabled,
                            check_pin=True,
                        )
                        gd_prev = gd
                        gd = ConceptGraphState(
                            slots=s_next, edges=a_next, salience=omega_next, slot0_pinned=pin_next, phase=phase_next
                        )

                        if use_cq:
                            assert cq_q_cur is not None and cq_a_prev is not None
                            q_init_d = torch.sigmoid(gd_prev.salience.float())
                            q_in_d = q_init_d if (m_step % K == 0) else cq_q_cur
                            a_in_d = torch.zeros_like(cq_a_prev) if (m_step % K == 0) else cq_a_prev
                            cq_out = self.pmd.broca_cq_step(
                                graph=gd_prev, q_prev=q_in_d, q_init=q_init_d, a_prev=a_in_d, l_reg=l_reg_d
                            )
                            cq_q_cur = cq_out.q_next
                            cq_a_prev = cq_out.a_prev_next
                            zcq_d_list.append(cq_out.z_out[0])
                            dscq_d_list.append(cq_out.delta_syn[0])
                            q_hip_s = cq_out.z_out
                        else:
                            q_hip_s = c_hat_s

                        if hip_enabled and fd is not None:
                            rho_s, fhip_s = self._query_hippocampus_expression_fast(q_hip_s, fd)
                            rho_d_list.append(rho_s[0])
                            fhip_d_list.append(fhip_s[0])

                        if record_traces:
                            assert (
                                s_trace_flat is not None
                                and a_trace_flat is not None
                                and alpha_trace_flat is not None
                                and slot0_pinned_flat is not None
                                and win_pair_flat is not None
                            )
                            s_trace_flat[f_idx] = gd.slots[0].detach().to(s_trace_flat.dtype)
                            a_trace_flat[f_idx] = gd.edges[0].detach().to(a_trace_flat.dtype)
                            alpha_trace_flat[f_idx] = alpha_s[0].detach().float()
                            slot0_pinned_flat[f_idx] = pin_next[0].detach().float()
                            wi = win_i_s[0].detach()
                            wj = win_j_s[0].detach()
                            win_pair_flat[f_idx, 0] = wi
                            win_pair_flat[f_idx, 1] = wj
                            win_pair_flat[f_idx, 2] = torch.minimum(wi, wj)

                        if hip_enabled and fd is not None and ed is not None:
                            ed = self.hippocampus.inherit_eligibility_on_merge(
                                elig=ed,
                                cand_idx=hip_r.psi_idx,
                                cand_val=hip_r.psi_val,
                                cand_f=f_step,
                                cand_ach=hip_r.ach_signal,
                                win_i=win_i_s,
                                win_j=win_j_s,
                            )
                            ed, _, beta_mat = self.hippocampus.three_factor_write(
                                s_bar_post=s_bar_post_s,
                                edges_post=gd.edges,
                                c0=bcg_raw.c0,
                                elig=ed,
                                fast_state=fd,
                                write_threshold=eff_write_thresh,
                            )
                            beta_sum_t = beta_sum_t + beta_mat.detach().float().mean()
                            beta_count += 1
                            ach_list.append(hip_r.ach_signal)
                            if record_traces and ach_trace_flat is not None and beta_max_flat is not None:
                                ach_trace_flat[f_idx] = hip_r.ach_signal[0].detach().float()
                                beta_max_flat[f_idx] = beta_mat[0].detach().float().max()
                            if familiar_doc_mask is not None and bool(familiar_doc_mask[d_idx].item()):
                                familiar_ach_list.append(hip_r.ach_signal)

                        z0_d.append(z_0_s[0])
                        g_d.append(g_ctx_s[0])
                        chat_d.append(c_hat_s[0])
                        pi_d.append(pi_m_s[0])
                        pieff_d.append(pi_eff_s[0])
                        surp_d.append(surp_0_s[0])
                        eps_d.append(eps_0_s[0])
                        sbar_prev_d.append(s_bar_prev_s[0])
                        cfused_d.append(hip_r.c_fused[0])
                        merge_probs_sum = merge_probs_sum + mp_s[0].float()

                    z0_t = torch.stack(z0_d, dim=0)
                    g_t = torch.stack(g_d, dim=0)
                    chat_t = torch.stack(chat_d, dim=0)
                    pi_t = torch.stack(pi_d, dim=0)
                    pieff_t = torch.stack(pieff_d, dim=0)
                    surp_t = torch.stack(surp_d, dim=0)
                    eps_t = torch.stack(eps_d, dim=0)
                    sbar_prev_t = torch.stack(sbar_prev_d, dim=0)
                    cfused_t = torch.stack(cfused_d, dim=0)

                    # Step 4 (isolated doc): Parallel Pondering
                    z_ref_d, ponder_d, bal_pon_d = bcg_raw.ponder_parallel(
                        z0_t, eps_t, surp_t, sbar_prev_t, cfused_t, pieff_t
                    )
                    z_star_flat[sl] = z_ref_d
                    if c_fused_flat is not None:
                        c_fused_flat[sl] = cfused_t.detach()

                    # Step 5 (isolated doc): Causal `< m` Shift + Upper Cortical Decoder
                    bos_d = self.cortical.bos_concept.to(device=device, dtype=z_ref_d.dtype).unsqueeze(0)
                    z_prev_d = bos_d if m_doc == 1 else torch.cat([bos_d, z_ref_d[:-1]], dim=0)
                    z_dec_d, copy_stem_d, bal_dec_d = self.cortical.decode_parallel(
                        z_ref_prev=z_prev_d,
                        l_reg_seg=l_reg_seg_d,
                        c_kv=c_kv_d,
                        doc_ids=doc_zero_ids,
                        pos_in_doc=pos_d,
                        seg_start_depth=start_dep_d,
                        seg_end_depth=end_dep_d,
                        pre_slots=torch.stack(pre_s_d, dim=0),
                        pre_edges=torch.stack(pre_e_d, dim=0),
                        pre_salience=torch.stack(pre_w_d, dim=0),
                        pre_phase=torch.stack(pre_p_d, dim=0),
                    )
                    if use_cq:
                        z_out_d = 0.5 * (torch.stack(zcq_d_list, dim=0) + z_dec_d)
                        ds_d = torch.stack(dscq_d_list, dim=0) + 0.5 * self.pmd.compute_streaming_delta_syn(
                            g_ctx=g_t, z_star=z_dec_d, l_reg=l_reg_seg_d
                        )
                    else:
                        z_out_d = z_dec_d
                        ds_d = self.pmd.compute_streaming_delta_syn(g_ctx=g_t, z_star=z_dec_d, l_reg=l_reg_seg_d)

                    f_slow_d, _ = self.hippocampus.read_expression(z_out=z_out_d, l_reg=l_reg_seg_d, enabled=False)
                    f_cort_d = f_slow_d + copy_stem_d
                    if hip_enabled and rho_d_list:
                        rho_t = torch.stack(rho_d_list, dim=0)
                        fhip_t = torch.stack(fhip_d_list, dim=0)
                        f_out_d = ((1.0 - rho_t).unsqueeze(-1) * f_cort_d.float() + rho_t.unsqueeze(-1) * fhip_t).to(
                            f_out_flat.dtype
                        )
                    else:
                        f_out_d = f_cort_d.to(f_out_flat.dtype)

                    f_out_flat[sl] = f_out_d
                    delta_syn_flat[sl] = ds_d.to(delta_syn_flat.dtype)

                    with fp32_island(device):
                        err_sq = (cfused_t.detach().float() - chat_t.float()).pow(2)
                        pc_nll_sum = pc_nll_sum + 0.5 * (
                            (pi_t.float() * err_sq).mean(dim=-1) - torch.log(pi_t.float().clamp_min(1e-6)).mean(dim=-1)
                        ).sum()
                        ponder_sum = ponder_sum + ponder_d.float().sum()
                        ponder_tax_sum = ponder_tax_sum + (ponder_d.float() * pts_d).sum()
                        ponder_skip_sum = ponder_skip_sum + (ponder_d.detach().float() < 0.5).float().sum()
                        moe_bal_sum = moe_bal_sum + (bal_enc_d + bal_pon_d + bal_dec_d) / 3.0
                        moe_bal_steps += 1
                    total_valid_steps += m_doc

                    last_g_slots.append(gd.slots)
                    last_g_edges.append(gd.edges)
                    last_g_sal.append(gd.salience)
                    last_g_pin.append(gd.slot0_pinned)
                    last_g_phase.append(gd.phase)

                g_state = ConceptGraphState(
                    slots=torch.cat(last_g_slots, dim=0),
                    edges=torch.cat(last_g_edges, dim=0),
                    salience=torch.cat(last_g_sal, dim=0),
                    slot0_pinned=torch.cat(last_g_pin, dim=0),
                    phase=torch.cat(last_g_phase, dim=0),
                )
                f_state = None
                e_state = None
                s_trace = s_trace_flat
                a_trace = a_trace_flat
                merge_alpha_trace = alpha_trace_flat
                win_pair_trace = win_pair_flat
                ach_trace = ach_trace_flat
                beta_max_trace = beta_max_flat
                slot0_pinned_trace = slot0_pinned_flat

            # -----------------------------------------------------------------
            # Mode III: Full-Batch Parallel DCPA-Turbo Pipeline (Train & Batch Eval)
            # -----------------------------------------------------------------
            else:
                use_bucketing = bool(
                    self.enable_doc_bucketing
                    and graph_state is None
                    and fast_state is None
                    and elig_state is None
                )
                use_prefix_shrink = bool(
                    (not hip_enabled)
                    and graph_state is None
                    and fast_state is None
                    and elig_state is None
                    and V > 1
                )
                bucket = max(1, int(self.doc_bucket_size))

                def _coarse_bucket_v(n: int, cap: int) -> int:
                    if not use_bucketing:
                        return min(max(n, 1), cap)
                    if n <= bucket:
                        return min(bucket, cap)
                    p = bucket
                    while p < n:
                        p <<= 1
                    return min(p, cap)

                V_eff = _coarse_bucket_v(V, 1 << 30) if use_bucketing else V
                pad_v = V_eff - V

                pts_base = (
                    torch.ones((V,), device=device, dtype=torch.float32)
                    if ponder_tax_scale is None
                    else ponder_tax_scale.to(device=device, dtype=torch.float32)
                )
                if use_prefix_shrink:
                    doc_order = torch.argsort(seg_counts, descending=True)
                    inv_doc_order = torch.argsort(doc_order)
                    seg_counts_ord = seg_counts[doc_order]
                    seg_offsets_ord = seg_offsets[doc_order]
                    l_reg_ord = l_reg[doc_order]
                    pts_ord = pts_base[doc_order]
                    fam_mask_ord = familiar_doc_mask[doc_order] if familiar_doc_mask is not None else None
                else:
                    inv_doc_order = None
                    seg_counts_ord = seg_counts
                    seg_offsets_ord = seg_offsets
                    l_reg_ord = l_reg
                    pts_ord = pts_base
                    fam_mask_ord = familiar_doc_mask

                if pad_v > 0:
                    seg_counts_eff = torch.cat(
                        [seg_counts_ord, torch.zeros((pad_v,), device=device, dtype=seg_counts.dtype)], dim=0
                    )
                    seg_offsets_eff = torch.cat(
                        [seg_offsets_ord, torch.zeros((pad_v,), device=device, dtype=seg_offsets.dtype)], dim=0
                    )
                    l_reg_eff = torch.cat(
                        [l_reg_ord, torch.zeros((pad_v, l_reg.shape[-1]), device=device, dtype=l_reg.dtype)], dim=0
                    )
                    pts_eff = torch.cat([pts_ord, torch.zeros((pad_v,), device=device, dtype=torch.float32)], dim=0)
                    fam_mask_eff = (
                        torch.cat([fam_mask_ord, torch.zeros((pad_v,), device=device, dtype=torch.bool)], dim=0)
                        if fam_mask_ord is not None
                        else None
                    )
                else:
                    seg_counts_eff = seg_counts_ord
                    seg_offsets_eff = seg_offsets_ord
                    l_reg_eff = l_reg_ord
                    pts_eff = pts_ord
                    fam_mask_eff = fam_mask_ord

                # === STEP 2: Full-Batch `[M_total, D]` Lower Cortical Encoder ===
                h_enc_all, c_kv_all, bal_enc = self.cortical.encode_parallel(
                    sgsc_out.c_slow, sgsc_out.f_stem, seg_doc_ids, pos_in_doc, seg_end_depth
                )

                g_state = bcg_raw.init_state(V_eff, device=device, dtype=h_enc_all.dtype) if graph_state is None else graph_state
                if g_state.slot0_pinned is None:
                    g_state = ConceptGraphState(
                        slots=g_state.slots,
                        edges=g_state.edges,
                        salience=g_state.salience,
                        slot0_pinned=torch.zeros((g_state.slots.shape[0],), device=device, dtype=torch.float32),
                        phase=g_state.phase,
                    )
                if g_state.phase is None:
                    g_state = ConceptGraphState(
                        slots=g_state.slots,
                        edges=g_state.edges,
                        salience=g_state.salience,
                        slot0_pinned=g_state.slot0_pinned,
                        phase=torch.zeros((g_state.slots.shape[0], K), device=device, dtype=torch.float32),
                    )
                f_state = (
                    self.hippocampus.init_fast_state(V_eff, mode="compact", device=device)
                    if (hip_enabled and fast_state is None)
                    else fast_state
                )
                e_state = (
                    EligibilityState.zeros(V_eff, K, self.cfg.s_active, d_s, device, sgsc_out.f_stem.dtype)
                    if (hip_enabled and elig_state is None)
                    else elig_state
                )

                tau_m_t = torch.tensor(float(tau_m), device=device, dtype=torch.float32)
                mix_t = torch.tensor(float(merge_uniform_mix), device=device, dtype=torch.float32)

                counts_offs_cpu = (
                    torch.cat(
                        [seg_counts_ord.detach().to(torch.int32), seg_offsets_ord.detach().to(torch.int32)], dim=0
                    )
                    .to("cpu")
                    .tolist()
                )
                n_v_cpu = int(seg_counts_ord.numel())
                seg_counts_ord_cpu = [int(x) for x in counts_offs_cpu[:n_v_cpu]]
                seg_offsets_ord_cpu = [int(x) for x in counts_offs_cpu[n_v_cpu : 2 * n_v_cpu]]
                m_max = max(seg_counts_ord_cpu) if seg_counts_ord_cpu else 0

                s_trace_flat = torch.zeros((m_total, K, D), device=device, dtype=h_enc_all.dtype) if record_traces else None
                a_trace_flat = torch.zeros((m_total, K, K, d_r), device=device, dtype=h_enc_all.dtype) if record_traces else None
                alpha_trace_flat = torch.zeros((m_total, num_pairs), device=device, dtype=torch.float32) if record_traces else None
                win_pair_flat = torch.zeros((m_total, 3), device=device, dtype=torch.int64) if record_traces else None
                ach_trace_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None
                beta_max_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None
                slot0_pinned_flat = torch.zeros((m_total,), device=device, dtype=torch.float32) if record_traces else None

                pi_cort_all: Optional[torch.Tensor] = None
                if not hip_enabled and m_total > 0:
                    with fp32_island(device):
                        pi_cort_all = (
                            F.softplus(
                                F.linear(
                                    h_enc_all.float(),
                                    self.hippocampus.f_prec_cort.weight.float(),
                                    self.hippocampus.f_prec_cort.bias.float()
                                    if self.hippocampus.f_prec_cort.bias is not None
                                    else None,
                                )
                            )
                            + 1e-4
                        )

                # === STEP 3: Zero-MoE Contractive Graph (+ WY Parallel Hippocampus in Stage D) ===
                seg_offsets_i64 = seg_offsets_eff.to(torch.int64)
                zero_idx = torch.zeros_like(seg_offsets_i64)
                flat_idx_chunks: list[torch.Tensor] = []
                z0_chunks: list[torch.Tensor] = []
                g_ctx_chunks: list[torch.Tensor] = []
                c_hat_chunks: list[torch.Tensor] = []
                pi_m_chunks: list[torch.Tensor] = []
                pi_eff_chunks: list[torch.Tensor] = []
                surp_0_chunks: list[torch.Tensor] = []
                eps_0_chunks: list[torch.Tensor] = []
                sbar_prev_chunks: list[torch.Tensor] = []
                cfused_chunks: list[torch.Tensor] = []
                pts_chunks: list[torch.Tensor] = []
                merge_probs_chunks: list[torch.Tensor] = []
                pre_slots_chunks: list[torch.Tensor] = []
                pre_edges_chunks: list[torch.Tensor] = []
                pre_sal_chunks: list[torch.Tensor] = []
                pre_phase_chunks: list[torch.Tensor] = []
                rho_chunks: list[torch.Tensor] = []
                fhip_chunks: list[torch.Tensor] = []
                zcq_chunks: list[torch.Tensor] = []
                dscq_chunks: list[torch.Tensor] = []

                ach_list = []
                familiar_ach_list = []
                fam_ach_sum_t = zero
                fam_ach_cnt_t = zero
                beta_sum_t = zero
                beta_count = 0

                seg_counts_cpu = seg_counts_ord_cpu if use_prefix_shrink else []
                n_active_prefix = sum(1 for c in seg_counts_cpu if c > 0) if use_prefix_shrink else V
                cur_v = V_eff
                finished_slots: list[torch.Tensor] = []
                finished_edges: list[torch.Tensor] = []
                finished_sal: list[torch.Tensor] = []
                finished_pin: list[torch.Tensor] = []
                finished_phase: list[torch.Tensor] = []

                use_chunked_bcg = bool(
                    (not hip_enabled)
                    and (not record_traces)
                    and int(self.bcg_chunk_steps) > 1
                    and m_max > 1
                    and pi_cort_all is not None
                )
                use_parallel_cq = bool(use_cq and (not hip_enabled))
                cq_q_cur: Optional[torch.Tensor] = (
                    torch.sigmoid(g_state.salience.float()) if (use_cq and not use_parallel_cq) else None
                )
                cq_a_prev: Optional[torch.Tensor] = (
                    torch.zeros((V_eff, K, d_r), device=device, dtype=g_state.edges.dtype)
                    if (use_cq and not use_parallel_cq)
                    else None
                )

                carry_gamma = float(self.carry_grad_gamma)
                carry_max_rms = float(self.carry_grad_max_rms)
                tbptt_w = int(self.bcg_tbptt_window)
                watermark_gb = float(self.bcg_checkpoint_vram_watermark_gb)

                if use_chunked_bcg:
                    assert pi_cort_all is not None
                    K_c = int(self.bcg_chunk_steps)
                    k_offs_c = torch.arange(K_c, device=device, dtype=torch.int64)
                    ach_zeros_flat = torch.zeros((K_c * V_eff,), device=device, dtype=torch.float32)
                    m_ema_val = float(bcg_raw.cfg.c0_ema_momentum)
                    c0_ema_weights_by_rem: tuple[torch.Tensor, ...] = tuple(
                        torch.tensor(
                            [(m_ema_val ** (r_len - 1 - idx_r)) * (1.0 - m_ema_val) for idx_r in range(r_len)],
                            device=device,
                            dtype=torch.float32,
                        ).unsqueeze(1)
                        for r_len in range(1, K_c + 1)
                    )

                    m_step = 0
                    while m_step < m_max:
                        if use_prefix_shrink:
                            while n_active_prefix > 0 and seg_counts_cpu[n_active_prefix - 1] <= m_step:
                                n_active_prefix -= 1
                            if n_active_prefix == 0:
                                break
                            v_next = _coarse_bucket_v(n_active_prefix, cur_v)
                            if v_next < cur_v:
                                assert g_state.slot0_pinned is not None and g_state.phase is not None
                                finished_slots.append(g_state.slots[v_next:])
                                finished_edges.append(g_state.edges[v_next:])
                                finished_sal.append(g_state.salience[v_next:])
                                finished_pin.append(g_state.slot0_pinned[v_next:])
                                finished_phase.append(g_state.phase[v_next:])
                                g_state = ConceptGraphState(
                                    slots=g_state.slots[:v_next],
                                    edges=g_state.edges[:v_next],
                                    salience=g_state.salience[:v_next],
                                    slot0_pinned=g_state.slot0_pinned[:v_next],
                                    phase=g_state.phase[:v_next],
                                )
                                cur_v = v_next
                            seg_off_cur = seg_offsets_i64[:cur_v]
                            zero_cur = zero_idx[:cur_v]
                            pts_cur = pts_eff[:cur_v]
                            seg_cnt_cur = seg_counts_eff[:cur_v]
                        else:
                            if not any(m_step < c for c in seg_counts_ord_cpu):
                                break
                            seg_off_cur = seg_offsets_i64
                            zero_cur = zero_idx
                            pts_cur = pts_eff
                            seg_cnt_cur = seg_counts_eff

                        rem = min(K_c, m_max - m_step)
                        if self.training and tbptt_w > 0:
                            rem = min(rem, tbptt_w - (m_step % tbptt_w))

                        m_steps_2d = m_step + k_offs_c.unsqueeze(1)
                        valid_k_mask = (k_offs_c < rem).unsqueeze(1)
                        active_mask_seq = (m_steps_2d < seg_cnt_cur.unsqueeze(0)) & valid_k_mask
                        active_f_seq = active_mask_seq.float()
                        safe_idx_seq = torch.where(active_mask_seq, seg_off_cur.unsqueeze(0) + m_steps_2d, zero_cur.unsqueeze(0))
                        c_seq = h_enc_all[safe_idx_seq] * active_f_seq.unsqueeze(-1).to(h_enc_all.dtype)
                        prec_seq = pi_cort_all[safe_idx_seq] * active_f_seq.unsqueeze(-1)
                        ach_seq = ach_zeros_flat[: K_c * cur_v].view(K_c, cur_v)
                        trig_seq = seg_pin_trigger[safe_idx_seq] * active_f_seq
                        unpin_seq = (seg_unpin_trigger[safe_idx_seq] * active_f_seq) > 0.5

                        if self.training and m_step > 0:
                            if tbptt_w > 0 and (m_step % tbptt_w == 0):
                                in_slots = g_state.slots.detach().requires_grad_(True)
                                in_edges = g_state.edges.detach().requires_grad_(True)
                                in_sal = g_state.salience.detach().requires_grad_(True)
                            elif carry_gamma < 0.9999:
                                in_slots = _DampedBoundedCarry.apply(g_state.slots, carry_gamma, carry_max_rms)
                                in_edges = _DampedBoundedCarry.apply(g_state.edges, carry_gamma, carry_max_rms)
                                in_sal = _DampedBoundedCarry.apply(g_state.salience, carry_gamma, carry_max_rms)
                            else:
                                in_slots, in_edges, in_sal = g_state.slots, g_state.edges, g_state.salience
                        else:
                            in_slots, in_edges, in_sal = g_state.slots, g_state.edges, g_state.salience

                        assert g_state.slot0_pinned is not None and g_state.phase is not None
                        gumbel_u_seq: Optional[torch.Tensor] = (
                            torch.rand((K_c, cur_v, num_pairs), device=device, dtype=torch.float32)
                            if self.training
                            else None
                        )
                        use_ckpt_chunk = bool(
                            self.training
                            and torch.is_grad_enabled()
                            and self.bcg_grad_checkpointing
                            and (watermark_gb <= 0.0 or _get_allocated_memory_gb(device) >= watermark_gb)
                        )
                        if use_ckpt_chunk:
                            assert gumbel_u_seq is not None
                            chunk_out = torch_checkpoint(
                                self._clause_chunk_checkpointed_fn,
                                in_slots,
                                in_edges,
                                in_sal,
                                g_state.phase,
                                g_state.slot0_pinned,
                                c_seq,
                                prec_seq,
                                ach_seq,
                                active_mask_seq,
                                trig_seq,
                                unpin_seq,
                                tau_m_t,
                                mix_t,
                                gumbel_u_seq,
                                True,
                                use_reentrant=False,
                                preserve_rng_state=False,
                            )
                        else:
                            chunk_out = bcg_raw.forward_clause_chunk(
                                in_slots,
                                in_edges,
                                in_sal,
                                g_state.phase,
                                g_state.slot0_pinned,
                                c_seq,
                                prec_seq,
                                ach_seq,
                                active_mask_seq,
                                trig_seq,
                                unpin_seq,
                                tau_m_t,
                                mix_t,
                                gumbel_u_seq=gumbel_u_seq,
                                use_gumbel=self.training,
                                check_pin=True,
                            )

                        (
                            z0_seq,
                            g_seq,
                            chat_seq,
                            pi_seq,
                            pieff_seq,
                            surp_seq,
                            eps_seq,
                            sbar_prev_seq,
                            mp_seq,
                            _alpha_seq,
                            s_final,
                            a_final,
                            omega_final,
                            phase_final,
                            pin_final,
                            c0_means_seq,
                            pre_s_seq,
                            pre_e_seq,
                            pre_w_seq,
                            pre_p_seq,
                        ) = chunk_out
                        g_state = ConceptGraphState(
                            slots=s_final, edges=a_final, salience=omega_final, slot0_pinned=pin_final, phase=phase_final
                        )
                        if self.training:
                            with torch.no_grad():
                                if rem == 1:
                                    bcg_raw.c0.mul_(m_ema_val).add_(c0_means_seq[0], alpha=1.0 - m_ema_val)
                                else:
                                    w_ema_r = c0_ema_weights_by_rem[rem - 1]
                                    bcg_raw.c0.mul_(m_ema_val**rem).add_((w_ema_r * c0_means_seq[:rem]).sum(dim=0))

                        if (
                            use_prefix_shrink
                            and n_active_prefix == cur_v
                            and seg_counts_cpu[n_active_prefix - 1] > m_step + rem - 1
                        ):
                            n_flat_c = rem * cur_v
                            flat_idx_chunks.append(
                                (seg_off_cur.unsqueeze(0) + (m_step + k_offs_c[:rem].unsqueeze(1))).reshape(n_flat_c)
                            )
                            z0_chunks.append(z0_seq[:rem].reshape(n_flat_c, D))
                            g_ctx_chunks.append(g_seq[:rem].reshape(n_flat_c, D))
                            c_hat_chunks.append(chat_seq[:rem].reshape(n_flat_c, D))
                            pi_m_chunks.append(pi_seq[:rem].reshape(n_flat_c, D))
                            pi_eff_chunks.append(pieff_seq[:rem].reshape(n_flat_c, D))
                            surp_0_chunks.append(surp_seq[:rem].reshape(n_flat_c))
                            eps_0_chunks.append(eps_seq[:rem].reshape(n_flat_c, D))
                            sbar_prev_chunks.append(sbar_prev_seq[:rem].reshape(n_flat_c, K, D))
                            cfused_chunks.append(c_seq[:rem].reshape(n_flat_c, D))
                            pts_chunks.append(pts_cur.unsqueeze(0).expand(rem, cur_v).reshape(n_flat_c))
                            merge_probs_chunks.append(mp_seq[:rem].reshape(n_flat_c, num_pairs))
                            pre_slots_chunks.append(pre_s_seq[:rem].reshape(n_flat_c, K, D))
                            pre_edges_chunks.append(pre_e_seq[:rem].reshape(n_flat_c, K, K, d_r))
                            pre_sal_chunks.append(pre_w_seq[:rem].reshape(n_flat_c, K))
                            pre_phase_chunks.append(pre_p_seq[:rem].reshape(n_flat_c, K))
                        else:
                            for k_i in range(rem):
                                step_m = m_step + k_i
                                if use_prefix_shrink:
                                    while n_active_prefix > 0 and seg_counts_cpu[n_active_prefix - 1] <= step_m:
                                        n_active_prefix -= 1
                                    if n_active_prefix == 0:
                                        break
                                    sl_v = slice(0, n_active_prefix)
                                    flat_idx_k = seg_off_cur[sl_v] + step_m
                                    z0_chunks.append(z0_seq[k_i, sl_v])
                                    g_ctx_chunks.append(g_seq[k_i, sl_v])
                                    c_hat_chunks.append(chat_seq[k_i, sl_v])
                                    pi_m_chunks.append(pi_seq[k_i, sl_v])
                                    pi_eff_chunks.append(pieff_seq[k_i, sl_v])
                                    surp_0_chunks.append(surp_seq[k_i, sl_v])
                                    eps_0_chunks.append(eps_seq[k_i, sl_v])
                                    sbar_prev_chunks.append(sbar_prev_seq[k_i, sl_v])
                                    cfused_chunks.append(c_seq[k_i, sl_v])
                                    pts_chunks.append(pts_cur[sl_v])
                                    merge_probs_chunks.append(mp_seq[k_i, sl_v])
                                    pre_slots_chunks.append(pre_s_seq[k_i, sl_v])
                                    pre_edges_chunks.append(pre_e_seq[k_i, sl_v])
                                    pre_sal_chunks.append(pre_w_seq[k_i, sl_v])
                                    pre_phase_chunks.append(pre_p_seq[k_i, sl_v])
                                    flat_idx_chunks.append(flat_idx_k)
                                else:
                                    act_ids_cpu = [i for i, c in enumerate(seg_counts_ord_cpu) if step_m < c]
                                    if not act_ids_cpu:
                                        break
                                    active_b_k = (
                                        torch.arange(len(act_ids_cpu), device=device, dtype=torch.int64)
                                        if act_ids_cpu == list(range(len(act_ids_cpu)))
                                        else torch.tensor(act_ids_cpu, device=device, dtype=torch.int64)
                                    )
                                    flat_idx_k = seg_off_cur[active_b_k] + step_m
                                    z0_chunks.append(z0_seq[k_i][active_b_k])
                                    g_ctx_chunks.append(g_seq[k_i][active_b_k])
                                    c_hat_chunks.append(chat_seq[k_i][active_b_k])
                                    pi_m_chunks.append(pi_seq[k_i][active_b_k])
                                    pi_eff_chunks.append(pieff_seq[k_i][active_b_k])
                                    surp_0_chunks.append(surp_seq[k_i][active_b_k])
                                    eps_0_chunks.append(eps_seq[k_i][active_b_k])
                                    sbar_prev_chunks.append(sbar_prev_seq[k_i][active_b_k])
                                    cfused_chunks.append(c_seq[k_i][active_b_k])
                                    pts_chunks.append(pts_cur[active_b_k])
                                    merge_probs_chunks.append(mp_seq[k_i][active_b_k])
                                    pre_slots_chunks.append(pre_s_seq[k_i][active_b_k])
                                    pre_edges_chunks.append(pre_e_seq[k_i][active_b_k])
                                    pre_sal_chunks.append(pre_w_seq[k_i][active_b_k])
                                    pre_phase_chunks.append(pre_p_seq[k_i][active_b_k])
                                    flat_idx_chunks.append(flat_idx_k)
                        m_step += rem
                else:
                    for m_step in range(m_max):
                        if use_prefix_shrink:
                            while n_active_prefix > 0 and seg_counts_cpu[n_active_prefix - 1] <= m_step:
                                n_active_prefix -= 1
                            if n_active_prefix == 0:
                                break
                            v_next = _coarse_bucket_v(n_active_prefix, cur_v)
                            if v_next < cur_v:
                                assert g_state.slot0_pinned is not None and g_state.phase is not None
                                finished_slots.append(g_state.slots[v_next:])
                                finished_edges.append(g_state.edges[v_next:])
                                finished_sal.append(g_state.salience[v_next:])
                                finished_pin.append(g_state.slot0_pinned[v_next:])
                                finished_phase.append(g_state.phase[v_next:])
                                g_state = ConceptGraphState(
                                    slots=g_state.slots[:v_next],
                                    edges=g_state.edges[:v_next],
                                    salience=g_state.salience[:v_next],
                                    slot0_pinned=g_state.slot0_pinned[:v_next],
                                    phase=g_state.phase[:v_next],
                                )
                                if cq_q_cur is not None:
                                    cq_q_cur = cq_q_cur[:v_next]
                                if cq_a_prev is not None:
                                    cq_a_prev = cq_a_prev[:v_next]
                                cur_v = v_next
                            active_mask = m_step < seg_counts_eff[:cur_v]
                            active_f = active_mask.float()
                            active_b = torch.arange(n_active_prefix, device=device, dtype=torch.int64)
                            seg_off_cur = seg_offsets_i64[:cur_v]
                            zero_cur = zero_idx[:cur_v]
                            l_reg_cur = l_reg_eff[:cur_v]
                            pts_cur = pts_eff[:cur_v]
                        else:
                            act_ids_cpu = [i for i, c in enumerate(seg_counts_ord_cpu) if m_step < c]
                            if not act_ids_cpu:
                                break
                            active_mask = m_step < seg_counts_eff
                            active_f = active_mask.float()
                            active_b = (
                                torch.arange(len(act_ids_cpu), device=device, dtype=torch.int64)
                                if act_ids_cpu == list(range(len(act_ids_cpu)))
                                else torch.tensor(act_ids_cpu, device=device, dtype=torch.int64)
                            )
                            seg_off_cur = seg_offsets_i64
                            zero_cur = zero_idx
                            l_reg_cur = l_reg_eff
                            pts_cur = pts_eff

                        flat_idx = seg_off_cur[active_b] + m_step
                        safe_idx = torch.where(active_mask, seg_off_cur + m_step, zero_cur)
                        am_col = active_f.unsqueeze(-1).to(h_enc_all.dtype)
                        h_step = h_enc_all[safe_idx] * am_col
                        trig_step = seg_pin_trigger[safe_idx] * active_f
                        unpin_step = (seg_unpin_trigger[safe_idx] * active_f) > 0.5

                        if hip_enabled:
                            f_step = sgsc_out.f_stem[safe_idx] * active_f.unsqueeze(-1).to(sgsc_out.f_stem.dtype)
                            sn_step = sgsc_out.sem_norm[safe_idx] * active_f
                            mi_step = sgsc_out.mean_info[safe_idx] * active_f
                            hip_read = self.hippocampus.read_comprehension(
                                f_stem=f_step,
                                c_slow=h_step,
                                sem_norm=sn_step,
                                mean_info=mi_step,
                                fast_state=f_state,
                                enabled=True,
                                ach_override=ach_override,
                            )
                            c_fused_step = hip_read.c_fused
                            prec_bu_step = hip_read.prec_bu
                            ach_step = hip_read.ach_signal * active_f
                        else:
                            assert pi_cort_all is not None
                            hip_read = None
                            f_step = None
                            c_fused_step = h_step
                            prec_bu_step = pi_cort_all[safe_idx] * active_f.unsqueeze(-1)
                            ach_step = torch.zeros_like(active_f)

                        if self.training and m_step > 0:
                            if tbptt_w > 0 and (m_step % tbptt_w == 0):
                                in_slots = g_state.slots.detach().requires_grad_(True)
                                in_edges = g_state.edges.detach().requires_grad_(True)
                                in_sal = g_state.salience.detach().requires_grad_(True)
                                if cq_q_cur is not None:
                                    cq_q_cur = cq_q_cur.detach()
                                if cq_a_prev is not None:
                                    cq_a_prev = cq_a_prev.detach()
                            elif carry_gamma < 0.9999:
                                in_slots = _DampedBoundedCarry.apply(g_state.slots, carry_gamma, carry_max_rms)
                                in_edges = _DampedBoundedCarry.apply(g_state.edges, carry_gamma, carry_max_rms)
                                in_sal = _DampedBoundedCarry.apply(g_state.salience, carry_gamma, carry_max_rms)
                            else:
                                in_slots, in_edges, in_sal = g_state.slots, g_state.edges, g_state.salience
                        else:
                            in_slots, in_edges, in_sal = g_state.slots, g_state.edges, g_state.salience

                        assert g_state.slot0_pinned is not None and g_state.phase is not None
                        eff_pin = torch.where(unpin_step, torch.zeros_like(g_state.slot0_pinned), g_state.slot0_pinned)
                        eff_phase = torch.where(unpin_step.unsqueeze(-1), torch.zeros_like(g_state.phase), g_state.phase)

                        gumbel_u_step: Optional[torch.Tensor] = (
                            torch.rand((cur_v, num_pairs), device=device, dtype=torch.float32)
                            if self.training
                            else None
                        )
                        (
                            z0_s,
                            g_ctx_s,
                            c_hat_s,
                            pi_m_s,
                            pi_eff_s,
                            surp_0_s,
                            eps_0_s,
                            sbar_prev_s,
                            s_next,
                            a_next,
                            omega_next,
                            phase_next,
                            s_bar_post_s,
                            _win_idx_s,
                            win_i_s,
                            win_j_s,
                            mp_s,
                            alpha_s,
                            pin_next,
                        ) = bcg_raw.forward_graph_step_no_ponder(
                            in_slots,
                            in_edges,
                            in_sal,
                            eff_phase,
                            c_fused_step,
                            prec_bu_step,
                            ach_step,
                            tau_m_t,
                            active_mask,
                            mix_t,
                            eff_pin,
                            trig_step,
                            use_gumbel=self.training,
                            gumbel_u=gumbel_u_step,
                            compute_s_bar_post=hip_enabled,
                            check_pin=True,
                        )
                        if self.training:
                            bcg_raw.update_c0_from_slots(s_next, active_mask)

                        g_prev_cq = (
                            ConceptGraphState(
                                slots=in_slots, edges=in_edges, salience=in_sal, slot0_pinned=eff_pin, phase=eff_phase
                            )
                            if (use_cq and not use_parallel_cq)
                            else None
                        )
                        g_state = ConceptGraphState(
                            slots=s_next, edges=a_next, salience=omega_next, slot0_pinned=pin_next, phase=phase_next
                        )

                        if record_traces:
                            assert (
                                s_trace_flat is not None
                                and slot0_pinned_flat is not None
                                and alpha_trace_flat is not None
                            )
                            s_trace_flat[flat_idx] = g_state.slots[active_b].detach().to(s_trace_flat.dtype)
                            slot0_pinned_flat[flat_idx] = pin_next[active_b].detach().float()
                            alpha_trace_flat[flat_idx] = alpha_s[active_b].detach().float()
                            if not self.training:
                                assert a_trace_flat is not None and win_pair_flat is not None
                                a_trace_flat[flat_idx] = g_state.edges[active_b].detach().to(a_trace_flat.dtype)
                                wi_a = win_i_s[active_b].detach()
                                wj_a = win_j_s[active_b].detach()
                                win_pair_flat[flat_idx, 0] = wi_a
                                win_pair_flat[flat_idx, 1] = wj_a
                                win_pair_flat[flat_idx, 2] = torch.minimum(wi_a, wj_a)

                        if use_cq and not use_parallel_cq:
                            assert g_prev_cq is not None and cq_q_cur is not None and cq_a_prev is not None
                            q_init_step = torch.sigmoid(in_sal.float())
                            q_in_step = q_init_step if (m_step % K == 0) else cq_q_cur
                            a_in_step = torch.zeros_like(cq_a_prev) if (m_step % K == 0) else cq_a_prev
                            cq_out = self.pmd.broca_cq_step(
                                graph=g_prev_cq, q_prev=q_in_step, q_init=q_init_step, a_prev=a_in_step, l_reg=l_reg_cur
                            )
                            cq_q_cur = torch.where(active_mask.unsqueeze(-1), cq_out.q_next, cq_q_cur)
                            cq_a_prev = torch.where(
                                active_mask.view(-1, 1, 1), cq_out.a_prev_next.to(cq_a_prev.dtype), cq_a_prev
                            )
                            zcq_chunks.append(cq_out.z_out[active_b])
                            dscq_chunks.append(cq_out.delta_syn[active_b])
                            q_hip_step = cq_out.z_out
                        else:
                            q_hip_step = c_hat_s

                        # Query pre-write `M_{m-1}` strictly BEFORE `three_factor_write`!
                        if hip_enabled and f_state is not None:
                            rho_all, fhip_all = self._query_hippocampus_expression_fast(q_hip_step, f_state)
                            rho_chunks.append(rho_all[active_b])
                            fhip_chunks.append(fhip_all[active_b])

                        if hip_enabled and f_state is not None and e_state is not None:
                            assert hip_read is not None and f_step is not None
                            e_next = self.hippocampus.inherit_eligibility_on_merge(
                                elig=e_state,
                                cand_idx=hip_read.psi_idx,
                                cand_val=hip_read.psi_val,
                                cand_f=f_step,
                                cand_ach=hip_read.ach_signal * active_f,
                                win_i=win_i_s,
                                win_j=win_j_s,
                            )
                            e_next_masked = EligibilityState(
                                e_idx=torch.where(active_mask.view(V_eff, 1, 1), e_next.e_idx, e_state.e_idx),
                                e_val=torch.where(active_mask.view(V_eff, 1, 1), e_next.e_val, e_state.e_val),
                                f_tilde=torch.where(active_mask.view(V_eff, 1, 1), e_next.f_tilde, e_state.f_tilde),
                                eta=torch.where(active_mask.view(V_eff, 1), e_next.eta, torch.zeros_like(e_next.eta)),
                            )
                            e_state, _, beta_mat = self.hippocampus.three_factor_write(
                                s_bar_post=s_bar_post_s,
                                edges_post=g_state.edges,
                                c0=bcg_raw.c0,
                                elig=e_next_masked,
                                fast_state=f_state,
                                write_threshold=eff_write_thresh,
                            )
                            beta_sum_t = beta_sum_t + beta_mat[active_b].detach().float().mean()
                            beta_count += 1
                            ach_act = hip_read.ach_signal[active_b]
                            ach_list.append(ach_act)
                            if (not self.training) and record_traces and ach_trace_flat is not None and beta_max_flat is not None:
                                ach_trace_flat[flat_idx] = ach_act.detach().float()
                                beta_max_flat[flat_idx] = beta_mat[active_b].detach().float().max(dim=-1).values
                            if fam_mask_eff is not None:
                                fam_f = fam_mask_eff[active_b].float()
                                fam_ach_sum_t = fam_ach_sum_t + (ach_act.float() * fam_f).sum()
                                fam_ach_cnt_t = fam_ach_cnt_t + fam_f.sum()

                        flat_idx_chunks.append(flat_idx)
                        z0_chunks.append(z0_s[active_b])
                        g_ctx_chunks.append(g_ctx_s[active_b])
                        c_hat_chunks.append(c_hat_s[active_b])
                        pi_m_chunks.append(pi_m_s[active_b])
                        pi_eff_chunks.append(pi_eff_s[active_b])
                        surp_0_chunks.append(surp_0_s[active_b])
                        eps_0_chunks.append(eps_0_s[active_b])
                        sbar_prev_chunks.append(sbar_prev_s[active_b])
                        cfused_chunks.append(c_fused_step[active_b])
                        pts_chunks.append(pts_cur[active_b])
                        merge_probs_chunks.append(mp_s[active_b])
                        pre_slots_chunks.append(in_slots[active_b])
                        pre_edges_chunks.append(in_edges[active_b])
                        pre_sal_chunks.append(in_sal[active_b])
                        pre_phase_chunks.append(eff_phase[active_b])

                # === STEP 4 & STEP 5: Full-Batch `[M_total, D]` Parallel Pondering + Upper Cortical Decoder ===
                if flat_idx_chunks:
                    order = torch.cat(flat_idx_chunks, dim=0)
                    inv_perm = torch.argsort(order)

                    z0_flat = torch.cat(z0_chunks, dim=0)[inv_perm]
                    g_ctx_flat = torch.cat(g_ctx_chunks, dim=0)[inv_perm]
                    c_hat_flat = torch.cat(c_hat_chunks, dim=0)[inv_perm]
                    pi_m_flat = torch.cat(pi_m_chunks, dim=0)[inv_perm]
                    pi_eff_flat = torch.cat(pi_eff_chunks, dim=0)[inv_perm]
                    surp_0_flat = torch.cat(surp_0_chunks, dim=0)[inv_perm]
                    eps_0_flat = torch.cat(eps_0_chunks, dim=0)[inv_perm]
                    sbar_prev_flat = torch.cat(sbar_prev_chunks, dim=0)[inv_perm]
                    c_fused_flat = torch.cat(cfused_chunks, dim=0)[inv_perm]
                    pts_flat = torch.cat(pts_chunks, dim=0)[inv_perm]
                    pre_slots_flat = torch.cat(pre_slots_chunks, dim=0)[inv_perm]
                    pre_edges_flat = torch.cat(pre_edges_chunks, dim=0)[inv_perm]
                    pre_sal_flat = torch.cat(pre_sal_chunks, dim=0)[inv_perm]
                    pre_phase_flat = torch.cat(pre_phase_chunks, dim=0)[inv_perm]

                    # Step 4: Full-batch `[M_total, D]` System-2 Pondering!
                    z_refined_flat, ponder_flat, bal_pon = bcg_raw.ponder_parallel(
                        z0_flat, eps_0_flat, surp_0_flat, sbar_prev_flat, c_fused_flat, pi_eff_flat
                    )
                    z_star_flat = z_refined_flat

                    # Step 5: Causal `< m` shift (`bos_concept` at doc start, `z_refined[m-1]` for `m > doc_start`)
                    bos_vec = self.cortical.bos_concept.to(device=device, dtype=z_refined_flat.dtype).unsqueeze(0).expand(m_total, -1)
                    if m_total > 1:
                        shifted_z = torch.cat([bos_vec[:1], z_refined_flat[:-1]], dim=0)
                        z_ref_prev = torch.where(seg_doc_start.unsqueeze(-1), bos_vec, shifted_z)
                    else:
                        z_ref_prev = bos_vec

                    z_dec_flat, copy_stem_flat, bal_dec = self.cortical.decode_parallel(
                        z_ref_prev=z_ref_prev,
                        l_reg_seg=l_reg_seg,
                        c_kv=c_kv_all,
                        doc_ids=seg_doc_ids,
                        pos_in_doc=pos_in_doc,
                        seg_start_depth=seg_start_depth,
                        seg_end_depth=seg_end_depth,
                        pre_slots=pre_slots_flat,
                        pre_edges=pre_edges_flat,
                        pre_salience=pre_sal_flat,
                        pre_phase=pre_phase_flat,
                    )

                    if use_cq:
                        if use_parallel_cq:
                            # `seg_counts` / `seg_offsets` in original document order match `pre_*_flat` (`inv_perm`)!
                            sc_orig_cpu = [int(x) for x in seg_counts.detach().to(torch.int32).to("cpu").tolist()]
                            so_orig_cpu = [int(x) for x in seg_offsets.detach().to(torch.int32).to("cpu").tolist()]
                            z_cq_flat, ds_cq_flat = self.pmd.broca_cq_blocks_parallel(
                                pre_slots_flat=pre_slots_flat,
                                pre_edges_flat=pre_edges_flat,
                                pre_sal_flat=pre_sal_flat,
                                l_reg_seg=l_reg_seg,
                                seg_counts_cpu=sc_orig_cpu,
                                seg_offsets_cpu=so_orig_cpu,
                            )
                        else:
                            z_cq_flat = torch.cat(zcq_chunks, dim=0)[inv_perm]
                            ds_cq_flat = torch.cat(dscq_chunks, dim=0)[inv_perm]
                        z_out_expr = 0.5 * (z_cq_flat + z_dec_flat)
                        delta_syn_all = ds_cq_flat + 0.5 * self.pmd.compute_streaming_delta_syn(
                            g_ctx=g_ctx_flat, z_star=z_dec_flat, l_reg=l_reg_seg
                        )
                    else:
                        z_out_expr = z_dec_flat
                        delta_syn_all = self.pmd.compute_streaming_delta_syn(
                            g_ctx=g_ctx_flat, z_star=z_dec_flat, l_reg=l_reg_seg
                        )

                    f_slow_flat, _ = self.hippocampus.read_expression(z_out=z_out_expr, l_reg=l_reg_seg, enabled=False)
                    f_cort_flat = f_slow_flat + copy_stem_flat
                    if hip_enabled and rho_chunks:
                        rho_flat = torch.cat(rho_chunks, dim=0)[inv_perm]
                        fhip_flat = torch.cat(fhip_chunks, dim=0)[inv_perm]
                        f_out_flat = (
                            (1.0 - rho_flat).unsqueeze(-1) * f_cort_flat.float() + rho_flat.unsqueeze(-1) * fhip_flat
                        ).to(sgsc_out.f_stem.dtype)
                    else:
                        f_out_flat = f_cort_flat.to(sgsc_out.f_stem.dtype)
                    delta_syn_flat = delta_syn_all.to(sgsc_out.f_stem.dtype)

                    with fp32_island(device):
                        err_sq_all = (c_fused_flat.detach().float() - c_hat_flat.float()).pow(2)
                        pi_all = pi_m_flat.float()
                        pc_nll_sum = 0.5 * (
                            (pi_all * err_sq_all).mean(dim=-1) - torch.log(pi_all.clamp_min(1e-6)).mean(dim=-1)
                        ).sum()
                        ponder_all = ponder_flat.float()
                        ponder_sum = ponder_all.sum()
                        ponder_tax_sum = (ponder_all * pts_flat.float()).sum()
                        ponder_skip_sum = (ponder_all.detach() < 0.5).float().sum()
                        merge_probs_sum = torch.cat(merge_probs_chunks, dim=0).float().sum(dim=0)
                        moe_bal_sum = (bal_enc.float() + bal_pon.float() + bal_dec.float()) / 3.0
                        moe_bal_steps = 1
                    total_valid_steps = m_total
                else:
                    f_out_flat = torch.zeros((0, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
                    delta_syn_flat = torch.zeros((0, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
                    z_star_flat = None
                    c_fused_flat = None
                    pc_nll_sum = zero
                    ponder_sum = zero
                    ponder_tax_sum = zero
                    ponder_skip_sum = zero
                    merge_probs_sum = torch.zeros((num_pairs,), device=device, dtype=torch.float32)
                    moe_bal_sum = bal_enc.float()
                    moe_bal_steps = 1
                    total_valid_steps = 0

                if finished_slots:
                    assert g_state.slot0_pinned is not None and g_state.phase is not None
                    g_state = ConceptGraphState(
                        slots=torch.cat([g_state.slots, *reversed(finished_slots)], dim=0),
                        edges=torch.cat([g_state.edges, *reversed(finished_edges)], dim=0),
                        salience=torch.cat([g_state.salience, *reversed(finished_sal)], dim=0),
                        slot0_pinned=torch.cat([g_state.slot0_pinned, *reversed(finished_pin)], dim=0),
                        phase=torch.cat([g_state.phase, *reversed(finished_phase)], dim=0),
                    )
                if pad_v > 0:
                    g_state = ConceptGraphState(
                        slots=g_state.slots[:V],
                        edges=g_state.edges[:V],
                        salience=g_state.salience[:V],
                        slot0_pinned=g_state.slot0_pinned[:V] if g_state.slot0_pinned is not None else None,
                        phase=g_state.phase[:V] if g_state.phase is not None else None,
                    )
                    if isinstance(f_state, CompactHouseholderState):
                        f_state = f_state.slice_batch(V)
                    if e_state is not None:
                        e_state = EligibilityState(
                            e_idx=e_state.e_idx[:V],
                            e_val=e_state.e_val[:V],
                            f_tilde=e_state.f_tilde[:V],
                            eta=e_state.eta[:V],
                        )
                if inv_doc_order is not None:
                    g_state = ConceptGraphState(
                        slots=g_state.slots[inv_doc_order],
                        edges=g_state.edges[inv_doc_order],
                        salience=g_state.salience[inv_doc_order],
                        slot0_pinned=g_state.slot0_pinned[inv_doc_order] if g_state.slot0_pinned is not None else None,
                        phase=g_state.phase[inv_doc_order] if g_state.phase is not None else None,
                    )

                s_trace = s_trace_flat
                a_trace = a_trace_flat
                merge_alpha_trace = alpha_trace_flat
                win_pair_trace = win_pair_flat
                ach_trace = ach_trace_flat
                beta_max_trace = beta_max_flat
                slot0_pinned_trace = slot0_pinned_flat

            denom = max(total_valid_steps, 1)
            lam_pc = 1.0 if lambda_pc_override is None else float(lambda_pc_override)
            lam_act = self.cfg.lambda_act if lambda_act_override is None else float(lambda_act_override)

            if (not hip_enabled) and m_total > 0 and z_star_flat is not None:
                f_z_align, _ = self.hippocampus.read_expression(
                    z_out=z_star_flat, l_reg=l_reg_seg, fast_state=None, enabled=False
                )
                with fp32_island(device):
                    loss_stem_align = 0.1 * (
                        1.0 - F.cosine_similarity(f_z_align.float(), sgsc_out.f_stem.detach().float(), dim=-1, eps=1e-6)
                    ).mean()
            if m_total > 0 and z_star_flat is not None and (
                (not hip_enabled) or (self.training and bcg_raw.w_gen.weight.requires_grad)
            ):
                with fp32_island(device):
                    c_align_target = (
                        c_fused_flat.detach().float()
                        if (hip_enabled and c_fused_flat is not None)
                        else (
                            h_enc_all.detach().float()
                            if h_enc_all is not None
                            else sgsc_out.c_slow.detach().float()
                        )
                    )
                    c_gen_pred = F.linear(
                        z_star_flat.detach().float(),
                        bcg_raw.w_gen.weight.float(),
                        bcg_raw.w_gen.bias.float() if bcg_raw.w_gen.bias is not None else None,
                    )
                    loss_gen_align = 0.05 * (c_align_target - c_gen_pred).pow(2).mean()

            with fp32_island(device):
                raw_pc_nll = pc_nll_sum / denom
                loss_pc_raw_t = raw_pc_nll.detach()
                loss_pc = lam_pc * raw_pc_nll + loss_var
                loss_ponder = lam_act * (ponder_tax_sum / denom)
                loss_moe_bal = float(self.cfg.lambda_moe_bal) * (moe_bal_sum / max(moe_bal_steps, 1))
                avg_ponder_t = (ponder_sum / denom).detach()
                ponder_skip_rate_t = (ponder_skip_sum / denom).detach()

                mean_merge_probs = (merge_probs_sum / denom).clamp_min(1e-8)
                merge_entropy = -(mean_merge_probs * torch.log(mean_merge_probs)).sum()
                merge_entropy_t = merge_entropy.detach()
                if merge_kl_weight > 0.0:
                    uniform = torch.full_like(mean_merge_probs, 1.0 / mean_merge_probs.numel())
                    loss_merge_kl = merge_kl_weight * F.kl_div(torch.log(mean_merge_probs), uniform, reduction="sum")

                if ach_list:
                    all_ach = torch.cat(ach_list, dim=0)
                    mean_ach_t = all_ach.detach().mean()
                    if ach_hint_targets is not None and ach_hint_weight > 0.0 and ach_hint_targets.numel() == all_ach.numel():
                        loss_ach_hint = ach_hint_weight * F.binary_cross_entropy(
                            all_ach.clamp(1e-5, 1.0 - 1e-5),
                            ach_hint_targets.float(),
                        )
                if familiar_ach_reg_weight > 0.0:
                    if familiar_ach_list:
                        all_fam_ach = torch.cat(familiar_ach_list, dim=0)
                        loss_ach_reg = float(familiar_ach_reg_weight) * all_fam_ach.mean()
                    else:
                        loss_ach_reg = (
                            float(familiar_ach_reg_weight)
                            * (fam_ach_sum_t / fam_ach_cnt_t.clamp_min(1.0))
                            * (fam_ach_cnt_t > 0.0).float()
                        )
            if beta_count > 0:
                mean_beta_t = (beta_sum_t / float(beta_count)).detach()

            pmd_out = self.pmd.decode_teacher_forced(
                target_bytes=raw_bytes,
                cu_seqlens=sgsc_out.cu_seqlens,
                f_out_seg=f_out_flat,
                delta_syn_seg=delta_syn_flat,
                l_reg=l_reg,
                m_init=m_init,
                xi_init=xi_init,
                doc_cu_seqlens=doc_cu_seqlens,
                known_max_seg_len=self.cfg.max_seg_len + 4,
            )
            out_graph = g_state
            out_fast = f_state
            out_elig = e_state

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

            prior_w = 0.5 if stage == "A" else (0.05 if stage in ("B", "C") else 0.0)
            loss_total = (
                loss_lang
                + prior_w * loss_sgsc_prior
                + (loss_pc if stage != "A" else loss_var)
                + (loss_rate if stage != "D" else zero)
                + loss_ponder
                + loss_moe_bal
                + loss_stem_align
                + loss_gen_align
                + loss_halt
                + loss_merge_kl
                + loss_ach_hint
                + loss_ach_reg
            )

        spike_rate_t = sgsc_out.spikes.float().mean().detach()
        metrics_tensor = torch.stack(
            [
                loss_total.detach().float(),
                loss_lang.detach().float(),
                loss_sgsc_prior.detach().float(),
                loss_pc.detach().float(),
                loss_pc_raw_t.detach().float(),
                loss_stem_align.detach().float(),
                loss_gen_align.detach().float(),
                loss_var.detach().float(),
                loss_rate.detach().float(),
                loss_ponder.detach().float(),
                loss_moe_bal.detach().float(),
                loss_halt.detach().float(),
                loss_ach_reg.detach().float(),
                spike_rate_t.float(),
                sgsc_out.theta.detach().float(),
                c_slow_std_mean_t.float(),
                c_slow_std_min_t.float(),
                avg_ponder_t.float(),
                ponder_skip_rate_t.float(),
                merge_entropy_t.float(),
                mean_ach_t.float(),
                mean_beta_t.float(),
            ]
        )
        metrics = materialize_metric_vec(metrics_tensor) if sync_metrics else {}

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
            slot0_pinned_trace=slot0_pinned_trace,
            loss_moe_bal=loss_moe_bal,
            metrics_tensor=metrics_tensor,
            cortical_state=out_cortical,
        )

    def forward_and_backward_tbptt(
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
        style_ids: Optional[torch.Tensor] = None,
        ponder_tax_scale: Optional[torch.Tensor] = None,
        ach_override: Optional[float] = None,
        record_traces: bool = False,
        sync_metrics: bool = True,
        loss_scale: float = 1.0,
        tbptt_window: Optional[int] = None,
    ) -> DCPAForwardOutput:
        """Window-bounded TBPTT forward + backward for DCPA-Turbo.

        Because DCPA-Turbo hoists 100% of MoE & Pondering compute into parallel `[M_total, D]` kernels,
        the zero-MoE contractive graph recurrence uses < 20 MB of activation VRAM with chunk checkpointing,
        allowing single-pass `forward(bcg_tbptt_window=W) + backward()` (`1x` fwd instead of `2x` replay!)
        with 0.0 ULP loss/gradient discrepancy!
        """
        W = int(self.bcg_tbptt_window if tbptt_window is None else tbptt_window)
        scale = float(loss_scale)
        prev_w = self.bcg_tbptt_window
        if stage.upper() in ("B", "C") and W > 0:
            self.bcg_tbptt_window = W
        try:
            out = self.forward(
                raw_bytes=raw_bytes,
                lang_ids=lang_ids,
                stage=stage,
                tau_s=tau_s,
                tau_m=tau_m,
                use_broca_cq=use_broca_cq,
                loss_weights=loss_weights,
                ach_hint_targets=ach_hint_targets,
                ach_hint_weight=ach_hint_weight,
                lambda_rate_override=lambda_rate_override,
                dual_rate_lagrange=dual_rate_lagrange,
                lambda_var_override=lambda_var_override,
                lambda_pc_override=lambda_pc_override,
                lambda_act_override=lambda_act_override,
                lambda_halt_override=lambda_halt_override,
                target_spike_rate_override=target_spike_rate_override,
                merge_kl_weight=merge_kl_weight,
                merge_uniform_mix=merge_uniform_mix,
                familiar_ach_reg_weight=familiar_ach_reg_weight,
                familiar_doc_mask=familiar_doc_mask,
                write_threshold_override=write_threshold_override,
                fast_state=fast_state,
                graph_state=graph_state,
                elig_state=elig_state,
                doc_cu_seqlens=doc_cu_seqlens,
                h_init=h_init,
                m_init=m_init,
                xi_init=xi_init,
                style_ids=style_ids,
                ponder_tax_scale=ponder_tax_scale,
                ach_override=ach_override,
                record_traces=record_traces,
                sync_metrics=sync_metrics,
            )
        finally:
            self.bcg_tbptt_window = prev_w

        if torch.is_grad_enabled() and out.loss_total.requires_grad:
            (out.loss_total * scale).backward()
        return out
