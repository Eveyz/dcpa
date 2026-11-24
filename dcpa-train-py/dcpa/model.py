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
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from dcpa.config import DCPAConfig
from dcpa.modules.bcg_pc import BCGPC, ConceptGraphState
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
from dcpa.ops.scan import parallel_diag_scan

try:
    from utils.device import get_allocated_memory_gb as _get_allocated_memory_gb
except ImportError:
    def _get_allocated_memory_gb(_device: torch.device) -> float:
        return 0.0


class _DampedBoundedCarry(torch.autograd.Function):
    """Exact bit-identical forward carry with spectral-safe per-sample backward RMS clamping.

    Forward: exact identity `return x` (0.0 ULP drift, preserves Slot-0 pinning & Rust parity).
    Backward: scales incoming recurrent carry gradient by `gamma` and clamps per-sample RMS to
    `max_rms` in float32 — mathematically preventing exponential Jacobian explosion (grad_norm=inf)
    across arbitrary document lengths (512B -> 4096B / 50 -> 430 concept steps)!
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: float, max_rms: float) -> torch.Tensor:
        ctx.gamma = float(gamma)
        ctx.max_rms = float(max_rms)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        g = grad_out.float() * ctx.gamma
        if g.ndim > 1:
            reduce_dims = tuple(range(1, g.ndim))
            # Scale by per-sample max-abs before squaring so even extreme gradients (>1e19)
            # can NEVER overflow float32 pow(2) into +inf!
            g_max = g.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
            g_scaled = g / g_max
            rms = g_max * torch.sqrt(g_scaled.pow(2).mean(dim=reduce_dims, keepdim=True) + 1e-12)
        else:
            rms = g.abs().clamp_min(1e-12)
        scale = torch.clamp(ctx.max_rms / rms.clamp_min(1e-12), max=1.0)
        return (g * scale).to(grad_out.dtype), None, None


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
    # If `\x1fA\n` is immediately followed by `\x1dS\n`, defer Slot-0 pinning to `\x1dE\n`
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
    slot0_pinned_trace: Optional[torch.Tensor] = None  # [M_total] float32 Slot-0 Thesis Pin latch state
    loss_moe_bal: Optional[torch.Tensor] = None      # scalar fp32 MoE load-balancing auxiliary loss


class DCPAModel(nn.Module):
    def __init__(self, cfg: DCPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sgsc = SGSC(cfg)
        self.bcg_pc = BCGPC(cfg)
        self.hippocampus = HippocampusDualMemory(cfg)
        # Strict weight tying between SGSC and PMD for Embed [256, d_s] and W_out [256, d_s]
        self.pmd = PMD(cfg, shared_embed=self.sgsc.embed, shared_w_out=self.sgsc.w_out)
        # Document-dimension bucketing (multiples of 16) enabled automatically under torch.compile
        self.enable_doc_bucketing: bool = False
        self.doc_bucket_size: int = 16
        # Activation checkpointing across unrolled BCG-PC concept steps:
        #   - On <= 80GB GPUs (`bcg_checkpoint_vram_watermark_gb == 0.0`), checkpoints every step.
        #   - On >= 88GB GPUs (e.g. 95.6GB G4 Blackwell, `watermark_gb == 52.0`), skips recomputation
        #     while live allocated VRAM < watermark_gb and automatically engages checkpointing above it!
        self.bcg_grad_checkpointing: bool = True
        self.bcg_checkpoint_vram_watermark_gb: float = 0.0

    def _bcg_step_checkpointed_fn(
        self,
        in_slots: torch.Tensor,
        in_edges: torch.Tensor,
        in_sal: torch.Tensor,
        c_fused: torch.Tensor,
        prec_bu: torch.Tensor,
        ach: torch.Tensor,
        tau_m_t: torch.Tensor,
        active_mask: torch.Tensor,
        mix_t: torch.Tensor,
        slot0_pinned: torch.Tensor,
        trig_step: torch.Tensor,
        w_gu1: torch.Tensor,
        w_d1: torch.Tensor,
        w_gu2: torch.Tensor,
        w_d2: torch.Tensor,
        gumbel_u: torch.Tensor,
        compute_s_bar_post: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        return self.bcg_pc(
            in_slots,
            in_edges,
            in_sal,
            c_fused,
            prec_bu,
            ach,
            tau_m_t,
            active_mask,
            mix_t,
            slot0_pinned,
            trig_step,
            use_gumbel=True,
            update_c0_ema=False,
            packed_moe_w_gu1=w_gu1,
            packed_moe_w_d1=w_d1,
            packed_moe_w_gu2=w_gu2,
            packed_moe_w_d2=w_d2,
            gumbel_u=gumbel_u,
            compute_s_bar_post=compute_s_bar_post,
        )

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
        all_experts = sum(p.numel() for p in self.bcg_pc.moe.experts.parameters())
        inactive_experts = all_experts * (self.cfg.num_experts - self.cfg.top_k_experts) // max(self.cfg.num_experts, 1)
        active_per_step = total - inactive_experts
        return {
            "sgsc": sgsc_n,
            "bcg_pc": bcg_n,
            "hippocampus_slow": hip_n,
            "pmd_unique": pmd_unique,
            "total": total,
            "active_per_step": active_per_step,
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
        style_ids: Optional[torch.Tensor] = None,
        ach_override: Optional[float] = None,
    ) -> tuple[DCPAForwardOutput, DCPACarryState]:
        """Run one chunk through all 4 modules while carrying strictly O(1) constant state."""
        B = raw_bytes.shape[0]
        if carry_state is None:
            carry_state = self.init_carry_state(batch_size=B, device=raw_bytes.device, dtype=torch.float32)

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
                slot0_pinned=(
                    out.graph_state.slot0_pinned.detach().clone()
                    if out.graph_state.slot0_pinned is not None
                    else None
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
        style_ids: Optional[torch.Tensor] = None,
        ponder_tax_scale: Optional[torch.Tensor] = None,
        ach_override: Optional[float] = None,
        record_traces: bool = True,
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
        l_reg = self.pmd.get_lang_vec(lang_ids, style_ids)  # [V, d_l] where V=D_docs if packed else B

        # Build per-document segment counts & offsets (V = num_docs in packed mode, else B)
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

        # Vectorized per-segment Slot-0 Thesis Pinning & Turn-Boundary Unpinning triggers (Spec 2)
        if m_total > 0 and T > 0:
            pin_byte_mask = _compute_pin_trigger_byte_mask(raw_bytes, doc_reset_mask)  # [B, T]
            unpin_byte_mask = _compute_unpin_trigger_byte_mask(raw_bytes)  # [B, T]
            seg_id_flat, local_pos, _, _ = ragged_local_offsets(sgsc_out.cu_seqlens, B * T)
            pos_pow = torch.pow(2.0, local_pos.clamp_max(22).float()).unsqueeze(-1)
            pin_w = ragged_segment_sum(
                pin_byte_mask.float().reshape(B * T, 1) * pos_pow,
                seg_id_flat,
                m_total,
            ).squeeze(-1)
            unpin_w = ragged_segment_sum(
                unpin_byte_mask.float().reshape(B * T, 1) * pos_pow,
                seg_id_flat,
                m_total,
            ).squeeze(-1)
            seg_unpin_trigger = (unpin_w > 0.5).float()  # [M_total]
            seg_pin_trigger = (pin_w > unpin_w).float()  # [M_total]
        else:
            seg_unpin_trigger = torch.zeros((m_total,), device=device, dtype=torch.float32)
            seg_pin_trigger = torch.zeros((m_total,), device=device, dtype=torch.float32)

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

            # Precision Island #4 (Red Line #1): symmetric dimension-wise std band hinge [1.0, 1.20] in fp32
            # Prevents BOTH latent collapse (<1.0) AND scale inflation (>1.20) that would inflate pc_nll/surprisal.
            lam_var = self.cfg.lambda_var if lambda_var_override is None else lambda_var_override
            if m_total > 1:
                c_std = torch.sqrt(sgsc_out.c_slow.float().var(dim=0, unbiased=False) + 1e-4)
                loss_var = lam_var * (F.relu(1.0 - c_std).mean() + F.relu(c_std - 1.20).pow(2).mean())
                c_slow_std_mean = float(c_std.mean().detach().item())
                c_slow_std_min = float(c_std.min().detach().item())
            else:
                loss_var = zero
                c_slow_std_mean = 1.0
                c_slow_std_min = 1.0

        loss_pc = zero
        loss_pc_raw = 0.0
        loss_stem_align = zero
        loss_gen_align = zero
        loss_ponder = zero
        loss_merge_kl = zero
        loss_ach_hint = zero
        loss_ach_reg = zero
        loss_moe_bal = zero
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
        slot0_pinned_trace: Optional[torch.Tensor] = None

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

            use_cq = bool(use_broca_cq and stage in ("C", "D"))
            eval_isolated_path = (
                (doc_cu_seqlens is not None) and (not self.training) and (graph_state is None) and (V <= 8)
            )
            # Parallel document-dimension bucketing (Performance Audit Task 2.3): pad V -> ceil(V/16)*16
            # when selective torch.compile is active so Dynamo caches <= ~12 fixed shapes across 100k steps!
            use_bucketing = bool(
                self.enable_doc_bucketing
                and not eval_isolated_path
                and graph_state is None
                and fast_state is None
                and elig_state is None
            )
            # Monotonic active-prefix shrinking (Stages B/C): sort docs by seg_counts descending so
            # active docs always occupy contiguous [:n_active], shrinking v_cur as shorter docs finish
            # (eliminates 2-3x straggler-tail wasted BCG-PC work & removes per-step torch.nonzero syncs!).
            use_prefix_shrink = bool(
                (not hip_enabled)
                and (not use_cq)
                and (not eval_isolated_path)
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
                if n <= bucket * 2:
                    return min(bucket * 2, cap)
                if n <= bucket * 16:
                    step_b = bucket * 2
                    return min(((n + step_b - 1) // step_b) * step_b, cap)
                if n <= bucket * 32:
                    step_b = bucket * 4
                    return min(((n + step_b - 1) // step_b) * step_b, cap)
                step_b = bucket * 8
                return min(((n + step_b - 1) // step_b) * step_b, cap)

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
                pts_eff = torch.cat(
                    [pts_ord, torch.zeros((pad_v,), device=device, dtype=torch.float32)], dim=0
                )
                fam_mask_eff = (
                    torch.cat(
                        [fam_mask_ord, torch.zeros((pad_v,), device=device, dtype=torch.bool)], dim=0
                    )
                    if fam_mask_ord is not None
                    else None
                )
            else:
                seg_counts_eff = seg_counts_ord
                seg_offsets_eff = seg_offsets_ord
                l_reg_eff = l_reg_ord
                pts_eff = pts_ord
                fam_mask_eff = fam_mask_ord

            g_state = self.bcg_pc.init_state(V_eff, device=device, dtype=sgsc_out.c_slow.dtype) if graph_state is None else graph_state
            if g_state.slot0_pinned is None:
                g_state = ConceptGraphState(
                    slots=g_state.slots,
                    edges=g_state.edges,
                    salience=g_state.salience,
                    slot0_pinned=torch.zeros((g_state.slots.shape[0],), device=device, dtype=torch.float32),
                )
            f_state = (
                self.hippocampus.init_fast_state(V_eff, mode="compact", device=device)
                if (hip_enabled and fast_state is None)
                else fast_state
            )
            e_state = (
                EligibilityState.zeros(V_eff, self.cfg.k_slots, self.cfg.s_active, self.cfg.d_s, device, sgsc_out.f_stem.dtype)
                if (hip_enabled and elig_state is None)
                else elig_state
            )

            tau_m_t = torch.tensor(float(tau_m), device=device, dtype=torch.float32)
            mix_t = torch.tensor(float(merge_uniform_mix), device=device, dtype=torch.float32)

            m_max = int(seg_counts.max().item())
            D = self.cfg.d_concept
            d_s = self.cfg.d_s
            K = self.cfg.k_slots
            d_r = self.cfg.d_r
            num_pairs = self.cfg.num_merge_pairs

            f_out_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
            delta_syn_flat = torch.zeros((m_total, d_s), device=device, dtype=sgsc_out.f_stem.dtype)
            z_star_flat: Optional[torch.Tensor] = (
                torch.zeros((m_total, D), device=device, dtype=sgsc_out.c_slow.dtype)
                if eval_isolated_path
                else None
            )
            c_fused_flat: Optional[torch.Tensor] = (
                torch.zeros((m_total, D), device=device, dtype=sgsc_out.c_slow.dtype)
                if (eval_isolated_path and hip_enabled)
                else None
            )

            s_trace_flat = (
                torch.zeros((m_total, K, D), device=device, dtype=sgsc_out.c_slow.dtype)
                if record_traces
                else None
            )
            a_trace_flat = (
                torch.zeros((m_total, K, K, d_r), device=device, dtype=sgsc_out.c_slow.dtype)
                if record_traces
                else None
            )
            alpha_trace_flat = (
                torch.zeros((m_total, num_pairs), device=device, dtype=torch.float32)
                if record_traces
                else None
            )
            win_pair_flat = (
                torch.zeros((m_total, 3), device=device, dtype=torch.int64)
                if record_traces
                else None
            )
            ach_trace_flat = (
                torch.zeros((m_total,), device=device, dtype=torch.float32)
                if record_traces
                else None
            )
            beta_max_flat = (
                torch.zeros((m_total,), device=device, dtype=torch.float32)
                if record_traces
                else None
            )
            slot0_pinned_flat = (
                torch.zeros((m_total,), device=device, dtype=torch.float32)
                if record_traces
                else None
            )

            pc_nll_sum = zero
            ponder_sum = zero
            ponder_tax_sum = zero
            ponder_skip_sum = zero
            moe_bal_sum = zero
            moe_bal_steps = 0
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
                last_g_pin = []
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
                    pts_d = pts_base[d_idx : d_idx + 1]
                    cq_q_cur = torch.sigmoid(gd.salience.float()) if use_cq else None
                    cq_a_prev = (
                        torch.zeros((1, self.cfg.k_slots, self.cfg.d_r), device=device, dtype=gd.edges.dtype)
                        if use_cq
                        else None
                    )
                    for m_step in range(m_doc):
                        f_idx = d_off + m_step
                        f_step = sgsc_out.f_stem[f_idx : f_idx + 1]
                        c_step = sgsc_out.c_slow[f_idx : f_idx + 1]
                        sn_step = sgsc_out.sem_norm[f_idx : f_idx + 1]
                        mi_step = sgsc_out.mean_info[f_idx : f_idx + 1]
                        trig_step = seg_pin_trigger[f_idx : f_idx + 1]
                        if float(seg_unpin_trigger[f_idx].item()) > 0.5 and gd.slot0_pinned is not None:
                            gd = ConceptGraphState(
                                slots=gd.slots,
                                edges=gd.edges,
                                salience=gd.salience,
                                slot0_pinned=torch.zeros_like(gd.slot0_pinned),
                            )
                        gd_prev = gd

                        hip_read = self.hippocampus.read_comprehension(
                            f_stem=f_step,
                            c_slow=c_step,
                            sem_norm=sn_step,
                            mean_info=mi_step,
                            fast_state=fd,
                            enabled=hip_enabled,
                            ach_override=ach_override,
                        )
                        bcg_out = self.bcg_pc.step(
                            c_fused=hip_read.c_fused,
                            prec_bu=hip_read.prec_bu,
                            ach=hip_read.ach_signal,
                            state=gd,
                            tau_m=tau_m,
                            use_gumbel=False,
                            merge_uniform_mix=merge_uniform_mix,
                            pin_trigger=trig_step,
                        )
                        gd = bcg_out.state
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
                            alpha_trace_flat[f_idx] = bcg_out.merge_alpha[0].detach().float()
                            if gd.slot0_pinned is not None:
                                slot0_pinned_flat[f_idx] = gd.slot0_pinned[0].detach().float()
                            wi = bcg_out.win_i[0].detach()
                            wj = bcg_out.win_j[0].detach()
                            win_pair_flat[f_idx, 0] = wi
                            win_pair_flat[f_idx, 1] = wj
                            win_pair_flat[f_idx, 2] = torch.minimum(wi, wj)

                        # Strictly causal expressive retrieval BEFORE segment-m hippocampal write:
                        # Mode 2 (Broca CQ) queries strictly from pre-step G_{m-1} and pre-write M_{m-1};
                        # Mode 1 queries from top-down prediction c_hat_m (G_{m-1}) and M_{m-1}.
                        if use_cq:
                            assert cq_q_cur is not None and cq_a_prev is not None
                            q_init_d = torch.sigmoid(gd_prev.salience.float())
                            if m_step % self.cfg.k_slots == 0:
                                q_in_d = q_init_d
                                a_in_d = torch.zeros_like(cq_a_prev)
                            else:
                                q_in_d = cq_q_cur
                                a_in_d = cq_a_prev
                            cq_out = self.pmd.broca_cq_step(
                                graph=gd_prev,
                                q_prev=q_in_d,
                                q_init=q_init_d,
                                a_prev=a_in_d,
                                l_reg=l_reg_d,
                            )
                            cq_q_cur = cq_out.q_next
                            cq_a_prev = cq_out.a_prev_next
                            f_out_step, _ = self.hippocampus.read_expression(
                                z_out=cq_out.z_out,
                                l_reg=l_reg_d,
                                fast_state=fd if hip_enabled else None,
                                enabled=hip_enabled,
                            )
                            delta_syn_step = cq_out.delta_syn
                        else:
                            f_out_step, _ = self.hippocampus.read_expression(
                                z_out=bcg_out.c_hat,
                                l_reg=l_reg_d,
                                fast_state=fd if hip_enabled else None,
                                enabled=hip_enabled,
                            )
                            delta_syn_step = self.pmd.compute_streaming_delta_syn(
                                g_ctx=bcg_out.g_ctx,
                                z_star=bcg_out.c_hat,
                                l_reg=l_reg_d,
                            )
                        f_out_flat[f_idx] = f_out_step[0].to(f_out_flat.dtype)
                        delta_syn_flat[f_idx] = delta_syn_step[0].to(delta_syn_flat.dtype)
                        assert z_star_flat is not None
                        z_star_flat[f_idx] = bcg_out.z_star[0].to(z_star_flat.dtype)
                        if hip_enabled and c_fused_flat is not None:
                            c_fused_flat[f_idx] = hip_read.c_fused[0].detach().to(c_fused_flat.dtype)

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
                            if record_traces:
                                assert ach_trace_flat is not None and beta_max_flat is not None
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
                            ponder_tax_sum = ponder_tax_sum + (bcg_out.ponder_steps.float() * pts_d).sum()
                            ponder_skip_sum = ponder_skip_sum + (bcg_out.ponder_steps.detach().float() < 0.5).float().sum()
                            merge_probs_sum = merge_probs_sum + bcg_out.merge_probs.float().sum(dim=0)
                            if bcg_out.moe_bal_loss is not None:
                                moe_bal_sum = moe_bal_sum + bcg_out.moe_bal_loss.float()
                                moe_bal_steps += 1
                        total_valid_steps += 1

                    last_g_slots.append(gd.slots)
                    last_g_edges.append(gd.edges)
                    last_g_sal.append(gd.salience)
                    last_g_pin.append(
                        gd.slot0_pinned
                        if gd.slot0_pinned is not None
                        else torch.zeros((1,), device=device, dtype=torch.float32)
                    )

                g_state = ConceptGraphState(
                    slots=torch.cat(last_g_slots, dim=0),
                    edges=torch.cat(last_g_edges, dim=0),
                    salience=torch.cat(last_g_sal, dim=0),
                    slot0_pinned=torch.cat(last_g_pin, dim=0),
                )
            else:
                # Zero-index_put gather + monotonic active-prefix tail shrinking + compiled Hotspot A kernel!
                seg_offsets_i64 = seg_offsets_eff.to(torch.int64)
                zero_idx = torch.zeros_like(seg_offsets_i64)
                f_out_chunks: list[torch.Tensor] = []
                delta_syn_chunks: list[torch.Tensor] = []
                c_hat_chunks: list[torch.Tensor] = []
                z_star_chunks: list[torch.Tensor] = []
                c_fused_chunks: list[torch.Tensor] = []
                g_ctx_chunks: list[torch.Tensor] = []
                pi_m_chunks: list[torch.Tensor] = []
                ponder_chunks: list[torch.Tensor] = []
                pts_chunks: list[torch.Tensor] = []
                merge_probs_chunks: list[torch.Tensor] = []
                moe_bal_chunks: list[torch.Tensor] = []
                flat_idx_chunks: list[torch.Tensor] = []

                # Pre-pack MoE expert weights into 4 batched-GEMM tensors ONCE per forward()!
                # Saves E*6 sequential GEMMs & eliminates m_max * L_max redundant fp32->bf16 weight casts.
                bcg_raw = getattr(self.bcg_pc, "_orig_mod", self.bcg_pc)
                w_gu1, w_d1, w_gu2, w_d2 = bcg_raw.moe.pack_weights(dtype=sgsc_out.c_slow.dtype)

                # Hoist cortical bottom-up precision pi_cort ONCE across all m_total segments when
                # Hippocampus is disabled (Stages B/C): skips m_max dead dg_lex + topk(4096, 32) calls!
                pi_cort_all: Optional[torch.Tensor] = None
                if not hip_enabled and m_total > 0:
                    with fp32_island(device):
                        pi_cort_all = (
                            F.softplus(
                                F.linear(
                                    sgsc_out.c_slow.float(),
                                    self.hippocampus.f_prec_cort.weight.float(),
                                    self.hippocampus.f_prec_cort.bias.float()
                                    if self.hippocampus.f_prec_cort.bias is not None
                                    else None,
                                )
                            )
                            + 1e-4
                        )  # [m_total, D] fp32

                seg_counts_cpu = seg_counts_ord.detach().cpu().tolist() if use_prefix_shrink else []
                n_active_prefix = sum(1 for c in seg_counts_cpu if c > 0) if use_prefix_shrink else V
                cur_v = V_eff
                finished_slots: list[torch.Tensor] = []
                finished_edges: list[torch.Tensor] = []
                finished_sal: list[torch.Tensor] = []
                finished_pin: list[torch.Tensor] = []
                cq_q_cur: Optional[torch.Tensor] = torch.sigmoid(g_state.salience.float()) if use_cq else None
                cq_a_prev: Optional[torch.Tensor] = (
                    torch.zeros((V_eff, self.cfg.k_slots, self.cfg.d_r), device=device, dtype=g_state.edges.dtype)
                    if use_cq
                    else None
                )

                for m_step in range(m_max):
                    if use_prefix_shrink:
                        while n_active_prefix > 0 and seg_counts_cpu[n_active_prefix - 1] <= m_step:
                            n_active_prefix -= 1
                        if n_active_prefix == 0:
                            break
                        v_next = _coarse_bucket_v(n_active_prefix, cur_v)
                        if v_next < cur_v:
                            assert g_state.slot0_pinned is not None
                            finished_slots.append(g_state.slots[v_next:])
                            finished_edges.append(g_state.edges[v_next:])
                            finished_sal.append(g_state.salience[v_next:])
                            finished_pin.append(g_state.slot0_pinned[v_next:])
                            g_state = ConceptGraphState(
                                slots=g_state.slots[:v_next],
                                edges=g_state.edges[:v_next],
                                salience=g_state.salience[:v_next],
                                slot0_pinned=g_state.slot0_pinned[:v_next],
                            )
                            cur_v = v_next
                        active_mask = m_step < seg_counts_eff[:cur_v]
                        active_f = active_mask.float()
                        active_b = torch.arange(n_active_prefix, device=device, dtype=torch.int64)
                        seg_off_cur = seg_offsets_i64[:cur_v]
                        zero_cur = zero_idx[:cur_v]
                        l_reg_cur = l_reg_eff[:cur_v]
                        pts_cur = pts_eff[:cur_v]
                    else:
                        active_mask = m_step < seg_counts_eff  # [V_eff] bool (strictly False for padded virtual docs!)
                        active_f = active_mask.float()
                        active_b = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
                        if active_b.numel() == 0:
                            break
                        seg_off_cur = seg_offsets_i64
                        zero_cur = zero_idx
                        l_reg_cur = l_reg_eff
                        pts_cur = pts_eff

                    flat_idx = seg_off_cur[active_b] + m_step  # [V_act]
                    safe_idx = torch.where(active_mask, seg_off_cur + m_step, zero_cur)

                    am_col_c = active_f.unsqueeze(-1).to(sgsc_out.c_slow.dtype)
                    c_step = sgsc_out.c_slow[safe_idx] * am_col_c
                    trig_step = seg_pin_trigger[safe_idx] * active_f
                    unpin_step = (seg_unpin_trigger[safe_idx] * active_f) > 0.5

                    # Module 3 Read Path (hoisted zero-overhead cortical pass-through in Stages B/C!)
                    if hip_enabled:
                        am_col_f = active_f.unsqueeze(-1).to(sgsc_out.f_stem.dtype)
                        f_step = sgsc_out.f_stem[safe_idx] * am_col_f
                        sn_step = sgsc_out.sem_norm[safe_idx] * active_f
                        mi_step = sgsc_out.mean_info[safe_idx] * active_f
                        hip_read = self.hippocampus.read_comprehension(
                            f_stem=f_step,
                            c_slow=c_step,
                            sem_norm=sn_step,
                            mean_info=mi_step,
                            fast_state=f_state,
                            enabled=True,
                            ach_override=ach_override,
                        )
                        c_fused_step = hip_read.c_fused
                        prec_bu_step = hip_read.prec_bu
                        ach_step = hip_read.ach_signal
                    else:
                        hip_read = None
                        f_step = None
                        c_fused_step = c_step
                        assert pi_cort_all is not None
                        prec_bu_step = pi_cort_all[safe_idx] * active_f.unsqueeze(-1)
                        ach_step = torch.zeros_like(active_f)

                    # Module 2 BCG-PC Step via Hotspot A (`self.bcg_pc(...)` routes through torch.compile!)
                    if self.training and m_step > 0:
                        # Exact bit-identical forward carry (`return x`, 0.0 ULP drift, preserving Slot-0
                        # pinning & Rust parity) with spectral-safe backward damping (gamma=0.5) AND
                        # tight per-sample RMS safety ceiling (max_rms=0.25).
                        in_slots = _DampedBoundedCarry.apply(g_state.slots, 0.5, 0.25)
                        in_edges = _DampedBoundedCarry.apply(g_state.edges, 0.5, 0.25)
                        in_sal = _DampedBoundedCarry.apply(g_state.salience, 0.5, 0.25)
                    else:
                        in_slots = g_state.slots
                        in_edges = g_state.edges
                        in_sal = g_state.salience
                    assert g_state.slot0_pinned is not None
                    eff_slot0_pinned = torch.where(
                        unpin_step, torch.zeros_like(g_state.slot0_pinned), g_state.slot0_pinned
                    )
                    g_prev_cq = (
                        ConceptGraphState(
                            slots=in_slots,
                            edges=in_edges,
                            salience=in_sal,
                            slot0_pinned=eff_slot0_pinned,
                        )
                        if use_cq
                        else None
                    )

                    # Pre-sample 72-pair Gumbel uniform noise ONCE outside checkpointing:
                    #   1. Eliminates 100% of fork_rng CUDA generator save/restore syncs (`preserve_rng_state=False`).
                    #   2. Guarantees bit-identical RNG trajectory whether watermark checkpointing is engaged or bypassed!
                    gumbel_u_step: Optional[torch.Tensor] = (
                        torch.rand((cur_v, num_pairs), device=device, dtype=torch.float32)
                        if self.training
                        else None
                    )
                    watermark_gb = float(self.bcg_checkpoint_vram_watermark_gb)
                    use_ckpt_step = bool(
                        self.training
                        and torch.is_grad_enabled()
                        and self.bcg_grad_checkpointing
                        and (watermark_gb <= 0.0 or _get_allocated_memory_gb(device) >= watermark_gb)
                    )
                    if use_ckpt_step:
                        assert gumbel_u_step is not None
                        bcg_step_out = torch_checkpoint(
                            self._bcg_step_checkpointed_fn,
                            in_slots,
                            in_edges,
                            in_sal,
                            c_fused_step,
                            prec_bu_step,
                            ach_step,
                            tau_m_t,
                            active_mask,
                            mix_t,
                            eff_slot0_pinned,
                            trig_step,
                            w_gu1,
                            w_d1,
                            w_gu2,
                            w_d2,
                            gumbel_u_step,
                            hip_enabled,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                        # Update neutral concept center c_0 ONCE outside checkpoint (never doubled on bwd!)
                        bcg_raw.update_c0_from_slots(bcg_step_out[6], active_mask)
                    else:
                        bcg_step_out = self.bcg_pc(
                            in_slots,
                            in_edges,
                            in_sal,
                            c_fused_step,
                            prec_bu_step,
                            ach_step,
                            tau_m_t,
                            active_mask,
                            mix_t,
                            eff_slot0_pinned,
                            trig_step,
                            use_gumbel=self.training,
                            update_c0_ema=self.training,
                            packed_moe_w_gu1=w_gu1,
                            packed_moe_w_d1=w_d1,
                            packed_moe_w_gu2=w_gu2,
                            packed_moe_w_d2=w_d2,
                            gumbel_u=gumbel_u_step,
                            compute_s_bar_post=hip_enabled,
                        )
                    (
                        z_star_step,
                        g_ctx_step,
                        c_hat_step,
                        pi_m_step,
                        _surp_step,
                        ponder_step,
                        s_next,
                        a_next,
                        omega_next,
                        s_bar_post_step,
                        _win_idx_step,
                        win_i_step,
                        win_j_step,
                        merge_probs_step,
                        merge_alpha_step,
                        slot0_pinned_next,
                        moe_bal_step,
                    ) = bcg_step_out
                    g_state = ConceptGraphState(
                        slots=s_next,
                        edges=a_next,
                        salience=omega_next,
                        slot0_pinned=slot0_pinned_next,
                    )

                    # Record detached per-segment graph trajectory & forward discrete merge alpha (only when requested)
                    if record_traces:
                        assert (
                            s_trace_flat is not None
                            and slot0_pinned_flat is not None
                            and alpha_trace_flat is not None
                        )
                        s_trace_flat[flat_idx] = g_state.slots[active_b].detach().to(s_trace_flat.dtype)
                        slot0_pinned_flat[flat_idx] = slot0_pinned_next[active_b].detach().float()
                        if not self.training:
                            assert a_trace_flat is not None and win_pair_flat is not None
                            a_trace_flat[flat_idx] = g_state.edges[active_b].detach().to(a_trace_flat.dtype)
                            wi_a = win_i_step[active_b].detach()
                            wj_a = win_j_step[active_b].detach()
                            win_pair_flat[flat_idx, 0] = wi_a
                            win_pair_flat[flat_idx, 1] = wj_a
                            win_pair_flat[flat_idx, 2] = torch.minimum(wi_a, wj_a)
                        alpha_trace_flat[flat_idx] = merge_alpha_step[active_b].detach().float()

                    # Strictly causal expressive retrieval BEFORE segment-m hippocampal write:
                    # - Mode 2 (use_cq): rolling causal Broca CQ from pre-step graph G_{m-1} and M_{m-1}.
                    # - Mode 1 Stage D (hip_enabled): per-step retrieval from M_{m-1} via c_hat_m.
                    # - Mode 1 Stages B/C: hoist expression & L_PC/ponder reductions into one pass after loop.
                    if use_cq:
                        assert g_prev_cq is not None and cq_q_cur is not None and cq_a_prev is not None
                        q_init_step = torch.sigmoid(in_sal.float())
                        if m_step % self.cfg.k_slots == 0:
                            q_in_step = q_init_step
                            a_in_step = torch.zeros_like(cq_a_prev)
                        else:
                            q_in_step = cq_q_cur
                            a_in_step = cq_a_prev
                        cq_out = self.pmd.broca_cq_step(
                            graph=g_prev_cq,
                            q_prev=q_in_step,
                            q_init=q_init_step,
                            a_prev=a_in_step,
                            l_reg=l_reg_cur,
                        )
                        cq_q_cur = torch.where(active_mask.unsqueeze(-1), cq_out.q_next, cq_q_cur)
                        cq_a_prev = torch.where(
                            active_mask.view(-1, 1, 1),
                            cq_out.a_prev_next.to(cq_a_prev.dtype),
                            cq_a_prev,
                        )
                        f_cq_all, _ = self.hippocampus.read_expression(
                            z_out=cq_out.z_out,
                            l_reg=l_reg_cur,
                            fast_state=f_state if hip_enabled else None,
                            enabled=hip_enabled,
                        )
                        f_out_chunks.append(f_cq_all[active_b].to(sgsc_out.f_stem.dtype))
                        delta_syn_chunks.append(cq_out.delta_syn[active_b].to(sgsc_out.f_stem.dtype))
                        if hip_enabled:
                            z_star_chunks.append(z_star_step[active_b].detach())
                            c_fused_chunks.append(c_fused_step[active_b].detach())
                        else:
                            c_hat_chunks.append(c_hat_step[active_b])
                            z_star_chunks.append(z_star_step[active_b])
                            g_ctx_chunks.append(g_ctx_step[active_b])
                            pi_m_chunks.append(pi_m_step[active_b])
                            ponder_chunks.append(ponder_step[active_b])
                            pts_chunks.append(pts_cur[active_b])
                            merge_probs_chunks.append(merge_probs_step[active_b])
                            moe_bal_chunks.append(moe_bal_step)
                    elif hip_enabled:
                        f_out_all, _ = self.hippocampus.read_expression(
                            z_out=c_hat_step,
                            l_reg=l_reg_cur,
                            fast_state=f_state,
                            enabled=True,
                        )
                        f_out_step = f_out_all[active_b]
                        delta_syn_step = self.pmd.compute_streaming_delta_syn(
                            g_ctx=g_ctx_step[active_b],
                            z_star=c_hat_step[active_b],
                            l_reg=l_reg_cur[active_b],
                        )
                        f_out_chunks.append(f_out_step.to(sgsc_out.f_stem.dtype))
                        delta_syn_chunks.append(delta_syn_step.to(sgsc_out.f_stem.dtype))
                        z_star_chunks.append(z_star_step[active_b].detach())
                        c_fused_chunks.append(c_fused_step[active_b].detach())
                    else:
                        c_hat_chunks.append(c_hat_step[active_b])
                        z_star_chunks.append(z_star_step[active_b])
                        g_ctx_chunks.append(g_ctx_step[active_b])
                        pi_m_chunks.append(pi_m_step[active_b])
                        ponder_chunks.append(ponder_step[active_b])
                        pts_chunks.append(pts_cur[active_b])
                        merge_probs_chunks.append(merge_probs_step[active_b])
                        moe_bal_chunks.append(moe_bal_step)
                    flat_idx_chunks.append(flat_idx)

                    # Module 3 Eligibility Inheritance & Three-Factor Gated Delta Write (Stage D)
                    if hip_enabled and f_state is not None and e_state is not None:
                        assert hip_read is not None and f_step is not None
                        e_next = self.hippocampus.inherit_eligibility_on_merge(
                            elig=e_state,
                            cand_idx=hip_read.psi_idx,
                            cand_val=hip_read.psi_val,
                            cand_f=f_step,
                            cand_ach=hip_read.ach_signal * active_f,
                            win_i=win_i_step,
                            win_j=win_j_step,
                        )
                        e_next_masked = EligibilityState(
                            e_idx=torch.where(active_mask.view(V_eff, 1, 1), e_next.e_idx, e_state.e_idx),
                            e_val=torch.where(active_mask.view(V_eff, 1, 1), e_next.e_val, e_state.e_val),
                            f_tilde=torch.where(active_mask.view(V_eff, 1, 1), e_next.f_tilde, e_state.f_tilde),
                            eta=torch.where(active_mask.view(V_eff, 1), e_next.eta, torch.zeros_like(e_next.eta)),
                        )
                        e_state, _, beta_mat = self.hippocampus.three_factor_write(
                            s_bar_post=s_bar_post_step,
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
                        if (not self.training) and record_traces:
                            assert ach_trace_flat is not None and beta_max_flat is not None
                            ach_trace_flat[flat_idx] = ach_act.detach().float()
                            beta_max_flat[flat_idx] = beta_mat[active_b].detach().float().max(dim=-1).values
                        if fam_mask_eff is not None:
                            fam_active = fam_mask_eff[active_b]
                            if fam_active.any():
                                familiar_ach_list.append(ach_act[fam_active])

                    if hip_enabled:
                        # Precision Island #4 (Red Line #1): Accumulate L_PC strictly in float32
                        with fp32_island(device):
                            c_target_sg = c_fused_step[active_b].detach().float()
                            c_pred = c_hat_step[active_b].float()
                            pi_act = pi_m_step[active_b].float()
                            err_sq = (c_target_sg - c_pred).pow(2)
                            nll_step = 0.5 * (
                                (pi_act * err_sq).mean(dim=-1) - torch.log(pi_act.clamp_min(1e-6)).mean(dim=-1)
                            )
                            pc_nll_sum = pc_nll_sum + nll_step.sum()
                            ponder_sum = ponder_sum + ponder_step[active_b].float().sum()
                            ponder_tax_sum = ponder_tax_sum + (ponder_step[active_b].float() * pts_cur[active_b]).sum()
                            ponder_skip_sum = (
                                ponder_skip_sum + (ponder_step[active_b].detach().float() < 0.5).float().sum()
                            )
                            merge_probs_sum = merge_probs_sum + merge_probs_step[active_b].float().sum(dim=0)
                            moe_bal_sum = moe_bal_sum + moe_bal_step.float()
                            moe_bal_steps += 1
                        total_valid_steps += int(active_b.numel())

                if flat_idx_chunks:
                    order = torch.cat(flat_idx_chunks, dim=0)
                    inv_perm = torch.argsort(order)
                    if hip_enabled:
                        f_out_flat = torch.cat(f_out_chunks, dim=0)[inv_perm]
                        delta_syn_flat = torch.cat(delta_syn_chunks, dim=0)[inv_perm]
                        z_star_flat = torch.cat(z_star_chunks, dim=0)[inv_perm]
                        c_fused_flat = torch.cat(c_fused_chunks, dim=0)[inv_perm]
                    else:
                        c_hat_all_ord = torch.cat(c_hat_chunks, dim=0)
                        c_hat_flat = c_hat_all_ord[inv_perm]
                        z_star_flat = torch.cat(z_star_chunks, dim=0)[inv_perm]
                        g_ctx_flat = torch.cat(g_ctx_chunks, dim=0)[inv_perm]
                        if use_cq:
                            f_out_flat = torch.cat(f_out_chunks, dim=0)[inv_perm]
                            delta_syn_flat = torch.cat(delta_syn_chunks, dim=0)[inv_perm]
                        else:
                            f_out_all_flat, _ = self.hippocampus.read_expression(
                                z_out=c_hat_flat,
                                l_reg=l_reg_seg,
                                fast_state=None,
                                enabled=False,
                            )
                            delta_syn_all_flat = self.pmd.compute_streaming_delta_syn(
                                g_ctx=g_ctx_flat,
                                z_star=c_hat_flat,
                                l_reg=l_reg_seg,
                            )
                            f_out_flat = f_out_all_flat.to(sgsc_out.f_stem.dtype)
                            delta_syn_flat = delta_syn_all_flat.to(sgsc_out.f_stem.dtype)
                        with fp32_island(device):
                            c_target_all = sgsc_out.c_slow.detach().float()[order]
                            c_pred_all = c_hat_all_ord.float()
                            pi_all = torch.cat(pi_m_chunks, dim=0).float()
                            err_sq_all = (c_target_all - c_pred_all).pow(2)
                            pc_nll_sum = (
                                0.5
                                * (
                                    (pi_all * err_sq_all).mean(dim=-1)
                                    - torch.log(pi_all.clamp_min(1e-6)).mean(dim=-1)
                                )
                            ).sum()
                            ponder_all = torch.cat(ponder_chunks, dim=0).float()
                            pts_all = torch.cat(pts_chunks, dim=0).float()
                            ponder_sum = ponder_all.sum()
                            ponder_tax_sum = (ponder_all * pts_all).sum()
                            ponder_skip_sum = (ponder_all.detach() < 0.5).float().sum()
                            merge_probs_sum = torch.cat(merge_probs_chunks, dim=0).float().sum(dim=0)
                            moe_bal_sum = torch.stack(moe_bal_chunks).float().sum()
                            moe_bal_steps = len(moe_bal_chunks)
                        total_valid_steps = int(order.numel())

                # Reassemble shrunk active-prefix tails back to [V_eff, ...]
                if finished_slots:
                    assert g_state.slot0_pinned is not None
                    g_state = ConceptGraphState(
                        slots=torch.cat([g_state.slots, *reversed(finished_slots)], dim=0),
                        edges=torch.cat([g_state.edges, *reversed(finished_edges)], dim=0),
                        salience=torch.cat([g_state.salience, *reversed(finished_sal)], dim=0),
                        slot0_pinned=torch.cat([g_state.slot0_pinned, *reversed(finished_pin)], dim=0),
                    )

                # Trim bucketed virtual slots back to exact document count V for downstream CQ / output
                if pad_v > 0:
                    g_state = ConceptGraphState(
                        slots=g_state.slots[:V],
                        edges=g_state.edges[:V],
                        salience=g_state.salience[:V],
                        slot0_pinned=(
                            g_state.slot0_pinned[:V] if g_state.slot0_pinned is not None else None
                        ),
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

                # Restore original document ordering if active-prefix descending sort was used
                if inv_doc_order is not None:
                    g_state = ConceptGraphState(
                        slots=g_state.slots[inv_doc_order],
                        edges=g_state.edges[inv_doc_order],
                        salience=g_state.salience[inv_doc_order],
                        slot0_pinned=(
                            g_state.slot0_pinned[inv_doc_order] if g_state.slot0_pinned is not None else None
                        ),
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
            # Lightweight scale-invariant stem-alignment anchor (Stages B/C) and generative W_gen anchor:
            # keeps w_lex_slow(z_star) calibrated to Stage-1 f_stem space in Stages B/C, AND keeps
            # W_gen calibrated to reconstruct c_slow / c_fused from z_star across Stages B/C & D/SFT
            # (stop-grad on BOTH targets & z_star -> trains ONLY W_gen so AdamW weight_decay never collapses it!).
            bcg_mod = getattr(self.bcg_pc, "_orig_mod", self.bcg_pc)
            if (not hip_enabled) and m_total > 0 and z_star_flat is not None:
                f_z_align, _ = self.hippocampus.read_expression(
                    z_out=z_star_flat,
                    l_reg=l_reg_seg,
                    fast_state=None,
                    enabled=False,
                )
                with fp32_island(device):
                    loss_stem_align = 0.1 * (
                        1.0
                        - F.cosine_similarity(
                            f_z_align.float(), sgsc_out.f_stem.detach().float(), dim=-1, eps=1e-6
                        )
                    ).mean()
            if m_total > 0 and z_star_flat is not None and (
                (not hip_enabled) or (self.training and bcg_mod.w_gen.weight.requires_grad)
            ):
                with fp32_island(device):
                    c_align_target = (
                        c_fused_flat.detach().float()
                        if (hip_enabled and c_fused_flat is not None)
                        else sgsc_out.c_slow.detach().float()
                    )
                    c_gen_pred = F.linear(
                        z_star_flat.detach().float(),
                        bcg_mod.w_gen.weight.float(),
                        bcg_mod.w_gen.bias.float() if bcg_mod.w_gen.bias is not None else None,
                    )
                    loss_gen_align = 0.05 * (c_align_target - c_gen_pred).pow(2).mean()
            with fp32_island(device):
                raw_pc_nll = pc_nll_sum / denom
                loss_pc_raw = float(raw_pc_nll.detach().item())
                loss_pc = lam_pc * raw_pc_nll + loss_var
                loss_ponder = lam_act * (ponder_tax_sum / denom)
                loss_moe_bal = float(self.cfg.lambda_moe_bal) * (moe_bal_sum / max(moe_bal_steps, 1))
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

            # Decoding features are already strictly causal from Mode 1 or rolling causal Mode 2 (Broca CQ)
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
                + loss_moe_bal
                + loss_stem_align
                + loss_gen_align
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
            "loss_pc_raw": loss_pc_raw,
            "loss_stem_align": float(loss_stem_align.detach().item()),
            "loss_gen_align": float(loss_gen_align.detach().item()),
            "loss_var": float(loss_var.detach().item()),
            "loss_rate": float(loss_rate.detach().item()),
            "loss_ponder": float(loss_ponder.detach().item()),
            "loss_moe_bal": float(loss_moe_bal.detach().item()),
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
            slot0_pinned_trace=slot0_pinned_trace,
            loss_moe_bal=loss_moe_bal,
        )
