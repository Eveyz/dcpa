"""WS-5: Top-Level DCPA-Flat v3.3 Unified Model (`DCPAModelFlat`).

Wires all four decoupled workstreams into one clean, static-shape architecture:
  1. `SGSCFlat` + `static_grid` (`S+1` overflow trash slot) -> `ScatterBridgeOut` `[P, S=640]`
  2. `CorticalStackFlat.forward_pre_g1` (L1: optional `W=128` causal local window + Base/Plugin MoE)
  3. `ContractiveGraphCarrier` (`G1`: free-flowing, post-update, `C_scan` chunked scan)
  4. `CorticalStackFlat.forward_post_g1` (L2-L3: shared `G1` KV graph cross-attn + optional `+32KB` `c_kv_sem`)
  5. `HippocampusFlat` <-> `ContractiveGraphCarrier` (`G2`: strictly causal prior + Slot-0 pinning):
     - Stages 1-3 (`hip_enabled=False`): fast `C_scan` chunked scan of `G2`.
     - Stages 4-5 / inference (`hip_enabled=True`): R4 causal chunk-interleaved `HIPR <-> G2` loop.
  6. `CorticalStackFlat.forward_predict_and_refine`:
     - L4-L6 (L4 & L6 unified `2*W + K + 1` single-softmax `(W=128 ∪ K=8 ∪ sink)`, L5 pure `K=8`)
     - Cognitive uncertainty head `u_n` with dual gradient paths (`L_unc` Huber regression + MoD gate)
     - L7-L8 refinement (training static Top-25% + EMA `tau_think`; eval/streaming `u_n > tau_think`)
  7. Three-way stem confluence: Slow Cortex + Hippocampus `M^{L<-C}` (triangular solve) + `HolographicCopyRing`
  8. `PMDFlat`: zero-sync in-place gather -> inverse RoPE demod -> 128B stack -> overflow-masked byte CE + `L_halt`

Global v3.3 red lines:
  - ZERO `torch.nonzero`, ZERO `repeat_interleave`, ZERO `.item()`/`.cpu()`/`.tolist()` in forward/backward.
  - Strictly decoupled Base Top-1 MoE + Additive Dead-Zone Plugin Pool (`0.0` bit drift on non-target inputs).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.interfaces import (
    CarrierGraphState,
    CarrierScanOut,
    DCPAFlatConfig,
    HippocampusChunkState,
    PluginExpertPack,
    ScatterBridgeOut,
    StaticGridBatch,
)
from dcpa.modules.copy_ring import HolographicCopyRing
from dcpa.modules.cortical_moe import CorticalPredOut, CorticalStackFlat
from dcpa.modules.graph_carrier import ContractiveGraphCarrier
from dcpa.modules.hippocampus_flat import (
    HippocampusFlat,
    HippocampusInterleavedOut,
    HippocampusWriteEvents,
)
from dcpa.modules.pmd import PMDFlat, PMDFlatOutput
from dcpa.modules.sgsc import SGSCFlat
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.rmsnorm import RMSNorm
from dcpa.ops.static_grid import static_segment_mean, static_segment_scatter_add

FlatStageMode = Literal["sensory", "cortex", "hippocampus", "full"]


@dataclass
class DCPAFlatForwardOutput:
    """Structured forward output of `DCPAModelFlat` (100% on-device tensors, zero host syncs)."""

    loss_total: torch.Tensor
    loss_lang: torch.Tensor
    loss_sgsc_prior: torch.Tensor
    loss_jepa: torch.Tensor
    loss_unc: torch.Tensor
    loss_var: torch.Tensor
    loss_rate: torch.Tensor
    loss_halt: torch.Tensor
    loss_dg: torch.Tensor
    bridge: ScatterBridgeOut
    pmd: PMDFlatOutput
    pred: Optional[CorticalPredOut] = None
    g1_scan: Optional[CarrierScanOut] = None
    g2_scan: Optional[CarrierScanOut] = None
    hip_out: Optional[HippocampusInterleavedOut] = None
    x3: Optional[torch.Tensor] = None
    f_out_seg: Optional[torch.Tensor] = None
    expert_counts: list[torch.Tensor] = field(default_factory=list)
    metrics_tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def materialize_metrics(self) -> dict[str, float]:
        """Single-pass host materialization for logging (never called inside hot forward/backward)."""
        keys = list(self.metrics_tensors.keys())
        if not keys:
            return {}
        vec = torch.stack([self.metrics_tensors[k].detach().float().reshape(()) for k in keys], dim=0)
        vals = vec.to("cpu").tolist()
        return {k: float(v) for k, v in zip(keys, vals)}


class DCPAModelFlat(nn.Module):
    """Top-level DCPA-Flat v3.3 architecture."""

    def __init__(self, cfg: DCPAFlatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_concept
        d_s = cfg.d_s

        # 1. Sensorimotor micro-cores with strictly tied `embed` and `w_out` (independent temperatures)
        self.sgsc = SGSCFlat(cfg)
        self.pmd = PMDFlat(cfg, shared_embed=self.sgsc.embed, shared_w_out=self.sgsc.w_out)

        # 2. Dual contractive concept-graph carriers (G1 free-flowing post; G2 Slot-0 pinnable causal prior)
        self.carrier_g1 = ContractiveGraphCarrier(cfg, pin_enabled=False, return_prior=False)
        self.carrier_g2 = ContractiveGraphCarrier(cfg, pin_enabled=True, return_prior=True)

        # 3. 8-layer deep cortical stack (L1 pre-G1, L2-L3 post-G1, L4-L6 G2_prior + W=128, L7-L8 refine)
        self.cortical = CorticalStackFlat(cfg)

        # 4. Sandwiched chunk-folded Hippocampus & residual-whitened dual-morpheme DG
        self.hippocampus = HippocampusFlat(cfg)

        # 5. Pure 48KB f_stem Holographic Copy Ring (W=128 + always-open sink column)
        self.copy_ring = HolographicCopyRing(cfg)

        # 6. Causal seed & three-way stem confluence heads
        self.bos_concept = nn.Parameter(torch.zeros((D,), dtype=torch.float32))
        self.w_lex_slow = nn.Linear(D, d_s, bias=False)
        self.w_delta_syn = nn.Linear(D + cfg.d_l, d_s, bias=True)
        # Sensory warmup bypass head (Stage 1: maps x0_prev -> f_out_seg when bypassing cortex/carriers)
        self.w_sensory_bypass = nn.Linear(D, d_s, bias=False)

        # 7. JEPA predictive head on strictly causal `z_hat` -> parameter-free Unit-RMSNorm(`x3.detach()`)
        self.jepa_pred = nn.Sequential(
            nn.Linear(D, D, bias=False),
            nn.SiLU(),
            nn.Linear(D, D, bias=False),
        )

        self._init_top_weights()

    def _init_top_weights(self) -> None:
        nn.init.normal_(self.bos_concept, std=0.02)
        nn.init.xavier_uniform_(self.w_lex_slow.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w_delta_syn.weight, gain=0.2)
        nn.init.zeros_(self.w_delta_syn.bias)
        nn.init.xavier_uniform_(self.w_sensory_bypass.weight, gain=0.5)
        nn.init.xavier_uniform_(self.jepa_pred[0].weight)
        nn.init.normal_(self.jepa_pred[2].weight, std=0.02)

    @staticmethod
    def _unit_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Parameter-free Unit-RMSNorm: norm = sqrt(D), O(1) per-dim MSE, immune to scale collapse."""
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)

    def _g2_carrier_step_adapter(
        self,
        state: CarrierGraphState,
        x_in: torch.Tensor,
        reset: torch.Tensor,
        valid: torch.Tensor,
        pin_trig: torch.Tensor,
        unpin_trig: torch.Tensor,
    ) -> tuple[CarrierGraphState, CarrierGraphState, torch.Tensor, torch.Tensor, Any]:
        """Bridge `ContractiveGraphCarrier.step_single(..., return_slot_info=True)` to `HippocampusFlat`."""
        next_state, prior_state, evict_weight, evict_target, _empty_write, _has_empty, slot_info = (
            self.carrier_g2.step_single(
                state=state,
                x_in=x_in,
                seg_reset=reset,
                seg_valid=valid,
                pin_trig=pin_trig,
                unpin_trig=unpin_trig,
                return_slot_info=True,
            )
        )
        return next_state, prior_state, evict_weight, evict_target, slot_info

    # ------------------------------------------------------------------
    # Plugin Hot-Mount & Stage Freeze Controls
    # ------------------------------------------------------------------

    def mount_expert_pack(self, pack: PluginExpertPack) -> None:
        """Hot-mount an Additive Dead-Zone plugin pack across all 8 cortical layers."""
        self.cortical.mount_expert_pack(pack)

    def unmount_expert_pack(self, pack_id: str) -> bool:
        """Unmount a mounted plugin pack cleanly (restoring bit-exact baseline)."""
        return self.cortical.unmount_expert_pack(pack_id)

    def freeze_base_for_plugin_training(self) -> None:
        """Freeze 100% of base model weights (for domain or personal-sleep plugin training)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.hippocampus.dg.freeze_whiten(True)

    def set_stage_mode(self, mode: FlatStageMode) -> None:
        """Configure trainable parameters for developmental stages."""
        for p in self.parameters():
            p.requires_grad_(True)
        if mode == "sensory":
            for mod in (self.carrier_g1, self.carrier_g2, self.cortical, self.hippocampus, self.copy_ring):
                for p in mod.parameters():
                    p.requires_grad_(False)
        elif mode == "cortex":
            for p in self.hippocampus.parameters():
                p.requires_grad_(False)
        elif mode == "hippocampus":
            # Stage 4: freeze sensory micro-cores & keep DG whitening + hippocampal gates/adapters trainable
            self.sgsc.embed.weight.requires_grad_(False)
            self.sgsc.theta_raw.requires_grad_(False)
            self.hippocampus.dg.freeze_whiten(False)
        elif mode == "full":
            # Stage 5 / inference stationarity: freeze DG whitening so fast-weight key geometry never drifts
            self.hippocampus.dg.freeze_whiten(True)

    def count_parameters(self) -> dict[str, int]:
        """Exact trainable parameter breakdown + average active parameters per segment."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        def _count_all(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        sgsc_n = _count_all(self.sgsc)
        tied_ids = {id(self.sgsc.embed.weight), id(self.sgsc.w_out.weight)}
        pmd_unique = sum(p.numel() for p in self.pmd.parameters() if id(p) not in tied_ids)
        g1_n = _count_all(self.carrier_g1)
        g2_n = _count_all(self.carrier_g2)
        cort_n = _count_all(self.cortical)
        hip_n = _count_all(self.hippocampus)
        ring_n = _count_all(self.copy_ring)
        trainable_total = sum(p.numel() for p in self.parameters())
        buffers_total = sum(b.numel() for b in self.buffers())

        # Average active parameters per segment:
        # L1-L6 run on 100% segments (1 shared + top_k/E routed); L7-L8 run on refine_frac (25%) segments
        inactive_routed = 0.0
        for idx, layer in enumerate(self.cortical.layers):
            moe = layer.moe
            routed_n = moe.w_gu.numel() + moe.w_d.numel()
            layer_total = _count_all(layer)
            if idx < self.cfg.n_layers - self.cfg.n_refine_layers:
                inactive_routed += routed_n * (1.0 - float(moe.top_k) / max(moe.num_experts, 1))
            else:
                # L7-L8 only execute on `refine_frac` of segments, and within those only top_k/E routed experts
                active_l = (layer_total - routed_n) + routed_n * (float(moe.top_k) / max(moe.num_experts, 1))
                inactive_routed += layer_total - active_l * float(self.cfg.refine_frac)

        active_avg = int(round(trainable_total - inactive_routed))
        return {
            "sgsc": sgsc_n,
            "pmd_unique": pmd_unique,
            "carrier_g1": g1_n,
            "carrier_g2": g2_n,
            "cortical": cort_n,
            "hippocampus": hip_n,
            "copy_ring": ring_n,
            "trainable_total": trainable_total,
            "buffers_total": buffers_total,
            "total_with_buffers": trainable_total + buffers_total,
            "active_avg_per_seg": active_avg,
        }

    def parameter_groups(self) -> list[dict[str, Any]]:
        """Five-group optimizer schedule with strict zero weight decay on all 1D/0D norms, biases,
        scalars, stack/carrier gate parameters, and initial graph/motor state vectors."""
        scalar_params: list[nn.Parameter] = [
            self.sgsc.theta_raw,
            self.sgsc.log_temp_sgsc,
            self.pmd.log_temp_pmd,
        ] + self.hippocampus.scalar_gate_parameters()
        scalar_ids = {id(p) for p in scalar_params}

        # Initial state matrices, embeddings, & row-L2-normalized heads (`ndim >= 2` but exempt from WD)
        no_decay_explicit_ids: set[int] = {
            id(self.carrier_g1.init_slots_raw),
            id(self.carrier_g1.init_edges_raw),
            id(self.carrier_g2.init_slots_raw),
            id(self.carrier_g2.init_edges_raw),
            id(self.sgsc.conv_w),
            id(self.sgsc.embed.weight),
            id(self.sgsc.w_out.weight),
            id(self.sgsc.lang_embed.weight),
            id(self.sgsc.style_embed.weight),
        }

        seen: set[int] = set(scalar_ids)
        micro_decay: list[nn.Parameter] = []
        micro_no_decay: list[nn.Parameter] = []
        for mod in (self.sgsc, self.pmd):
            for p in mod.parameters():
                pid = id(p)
                if pid in seen or not p.requires_grad:
                    continue
                seen.add(pid)
                if p.ndim < 2 or pid in no_decay_explicit_ids:
                    micro_no_decay.append(p)
                else:
                    micro_decay.append(p)

        concept_decay: list[nn.Parameter] = []
        concept_no_decay: list[nn.Parameter] = []
        for p in self.parameters():
            pid = id(p)
            if pid in seen or not p.requires_grad:
                continue
            seen.add(pid)
            if p.ndim < 2 or pid in no_decay_explicit_ids:
                concept_no_decay.append(p)
            else:
                concept_decay.append(p)

        scalar_trainable = [p for p in scalar_params if p.requires_grad]
        groups: list[dict[str, Any]] = [
            {"name": "micro_cores_decay", "params": micro_decay, "lr": self.cfg.lr_micro, "base_lr": self.cfg.lr_micro, "weight_decay": self.cfg.weight_decay},
            {"name": "micro_cores_no_decay", "params": micro_no_decay, "lr": self.cfg.lr_micro, "base_lr": self.cfg.lr_micro, "weight_decay": 0.0},
            {"name": "concept_core_decay", "params": concept_decay, "lr": self.cfg.lr_concept, "base_lr": self.cfg.lr_concept, "weight_decay": self.cfg.weight_decay},
            {"name": "concept_core_no_decay", "params": concept_no_decay, "lr": self.cfg.lr_concept, "base_lr": self.cfg.lr_concept, "weight_decay": 0.0},
            {"name": "scalar_gates", "params": scalar_trainable, "lr": self.cfg.lr_scalar_gate, "base_lr": self.cfg.lr_scalar_gate, "weight_decay": 0.0},
        ]
        return [g for g in groups if g["params"]]

    def enable_compiled_submodules(
        self,
        device: torch.device,
        *,
        mode: str = "default",
        cache_size_limit: int = 64,
    ) -> list[str]:
        """Selectively `torch.compile` static-grid submodules (CUDA only) while preserving clean checkpointing."""
        if device.type != "cuda" or not hasattr(torch, "compile"):
            return []
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            torch._dynamo.config.cache_size_limit = max(int(cache_size_limit), 64)
            if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
                torch._dynamo.config.accumulated_cache_size_limit = max(int(cache_size_limit) * 4, 256)
        compiled: list[str] = []
        compile_kw: dict[str, Any] = {"dynamic": False, "mode": mode}
        self.sgsc = torch.compile(self.sgsc, **compile_kw)  # type: ignore[assignment]
        compiled.append("sgsc")
        self.carrier_g1.enable_compiled_chunk(**compile_kw)
        compiled.append("carrier_g1.chunk")
        self.carrier_g2.enable_compiled_chunk(**compile_kw)
        compiled.append("carrier_g2.chunk")
        self.cortical.enable_compiled_methods(**compile_kw)
        compiled.append("cortical")
        self.copy_ring = torch.compile(self.copy_ring, **compile_kw)  # type: ignore[assignment]
        compiled.append("copy_ring")
        self.pmd = torch.compile(self.pmd, **compile_kw)  # type: ignore[assignment]
        compiled.append("pmd")
        return compiled

    def disable_compiled_submodules(self) -> None:
        """Unwrap any `torch.compile` submodule wrappers cleanly (e.g. before CPU export or shape-change eval)."""
        from dcpa.compile_utils import unwrap_compiled_module

        self.sgsc = unwrap_compiled_module(self.sgsc)  # type: ignore[assignment]
        self.copy_ring = unwrap_compiled_module(self.copy_ring)  # type: ignore[assignment]
        self.pmd = unwrap_compiled_module(self.pmd)  # type: ignore[assignment]
        self.carrier_g1.disable_compiled_chunk()
        self.carrier_g2.disable_compiled_chunk()
        self.cortical.disable_compiled_methods()

    @torch.no_grad()
    def post_optimizer_step(self, out: DCPAFlatForwardOutput, expert_counts_override: Optional[list[torch.Tensor]] = None) -> None:
        """Non-gradient post-step updates: MoE `b_bal` (supports micro-batch accumulated counts)."""
        counts = out.expert_counts if expert_counts_override is None else expert_counts_override
        if counts:
            self.cortical.update_all_balance_biases(counts)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: StaticGridBatch,
        *,
        stage: FlatStageMode = "cortex",
        tau_s: float = 1.0,
        hip_enabled: Optional[bool] = None,
        use_checkpoint: bool = False,
        forced_spikes: Optional[torch.Tensor] = None,
        write_threshold: Optional[float] = None,
    ) -> DCPAFlatForwardOutput:
        """Execute full static-grid forward pass `[P, T] -> [P, S] -> [P, T, 256]`.

        Args:
          batch:       Frozen `StaticGridBatch` `[P, T]`.
          stage:       `"sensory"` (SGSC+PMD warmup bypass), `"cortex"` (Stages 2-3, no hip),
                       `"hippocampus"` or `"full"` (Stages 4-5, full `HIPR <-> G2` + `M(L<-C)`).
          tau_s:       SGSC STE spike temperature.
          hip_enabled: Explicit override for Hippocampus activation (defaults to `stage in ("hippocampus", "full")`).
          use_checkpoint: Enable activation checkpointing across `C_scan` carrier chunks.
        """
        cfg = self.cfg
        P, T = batch.x_bytes.shape
        S = cfg.grid_segs
        device = batch.x_bytes.device
        zero = torch.zeros((), device=device, dtype=torch.float32)

        use_hip = bool(stage in ("hippocampus", "full")) if hip_enabled is None else bool(hip_enabled)

        # 1. SGSCFlat + static S+1 overflow-protected prefix scatter bridge
        bridge: ScatterBridgeOut = self.sgsc(batch, tau_s=tau_s, forced_spikes=forced_spikes, grid_segs=S)

        # Auxiliary SGSC causal byte prior loss (strictly masked over valid non-overflow bytes)
        with fp32_island(device):
            valid_byte = (batch.doc_id_byte >= 0) & (~bridge.byte_overflow)
            eff_w_byte = batch.loss_weight.float() * valid_byte.float()
            prior_ce_raw = F.cross_entropy(
                bridge.prior_logits.float().reshape(P * T, cfg.vocab_size),
                batch.x_bytes.to(torch.long).reshape(P * T),
                reduction="none",
            ).view(P, T)
            loss_sgsc_prior = (prior_ce_raw * eff_w_byte).sum() / eff_w_byte.sum().clamp_min(1.0)

            # Spike rate regularization towards target_spike_rate (~1/9.5)
            valid_f = valid_byte.float()
            spike_rate_smooth = (bridge.s_tilde.float() * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            loss_rate = (spike_rate_smooth - float(cfg.target_spike_rate)).pow(2)
            overflow_ratio = bridge.byte_overflow.float().mean()

        # Fast Stage-1 sensory warmup path (identical warmup snapshot for 3-way 5k killgate!)
        if stage == "sensory":
            x0_prev = torch.cat(
                [torch.zeros_like(bridge.x0[:, :1, :]), bridge.x0[:, :-1, :]], dim=1
            )
            bos = self.bos_concept.to(device=device, dtype=bridge.x0.dtype).view(1, 1, cfg.d_concept)
            x0_causal = torch.where(bridge.seg_reset.unsqueeze(-1), bos.expand(P, S, -1), x0_prev)
            f_out_seg = self.w_sensory_bypass(x0_causal)
            delta_syn_seg = self.w_delta_syn(torch.cat([x0_causal, bridge.seg_l_reg.to(x0_causal.dtype)], dim=-1))
            pmd_out = self.pmd(batch, bridge, f_out_seg, delta_syn_seg=delta_syn_seg)
            loss_lang = pmd_out.ce_loss
            loss_halt = pmd_out.halt_loss
            loss_total = (
                loss_lang
                + float(cfg.lambda_sgsc_prior) * loss_sgsc_prior
                + float(cfg.lambda_rate) * loss_rate
                + float(cfg.lambda_halt) * loss_halt
            )
            metrics = {
                "loss_total": loss_total,
                "loss_lang": loss_lang,
                "loss_sgsc_prior": loss_sgsc_prior,
                "loss_jepa": zero,
                "loss_unc": zero,
                "loss_var": zero,
                "loss_rate": loss_rate,
                "loss_halt": loss_halt,
                "loss_dg": zero,
                "spike_rate": (bridge.spikes.float() * valid_f).sum() / valid_f.sum().clamp_min(1.0),
                "overflow_byte_ratio": overflow_ratio,
                "tau_think": self.cortical.tau_think.float(),
                "empty_write_rate": zero,
                "mean_ach": zero,
                "mean_beta": zero,
            }
            return DCPAFlatForwardOutput(
                loss_total=loss_total,
                loss_lang=loss_lang,
                loss_sgsc_prior=loss_sgsc_prior,
                loss_jepa=zero,
                loss_unc=zero,
                loss_var=zero,
                loss_rate=loss_rate,
                loss_halt=loss_halt,
                loss_dg=zero,
                bridge=bridge,
                pmd=pmd_out,
                f_out_seg=f_out_seg,
                metrics_tensors=metrics,
            )

        # 2. L1 (pre-G1): optional W=128 causal local window + Base/Plugin MoE
        x1, aux_pre = self.cortical.forward_pre_g1(
            x0=bridge.x0,
            f_stem=bridge.f_stem,
            seg_doc_id=bridge.seg_doc_id,
            seg_end_depth=bridge.seg_end_depth,
            seg_valid=bridge.seg_valid,
        )

        # 3. Carrier G1: free-flowing post-update contractive graph scan
        g1_scan = self.carrier_g1.forward_scan(
            x_seq=x1,
            seg_reset=bridge.seg_reset,
            seg_valid=bridge.seg_valid,
            use_checkpoint=use_checkpoint,
        )

        # 4. L2-L3 (post-G1): shared G1 KV graph cross-attention + MoE + optional +32KB semantic ring
        x3, c_kv_sem, aux_post = self.cortical.forward_post_g1(
            x1=x1,
            g1_out=g1_scan,
            seg_l_reg=bridge.seg_l_reg,
            seg_valid=bridge.seg_valid,
        )

        # 5. Carrier G2 (+ R4 causal chunk-interleaved Hippocampus when enabled)
        hip_out: Optional[HippocampusInterleavedOut] = None
        write_events: Optional[HippocampusWriteEvents] = None
        if use_hip:
            g2_init = self.carrier_g2.init_state(P, device=device, dtype=x3.dtype)
            hip_out = self.hippocampus.forward_interleaved_with_carrier_g2(
                x3=x3,
                f_stem=bridge.f_stem,
                sem_norm=bridge.sem_norm,
                mean_info=bridge.mean_info,
                seg_doc_id=bridge.seg_doc_id,
                seg_reset=bridge.seg_reset,
                seg_valid=bridge.seg_valid,
                seg_pin_trig=bridge.seg_pin_trig,
                seg_unpin_trig=bridge.seg_unpin_trig,
                g2_state=g2_init,
                carrier_step_fn=self._g2_carrier_step_adapter,
                hip_enabled=True,
                write_threshold=write_threshold,
                update_homeostasis=self.training,
            )
            g2_scan = hip_out.g2_scan
            write_events = hip_out.write_events
            loss_dg = hip_out.l_dg
            mean_ach = hip_out.mean_ach
            mean_beta = hip_out.mean_beta
        else:
            g2_scan = self.carrier_g2.forward_scan(
                x_seq=x3,
                seg_reset=bridge.seg_reset,
                seg_valid=bridge.seg_valid,
                seg_pin_trig=bridge.seg_pin_trig,
                seg_unpin_trig=bridge.seg_unpin_trig,
                use_checkpoint=use_checkpoint,
            )
            loss_dg = zero
            mean_ach = zero
            mean_beta = zero

        # 6. Strictly causal shift of `x3` for prior prediction seed & soft IoR
        x3_shifted = torch.cat([torch.zeros_like(x3[:, :1, :]), x3[:, :-1, :]], dim=1)
        x3_prev = torch.where(bridge.seg_reset.unsqueeze(-1), torch.zeros_like(x3_shifted), x3_shifted)
        bos = self.bos_concept.to(device=device, dtype=x3.dtype).view(1, 1, cfg.d_concept)
        x_seed = torch.where(bridge.seg_reset.unsqueeze(-1), bos.expand(P, S, -1), x3_prev)

        # Ring KV for L4/L6 unified window attention (`c_kv_sem` +32KB default, or shared `f_stem` 0KB)
        ring_kv = bridge.f_stem if (c_kv_sem is None) else c_kv_sem

        # 7. L4-L6 (full prior prediction + uncertainty u_n) + L7-L8 (Top-25% / threshold refinement)
        pred_out: CorticalPredOut = self.cortical.forward_predict_and_refine(
            x_seed=x_seed,
            x3_prev=x3_prev,
            g2_prior=g2_scan,
            ring_kv=ring_kv,
            seg_doc_id=bridge.seg_doc_id,
            seg_start_depth=bridge.seg_start_depth,
            seg_end_depth=bridge.seg_end_depth,
            seg_l_reg=bridge.seg_l_reg,
            seg_valid=bridge.seg_valid,
        )

        # Collect all 8 layers' base expert load counts for post-step non-grad `b_bal` update
        all_expert_counts: list[torch.Tensor] = (
            list(aux_pre.get("expert_counts", []))
            + list(aux_post.get("expert_counts", []))
            + list(pred_out.expert_counts)
        )

        # 8. Three-way stem confluence: Slow Cortex + HolographicCopyRing + Hippocampus M^{L<-C}
        f_slow = self.w_lex_slow(pred_out.z_hat)
        f_copy = self.copy_ring(
            q_state=pred_out.z_hat,
            f_stem=bridge.f_stem,
            seg_doc_id=bridge.seg_doc_id,
            seg_start_depth=bridge.seg_start_depth,
            seg_end_depth=bridge.seg_end_depth,
        )
        f_cort = f_slow + f_copy
        if use_hip and write_events is not None:
            f_hip, rho_out = self.hippocampus.read_expression_chunk_triangular(
                q_concept=pred_out.z_hat,
                seg_doc_id=bridge.seg_doc_id,
                seg_valid=bridge.seg_valid,
                write_events=write_events,
                seg_reset=bridge.seg_reset,
                hip_enabled=True,
                update_homeostasis=self.training,
            )
            rho_u = rho_out.unsqueeze(-1).to(dtype=torch.float32)
            f_out_seg = ((1.0 - rho_u) * f_cort.float() + rho_u * f_hip.float()).to(f_cort.dtype)
        else:
            f_out_seg = f_cort

        delta_syn_seg = self.w_delta_syn(
            torch.cat([pred_out.z_hat, bridge.seg_l_reg.to(pred_out.z_hat.dtype)], dim=-1)
        )

        # 9. PMDFlat: zero-sync gather -> inverse RoPE demod -> 128B stack -> byte CE + L_halt
        pmd_out: PMDFlatOutput = self.pmd(batch, bridge, f_out_seg, delta_syn_seg=delta_syn_seg)
        loss_lang = pmd_out.ce_loss
        loss_halt = pmd_out.halt_loss

        # 10. JEPA, Uncertainty Regression (R5), and Concept Variance Hinge losses
        with fp32_island(device):
            seg_v_f = bridge.seg_valid.float()
            seg_v_denom = seg_v_f.sum().clamp_min(1.0)

            # JEPA predictive target loss using parameter-free Unit-RMSNorm (norm = sqrt(D), O(1) MSE, collapse-proof)
            jepa_hat = self._unit_rmsnorm(self.jepa_pred(pred_out.z_hat), eps=cfg.eps)
            jepa_tgt = self._unit_rmsnorm(x3.detach(), eps=cfg.eps)
            seg_jepa_err = (jepa_hat - jepa_tgt).pow(2).mean(dim=-1)  # [P, S] O(1)
            loss_jepa = (seg_jepa_err * seg_v_f).sum() / seg_v_denom

            # Decoupled pre-refinement (L6) JEPA difficulty term to avoid L7-L8 thermostat feedback
            with torch.no_grad():
                jepa_hat_6 = self._unit_rmsnorm(self.jepa_pred(pred_out.z_hat_6.detach()), eps=cfg.eps)
                seg_jepa_err_6 = (jepa_hat_6 - jepa_tgt).pow(2).mean(dim=-1)

            # R5: Self-supervised cognitive uncertainty regression target from per-segment NLL + pre-refine JEPA error
            seg_lens_nll = static_segment_scatter_add(valid_byte.float(), bridge.byte_to_seg, S)
            seg_nll = static_segment_mean(
                pmd_out.per_byte_ce.float() * valid_byte.float(),
                bridge.byte_to_seg,
                seg_lens_nll,
                S,
            )  # [P, S]
            seg_difficulty = (0.5 * seg_nll + 0.5 * seg_jepa_err_6).detach().clamp(0.0, 12.0)
            unc_err = F.smooth_l1_loss(pred_out.u_n.float(), seg_difficulty, reduction="none")
            loss_unc = (unc_err * seg_v_f).sum() / seg_v_denom

            # Concept variance hinge on `x3` across valid segments to prevent latent collapse
            x3_f = x3.float()
            x3_mean = (x3_f * seg_v_f.unsqueeze(-1)).sum(dim=(0, 1)) / seg_v_denom
            x3_var = ((x3_f - x3_mean.view(1, 1, -1)).pow(2) * seg_v_f.unsqueeze(-1)).sum(dim=(0, 1)) / seg_v_denom
            x3_std = torch.sqrt(x3_var + 1e-6)
            loss_var = F.relu(1.0 - x3_std).mean()

            loss_total = (
                loss_lang
                + float(cfg.lambda_sgsc_prior) * loss_sgsc_prior
                + float(cfg.lambda_jepa) * loss_jepa
                + float(cfg.lambda_unc) * loss_unc
                + float(cfg.lambda_var) * loss_var
                + float(cfg.lambda_rate) * loss_rate
                + float(cfg.lambda_halt) * loss_halt
                + (float(cfg.lambda_dg) * loss_dg if use_hip else 0.0)
            )

            empty_write_rate = (
                g1_scan.empty_write_rate.float()
                if g1_scan.empty_write_rate is not None
                else zero
            )
            metrics = {
                "loss_total": loss_total,
                "loss_lang": loss_lang,
                "loss_sgsc_prior": loss_sgsc_prior,
                "loss_jepa": loss_jepa,
                "loss_unc": loss_unc,
                "loss_var": loss_var,
                "loss_rate": loss_rate,
                "loss_halt": loss_halt,
                "loss_dg": loss_dg,
                "spike_rate": (bridge.spikes.float() * valid_f).sum() / valid_f.sum().clamp_min(1.0),
                "overflow_byte_ratio": overflow_ratio,
                "x3_std_mean": x3_std.mean(),
                "x3_std_min": x3_std.min(),
                "tau_think": pred_out.tau_think.float(),
                "refine_rate": (pred_out.refine_mask.float() * seg_v_f).sum() / seg_v_denom,
                "empty_write_rate": empty_write_rate,
                "mean_ach": mean_ach,
                "mean_beta": mean_beta,
            }

        return DCPAFlatForwardOutput(
            loss_total=loss_total,
            loss_lang=loss_lang,
            loss_sgsc_prior=loss_sgsc_prior,
            loss_jepa=loss_jepa,
            loss_unc=loss_unc,
            loss_var=loss_var,
            loss_rate=loss_rate,
            loss_halt=loss_halt,
            loss_dg=loss_dg,
            bridge=bridge,
            pmd=pmd_out,
            pred=pred_out,
            g1_scan=g1_scan,
            g2_scan=g2_scan,
            hip_out=hip_out,
            x3=x3,
            f_out_seg=f_out_seg,
            expert_counts=all_expert_counts,
            metrics_tensors=metrics,
        )

    @torch.no_grad()
    def forward_streaming_sequence(
        self,
        raw_bytes_1d: torch.Tensor,
        *,
        lang_id: int = 0,
        style_id: int = 0,
        hip_enabled: bool = True,
        write_threshold: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """Exact single-stream O(1) reference execution for Gate 6 parity (`Python <-> dcpa-core-rs`).

        Runs byte-by-byte SGSC -> segment-by-segment L1 / G1 / L2-L3 / Hippocampus+G2 / L4-L8 /
        CopyRing / PMD in eval threshold mode (`u_n > tau_think`), returning per-module reference
        tensors matching both batched `DCPAModelFlat.forward` (eval) and Rust `DcpaFlatEngine` `< 1e-5`.
        """
        was_training = self.training
        self.eval()
        try:
            cfg = self.cfg
            device = next(self.parameters()).device
            rb = raw_bytes_1d.to(device=device, dtype=torch.uint8).reshape(-1)
            T = int(rb.shape[0])
            W = cfg.window_size
            D = cfg.d_concept
            d_s = cfg.d_s

            # 1. Exact streaming SGSC byte ingest -> completed segments
            sgsc_res = self.sgsc.encode_streaming_segments(rb, lang_id=int(lang_id), style_id=int(style_id))
            M = int(sgsc_res["f_stem"].shape[0])
            l_reg = self.sgsc.get_lang_vec(
                torch.tensor([int(lang_id)], device=device, dtype=torch.long),
                torch.tensor([int(style_id)], device=device, dtype=torch.long),
            ).float()  # [1, 64]

            # Initialize strictly O(1) streaming runtime states
            g1_state = self.carrier_g1.init_state(1, device=device, dtype=torch.float32)
            g2_state = self.carrier_g2.init_state(1, device=device, dtype=torch.float32)
            hip_state = self.hippocampus.init_chunk_state(1, device=device, dtype=torch.float32)

            ring_stem, ring_pos, ring_depth, ring_valid = self.copy_ring.init_ring_state(
                1, device=device, dtype=torch.float32
            )
            ring_sem: Optional[torch.Tensor] = (
                None
                if cfg.share_stem_ring
                else torch.zeros((1, W, cfg.d_sem_ring), device=device, dtype=torch.float32)
            )
            prev_stem = torch.zeros((1, d_s), device=device, dtype=torch.float32)
            x3_prev = torch.zeros((1, D), device=device, dtype=torch.float32)

            # PMD streaming motor & 128B stack states
            pmd_m = self.pmd.m0.detach().to(device=device, dtype=torch.float32).unsqueeze(0).clone()
            pmd_xi = torch.zeros((1, cfg.stack_depth, cfg.stack_tag_dim), device=device, dtype=torch.float32)

            # Collect per-segment & per-byte reference traces
            x1_list: list[torch.Tensor] = []
            g1_slots_list: list[torch.Tensor] = []
            x3_list: list[torch.Tensor] = []
            g2_prior_slots_list: list[torch.Tensor] = []
            z_hat_list: list[torch.Tensor] = []
            u_n_list: list[torch.Tensor] = []
            f_copy_list: list[torch.Tensor] = []
            f_hip_list: list[torch.Tensor] = []
            rho_out_list: list[torch.Tensor] = []
            f_out_list: list[torch.Tensor] = []
            max_beta_list: list[torch.Tensor] = []

            byte_logits_all = torch.zeros((T, cfg.vocab_size), device=device, dtype=torch.float32)
            p_seg_end_all = torch.zeros((T,), device=device, dtype=torch.float32)
            charge_e_all = torch.zeros((T,), device=device, dtype=torch.float32)

            # Determine segment byte ranges from `spikes`
            spikes_1d = sgsc_res["spikes"].bool()  # [T]
            seg_starts: list[int] = []
            seg_ends: list[int] = []
            cur_st = 0
            for t in range(T):
                if bool(spikes_1d[t].item()):
                    seg_starts.append(cur_st)
                    seg_ends.append(t + 1)
                    cur_st = t + 1
            assert len(seg_starts) == M, f"Segment count mismatch: {len(seg_starts)} vs {M}"

            valid_1 = torch.ones((1,), device=device, dtype=torch.bool)
            doc0 = torch.zeros((1,), device=device, dtype=torch.int32)
            bos = self.bos_concept.detach().to(device=device, dtype=torch.float32).view(1, D)

            for s in range(M):
                reset_s = torch.tensor([s == 0], device=device, dtype=torch.bool)
                pos_s = torch.tensor([s], device=device, dtype=torch.int32)
                f_s = sgsc_res["f_stem"][s : s + 1].float()
                x0_s = sgsc_res["x0"][s : s + 1].float()
                sn_s = sgsc_res["sem_norm"][s : s + 1].float()
                mi_s = sgsc_res["mean_info"][s : s + 1].float()
                st_dep_s = sgsc_res["seg_start_depth"][s : s + 1].to(torch.int32)
                en_dep_s = sgsc_res["seg_end_depth"][s : s + 1].to(torch.int32)
                pin_s = sgsc_res["seg_pin_trig"][s : s + 1].bool()
                unpin_s = sgsc_res["seg_unpin_trig"][s : s + 1].bool()

                # (A) Strictly causal `< s` prediction BEFORE absorbing segment `s`
                x_seed_s = bos if s == 0 else x3_prev
                ring_kv_pre = ring_stem if ring_sem is None else ring_sem
                # Post-reset pre-update G2 prior matches batched `Carrier G2` when `s == 0`
                g2_prior_for_pred = (
                    self.carrier_g2.init_state(1, device=device, dtype=torch.float32) if s == 0 else g2_state
                )
                if s > 0 and bool(unpin_s.item()) and self.carrier_g2.pin_enabled:
                    g2_prior_for_pred = CarrierGraphState(
                        slots=g2_state.slots,
                        edges=g2_state.edges,
                        occ=g2_state.occ,
                        slot0_pinned=torch.zeros_like(g2_state.slot0_pinned),
                    )

                pred_step = self.cortical.step_streaming_predict_and_refine(
                    x_seed_step=x_seed_s,
                    x3_prev_step=x3_prev,
                    g2_prior_state=g2_prior_for_pred,
                    ring_kv=ring_kv_pre,
                    ring_pos=ring_pos,
                    ring_depth=ring_depth,
                    ring_valid=ring_valid,
                    cur_pos=pos_s,
                    cur_start_depth=st_dep_s,
                    l_reg_step=l_reg,
                )
                z_hat_s = pred_step.z_hat.float()
                f_slow_s = self.w_lex_slow(z_hat_s).float()
                f_copy_s = self.copy_ring.step_streaming(
                    q_state=z_hat_s,
                    ring_stem=ring_stem,
                    ring_pos=ring_pos,
                    ring_valid=ring_valid,
                    cur_pos=pos_s,
                    ring_depth=ring_depth,
                    cur_depth=st_dep_s,
                ).float()
                f_cort_s = f_slow_s + f_copy_s

                # Strictly causal expression read from pre-write `hip_state`
                if hip_enabled:
                    q_idx, q_val = self.hippocampus.dg.dg_sem(z_hat_s)
                    m_l_pre = torch.zeros_like(hip_state.m_l_base) if s == 0 else hip_state.m_l_base
                    u_sem_pre = torch.zeros_like(hip_state.u_sem) if s == 0 else hip_state.u_sem
                    cols_l = m_l_pre.gather(1, q_idx.unsqueeze(-1).expand(1, cfg.s_active, d_s))
                    f_hip_s = torch.einsum("ps,psd->pd", q_val.float(), cols_l)
                    g_sem = (q_val.float().pow(2) * u_sem_pre.gather(1, q_idx)).sum(dim=-1).clamp(0.0, 1.0)
                    rho_s = g_sem * torch.sigmoid(self.hippocampus.alpha_out * (g_sem - self.hippocampus.theta_out))
                    f_out_s = (1.0 - rho_s.unsqueeze(-1)) * f_cort_s + rho_s.unsqueeze(-1) * f_hip_s
                else:
                    f_hip_s = torch.zeros((1, d_s), device=device, dtype=torch.float32)
                    rho_s = torch.zeros((1,), device=device, dtype=torch.float32)
                    f_out_s = f_cort_s

                delta_syn_s = self.w_delta_syn(torch.cat([z_hat_s, l_reg], dim=-1)).float()

                # (B) PMD byte-by-byte motor decoding over bytes belonging to segment `s`
                charge_cur = self.pmd.init_segment_charge(f_out_s, l_reg, delta_syn=delta_syn_s)
                t0, t1 = seg_starts[s], seg_ends[s]
                for local_i, t in enumerate(range(t0, t1)):
                    pb_val = 0 if t == 0 else int(rb[t - 1].item())
                    prev_b = torch.tensor([pb_val], device=device, dtype=torch.uint8)
                    scratch_end_reset = bool(
                        t >= 3
                        and int(rb[t - 3].item()) == 0x1D
                        and int(rb[t - 2].item()) == ord("E")
                        and int(rb[t - 1].item()) == 0x0A
                    )
                    extra_rst = torch.tensor([scratch_end_reset or (t == 0)], device=device, dtype=torch.bool)
                    b_logits, pmd_m, pmd_xi, charge_cur, p_end = self.pmd.step_streaming_byte(
                        prev_byte=prev_b,
                        f_out=f_out_s,
                        l_reg=l_reg,
                        tau_step=local_i + 1,
                        m_prev=pmd_m,
                        xi_prev=pmd_xi,
                        charge_prev=charge_cur,
                        delta_syn=delta_syn_s,
                        extra_stack_reset=extra_rst,
                    )
                    byte_logits_all[t] = b_logits[0]
                    p_seg_end_all[t] = p_end[0]
                    charge_e_all[t] = charge_cur[0]

                # (C) Absorb completed segment `s` into rings, L1, G1, L2-L3, and Hippocampus + G2
                ring_stem, ring_pos, ring_depth, ring_valid = self.copy_ring.push_ring(
                    ring_stem, ring_pos, ring_depth, ring_valid, f_s, pos_s, en_dep_s
                )
                x1_s = self.cortical.step_streaming_pre_g1(
                    x0_step=x0_s,
                    ring_stem=ring_stem,
                    ring_pos=ring_pos,
                    ring_depth=ring_depth,
                    ring_valid=ring_valid,
                    cur_pos=pos_s,
                    cur_end_depth=en_dep_s,
                ).float()
                g1_state, g1_post, _, _, _, _ = self.carrier_g1.step_single(
                    state=g1_state,
                    x_in=x1_s,
                    seg_reset=reset_s,
                    seg_valid=valid_1,
                )
                x3_s, c_kv_sem_s = self.cortical.step_streaming_post_g1(
                    x1_step=x1_s,
                    g1_state=g1_post,
                    l_reg_step=l_reg,
                )
                x3_s = x3_s.float()

                if hip_enabled:
                    hip_step = self.hippocampus.step_streaming(
                        x3=x3_s,
                        f_stem=f_s,
                        prev_stem=prev_stem,
                        sem_norm=sn_s,
                        mean_info=mi_s,
                        doc_id=doc0,
                        reset=reset_s,
                        valid=valid_1,
                        pin_trig=pin_s,
                        unpin_trig=unpin_s,
                        g2_state=g2_state,
                        hip_state=hip_state,
                        carrier_step_fn=self._g2_carrier_step_adapter,
                        q_concept=None,
                        hip_enabled=True,
                        write_threshold=write_threshold,
                    )
                    g2_state = hip_step.next_g2_state
                    hip_state = hip_step.next_hip_state
                    prev_stem = hip_step.next_prev_stem
                    g2_prior_slots_list.append(hip_step.prior_g2_state.slots[0])
                    max_beta_list.append(hip_step.max_beta[0])
                else:
                    g2_state, g2_prior_s, _, _, _, _ = self.carrier_g2.step_single(
                        state=g2_state,
                        x_in=x3_s,
                        seg_reset=reset_s,
                        seg_valid=valid_1,
                        pin_trig=pin_s,
                        unpin_trig=unpin_s,
                    )
                    g2_prior_slots_list.append(g2_prior_s.slots[0])
                    max_beta_list.append(torch.zeros((), device=device, dtype=torch.float32))

                if ring_sem is not None and c_kv_sem_s is not None:
                    ring_sem = torch.cat([ring_sem[:, 1:, :], c_kv_sem_s.float().unsqueeze(1)], dim=1)
                x3_prev = x3_s

                x1_list.append(x1_s[0])
                g1_slots_list.append(g1_post.slots[0])
                x3_list.append(x3_s[0])
                z_hat_list.append(z_hat_s[0])
                u_n_list.append(pred_step.u_n[0])
                f_copy_list.append(f_copy_s[0])
                f_hip_list.append(f_hip_s[0])
                rho_out_list.append(rho_s[0])
                f_out_list.append(f_out_s[0])

            return {
                "raw_bytes": rb,
                "spikes": spikes_1d.to(torch.uint8),
                "prior_logits": sgsc_res["prior_logits"].float(),
                "f_stem": sgsc_res["f_stem"].float(),
                "x0": sgsc_res["x0"].float(),
                "x1": torch.stack(x1_list, dim=0) if x1_list else torch.zeros((0, D), device=device),
                "g1_slots": torch.stack(g1_slots_list, dim=0) if g1_slots_list else torch.zeros((0, cfg.k_slots, D), device=device),
                "x3": torch.stack(x3_list, dim=0) if x3_list else torch.zeros((0, D), device=device),
                "g2_prior_slots": torch.stack(g2_prior_slots_list, dim=0) if g2_prior_slots_list else torch.zeros((0, cfg.k_slots, D), device=device),
                "z_hat": torch.stack(z_hat_list, dim=0) if z_hat_list else torch.zeros((0, D), device=device),
                "u_n": torch.stack(u_n_list, dim=0) if u_n_list else torch.zeros((0,), device=device),
                "f_copy": torch.stack(f_copy_list, dim=0) if f_copy_list else torch.zeros((0, d_s), device=device),
                "f_hip": torch.stack(f_hip_list, dim=0) if f_hip_list else torch.zeros((0, d_s), device=device),
                "rho_out": torch.stack(rho_out_list, dim=0) if rho_out_list else torch.zeros((0,), device=device),
                "f_out": torch.stack(f_out_list, dim=0) if f_out_list else torch.zeros((0, d_s), device=device),
                "max_beta": torch.stack(max_beta_list, dim=0) if max_beta_list else torch.zeros((0,), device=device),
                "byte_logits": byte_logits_all,
                "p_seg_end": p_seg_end_all,
                "charge_e": charge_e_all,
                "final_m_c": hip_state.m_c_base[0],
                "final_m_l": hip_state.m_l_base[0],
                "final_u_lex": hip_state.u_lex[0],
            }
        finally:
            self.train(was_training)


__all__ = [
    "FlatStageMode",
    "DCPAFlatForwardOutput",
    "DCPAModelFlat",
]
