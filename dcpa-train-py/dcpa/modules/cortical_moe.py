"""WS-3B: Decoupled Pluggable MoE (R2) and 8-Layer Cortical Stack (R1 + R5) for DCPA-Flat v3.3.

Implements:
  1. `DecoupledPluggableMoE`:
     - Base Pool: 1 Always-On Shared SwiGLU Expert + E Contiguous 3D Routed Top-1 SwiGLU Experts
       (`F.grouped_mm` on SM80+ CUDA BF16, static BMM fallback on CPU/MPS, exact row-independent
       eval path for Gate 6 parity).
     - Auxiliary-loss-free load balancing via non-grad `b_bal` (selection-only, NEVER multiplied
       into synthesis gate `g_base = sigmoid(selected_logit)`), with lower logit floor preventing
       dead-weight collapse.
     - Additive Dead-Zone Plugin Pool (`mount_expert_pack` / `unmount_expert_pack`):
       completely orthogonal to Base Top-1; hard dead-zone `d = ReLU(<h_hat, k_hat_p> - tau_p)`,
       bounded gain `g = tanh(alpha_p * d)`, Top-`plugin_top_k` active plugins added onto `y_base`.
       Non-target inputs (`d == 0`) have bit-exact `0.0` drift and `0` extra FLOPs in eval/streaming.
  2. `CorticalStackFlat`:
     - Shared `build_graph_kv` computed ONCE per graph (`G1` for L2–L3, `G2_prior` for L4–L8).
     - `forward_pre_g1` (L1, layer 0): optional `W=128` causal window (`0 <= Δs < W`, 2D Concept-RoPE,
       KV from `f_stem`) + `DecoupledPluggableMoE`.
     - `forward_post_g1` (L2–L3, layers 1–2): relation-biased `K=8` graph cross-attention over shared
       `G1` KV (with `log(occ)` prior and `seg_l_reg` register precedence) + MoE; optional projection
       to `c_kv_sem [P, S, d_sem_ring]` when `share_stem_ring=False`.
     - `forward_predict_and_refine` (L4–L6 + L7–L8, layers 3–7):
       * L4 (idx 3) and L6 (idx 5): when `use_local_window_layers=True`, unified `2*W + K + 1` (=265)
         single-softmax attention over strict-past `1 <= Δs <= W` local ring UNION `K=8` `G2_prior`
         UNION always-open learnable sink `b_sink`, plus zero-state soft IoR against `x3_prev`.
       * L5 (idx 4): pure `K=8` `G2_prior` cross-attention + MoE.
       * Uncertainty head `u_n = W_unc(z_hat_6) + b_unc`.
       * L7–L8 (layers 6–7, R5): static Top-25% MoD-gated refinement during training (gradients flow
         directly through `sigmoid(u_sel - tau_think.detach())` into `u_n`, EMA updating `tau_think`),
         strictly threshold-gated (`seg_valid & (u_n > tau_think)`) during eval/streaming/Gate 6.
     - Single-step `step_streaming_*` methods for exact `< 1e-5` Gate 6 streaming parity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.interfaces import (
    CarrierGraphState,
    CarrierScanOut,
    DCPAFlatConfig,
    PluginExpertPack,
)
from dcpa.ops.fp32 import fp32_island
from dcpa.ops.rmsnorm import RMSNorm
from dcpa.ops.rope import apply_concept_rope_2d

_SM80_GROUPED_MM_CACHE: dict[int, bool] = {}


def _device_supports_grouped_mm(device: torch.device, allow_cpu_bf16: bool = False) -> bool:
    """Return True if `F.grouped_mm` BF16 grouped GEMM is supported (SM >= 80 CUDA)."""
    if not hasattr(F, "grouped_mm"):
        return False
    if device.type == "cuda":
        idx = 0 if device.index is None else int(device.index)
        cached = _SM80_GROUPED_MM_CACHE.get(idx)
        if cached is None:
            try:
                major, _ = torch.cuda.get_device_capability(idx)
                cached = bool(major >= 8)
            except Exception:
                cached = False
            _SM80_GROUPED_MM_CACHE[idx] = cached
        return cached
    if device.type == "cpu" and allow_cpu_bf16:
        return True
    return False


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedGraphKV:
    """Shared multi-head Graph KV projections computed ONCE per carrier (`G1` or `G2_prior`)."""

    k_g: torch.Tensor            # [..., H, K, Dh]
    v_g: torch.Tensor            # [..., H, K, Dh]
    occ_log_prior: torch.Tensor  # [..., K] float32 bounded log(occ)
    rel_bias: torch.Tensor       # [..., H, K] float32 directed edge relation bias

    def gather_flat(self, flat_idx: torch.Tensor) -> SharedGraphKV:
        """Gather selected flat indices `[M_sel]` from leading `[P, S, ...]` dims."""
        H, K, Dh = self.k_g.shape[-3], self.k_g.shape[-2], self.k_g.shape[-1]
        return SharedGraphKV(
            k_g=self.k_g.reshape(-1, H, K, Dh)[flat_idx],
            v_g=self.v_g.reshape(-1, H, K, Dh)[flat_idx],
            occ_log_prior=self.occ_log_prior.reshape(-1, K)[flat_idx],
            rel_bias=self.rel_bias.reshape(-1, H, K)[flat_idx],
        )

    def reshape_flat(self) -> SharedGraphKV:
        H, K, Dh = self.k_g.shape[-3], self.k_g.shape[-2], self.k_g.shape[-1]
        return SharedGraphKV(
            k_g=self.k_g.reshape(-1, H, K, Dh),
            v_g=self.v_g.reshape(-1, H, K, Dh),
            occ_log_prior=self.occ_log_prior.reshape(-1, K),
            rel_bias=self.rel_bias.reshape(-1, H, K),
        )


@dataclass(frozen=True)
class CorticalPredOut:
    """Output of `CorticalStackFlat.forward_predict_and_refine` (L4–L6 + L7–L8)."""

    z_hat_6: torch.Tensor                # [P, S, D] L6 pre-refinement prediction
    z_hat: torch.Tensor                  # [P, S, D] post-L7-L8 refined prediction
    u_n: torch.Tensor                    # [P, S] float32 uncertainty logit
    refine_mask: torch.Tensor            # [P, S] bool selected/active segments for L7–L8
    refine_gate: torch.Tensor            # [P, S] float32 MoD gate applied to L7–L8 residual delta
    tau_think: torch.Tensor              # scalar float32 threshold snapshot
    expert_counts: list[torch.Tensor]    # per-layer [E] float32 base MoE dispatch counts (L4–L8)


@dataclass(frozen=True)
class CorticalStepPredOut:
    """Single-step streaming output matching `CorticalPredOut` for Gate 6 parity."""

    z_hat_6: torch.Tensor      # [B, D]
    z_hat: torch.Tensor        # [B, D]
    u_n: torch.Tensor          # [B] float32
    refine_mask: torch.Tensor  # [B] bool
    refine_gate: torch.Tensor  # [B] float32


# ---------------------------------------------------------------------------
# R2: Decoupled Base Top-1 MoE + Additive Dead-Zone Plugin Pool
# ---------------------------------------------------------------------------


class DecoupledPluggableMoE(nn.Module):
    """Pre-RMSNorm Base MoE (1 Shared + E Routed Top-1) + Orthogonal Additive Dead-Zone Plugin Pool."""

    def __init__(
        self,
        d_model: int,
        shared_hidden: int,
        routed_hidden: int,
        num_experts: int,
        top_k: int = 1,
        router_bias_lr: float = 1e-3,
        router_logit_min: float = -4.0,
        plugin_top_k: int = 2,
        plugin_tau_min: float = 0.35,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.shared_hidden = int(shared_hidden)
        self.routed_hidden = int(routed_hidden)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_bias_lr = float(router_bias_lr)
        self.router_logit_min = float(router_logit_min)
        self.plugin_top_k = int(plugin_top_k)
        self.plugin_tau_min = float(plugin_tau_min)
        self.eps = float(eps)
        self.allow_cpu_grouped_mm: bool = False

        self.norm = RMSNorm(self.d_model, eps=self.eps)

        # Always-on Shared SwiGLU Expert
        self.shared_w_gu = nn.Linear(self.d_model, 2 * self.shared_hidden, bias=False)
        self.shared_w_d = nn.Linear(self.shared_hidden, self.d_model, bias=False)

        # Base Routed SwiGLU Experts (contiguous 3D weights for F.grouped_mm / BMM)
        self.router = nn.Linear(self.d_model, self.num_experts, bias=True)
        self.w_gu = nn.Parameter(torch.empty((self.num_experts, 2 * self.routed_hidden, self.d_model)))
        self.w_d = nn.Parameter(torch.empty((self.num_experts, self.d_model, self.routed_hidden)))

        # Non-grad load balance bias (selection ONLY; NEVER multiplied into synthesis gate)
        self.register_buffer("b_bal", torch.zeros(self.num_experts, dtype=torch.float32))

        # Hot-mounted additive dead-zone plugin packs (ordered dict pack_id -> layer tensors)
        self._plugin_packs: dict[str, dict[str, torch.Tensor]] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        hs = self.shared_hidden
        nn.init.kaiming_uniform_(self.shared_w_gu.weight[:hs], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_w_gu.weight[hs:], a=math.sqrt(5))
        nn.init.normal_(self.shared_w_d.weight, std=0.02)

        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)
        hr = self.routed_hidden
        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w_gu.data[e, :hr], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_gu.data[e, hr:], a=math.sqrt(5))
            nn.init.normal_(self.w_d.data[e], std=0.02)

    @property
    def mounted_pack_ids(self) -> tuple[str, ...]:
        return tuple(self._plugin_packs.keys())

    def mount_expert_pack(self, pack: PluginExpertPack, layer_idx: int) -> None:
        """Hot-mount one layer slice of a `PluginExpertPack` into the additive dead-zone pool."""
        if not (0 <= layer_idx < pack.proto_keys.shape[0]):
            raise ValueError(f"layer_idx={layer_idx} out of range for pack with {pack.proto_keys.shape[0]} layers")
        if pack.proto_keys.shape[2] != self.d_model:
            raise ValueError(f"Pack dim {pack.proto_keys.shape[2]} != layer d_model={self.d_model}")
        tau_l = pack.tau[layer_idx]
        if bool((tau_l < self.plugin_tau_min - 1e-6).any()):
            raise ValueError(f"Plugin tau below plugin_tau_min={self.plugin_tau_min} at layer {layer_idx}")

        keys_l = pack.proto_keys[layer_idx]
        keys_unit = F.normalize(keys_l.float(), dim=-1, eps=self.eps)
        self._plugin_packs[pack.pack_id] = {
            "proto_keys": keys_unit,
            "tau": tau_l.float(),
            "alpha": pack.alpha[layer_idx].float(),
            "w_gu": pack.w_gu[layer_idx],
            "w_d": pack.w_d[layer_idx],
        }

    def unmount_expert_pack(self, pack_id: str) -> bool:
        """Remove a mounted plugin pack by `pack_id`, restoring bit-exact baseline."""
        return self._plugin_packs.pop(pack_id, None) is not None

    @torch.no_grad()
    def update_balance_bias(self, counts: torch.Tensor, lr: Optional[float] = None) -> None:
        """Update non-grad selection-only balance bias `b_bal` from observed per-expert token counts."""
        step_lr = float(self.router_bias_lr if lr is None else lr)
        c = counts.detach().to(device=self.b_bal.device, dtype=torch.float32)
        total = float(c.sum())
        if total <= 0.0:
            return
        target = total / float(self.num_experts)
        updated = self.b_bal.float() + step_lr * torch.sign(target - c)
        updated = (updated - updated.mean()).clamp(-2.0, 2.0)
        self.b_bal.copy_(updated.to(self.b_bal.dtype))

    def _compute_plugin_residual(self, x_n: torch.Tensor, base_dtype: torch.dtype) -> torch.Tensor:
        """Compute strictly orthogonal additive dead-zone plugin residual `[N, D]`."""
        N, D = x_n.shape
        dev = x_n.device
        packs = list(self._plugin_packs.values())

        with fp32_island(dev):
            h_hat = F.normalize(x_n.float(), dim=-1, eps=self.eps)  # [N, D]
            all_keys = torch.cat([p["proto_keys"].to(device=dev, dtype=torch.float32) for p in packs], dim=0)
            all_tau = torch.cat([p["tau"].to(device=dev, dtype=torch.float32) for p in packs], dim=0)
            all_alpha = torch.cat([p["alpha"].to(device=dev, dtype=torch.float32) for p in packs], dim=0)

            cos_sim = F.linear(h_hat, all_keys)  # [N, P_tot]
            d = F.relu(cos_sim - all_tau.unsqueeze(0))  # hard dead-zone: 0.0 when cos <= tau
            g_all = torch.tanh(all_alpha.unsqueeze(0) * d)  # [N, P_tot], 0.0 iff d == 0.0

            p_tot = all_keys.shape[0]
            k_plug = min(self.plugin_top_k, p_tot)
            top_g, top_p = torch.topk(g_all, k=k_plug, dim=-1)  # [N, k_plug]

        # Eval / single-token streaming fast path: skip ALL plugin matmuls when no plugin activates (0 FLOPs!)
        if not self.training and not torch.compiler.is_compiling():
            if not bool((top_g > 0.0).any()):
                return torch.zeros((N, D), device=dev, dtype=base_dtype)

        with fp32_island(dev):
            g_selected = torch.zeros_like(g_all).scatter(1, top_p, top_g)  # [N, P_tot]

        y_plug_f = torch.zeros((N, D), device=dev, dtype=torch.float32)
        offset = 0
        for p in packs:
            e_p = int(p["proto_keys"].shape[0])
            g_pack = g_selected[:, offset : offset + e_p]  # [N, e_p] fp32
            offset += e_p

            if not self.training and not torch.compiler.is_compiling():
                if not bool((g_pack > 0.0).any()):
                    continue

            w_gu_p = p["w_gu"].to(device=dev, dtype=x_n.dtype)  # [e_p, 2*H_p, D]
            w_d_p = p["w_d"].to(device=dev, dtype=x_n.dtype)    # [e_p, D, H_p]
            x_e = x_n.unsqueeze(0).expand(e_p, -1, -1)          # [e_p, N, D]
            gu = torch.bmm(x_e, w_gu_p.transpose(1, 2))         # [e_p, N, 2*H_p]
            gp, up = gu.chunk(2, dim=-1)
            y_p = torch.bmm(F.silu(gp) * up, w_d_p.transpose(1, 2))  # [e_p, N, D]
            with fp32_island(dev):
                delta_p = torch.einsum("ne,end->nd", g_pack, y_p.float())
                active_tok = (g_pack.sum(dim=-1, keepdim=True) > 0.0)
                y_plug_f = y_plug_f + torch.where(active_tok, delta_p, torch.zeros_like(delta_p))

        return y_plug_f.to(base_dtype)

    def forward(
        self,
        x: torch.Tensor,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: `x [N, D]` -> `(y [N, D], expert_counts [E] fp32)`."""
        N, D = x.shape
        dev = x.device
        E = self.num_experts
        top_k = self.top_k

        x_n = self.norm(x)

        # 1. Always-on Shared SwiGLU Expert
        s_g, s_u = self.shared_w_gu(x_n).chunk(2, dim=-1)
        y_shared = self.shared_w_d(F.silu(s_g) * s_u)

        # 2. Base Router (fp32 island): b_bal used ONLY for selection, NEVER multiplied into gate!
        with fp32_island(dev):
            raw_logits = F.linear(
                x_n.float(),
                self.router.weight.float(),
                self.router.bias.float() if self.router.bias is not None else None,
            )  # [N, E]
            # Floor raw logits to prevent dead-weight zero-gate collapse
            logits = raw_logits.clamp_min(self.router_logit_min)
            sel_scores = logits + self.b_bal.float().unsqueeze(0)
            top_idx = torch.topk(sel_scores, k=top_k, dim=-1).indices  # [N, top_k]
            sel_logits = logits.gather(1, top_idx)  # [N, top_k] RAW logits without b_bal!
            if top_k == 1:
                g_base = torch.sigmoid(sel_logits)  # [N, 1] in [sigmoid(logit_min), 1)
            else:
                g_base = F.softmax(sel_logits, dim=-1)

            oh = F.one_hot(top_idx, num_classes=E).float().sum(dim=1)  # [N, E]
            if sample_mask is not None:
                counts = (oh * sample_mask.float().view(N, 1)).sum(dim=0)
            else:
                counts = oh.sum(dim=0)

        # 3. Base Routed SwiGLU dispatch (SM80+ grouped_mm -> row-independent eval -> static BMM)
        w_gu_t = self.w_gu.to(dtype=x_n.dtype).transpose(1, 2)  # [E, D, 2*H]
        w_d_t = self.w_d.to(dtype=x_n.dtype).transpose(1, 2)    # [E, H, D]

        if (
            x_n.dtype == torch.bfloat16
            and E >= 4
            and E > top_k
            and _device_supports_grouped_mm(dev, allow_cpu_bf16=self.allow_cpu_grouped_mm)
        ):
            flat_e_raw = top_idx.reshape(-1)
            if sample_mask is not None:
                pair_active = sample_mask.bool().view(N, 1).expand(N, top_k).reshape(-1)
                sentinel = torch.full_like(flat_e_raw, E)
                flat_e = torch.where(pair_active, flat_e_raw, sentinel)
                perm = torch.argsort(flat_e, stable=True)
                e_sorted = flat_e[perm]
                tok_sorted = torch.div(perm, top_k, rounding_mode="floor")
                x_sorted = x_n[tok_sorted].contiguous()
                counts_ext = torch.zeros((E + 1,), device=dev, dtype=torch.int32).scatter_add(
                    0, e_sorted, torch.ones_like(e_sorted, dtype=torch.int32)
                )
                offs = torch.cumsum(counts_ext[:E], dim=0, dtype=torch.int32)
                active_row = (e_sorted < E).unsqueeze(-1)
                x_in_gmm = torch.where(active_row, x_sorted, torch.zeros_like(x_sorted))
                gu_raw = F.grouped_mm(x_in_gmm, w_gu_t, offs=offs)
                gu_out = torch.where(active_row, gu_raw, torch.zeros_like(gu_raw))
                g_r, u_r = gu_out.chunk(2, dim=-1)
                act_r = (F.silu(g_r) * u_r).contiguous()
                y_raw = F.grouped_mm(act_r, w_d_t, offs=offs)
                y_sorted = torch.where(active_row, y_raw, torch.zeros_like(y_raw))
                gate_sorted = torch.where(
                    active_row,
                    g_base.reshape(-1)[perm].unsqueeze(-1),
                    torch.zeros((N * top_k, 1), device=dev, dtype=torch.float32),
                )
            else:
                perm = torch.argsort(flat_e_raw, stable=True)
                e_sorted = flat_e_raw[perm]
                tok_sorted = torch.div(perm, top_k, rounding_mode="floor")
                x_sorted = x_n[tok_sorted].contiguous()
                cnt = torch.zeros((E,), device=dev, dtype=torch.int32).scatter_add(
                    0, e_sorted, torch.ones_like(e_sorted, dtype=torch.int32)
                )
                offs = torch.cumsum(cnt, dim=0, dtype=torch.int32)
                gu_out = F.grouped_mm(x_sorted, w_gu_t, offs=offs)
                g_r, u_r = gu_out.chunk(2, dim=-1)
                act_r = (F.silu(g_r) * u_r).contiguous()
                y_sorted = F.grouped_mm(act_r, w_d_t, offs=offs)
                gate_sorted = g_base.reshape(-1)[perm].unsqueeze(-1)
            y_routed = (
                torch.zeros((N, D), device=dev, dtype=torch.float32)
                .index_add(0, tok_sorted, y_sorted.float() * gate_sorted)
                .to(x.dtype)
            )
        elif not self.training and x_n.dtype == torch.float32 and not torch.compiler.is_compiling():
            # Row-independent fp32 path for exact Gate 6 batch-vs-streaming parity
            y_routed_f = torch.zeros((N, D), device=dev, dtype=torch.float32)
            w_gu_f = self.w_gu.float()
            w_d_f = self.w_d.float()
            x_f = x_n.float()
            for k_i in range(top_k):
                e_ids = top_idx[:, k_i]  # [N]
                gu_k = torch.bmm(x_f.unsqueeze(1), w_gu_f[e_ids].transpose(1, 2)).squeeze(1)
                gk, uk = gu_k.chunk(2, dim=-1)
                yk = torch.bmm((F.silu(gk) * uk).unsqueeze(1), w_d_f[e_ids].transpose(1, 2)).squeeze(1)
                y_routed_f = y_routed_f + g_base[:, k_i : k_i + 1] * yk
            y_routed = y_routed_f.to(x.dtype)
        else:
            gate_full = torch.zeros((N, E), device=dev, dtype=torch.float32).scatter_add(1, top_idx, g_base)
            x_e = x_n.unsqueeze(0).expand(E, -1, -1)  # [E, N, D]
            g_all, u_all = torch.bmm(x_e, w_gu_t).chunk(2, dim=-1)
            y_all = torch.bmm(F.silu(g_all) * u_all, w_d_t)  # [E, N, D]
            y_routed = torch.einsum("ne,end->nd", gate_full, y_all.float()).to(x.dtype)

        y_base = x + y_shared + y_routed

        # 4. Orthogonal Additive Dead-Zone Plugin Pool (0.0 bit-exact drift when inactive!)
        if len(self._plugin_packs) == 0:
            return y_base, counts
        y_plug = self._compute_plugin_residual(x_n, y_base.dtype)
        return y_base + y_plug, counts


# ---------------------------------------------------------------------------
# Graph KV Builder (Computed ONCE per carrier graph)
# ---------------------------------------------------------------------------


class _GraphKVProjection(nn.Module):
    """Read-only side-path projection of graph `(slots, edges, occ)` into shared multi-head `(K_G, V_G)`."""

    def __init__(self, cfg: DCPAFlatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_concept
        d_r = cfg.d_r
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head

        # Read-only side-path RMSNorm (never writes back into recurrent graph state!)
        self.norm_slot = RMSNorm(D, eps=cfg.eps)
        self.w_edge_in = nn.Linear(d_r, D, bias=False)
        self.w_edge_out = nn.Linear(d_r, D, bias=False)
        self.w_gk = nn.Linear(D, D, bias=False)
        self.w_gv = nn.Linear(D, D, bias=False)
        self.w_rel_bias = nn.Linear(d_r, self.n_heads, bias=False)

        nn.init.normal_(self.w_edge_in.weight, std=0.02)
        nn.init.normal_(self.w_edge_out.weight, std=0.02)
        nn.init.normal_(self.w_rel_bias.weight, std=0.02)

    def forward(
        self,
        slots: torch.Tensor,
        edges: torch.Tensor,
        occ: torch.Tensor,
    ) -> SharedGraphKV:
        """slots `[..., K, D]`, edges `[..., K, K, d_r]`, occ `[..., K]` -> `SharedGraphKV`."""
        lead = slots.shape[:-2]
        K = slots.shape[-2]
        H = self.n_heads
        Dh = self.d_head
        dtype = slots.dtype

        with fp32_island(slots.device):
            occ_f = occ.float().clamp(self.cfg.occ_min, self.cfg.occ_max)
            w_occ = occ_f / occ_f.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
            edges_f = edges.float()
            edge_in = torch.einsum("...i,...ijr->...jr", w_occ, edges_f)   # [..., K, d_r]
            edge_out = torch.einsum("...j,...ijr->...ir", w_occ, edges_f)  # [..., K, d_r]
            occ_log_prior = torch.log(occ_f)                               # [..., K]

        s_enr = (
            self.norm_slot(slots)
            + torch.tanh(self.w_edge_in(edge_in.to(dtype)))
            + torch.tanh(self.w_edge_out(edge_out.to(dtype)))
        )
        k_g = self.w_gk(s_enr).view(*lead, K, H, Dh).transpose(-3, -2)  # [..., H, K, Dh]
        v_g = self.w_gv(s_enr).view(*lead, K, H, Dh).transpose(-3, -2)  # [..., H, K, Dh]
        with fp32_island(slots.device):
            rel_bias = 2.0 * torch.tanh(
                F.linear(edge_in, self.w_rel_bias.weight.float())
            ).transpose(-2, -1)  # [..., H, K]
        return SharedGraphKV(k_g=k_g, v_g=v_g, occ_log_prior=occ_log_prior, rel_bias=rel_bias)


# ---------------------------------------------------------------------------
# Cortical Layers (L1, L2–L3, L4–L6, L7–L8)
# ---------------------------------------------------------------------------


def _make_moe(cfg: DCPAFlatConfig) -> DecoupledPluggableMoE:
    return DecoupledPluggableMoE(
        d_model=cfg.d_concept,
        shared_hidden=cfg.shared_hidden,
        routed_hidden=cfg.routed_hidden,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k_experts,
        router_bias_lr=cfg.router_bias_lr,
        router_logit_min=cfg.router_logit_min,
        plugin_top_k=cfg.plugin_top_k,
        plugin_tau_min=cfg.plugin_tau_min,
        eps=cfg.eps,
    )


class PreG1CorticalLayer(nn.Module):
    """L1 (layer 0): Optional `W=128` causal local window attention (`0 <= Δs < W`) over `f_stem` + MoE."""

    def __init__(self, cfg: DCPAFlatConfig, enable_window: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.enable_window = bool(enable_window)
        D = cfg.d_concept
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.window_size = cfg.window_size

        if self.enable_window:
            self.norm_attn = RMSNorm(D, eps=cfg.eps)
            self.w_q = nn.Linear(D, D, bias=False)
            self.w_uk = nn.Linear(cfg.d_s, D, bias=False)
            self.w_uv = nn.Linear(cfg.d_s, D, bias=False)
            self.w_o = nn.Linear(D, D, bias=False)
            nn.init.normal_(self.w_o.weight, std=0.02)

        self.moe = _make_moe(cfg)

    def forward(
        self,
        x: torch.Tensor,
        f_stem: torch.Tensor,
        seg_doc_id: torch.Tensor,
        seg_end_depth: torch.Tensor,
        seg_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        P, S, D = x.shape
        dev = x.device
        orig_dtype = x.dtype

        if self.enable_window:
            W = self.window_size
            H = self.n_heads
            Dh = self.d_head
            if S % W != 0:
                raise ValueError(f"L1 window attention requires S={S} divisible by W={W}")
            n_chunks = S // W

            x_n = self.norm_attn(x)
            q = self.w_q(x_n).view(P, S, H, Dh)
            k = self.w_uk(f_stem.to(x_n.dtype)).view(P, S, H, Dh)
            v = self.w_uv(f_stem.to(x_n.dtype)).view(P, S, H, Dh)

            seg_pos = torch.arange(S, device=dev, dtype=torch.int64).unsqueeze(0).expand(P, S)
            q_rot = apply_concept_rope_2d(
                q, seg_pos, seg_end_depth, theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period
            )
            k_rot = apply_concept_rope_2d(
                k, seg_pos, seg_end_depth, theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period
            )

            # Chunk within each lane `[P, Nc, H, W, Dh]` — NEVER cross lane boundaries!
            q_t = q_rot.view(P, n_chunks, W, H, Dh).permute(0, 1, 3, 2, 4)  # [P, Nc, H, W, Dh]
            k_t = k_rot.view(P, n_chunks, W, H, Dh).permute(0, 1, 3, 2, 4)
            v_t = v.view(P, n_chunks, W, H, Dh).permute(0, 1, 3, 2, 4)
            doc_t = seg_doc_id.view(P, n_chunks, W)
            val_t = seg_valid.bool().view(P, n_chunks, W)

            z_kv = torch.zeros((P, 1, H, W, Dh), device=dev, dtype=k_t.dtype)
            k_2w = torch.cat([torch.cat([z_kv, k_t[:, :-1]], dim=1), k_t], dim=3)  # [P, Nc, H, 2W, Dh]
            v_2w = torch.cat([torch.cat([z_kv.to(v_t.dtype), v_t[:, :-1]], dim=1), v_t], dim=3)
            neg_d = torch.full((P, 1, W), -1, device=dev, dtype=doc_t.dtype)
            doc_2w = torch.cat([torch.cat([neg_d, doc_t[:, :-1]], dim=1), doc_t], dim=2)  # [P, Nc, 2W]

            q_loc = torch.arange(W, device=dev, dtype=torch.int64) + W
            k_loc = torch.arange(2 * W, device=dev, dtype=torch.int64)
            diff = q_loc.unsqueeze(1) - k_loc.unsqueeze(0)  # [W, 2W]
            in_win = (diff >= 0) & (diff < W)
            same_doc = (doc_t.unsqueeze(-1) == doc_2w.unsqueeze(-2)) & (doc_t.unsqueeze(-1) >= 0) & val_t.unsqueeze(-1)
            valid = same_doc & in_win.view(1, 1, W, 2 * W)
            # Safe self-diagonal fallback for padded invalid rows to prevent 0/0 NaN
            diag_fb = (diff == 0).view(1, 1, W, 2 * W) & (~val_t.unsqueeze(-1) | (doc_t.unsqueeze(-1) < 0))
            mask = (valid | diag_fb).unsqueeze(2)  # [P, Nc, 1, W, 2W]

            scale = 1.0 / math.sqrt(float(Dh))
            with fp32_island(dev):
                scores = torch.matmul(q_t.float(), k_2w.float().transpose(-2, -1)) * scale
                scores = torch.where(mask, scores, torch.full_like(scores, -1.0e9))
                attn = F.softmax(scores, dim=-1)
                ctx_t = torch.matmul(attn, v_2w.float())  # [P, Nc, H, W, Dh]
            ctx = ctx_t.permute(0, 1, 3, 2, 4).reshape(P, S, D).to(orig_dtype)
            x = x + self.w_o(ctx)

        y_flat, counts = self.moe(x.reshape(P * S, D), sample_mask=seg_valid.reshape(P * S))
        return y_flat.view(P, S, D), counts

    def step_streaming(
        self,
        x_cur: torch.Tensor,
        ring_stem: torch.Tensor,
        ring_pos: torch.Tensor,
        ring_depth: torch.Tensor,
        ring_valid: torch.Tensor,
        cur_pos: torch.Tensor,
        cur_end_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step streaming L1 (`ring_stem` already includes current segment at `cur_pos`)."""
        B, D = x_cur.shape
        dev = x_cur.device
        orig_dtype = x_cur.dtype

        if self.enable_window:
            H = self.n_heads
            Dh = self.d_head
            W = ring_stem.shape[1]

            x_n = self.norm_attn(x_cur)
            q = self.w_q(x_n).view(B, 1, H, Dh)
            k = self.w_uk(ring_stem.to(x_n.dtype)).view(B, W, H, Dh)
            v = self.w_uv(ring_stem.to(x_n.dtype)).view(B, W, H, Dh).permute(0, 2, 1, 3)

            q_rot = apply_concept_rope_2d(
                q, cur_pos.view(B, 1), cur_end_depth.view(B, 1),
                theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period,
            ).permute(0, 2, 1, 3)  # [B, H, 1, Dh]
            k_rot = apply_concept_rope_2d(
                k, ring_pos, ring_depth,
                theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period,
            ).permute(0, 2, 1, 3)  # [B, H, W, Dh]

            diff = cur_pos.view(B, 1) - ring_pos
            valid = (ring_valid.bool() & (diff >= 0) & (diff < self.window_size)).view(B, 1, 1, W)
            scale = 1.0 / math.sqrt(float(Dh))
            with fp32_island(dev):
                scores = torch.matmul(q_rot.float(), k_rot.float().transpose(-2, -1)) * scale
                scores = torch.where(valid, scores, torch.full_like(scores, -1.0e9))
                attn = F.softmax(scores, dim=-1)
                ctx = torch.matmul(attn, v.float()).squeeze(2).reshape(B, D)
            x_cur = x_cur + self.w_o(ctx.to(orig_dtype))

        y, _ = self.moe(x_cur)
        return y


class GraphCrossAttnCorticalLayer(nn.Module):
    """Pure `K=8` relation-biased graph cross-attention (shared KV) + optional soft IoR + MoE.

    Used by L2–L3 (over `G1`), L5, L7–L8 (over `G2_prior`), and L4/L6 when window layers are off.
    """

    def __init__(self, cfg: DCPAFlatConfig, use_ior: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_ior = bool(use_ior)
        D = cfg.d_concept
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.k_slots = cfg.k_slots

        self.norm_attn = RMSNorm(D, eps=cfg.eps)
        self.w_q = nn.Linear(D, D, bias=False)
        self.w_reg_bias = nn.Linear(cfg.d_l, self.n_heads * self.k_slots, bias=False)
        self.w_o = nn.Linear(D, D, bias=False)
        nn.init.normal_(self.w_reg_bias.weight, std=0.02)
        nn.init.normal_(self.w_o.weight, std=0.02)

        if self.use_ior:
            self.w_ior_q = nn.Linear(D, D, bias=False)
            self.ior_strength = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            nn.init.normal_(self.w_ior_q.weight, std=0.02)

        self.moe = _make_moe(cfg)

    def _compute_ior_penalty(
        self,
        x3_prev: Optional[torch.Tensor],
        k_g: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_ior or x3_prev is None:
            return None
        lead = x3_prev.shape[:-1]
        H, Dh = self.n_heads, self.d_head
        with fp32_island(x3_prev.device):
            q_ior = F.linear(x3_prev.float(), self.w_ior_q.weight.float()).view(*lead, H, Dh)
            # Zero-state soft IoR: when x3_prev == 0 (doc start), q_norm == 0 -> penalty == 0!
            q_norm = torch.sqrt(q_ior.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
            q_hat = q_ior / q_norm
            k_hat = F.normalize(k_g.float(), dim=-1, eps=1e-6)
            sim_prev = (q_hat.unsqueeze(-2) * k_hat).sum(dim=-1)  # [..., H, K]
            strength = F.softplus(self.ior_strength.float())
            return -strength * F.relu(sim_prev)

    def forward(
        self,
        x: torch.Tensor,
        graph_kv: SharedGraphKV,
        seg_l_reg: torch.Tensor,
        sample_mask: Optional[torch.Tensor] = None,
        x3_prev: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Supports both `[P, S, D]` grid and `[N, D]` gathered/streaming inputs."""
        lead = x.shape[:-1]
        D = x.shape[-1]
        H = self.n_heads
        Dh = self.d_head
        K = self.k_slots
        dev = x.device
        orig_dtype = x.dtype

        x_n = self.norm_attn(x)
        q = self.w_q(x_n).view(*lead, H, Dh)

        scale = 1.0 / math.sqrt(float(Dh))
        with fp32_island(dev):
            reg_bias = 2.0 * torch.tanh(
                F.linear(seg_l_reg.float(), self.w_reg_bias.weight.float())
            ).view(*lead, H, K)
            scores = (
                torch.einsum("...hd,...hkd->...hk", q.float(), graph_kv.k_g.float()) * scale
                + graph_kv.occ_log_prior.unsqueeze(-2)
                + graph_kv.rel_bias
                + reg_bias
            )
            ior = self._compute_ior_penalty(x3_prev, graph_kv.k_g)
            if ior is not None:
                scores = scores + ior
            attn = F.softmax(scores, dim=-1)  # [..., H, K]
            ctx = torch.einsum("...hk,...hkd->...hd", attn, graph_kv.v_g.float()).reshape(*lead, D)
            max_slot_w = attn.mean(dim=-2).amax(dim=-1)  # [...] diagnostic

        x_mid = x + self.w_o(ctx.to(orig_dtype))
        y_flat, counts = self.moe(x_mid.reshape(-1, D), sample_mask=sample_mask)
        return y_flat.view(*lead, D), counts, max_slot_w


class UnifiedWindowGraphCorticalLayer(nn.Module):
    """L4 / L6 Unified `2*W + K + 1` (=265) Single-Softmax Layer (R1):

    Strict-past `1 <= Δs <= W` local window (2D Concept-RoPE over `c_kv_sem` or `f_stem`)
    UNION `K=8` `G2_prior` shared KV (with `log(occ)`, relation bias, register precedence, soft IoR)
    UNION always-open learnable sink `b_sink` (zero value vector).
    """

    def __init__(self, cfg: DCPAFlatConfig, ring_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_concept
        self.ring_dim = int(ring_dim)
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.k_slots = cfg.k_slots
        self.window_size = cfg.window_size

        self.norm_attn = RMSNorm(D, eps=cfg.eps)
        self.w_q = nn.Linear(D, D, bias=False)
        self.w_uk = nn.Linear(self.ring_dim, D, bias=False)
        self.w_uv = nn.Linear(self.ring_dim, D, bias=False)
        self.w_reg_bias = nn.Linear(cfg.d_l, self.n_heads * self.k_slots, bias=False)
        self.w_ior_q = nn.Linear(D, D, bias=False)
        self.ior_strength = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.b_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.w_o = nn.Linear(D, D, bias=False)

        nn.init.normal_(self.w_reg_bias.weight, std=0.02)
        nn.init.normal_(self.w_ior_q.weight, std=0.02)
        nn.init.normal_(self.w_o.weight, std=0.02)

        self.moe = _make_moe(cfg)

    def _compute_ior_penalty(self, x3_prev: torch.Tensor, k_g: torch.Tensor) -> torch.Tensor:
        lead = x3_prev.shape[:-1]
        H, Dh = self.n_heads, self.d_head
        with fp32_island(x3_prev.device):
            q_ior = F.linear(x3_prev.float(), self.w_ior_q.weight.float()).view(*lead, H, Dh)
            q_hat = q_ior / torch.sqrt(q_ior.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
            k_hat = F.normalize(k_g.float(), dim=-1, eps=1e-6)
            sim_prev = (q_hat.unsqueeze(-2) * k_hat).sum(dim=-1)
            return -F.softplus(self.ior_strength.float()) * F.relu(sim_prev)

    def forward(
        self,
        x: torch.Tensor,
        x3_prev: torch.Tensor,
        graph_kv: SharedGraphKV,
        ring_kv: torch.Tensor,
        seg_doc_id: torch.Tensor,
        seg_start_depth: torch.Tensor,
        seg_end_depth: torch.Tensor,
        seg_l_reg: torch.Tensor,
        seg_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        P, S, D = x.shape
        W = self.window_size
        H = self.n_heads
        Dh = self.d_head
        K = self.k_slots
        dev = x.device
        orig_dtype = x.dtype

        if S % W != 0:
            raise ValueError(f"Unified window layer requires S={S} divisible by W={W}")
        n_chunks = S // W

        x_n = self.norm_attn(x)
        q_raw = self.w_q(x_n).view(P, S, H, Dh)
        k_raw = self.w_uk(ring_kv.to(x_n.dtype)).view(P, S, H, Dh)
        v_win = self.w_uv(ring_kv.to(x_n.dtype)).view(P, S, H, Dh)

        # Strictly causal RoPE: query uses seg_start_depth[s], completed strict-past keys use seg_end_depth[n<s]
        seg_pos = torch.arange(S, device=dev, dtype=torch.int64).unsqueeze(0).expand(P, S)
        q_rot = apply_concept_rope_2d(
            q_raw, seg_pos, seg_start_depth,
            theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period,
        )
        k_rot = apply_concept_rope_2d(
            k_raw, seg_pos, seg_end_depth,
            theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period,
        )

        # Chunk within each lane `[P, Nc, H, W, Dh]`
        q_t = q_rot.view(P, n_chunks, W, H, Dh).permute(0, 1, 3, 2, 4)
        k_t = k_rot.view(P, n_chunks, W, H, Dh).permute(0, 1, 3, 2, 4)
        v_t = v_win.view(P, n_chunks, W, H, Dh).permute(0, 1, 3, 2, 4)
        doc_t = seg_doc_id.view(P, n_chunks, W)
        val_t = seg_valid.bool().view(P, n_chunks, W)

        z_kv = torch.zeros((P, 1, H, W, Dh), device=dev, dtype=k_t.dtype)
        k_2w = torch.cat([torch.cat([z_kv, k_t[:, :-1]], dim=1), k_t], dim=3)  # [P, Nc, H, 2W, Dh]
        v_2w = torch.cat([torch.cat([z_kv.to(v_t.dtype), v_t[:, :-1]], dim=1), v_t], dim=3)
        neg_d = torch.full((P, 1, W), -1, device=dev, dtype=doc_t.dtype)
        doc_2w = torch.cat([torch.cat([neg_d, doc_t[:, :-1]], dim=1), doc_t], dim=2)  # [P, Nc, 2W]

        q_loc = torch.arange(W, device=dev, dtype=torch.int64) + W
        k_loc = torch.arange(2 * W, device=dev, dtype=torch.int64)
        diff = q_loc.unsqueeze(1) - k_loc.unsqueeze(0)  # [W, 2W]
        # STRICT PAST: 1 <= Δs <= W (never peeks at current segment s!)
        in_strict_past = (diff >= 1) & (diff <= W)
        valid_win = (
            (doc_t.unsqueeze(-1) == doc_2w.unsqueeze(-2))
            & (doc_t.unsqueeze(-1) >= 0)
            & val_t.unsqueeze(-1)
            & in_strict_past.view(1, 1, W, 2 * W)
        ).unsqueeze(2)  # [P, Nc, 1, W, 2W]

        scale = 1.0 / math.sqrt(float(Dh))
        with fp32_island(dev):
            scores_win = torch.matmul(q_t.float(), k_2w.float().transpose(-2, -1)) * scale  # [P, Nc, H, W, 2W]
            scores_win = torch.where(valid_win, scores_win, torch.full_like(scores_win, -1.0e9))

            reg_bias = 2.0 * torch.tanh(
                F.linear(seg_l_reg.float(), self.w_reg_bias.weight.float())
            ).view(P, S, H, K)
            ior = self._compute_ior_penalty(x3_prev, graph_kv.k_g)  # [P, S, H, K]
            scores_g_full = (
                torch.einsum("pshd,pshkd->pshk", q_raw.float(), graph_kv.k_g.float()) * scale
                + graph_kv.occ_log_prior.unsqueeze(2)
                + graph_kv.rel_bias
                + reg_bias
                + ior
            )  # [P, S, H, K]
            scores_g = scores_g_full.view(P, n_chunks, W, H, K).permute(0, 1, 3, 2, 4)  # [P, Nc, H, W, K]

            # Always-open learnable sink column (column 2W + K, zero value vector)
            scores_sink = self.b_sink.float().view(1, 1, H, 1, 1).expand(P, n_chunks, H, W, 1)

            # Unified single-softmax across all 2*W + K + 1 columns (=265 at W=128, K=8)
            all_scores = torch.cat([scores_win, scores_g, scores_sink], dim=-1)
            all_attn = F.softmax(all_scores, dim=-1)
            attn_win = all_attn[..., : 2 * W]
            attn_g = all_attn[..., 2 * W : 2 * W + K]

            ctx_win = torch.matmul(attn_win, v_2w.float())  # [P, Nc, H, W, Dh]
            gv_t = graph_kv.v_g.float().view(P, n_chunks, W, H, K, Dh).permute(0, 1, 3, 2, 4, 5)
            ctx_g = torch.einsum("pnhwk,pnhwkd->pnhwd", attn_g, gv_t)
            ctx = (ctx_win + ctx_g).permute(0, 1, 3, 2, 4).reshape(P, S, D)

        x_mid = x + self.w_o(ctx.to(orig_dtype))
        y_flat, counts = self.moe(x_mid.reshape(P * S, D), sample_mask=seg_valid.reshape(P * S))
        return y_flat.view(P, S, D), counts

    def step_streaming(
        self,
        x_cur: torch.Tensor,
        x3_prev: torch.Tensor,
        graph_kv: SharedGraphKV,
        ring_kv: torch.Tensor,
        ring_pos: torch.Tensor,
        ring_depth: torch.Tensor,
        ring_valid: torch.Tensor,
        cur_pos: torch.Tensor,
        cur_start_depth: torch.Tensor,
        l_reg: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step streaming unified `(strict-past W ∪ K=8 G2_prior ∪ 1 sink)` softmax."""
        B, D = x_cur.shape
        W = ring_kv.shape[1]
        H = self.n_heads
        Dh = self.d_head
        K = self.k_slots
        dev = x_cur.device
        orig_dtype = x_cur.dtype

        x_n = self.norm_attn(x_cur)
        q_raw = self.w_q(x_n).view(B, 1, H, Dh)
        k_raw = self.w_uk(ring_kv.to(x_n.dtype)).view(B, W, H, Dh)
        v_win = self.w_uv(ring_kv.to(x_n.dtype)).view(B, W, H, Dh).permute(0, 2, 1, 3)

        q_rot = apply_concept_rope_2d(
            q_raw, cur_pos.view(B, 1), cur_start_depth.view(B, 1),
            theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period,
        ).permute(0, 2, 1, 3)  # [B, H, 1, Dh]
        k_rot = apply_concept_rope_2d(
            k_raw, ring_pos, ring_depth,
            theta_base=self.cfg.rope_theta_concept, scope_period=self.cfg.rope_period,
        ).permute(0, 2, 1, 3)  # [B, H, W, Dh]

        diff = cur_pos.view(B, 1) - ring_pos
        valid_w = (ring_valid.bool() & (diff >= 1) & (diff <= self.window_size)).view(B, 1, 1, W)

        scale = 1.0 / math.sqrt(float(Dh))
        with fp32_island(dev):
            scores_w = torch.matmul(q_rot.float(), k_rot.float().transpose(-2, -1)) * scale
            scores_w = torch.where(valid_w, scores_w, torch.full_like(scores_w, -1.0e9))

            reg_bias = 2.0 * torch.tanh(F.linear(l_reg.float(), self.w_reg_bias.weight.float())).view(B, H, K)
            ior = self._compute_ior_penalty(x3_prev, graph_kv.k_g)
            scores_g = (
                torch.einsum("bhd,bhkd->bhk", q_raw.squeeze(1).float(), graph_kv.k_g.float()) * scale
                + graph_kv.occ_log_prior.unsqueeze(1)
                + graph_kv.rel_bias
                + reg_bias
                + ior
            ).unsqueeze(2)  # [B, H, 1, K]
            scores_sink = self.b_sink.float().view(1, H, 1, 1).expand(B, H, 1, 1)

            all_attn = F.softmax(torch.cat([scores_w, scores_g, scores_sink], dim=-1), dim=-1)
            attn_w = all_attn[..., :W]
            attn_g = all_attn[..., W : W + K]
            ctx_w = torch.matmul(attn_w, v_win.float())
            ctx_g = torch.matmul(attn_g, graph_kv.v_g.float())
            ctx = (ctx_w + ctx_g).squeeze(2).reshape(B, D)

        x_mid = x_cur + self.w_o(ctx.to(orig_dtype))
        y, _ = self.moe(x_mid)
        return y


# ---------------------------------------------------------------------------
# Full 8-Layer Cortical Stack (R1 + R2 + R5)
# ---------------------------------------------------------------------------


class CorticalStackFlat(nn.Module):
    """DCPA-Flat v3.3 8-layer Cortical Stack (L1 | L2–L3 | L4–L6 | L7–L8 MoD Refine)."""

    def __init__(self, cfg: DCPAFlatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_concept
        self.n_layers = cfg.n_layers
        self.window_layers = frozenset(cfg.local_window_layers) if cfg.use_local_window_layers else frozenset()
        self.ring_dim = int(cfg.d_s if cfg.share_stem_ring else cfg.d_sem_ring)

        # Shared graph KV builders (computed ONCE per graph: G1 for L2–L3, G2_prior for L4–L8)
        self.kv_g1 = _GraphKVProjection(cfg)
        self.kv_g2 = _GraphKVProjection(cfg)

        # Build the 8 cortical layers (0: L1, 1..2: L2-L3, 3..5: L4-L6, 6..7: L7-L8)
        layers: list[nn.Module] = []
        # Layer 0 (L1): pre-G1
        layers.append(PreG1CorticalLayer(cfg, enable_window=(0 in self.window_layers)))
        # Layers 1..2 (L2-L3): post-G1
        for _ in range(cfg.n_post_g1_layers):
            layers.append(GraphCrossAttnCorticalLayer(cfg, use_ior=False))
        # Layers 3..5 (L4-L6): predict full around G2_prior
        pred_start = cfg.n_pre_g1_layers + cfg.n_post_g1_layers
        for offset in range(cfg.n_predict_full_layers):
            layer_idx = pred_start + offset
            if layer_idx in self.window_layers:
                layers.append(UnifiedWindowGraphCorticalLayer(cfg, ring_dim=self.ring_dim))
            else:
                layers.append(GraphCrossAttnCorticalLayer(cfg, use_ior=True))
        # Layers 6..7 (L7-L8): MoD / threshold refinement around G2_prior
        for _ in range(cfg.n_refine_layers):
            layers.append(GraphCrossAttnCorticalLayer(cfg, use_ior=True))

        self.layers = nn.ModuleList(layers)

        # L3 output norm + optional +32KB semantic ring projection `c_kv_sem`
        self.norm_l3_out = RMSNorm(D, eps=cfg.eps)
        if not cfg.share_stem_ring:
            self.w_sem_down = nn.Linear(D, cfg.d_sem_ring, bias=False)
            self.norm_sem_ring = RMSNorm(cfg.d_sem_ring, eps=cfg.eps)
            nn.init.normal_(self.w_sem_down.weight, std=0.02)
        else:
            self.w_sem_down = None
            self.norm_sem_ring = None

        # L4 entry register conditioning + L6 output norm + R5 uncertainty head
        self.w_reg_pred = nn.Linear(cfg.d_l, D, bias=False)
        nn.init.normal_(self.w_reg_pred.weight, std=0.02)
        self.norm_l6_out = RMSNorm(D, eps=cfg.eps)
        self.w_unc = nn.Linear(D, 1, bias=True)
        nn.init.normal_(self.w_unc.weight, std=0.02)
        nn.init.zeros_(self.w_unc.bias)

        # Online EMA threshold for L7–L8 refinement (shared between training & eval/streaming)
        self.register_buffer("tau_think", torch.tensor(0.0, dtype=torch.float32))

        # Cached detached expert counts for convenient zero-arg `update_all_balance_biases()` (eager only)
        self._last_expert_counts: list[Optional[torch.Tensor]] = [None] * cfg.n_layers

    def build_graph_kv(
        self,
        slots_seq: torch.Tensor,
        edges_seq: torch.Tensor,
        occ_seq: torch.Tensor,
        which: Literal["g1", "g2"] = "g1",
    ) -> SharedGraphKV:
        """Compute shared multi-head `(K_G, V_G, occ_log_prior, rel_bias)` ONCE per carrier graph."""
        proj = self.kv_g1 if which == "g1" else self.kv_g2
        return proj(slots_seq, edges_seq, occ_seq)

    def mount_expert_pack(self, pack: PluginExpertPack) -> None:
        """Hot-mount a `PluginExpertPack` across all 8 cortical layers."""
        pack.validate(self.n_layers, self.cfg.d_concept, tau_min=self.cfg.plugin_tau_min)
        for idx, layer in enumerate(self.layers):
            layer.moe.mount_expert_pack(pack, layer_idx=idx)  # type: ignore[union-attr]

    def unmount_expert_pack(self, pack_id: str) -> bool:
        """Unmount a plugin pack from all 8 layers, restoring bit-exact base output."""
        removed = False
        for layer in self.layers:
            if layer.moe.unmount_expert_pack(pack_id):  # type: ignore[union-attr]
                removed = True
        return removed

    @torch.no_grad()
    def update_all_balance_biases(
        self,
        counts_per_layer: Optional[Sequence[Optional[torch.Tensor]]] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Update selection-only `b_bal` across all 8 MoE layers."""
        src = self._last_expert_counts if counts_per_layer is None else counts_per_layer
        for idx, layer in enumerate(self.layers):
            if idx < len(src) and src[idx] is not None:
                layer.moe.update_balance_bias(src[idx], lr=lr)  # type: ignore[union-attr, arg-type]

    # ------------------------------------------------------------------
    # Stage 3A-pre: L1 (layer 0) before Carrier G1
    # ------------------------------------------------------------------

    def forward_pre_g1(
        self,
        x0: torch.Tensor,
        f_stem: torch.Tensor,
        seg_doc_id: torch.Tensor,
        seg_end_depth: torch.Tensor,
        seg_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """L1 (layer 0): causal `W=128` local window over `f_stem` (if enabled) + MoE."""
        l1: PreG1CorticalLayer = self.layers[0]  # type: ignore[assignment]
        x1, counts0 = l1(x0, f_stem, seg_doc_id, seg_end_depth, seg_valid)
        if not torch.compiler.is_compiling():
            self._last_expert_counts[0] = counts0.detach()
        return x1, {"expert_counts": [counts0]}

    # ------------------------------------------------------------------
    # Stage 3A-post: L2–L3 (layers 1–2) after Carrier G1
    # ------------------------------------------------------------------

    def forward_post_g1(
        self,
        x1: torch.Tensor,
        g1_out: CarrierScanOut,
        seg_l_reg: torch.Tensor,
        seg_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
        """L2–L3 (layers 1–2): shared `G1` KV graph cross-attention + MoE + optional `c_kv_sem`."""
        P, S, _ = x1.shape
        kv_g1 = self.build_graph_kv(g1_out.slots_seq, g1_out.edges_seq, g1_out.occ_seq, which="g1")
        valid_flat = seg_valid.reshape(P * S)

        x = x1
        counts_list: list[torch.Tensor] = []
        max_slot_attn_list: list[torch.Tensor] = []
        for idx in range(1, 1 + self.cfg.n_post_g1_layers):
            layer: GraphCrossAttnCorticalLayer = self.layers[idx]  # type: ignore[assignment]
            x, c_l, msw = layer(x, kv_g1, seg_l_reg, sample_mask=valid_flat)
            counts_list.append(c_l)
            max_slot_attn_list.append(msw)
            if not torch.compiler.is_compiling():
                self._last_expert_counts[idx] = c_l.detach()

        x3 = self.norm_l3_out(x)
        if self.w_sem_down is not None and self.norm_sem_ring is not None:
            c_kv_sem: Optional[torch.Tensor] = self.norm_sem_ring(self.w_sem_down(x3))
        else:
            c_kv_sem = None

        aux = {
            "expert_counts": counts_list,
            "max_slot_attn": torch.stack(max_slot_attn_list, dim=0).amax(dim=0),
        }
        return x3, c_kv_sem, aux

    # ------------------------------------------------------------------
    # Stage 3B + 3C: L4–L6 + L7–L8 (layers 3–7) around G2_prior
    # ------------------------------------------------------------------

    def forward_predict_and_refine(
        self,
        x_seed: torch.Tensor,
        x3_prev: torch.Tensor,
        g2_prior: CarrierScanOut,
        ring_kv: Optional[torch.Tensor],
        seg_doc_id: torch.Tensor,
        seg_start_depth: torch.Tensor,
        seg_end_depth: torch.Tensor,
        seg_l_reg: torch.Tensor,
        seg_valid: torch.Tensor,
    ) -> CorticalPredOut:
        """L4–L6 full prediction (with unified 265-col softmax on L4/L6 if enabled) + R5 L7–L8 refine."""
        P, S, D = x_seed.shape
        N_flat = P * S
        dev = x_seed.device
        valid_flat = seg_valid.reshape(N_flat)

        kv_g2 = self.build_graph_kv(g2_prior.slots_seq, g2_prior.edges_seq, g2_prior.occ_seq, which="g2")
        x = x_seed + self.w_reg_pred(seg_l_reg.to(x_seed.dtype))

        counts_list: list[torch.Tensor] = []
        pred_start = self.cfg.n_pre_g1_layers + self.cfg.n_post_g1_layers
        pred_end = pred_start + self.cfg.n_predict_full_layers

        for idx in range(pred_start, pred_end):
            layer = self.layers[idx]
            if isinstance(layer, UnifiedWindowGraphCorticalLayer):
                if ring_kv is None:
                    raise ValueError(f"Layer {idx} has local window attention enabled but ring_kv is None")
                x, c_l = layer(
                    x=x,
                    x3_prev=x3_prev,
                    graph_kv=kv_g2,
                    ring_kv=ring_kv,
                    seg_doc_id=seg_doc_id,
                    seg_start_depth=seg_start_depth,
                    seg_end_depth=seg_end_depth,
                    seg_l_reg=seg_l_reg,
                    seg_valid=seg_valid,
                )
            else:
                g_layer: GraphCrossAttnCorticalLayer = layer  # type: ignore[assignment]
                x, c_l, _ = g_layer(x, kv_g2, seg_l_reg, sample_mask=valid_flat, x3_prev=x3_prev)
            counts_list.append(c_l)
            if not torch.compiler.is_compiling():
                self._last_expert_counts[idx] = c_l.detach()

        z_hat_6 = self.norm_l6_out(x)  # [P, S, D]

        # R5 Uncertainty head in fp32
        with fp32_island(dev):
            u_n = F.linear(z_hat_6.float(), self.w_unc.weight.float(), self.w_unc.bias.float()).squeeze(-1)  # [P, S]

        # L7–L8 Refinement Stack (R5 Dual Mode: Train Top-25% MoD vs Eval/Streaming Threshold)
        refine_start = pred_end
        refine_end = refine_start + self.cfg.n_refine_layers

        if self.training:
            k_ref = max(1, min(N_flat, int(round(N_flat * self.cfg.refine_frac))))
            with fp32_island(dev):
                u_flat = u_n.reshape(N_flat)
                u_masked = torch.where(valid_flat, u_flat, torch.full_like(u_flat, -1.0e9))
                top_u, top_idx = torch.topk(u_masked, k=k_ref, dim=0, sorted=True)  # [k_ref]
                valid_sel = valid_flat[top_idx]  # [k_ref] bool
                u_sel = u_flat[top_idx]  # [k_ref] differentiable -> grads flow directly into u_n!
                tau_det = self.tau_think.detach().float()
                gate_sel = torch.sigmoid(u_sel - tau_det) * valid_sel.float()  # [k_ref]

            z_flat = z_hat_6.reshape(N_flat, D)
            z_sel = z_flat[top_idx]  # [k_ref, D]
            x3_prev_sel = x3_prev.reshape(N_flat, D)[top_idx]
            reg_sel = seg_l_reg.reshape(N_flat, self.cfg.d_l)[top_idx]
            kv_sel = kv_g2.gather_flat(top_idx)

            h = z_sel
            for idx in range(refine_start, refine_end):
                r_layer: GraphCrossAttnCorticalLayer = self.layers[idx]  # type: ignore[assignment]
                h, c_l, _ = r_layer(h, kv_sel, reg_sel, sample_mask=valid_sel, x3_prev=x3_prev_sel)
                counts_list.append(c_l)
                if not torch.compiler.is_compiling():
                    self._last_expert_counts[idx] = c_l.detach()

            delta_sel = h - z_sel
            weighted_delta = (gate_sel.unsqueeze(-1) * delta_sel.float()).to(z_hat_6.dtype)
            z_hat = z_flat.index_add(0, top_idx, weighted_delta).view(P, S, D)

            refine_mask_flat = torch.zeros((N_flat,), device=dev, dtype=torch.bool).scatter(0, top_idx, valid_sel)
            refine_gate_flat = torch.zeros((N_flat,), device=dev, dtype=torch.float32).scatter(0, top_idx, gate_sel)
            refine_mask = refine_mask_flat.view(P, S)
            refine_gate = refine_gate_flat.view(P, S)

            # EMA update scalar buffer `tau_think` from k-th selected valid threshold
            with torch.no_grad():
                kth_valid = valid_sel[-1]
                kth_val = top_u[-1].detach().float()
                tau_f = self.tau_think.float()
                new_tau = self.cfg.tau_think_ema * tau_f + (1.0 - self.cfg.tau_think_ema) * kth_val
                self.tau_think.copy_(torch.where(kth_valid, new_tau, tau_f).to(self.tau_think.dtype))
        else:
            # Eval / streaming / Gate 6 mode: strictly per-position threshold gating (zero batch coupling!)
            with fp32_island(dev):
                tau_cur = self.tau_think.float()
                refine_mask = seg_valid.bool() & (u_n > tau_cur)  # [P, S]
                refine_gate = torch.sigmoid(u_n - tau_cur) * refine_mask.float()  # [P, S]

            # L7–L8 are 100% row-independent (pure K=8 graph + per-token MoE); static full-shape eval
            # guarantees zero dynamic slicing while matching per-token active execution bit-for-bit!
            z_flat = z_hat_6.reshape(N_flat, D)
            x3_prev_flat = x3_prev.reshape(N_flat, D)
            reg_flat = seg_l_reg.reshape(N_flat, self.cfg.d_l)
            kv_flat = kv_g2.reshape_flat()
            active_flat = refine_mask.reshape(N_flat)

            h = z_flat
            for idx in range(refine_start, refine_end):
                r_layer = self.layers[idx]  # type: ignore[assignment]
                h, c_l, _ = r_layer(h, kv_flat, reg_flat, sample_mask=active_flat, x3_prev=x3_prev_flat)
                counts_list.append(c_l)

            delta_full = (h - z_flat).view(P, S, D)
            z_hat = z_hat_6 + (refine_gate.unsqueeze(-1) * delta_full.float()).to(z_hat_6.dtype)

        return CorticalPredOut(
            z_hat_6=z_hat_6,
            z_hat=z_hat,
            u_n=u_n,
            refine_mask=refine_mask,
            refine_gate=refine_gate,
            tau_think=self.tau_think.detach().clone(),
            expert_counts=counts_list,
        )

    # ------------------------------------------------------------------
    # Single-Step O(1) Streaming API (Gate 6 Parity)
    # ------------------------------------------------------------------

    def step_streaming_pre_g1(
        self,
        x0_step: torch.Tensor,
        ring_stem: torch.Tensor,
        ring_pos: torch.Tensor,
        ring_depth: torch.Tensor,
        ring_valid: torch.Tensor,
        cur_pos: torch.Tensor,
        cur_end_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step streaming counterpart of `forward_pre_g1` (L1)."""
        l1: PreG1CorticalLayer = self.layers[0]  # type: ignore[assignment]
        return l1.step_streaming(x0_step, ring_stem, ring_pos, ring_depth, ring_valid, cur_pos, cur_end_depth)

    def step_streaming_post_g1(
        self,
        x1_step: torch.Tensor,
        g1_state: CarrierGraphState,
        l_reg_step: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Single-step streaming counterpart of `forward_post_g1` (L2–L3)."""
        kv_g1 = self.build_graph_kv(g1_state.slots, g1_state.edges, g1_state.occ, which="g1")
        x = x1_step
        for idx in range(1, 1 + self.cfg.n_post_g1_layers):
            layer: GraphCrossAttnCorticalLayer = self.layers[idx]  # type: ignore[assignment]
            x, _, _ = layer(x, kv_g1, l_reg_step)
        x3 = self.norm_l3_out(x)
        if self.w_sem_down is not None and self.norm_sem_ring is not None:
            return x3, self.norm_sem_ring(self.w_sem_down(x3))
        return x3, None

    def step_streaming_predict_and_refine(
        self,
        x_seed_step: torch.Tensor,
        x3_prev_step: torch.Tensor,
        g2_prior_state: CarrierGraphState,
        ring_kv: Optional[torch.Tensor],
        ring_pos: torch.Tensor,
        ring_depth: torch.Tensor,
        ring_valid: torch.Tensor,
        cur_pos: torch.Tensor,
        cur_start_depth: torch.Tensor,
        l_reg_step: torch.Tensor,
    ) -> CorticalStepPredOut:
        """Single-step streaming counterpart of `forward_predict_and_refine` (L4–L6 + threshold L7–L8)."""
        dev = x_seed_step.device
        kv_g2 = self.build_graph_kv(
            g2_prior_state.slots, g2_prior_state.edges, g2_prior_state.occ, which="g2"
        )
        x = x_seed_step + self.w_reg_pred(l_reg_step.to(x_seed_step.dtype))

        pred_start = self.cfg.n_pre_g1_layers + self.cfg.n_post_g1_layers
        pred_end = pred_start + self.cfg.n_predict_full_layers
        for idx in range(pred_start, pred_end):
            layer = self.layers[idx]
            if isinstance(layer, UnifiedWindowGraphCorticalLayer):
                if ring_kv is None:
                    raise ValueError(f"Layer {idx} has local window enabled but ring_kv is None")
                x = layer.step_streaming(
                    x_cur=x,
                    x3_prev=x3_prev_step,
                    graph_kv=kv_g2,
                    ring_kv=ring_kv,
                    ring_pos=ring_pos,
                    ring_depth=ring_depth,
                    ring_valid=ring_valid,
                    cur_pos=cur_pos,
                    cur_start_depth=cur_start_depth,
                    l_reg=l_reg_step,
                )
            else:
                g_layer: GraphCrossAttnCorticalLayer = layer  # type: ignore[assignment]
                x, _, _ = g_layer(x, kv_g2, l_reg_step, x3_prev=x3_prev_step)

        z_hat_6 = self.norm_l6_out(x)  # [B, D]
        with fp32_island(dev):
            u_n = F.linear(z_hat_6.float(), self.w_unc.weight.float(), self.w_unc.bias.float()).squeeze(-1)  # [B]
            tau_cur = self.tau_think.float()
            refine_mask = u_n > tau_cur
            refine_gate = torch.sigmoid(u_n - tau_cur) * refine_mask.float()

        # Skip L7–L8 completely when inactive in single-step streaming (0 extra FLOPs!)
        if not torch.compiler.is_compiling() and not bool(refine_mask.any()):
            z_hat = z_hat_6
        else:
            h = z_hat_6
            refine_start = pred_end
            refine_end = refine_start + self.cfg.n_refine_layers
            for idx in range(refine_start, refine_end):
                r_layer = self.layers[idx]  # type: ignore[assignment]
                h, _, _ = r_layer(h, kv_g2, l_reg_step, x3_prev=x3_prev_step)
            delta = h - z_hat_6
            z_hat = z_hat_6 + (refine_gate.unsqueeze(-1) * delta.float()).to(z_hat_6.dtype)

        return CorticalStepPredOut(
            z_hat_6=z_hat_6,
            z_hat=z_hat,
            u_n=u_n,
            refine_mask=refine_mask,
            refine_gate=refine_gate,
        )

    # Alias for direct streaming invocation
    step_streaming = step_streaming_predict_and_refine


__all__ = [
    "SharedGraphKV",
    "CorticalPredOut",
    "CorticalStepPredOut",
    "DecoupledPluggableMoE",
    "PreG1CorticalLayer",
    "GraphCrossAttnCorticalLayer",
    "UnifiedWindowGraphCorticalLayer",
    "CorticalStackFlat",
]
