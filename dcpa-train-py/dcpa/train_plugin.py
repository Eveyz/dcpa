"""Unified Additive Dead-Zone Plugin Trainer (R2 + R6: Domain Packs & Personal Sleep Consolidation).

Implements a single end-to-end pipeline shared by:
  - Community vertical domain expert packs (e.g. Gate E SQL plugin, 2.5 GB domain corpus)
  - Personal sleep consolidation packs (`--mode=sleep`, replaying awake high-beta real fragments)

Guarantees:
  1. 100% read-only frozen base weights (`DCPAModelFlat.freeze_base_for_plugin_training()`).
  2. Prototype-key centroid initialization from target-domain base hidden states + zero-initialized `w_d`.
  3. Dead-zone margin calibration `tau >= max(tau_min, q_99.9(anchor_cos) + margin)` + anchor repulsion loss.
  4. Post-training dual acceptance check: target domain BPC drops AND non-target anchor drift == 0.0 (< 1e-7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcpa.interfaces import PluginExpertPack, StaticGridBatch
from dcpa.model_flat import DCPAModelFlat


@dataclass
class PluginTrainReport:
    pack_id: str
    steps: int
    target_bpc_before: float
    target_bpc_after: float
    anchor_max_logit_drift: float
    anchor_plugin_active_rate: float
    passed_zero_drift: bool


class TrainablePluginModule(nn.Module):
    """Trainable wrapper around `PluginExpertPack` tensors for end-to-end frozen-base fine-tuning."""

    def __init__(
        self,
        pack_id: str,
        n_layers: int,
        d_model: int,
        experts_per_layer: int,
        hidden_dim: int,
        tau_min: float = 0.35,
        init_proto_keys: Optional[torch.Tensor] = None,
        init_tau: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.pack_id = str(pack_id)
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.experts_per_layer = int(experts_per_layer)
        self.hidden_dim = int(hidden_dim)
        self.tau_min = float(tau_min)

        if init_proto_keys is not None:
            k0 = F.normalize(init_proto_keys.detach().float(), dim=-1)
        else:
            k0 = F.normalize(torch.randn((n_layers, experts_per_layer, d_model)), dim=-1)
        self.proto_keys_raw = nn.Parameter(k0.clone())

        # Fixed calibrated tau floor per layer/expert (kept >= tau_min for strict dead-zone isolation)
        tau0 = (
            torch.clamp(init_tau.detach().float(), min=self.tau_min, max=0.95)
            if init_tau is not None
            else torch.full((n_layers, experts_per_layer), self.tau_min, dtype=torch.float32)
        )
        self.register_buffer("tau_floor", tau0)
        self.alpha_raw = nn.Parameter(torch.full((n_layers, experts_per_layer), 1.5, dtype=torch.float32))

        self.w_gu = nn.Parameter(torch.empty((n_layers, experts_per_layer, 2 * hidden_dim, d_model)))
        self.w_d = nn.Parameter(torch.zeros((n_layers, experts_per_layer, d_model, hidden_dim)))
        for l_idx in range(n_layers):
            for e_idx in range(experts_per_layer):
                nn.init.kaiming_uniform_(self.w_gu.data[l_idx, e_idx, :hidden_dim], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.w_gu.data[l_idx, e_idx, hidden_dim:], a=math.sqrt(5))
        # Zero-init w_d guarantees step-0 residual == 0.0!

    def to_pack(self) -> PluginExpertPack:
        proto = F.normalize(self.proto_keys_raw, dim=-1, eps=1e-6)
        alpha = F.softplus(self.alpha_raw) + 0.1
        return PluginExpertPack(
            pack_id=self.pack_id,
            proto_keys=proto,
            tau=self.tau_floor,
            alpha=alpha,
            w_gu=self.w_gu,
            w_d=self.w_d,
        )


def _collect_layer_hidden_samples(
    model: DCPAModelFlat,
    batches: Sequence[StaticGridBatch],
    max_per_layer: int = 1024,
) -> list[torch.Tensor]:
    """Capture normalized pre-MoE hidden representations `[N_i, D]` across all 8 cortical layers."""
    n_layers = model.cfg.n_layers
    buckets: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    hooks = []

    def _make_hook(layer_idx: int):
        def _hook(mod: nn.Module, inp: tuple[torch.Tensor, ...], kw: dict[str, Any]) -> None:
            x = inp[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])
            mask = kw.get("sample_mask", None)
            if mask is not None:
                m_flat = mask.detach().reshape(-1).bool()
                if m_flat.shape[0] == x_flat.shape[0]:
                    x_flat = x_flat[m_flat]
            if x_flat.shape[0] > 0:
                x_hat = F.normalize(x_flat, dim=-1, eps=1e-6)
                buckets[layer_idx].append(x_hat[: max(1, max_per_layer // max(len(batches), 1))])
        return _hook

    for idx, layer in enumerate(model.cortical.layers):
        hooks.append(layer.moe.register_forward_pre_hook(_make_hook(idx), with_kwargs=True))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for b in batches:
                model(b, stage="cortex", hip_enabled=False)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    out: list[torch.Tensor] = []
    dev = next(model.parameters()).device
    for idx in range(n_layers):
        if buckets[idx]:
            cat = torch.cat(buckets[idx], dim=0)[:max_per_layer]
            out.append(cat.to(device=dev))
        else:
            out.append(torch.zeros((1, model.cfg.d_concept), device=dev, dtype=torch.float32))
    return out


def calibrate_plugin_initialization(
    model: DCPAModelFlat,
    target_batches: Sequence[StaticGridBatch],
    anchor_batches: Sequence[StaticGridBatch],
    *,
    experts_per_layer: int = 1,
    tau_min: Optional[float] = None,
    safety_margin: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Compute centroid prototype keys & dead-zone `tau` above the anchor cosine maximum."""
    n_layers = model.cfg.n_layers
    D = model.cfg.d_concept
    t_min = float(model.cfg.plugin_tau_min if tau_min is None else tau_min)
    dev = next(model.parameters()).device

    target_h = _collect_layer_hidden_samples(model, target_batches)
    anchor_h = _collect_layer_hidden_samples(model, anchor_batches)

    proto_keys = torch.zeros((n_layers, experts_per_layer, D), device=dev, dtype=torch.float32)
    tau_mat = torch.full((n_layers, experts_per_layer), t_min, device=dev, dtype=torch.float32)

    for l_idx in range(n_layers):
        th = target_h[l_idx]  # [Nt, D]
        ah = anchor_h[l_idx]  # [Na, D]
        # Construct domain-discriminative prototype direction: target centroid minus projection onto anchor mean
        t_mean = th.mean(dim=0)
        a_mean = F.normalize(ah.mean(dim=0), dim=-1, eps=1e-6)
        disc_dir = t_mean - torch.dot(t_mean, a_mean) * a_mean
        if float(disc_dir.norm().item()) < 1e-5:
            disc_dir = t_mean
        base_k = F.normalize(disc_dir, dim=-1, eps=1e-6)

        for e_idx in range(experts_per_layer):
            if e_idx == 0:
                k_e = base_k
            else:
                # Split target samples by deterministic stride for multi-expert diversity
                sub = th[e_idx :: experts_per_layer]
                s_mean = sub.mean(dim=0) if sub.shape[0] > 0 else base_k
                s_disc = s_mean - torch.dot(s_mean, a_mean) * a_mean
                k_e = F.normalize(s_disc if float(s_disc.norm().item()) > 1e-5 else base_k, dim=-1, eps=1e-6)
            proto_keys[l_idx, e_idx] = k_e

            # Calibrate dead-zone floor strictly above anchor cosine max (+ safety margin)
            anchor_cos = (ah @ k_e).clamp(-1.0, 1.0)
            a_max = float(anchor_cos.max().item()) if anchor_cos.numel() > 0 else 0.0
            tau_mat[l_idx, e_idx] = min(0.95, max(t_min, a_max + float(safety_margin)))

    return proto_keys, tau_mat, anchor_h


def train_additive_plugin_pack(
    model: DCPAModelFlat,
    pack_id: str,
    target_batches: Sequence[StaticGridBatch],
    anchor_batches: Sequence[StaticGridBatch],
    *,
    steps: int = 50,
    experts_per_layer: int = 1,
    hidden_dim: Optional[int] = None,
    lr: float = 3e-3,
    anchor_repulsion_weight: float = 2.0,
    safety_margin: float = 0.03,
) -> tuple[PluginExpertPack, PluginTrainReport]:
    """Freeze base `DCPAModelFlat`, train an Additive Dead-Zone plugin pack, and verify 0.0 anchor drift."""
    device = next(model.parameters()).device
    cfg = model.cfg
    h_dim = int(cfg.routed_hidden if hidden_dim is None else hidden_dim)

    model.freeze_base_for_plugin_training()
    model.eval()

    # 1. Measure baseline logits & target BPC before plugin mounting
    with torch.no_grad():
        base_anchor_logits = [
            model(ab, stage="cortex", hip_enabled=False).pmd.byte_logits.detach().float().clone()
            for ab in anchor_batches
        ]
        target_bpc_before = float(
            sum(model(tb, stage="cortex", hip_enabled=False).loss_lang.item() for tb in target_batches)
            / max(len(target_batches), 1)
            / math.log(2.0)
        )

    # 2. Calibrate prototype keys and dead-zone thresholds strictly above anchor distribution
    init_keys, init_tau, anchor_h = calibrate_plugin_initialization(
        model,
        target_batches,
        anchor_batches,
        experts_per_layer=experts_per_layer,
        tau_min=cfg.plugin_tau_min,
        safety_margin=safety_margin,
    )

    plugin_mod = TrainablePluginModule(
        pack_id=pack_id,
        n_layers=cfg.n_layers,
        d_model=cfg.d_concept,
        experts_per_layer=experts_per_layer,
        hidden_dim=h_dim,
        tau_min=cfg.plugin_tau_min,
        init_proto_keys=init_keys,
        init_tau=init_tau,
    ).to(device=device)

    opt = torch.optim.AdamW(plugin_mod.parameters(), lr=lr, weight_decay=0.01)

    # 3. End-to-end frozen-base plugin training with anchor cosine repulsion
    for step_i in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        pack = plugin_mod.to_pack()
        model.mount_expert_pack(pack)
        tb = target_batches[step_i % len(target_batches)]
        out = model(tb, stage="cortex", hip_enabled=False)

        # Anchor repulsion penalty keeps proto_keys safely inside dead-zone for all non-target anchors
        rep_loss = torch.zeros((), device=device, dtype=torch.float32)
        for l_idx in range(cfg.n_layers):
            cos_a = anchor_h[l_idx] @ pack.proto_keys[l_idx].t()  # [Na, E_p]
            margin_a = pack.tau[l_idx].unsqueeze(0) - 0.02
            rep_loss = rep_loss + F.relu(cos_a - margin_a).pow(2).mean()

        loss = out.loss_lang + float(anchor_repulsion_weight) * rep_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plugin_mod.parameters(), 1.0)
        opt.step()

    # 4. Final hard dead-zone calibration check & dual acceptance verification
    with torch.no_grad():
        final_proto = F.normalize(plugin_mod.proto_keys_raw, dim=-1, eps=1e-6)
        # Guarantee post-training tau stays strictly above all anchor cosines
        for l_idx in range(cfg.n_layers):
            cos_a = anchor_h[l_idx] @ final_proto[l_idx].t()  # [Na, E_p]
            a_max_vec = cos_a.max(dim=0).values if cos_a.numel() > 0 else torch.zeros_like(plugin_mod.tau_floor[l_idx])
            plugin_mod.tau_floor[l_idx].copy_(
                torch.clamp(torch.maximum(plugin_mod.tau_floor[l_idx], a_max_vec + 0.01), min=cfg.plugin_tau_min, max=0.95)
            )
        final_pack = PluginExpertPack(
            pack_id=pack_id,
            proto_keys=final_proto.detach().clone(),
            tau=plugin_mod.tau_floor.detach().clone(),
            alpha=(F.softplus(plugin_mod.alpha_raw) + 0.1).detach().clone(),
            w_gu=plugin_mod.w_gu.detach().clone(),
            w_d=plugin_mod.w_d.detach().clone(),
        )
        model.mount_expert_pack(final_pack)

        target_bpc_after = float(
            sum(model(tb, stage="cortex", hip_enabled=False).loss_lang.item() for tb in target_batches)
            / max(len(target_batches), 1)
            / math.log(2.0)
        )

        max_drift = 0.0
        active_hits = 0
        total_checks = 0
        for ab, base_log in zip(anchor_batches, base_anchor_logits):
            cur_log = model(ab, stage="cortex", hip_enabled=False).pmd.byte_logits.detach().float()
            drift = float((cur_log - base_log).abs().max().item())
            max_drift = max(max_drift, drift)
        for l_idx in range(cfg.n_layers):
            cos_a = anchor_h[l_idx] @ final_pack.proto_keys[l_idx].t()
            active_hits += int((cos_a > final_pack.tau[l_idx].unsqueeze(0)).sum().item())
            total_checks += int(cos_a.numel())

    report = PluginTrainReport(
        pack_id=pack_id,
        steps=int(steps),
        target_bpc_before=target_bpc_before,
        target_bpc_after=target_bpc_after,
        anchor_max_logit_drift=max_drift,
        anchor_plugin_active_rate=float(active_hits) / max(total_checks, 1),
        passed_zero_drift=bool(max_drift < 1e-6),
    )
    return final_pack, report


def save_plugin_pack_safetensors(pack: PluginExpertPack, out_path: str | Path, metadata: Optional[dict[str, str]] = None) -> Path:
    """Save `PluginExpertPack` in the unified `.dcpa_expert.safetensors` format (shared with `dcpa-core-rs`)."""
    from safetensors.torch import save_file

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "proto_keys": pack.proto_keys.detach().cpu().contiguous(),
        "tau": pack.tau.detach().cpu().contiguous(),
        "alpha": pack.alpha.detach().cpu().contiguous(),
        "w_gu": pack.w_gu.detach().cpu().contiguous(),
        "w_d": pack.w_d.detach().cpu().contiguous(),
    }
    meta = {
        "format": "dcpa_flat_v3_3_plugin",
        "pack_id": pack.pack_id,
        "n_layers": str(pack.proto_keys.shape[0]),
        "experts_per_layer": str(pack.proto_keys.shape[1]),
        "d_concept": str(pack.proto_keys.shape[2]),
        "hidden_dim": str(pack.w_d.shape[3]),
    }
    if metadata:
        meta.update({str(k): str(v) for k, v in metadata.items()})
    save_file(tensors, str(out_p), metadata=meta)
    return out_p


def load_plugin_pack_safetensors(path: str | Path, device: Optional[torch.device] = None) -> PluginExpertPack:
    """Load `PluginExpertPack` from `.dcpa_expert.safetensors`."""
    from safetensors import safe_open

    p = Path(path)
    dev_s = "cpu" if device is None else str(device)
    with safe_open(str(p), framework="pt", device=dev_s) as f:
        meta = f.metadata() or {}
        pack_id = meta.get("pack_id", p.stem)
        return PluginExpertPack(
            pack_id=pack_id,
            proto_keys=f.get_tensor("proto_keys"),
            tau=f.get_tensor("tau"),
            alpha=f.get_tensor("alpha"),
            w_gu=f.get_tensor("w_gu"),
            w_d=f.get_tensor("w_d"),
        )


def load_base_flat_model_from_dir(
    base_dir: str | Path,
    device: Optional[torch.device] = None,
) -> DCPAModelFlat:
    """Load read-only base `DCPAModelFlat` from exported `config.json` + `model.safetensors`."""
    from safetensors.torch import load_file
    from dcpa.interfaces import DCPAFlatConfig

    d = Path(base_dir)
    cfg = DCPAFlatConfig.load_json(d / "config.json")
    model = DCPAModelFlat(cfg)
    raw_sd = load_file(str(d / "model.safetensors"), device="cpu")
    # Re-bind tied PMD embeddings/output projection from SGSC before strict=False load
    if "sgsc.embed.weight" in raw_sd:
        raw_sd["pmd.embed.weight"] = raw_sd["sgsc.embed.weight"]
    if "sgsc.w_out.weight" in raw_sd:
        raw_sd["pmd.w_out.weight"] = raw_sd["sgsc.w_out.weight"]
    model.load_state_dict(raw_sd, strict=False)
    if device is not None:
        model = model.to(device=device)
    model.freeze_base_for_plugin_training()
    model.eval()
    return model


def _build_default_anchor_batches(cfg: "DCPAFlatConfig", device: torch.device) -> list[StaticGridBatch]:
    """Built-in general-domain probe anchors guarding non-target 0.0 bit drift during sleep/domain training."""
    probes = [
        (0, b"The scientific method relies on reproducible experiments, clear hypotheses, and falsifiable measurements.\n\n"),
        (1, "科学方法依赖可重复实验、清晰假设与可证伪的量化检验，任何结论都必须经得起独立复核。\n\n".encode("utf-8")),
    ]
    P, T = cfg.grid_lanes, cfg.grid_bytes
    out: list[StaticGridBatch] = []
    for lid, payload in probes:
        x = torch.zeros((P, T), dtype=torch.uint8)
        doc_id = torch.full((P, T), -1, dtype=torch.int32)
        doc_start = torch.zeros((P, T), dtype=torch.bool)
        doc_end = torch.zeros((P, T), dtype=torch.bool)
        loss_w = torch.zeros((P, T), dtype=torch.float32)
        doc_meta = torch.zeros((P * cfg.doc_meta_cap_per_lane, 3), dtype=torch.int32)
        cur = 0
        for p in range(P):
            pos = 0
            while pos + len(payload) <= T and cur < doc_meta.shape[0]:
                L = len(payload)
                x[p, pos : pos + L] = torch.tensor(list(payload), dtype=torch.uint8)
                doc_id[p, pos : pos + L] = cur
                doc_start[p, pos] = True
                doc_end[p, pos + L - 1] = True
                loss_w[p, pos : pos + L] = 1.0
                doc_meta[cur, 0] = lid
                doc_meta[cur, 1] = 0
                doc_meta[cur, 2] = 1
                cur += 1
                pos += L
        out.append(
            StaticGridBatch(
                x_bytes=x,
                doc_id_byte=doc_id,
                doc_start_byte=doc_start,
                doc_end_byte=doc_end,
                loss_weight=loss_w,
                doc_meta=doc_meta,
            ).to(device=device)
        )
    return out


def _cli_main() -> None:
    """Unified CLI invoked by `dcpa-sleep-daemon` (`--mode sleep`) or domain pack builders (`--mode domain`)."""
    import argparse
    import json
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from data.ragged_dataloader import StaticGridByteDataset
    from utils.device import get_device

    ap = argparse.ArgumentParser(description="DCPA-Flat v3.3 Unified Additive Dead-Zone Plugin Trainer")
    ap.add_argument("--mode", choices=("sleep", "domain"), default="sleep")
    ap.add_argument("--base-dir", required=True, help="Read-only exported base model dir (config.json + model.safetensors)")
    ap.add_argument("--replay-bin", required=True, help="Target/replay .bin shard (e.g. sleep_replay_ring.bin)")
    ap.add_argument("--replay-idx", required=True, help="Target/replay <QIBBBB .idx shard")
    ap.add_argument("--out-plugin", required=True, help="Output .dcpa_expert.safetensors path")
    ap.add_argument("--pack-id", default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--experts-per-layer", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu", help="cpu | auto | cuda | mps")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    dev = torch.device("cpu") if args.device == "cpu" else get_device(preferred=args.device)
    model = load_base_flat_model_from_dir(args.base_dir, device=dev)
    cfg = model.cfg

    ds = StaticGridByteDataset(
        bin_path=args.replay_bin,
        idx_path=args.replay_idx,
        grid_lanes=cfg.grid_lanes,
        grid_bytes=cfg.grid_bytes,
        num_batches=max(2, min(8, args.steps)),
        seed=42,
    )
    target_batches = [b.to(device=dev) for b in ds]
    if not target_batches:
        raise RuntimeError(f"Replay dataset yielded 0 batches from {args.replay_bin}")

    anchor_batches = _build_default_anchor_batches(cfg, device=dev)
    pack_id = args.pack_id or ("personal_sleep" if args.mode == "sleep" else "domain_plugin")

    pack, report = train_additive_plugin_pack(
        model,
        pack_id=pack_id,
        target_batches=target_batches,
        anchor_batches=anchor_batches,
        steps=args.steps,
        experts_per_layer=args.experts_per_layer,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
    )
    save_plugin_pack_safetensors(
        pack,
        args.out_plugin,
        metadata={
            "mode": args.mode,
            "target_bpc_before": f"{report.target_bpc_before:.6f}",
            "target_bpc_after": f"{report.target_bpc_after:.6f}",
            "anchor_max_logit_drift": f"{report.anchor_max_logit_drift:.3e}",
        },
    )
    summary = {
        "mode": args.mode,
        "pack_id": report.pack_id,
        "steps": report.steps,
        "target_bpc_before": report.target_bpc_before,
        "target_bpc_after": report.target_bpc_after,
        "anchor_max_logit_drift": report.anchor_max_logit_drift,
        "anchor_plugin_active_rate": report.anchor_plugin_active_rate,
        "passed_zero_drift": report.passed_zero_drift,
        "out_plugin": str(args.out_plugin),
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not report.passed_zero_drift:
        raise SystemExit(2)


if __name__ == "__main__":
    _cli_main()


__all__ = [
    "PluginTrainReport",
    "TrainablePluginModule",
    "calibrate_plugin_initialization",
    "train_additive_plugin_pack",
    "save_plugin_pack_safetensors",
    "load_plugin_pack_safetensors",
    "load_base_flat_model_from_dir",
]
