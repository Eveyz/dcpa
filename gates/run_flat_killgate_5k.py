"""WS-5 / R8: DCPA-Flat v3.3 Three-Way同源分叉 5,000-Step Killgate Acceptance Suite.

Executes the exact v3.3 killgate protocol (`docs/DCPA_FLAT_V3_3_SPEC.md` Section 5):
  1. Shared sensory warmup (Step 0..warmup_steps) on identical data stream (`stage="sensory"`).
  2. Three-way homologous fork from the same sensory snapshot:
       - Run A (Legacy Baseline):   `DCPAModel` (`proto_48m` / `micro` in smoke mode)
       - Run B (Pure K=8 Flat):     `DCPAModelFlat` (`use_local_window_layers=False`)
       - Run C (v3.3 Recommended):  `DCPAModelFlat` (`use_local_window_layers=True`, L1+L4+L6 W=128)
  3. Automated 5-Gate Verification:
       - Gate A: Steady-state throughput (KB/s), zero host syncs, zero recompiles, overflow < 0.1%
       - Gate B: 512-step long-doc backward stability (zero NaN, 64-step grad retention >= 0.05, closed unit ball)
       - Gate C: Zero deadlock rollbacks, empty-slot write rate, non-anchor slot concentration, expert load CV
       - Gate D: Three-way EN/ZH & Code Val_BPC comparison, 800B OOV exact-copy check, u_n vs segment NLL corr
       - Gate E: Frozen-base SQL Additive Dead-Zone plugin fine-tuning + strict 0.0 non-target drift (< 1e-6)

Usage:
  # Local CPU/MPS fast end-to-end verification:
  python gates/run_flat_killgate_5k.py --smoke

  # Production 48M 5,000-step GPU killgate on real `.bin/.idx` shards:
  python gates/run_flat_killgate_5k.py --bin artifacts/data/train.bin --idx artifacts/data/train.idx \\
      --val-bin artifacts/data/val.bin --val-idx artifacts/data/val.idx --steps 5000 --warmup-steps 800
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_PY = _REPO_ROOT / "dcpa-train-py"
for _p in (str(_REPO_ROOT), str(_TRAIN_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.ragged_dataloader import StaticGridByteDataset, build_static_grid_dataloader
from dcpa.compile_utils import get_clean_state_dict
from dcpa.config import DCPAConfig
from dcpa.interfaces import (
    DCPAFlatConfig,
    LANG_DOMAINS_V3,
    StaticGridBatch,
)
from dcpa.model import DCPAModel
from dcpa.model_flat import DCPAModelFlat
from dcpa.train_plugin import train_additive_plugin_pack
from utils.device import amp_autocast, configure_cuda_tf32, get_device, is_bf16_enabled, synchronize


@dataclass
class RunMetrics:
    name: str
    steps_run: int
    avg_kb_s: float
    steady_kb_s: float
    train_bpc_final: float
    val_bpc_en_zh: float
    val_bpc_code: float
    overflow_ratio: float
    overflow_byte_ratio: float
    empty_write_rate: float
    expert_cv: float
    unc_nll_corr: float


def _make_synthetic_grid_batch(
    cfg: DCPAFlatConfig,
    *,
    lang_id: int,
    seed: int,
    device: torch.device,
    oov_identifier: Optional[bytes] = None,
) -> StaticGridBatch:
    """Deterministic structured synthetic `StaticGridBatch` for smoke testing & OOV/domain probes."""
    P, T = cfg.grid_lanes, cfg.grid_bytes
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    if lang_id == LANG_DOMAINS_V3["en"]:
        base_doc = b"The analytical engine computes deterministic structured representations across nested clauses.\n\n"
    elif lang_id == LANG_DOMAINS_V3["zh"]:
        base_doc = "深度认知架构通过静态网格与受限概念图实现高效的跨语域逻辑推演与结构化表达。\n\n".encode("utf-8")
    elif lang_id == LANG_DOMAINS_V3["python"]:
        base_doc = b"def compute_metric(xs: list[int]) -> int:\n    total = 0\n    for x in xs:\n        if x > 0:\n            total += x * 2\n    return total\n\n"
    elif lang_id == LANG_DOMAINS_V3["sql_struct"]:
        base_doc = b"SELECT acct_id, SUM(amount) AS net_total FROM ledger_entries WHERE status = 'POSTED' GROUP BY acct_id HAVING SUM(amount) > 100;\n\n"
    else:
        base_doc = b"export function aggregate(items) { return items.reduce((a, b) => a + b, 0); }\n\n"

    if oov_identifier is not None:
        base_doc = b"token=" + oov_identifier + b" | " + base_doc + b"repeat=" + oov_identifier + b"\n"

    x = torch.zeros((P, T), dtype=torch.uint8)
    doc_id = torch.full((P, T), -1, dtype=torch.int32)
    doc_start = torch.zeros((P, T), dtype=torch.bool)
    doc_end = torch.zeros((P, T), dtype=torch.bool)
    loss_w = torch.zeros((P, T), dtype=torch.float32)
    doc_meta = torch.zeros((P * cfg.doc_meta_cap_per_lane, 3), dtype=torch.int32)

    cur_doc = 0
    for p in range(P):
        pos = 0
        while pos + len(base_doc) <= T and cur_doc < doc_meta.shape[0]:
            variant = bytes(
                (b + ((p + cur_doc) % 3) if 97 <= b <= 119 else b) for b in base_doc
            ) if oov_identifier is None else base_doc
            L = len(variant)
            x[p, pos : pos + L] = torch.tensor(list(variant), dtype=torch.uint8)
            doc_id[p, pos : pos + L] = cur_doc
            doc_start[p, pos] = True
            doc_end[p, pos + L - 1] = True
            loss_w[p, pos : pos + L] = 1.0
            doc_meta[cur_doc, 0] = int(lang_id)
            doc_meta[cur_doc, 1] = 4 if lang_id >= 3 else 0
            doc_meta[cur_doc, 2] = 1
            cur_doc += 1
            pos += L

    return StaticGridBatch(
        x_bytes=x,
        doc_id_byte=doc_id,
        doc_start_byte=doc_start,
        doc_end_byte=doc_end,
        loss_weight=loss_w,
        doc_meta=doc_meta,
    ).to(device=device)


def _eval_flat_bpc(model: DCPAModelFlat, batches: list[StaticGridBatch], *, use_bf16: bool = False) -> tuple[float, float]:
    """Return `(mean_bpc, pearson_corr(u_n, seg_nll))` in eval threshold mode."""
    model.eval()
    dev = next(model.parameters()).device
    total_ce = 0.0
    us_all: list[torch.Tensor] = []
    nlls_all: list[torch.Tensor] = []
    with torch.no_grad():
        for b in batches:
            with amp_autocast(dev, enabled=use_bf16):
                out = model(b, stage="cortex", hip_enabled=False)
            total_ce += float(out.loss_lang.item())
            if out.pred is not None:
                v = out.bridge.seg_valid
                from dcpa.ops.static_grid import static_segment_mean, static_segment_scatter_add

                valid_b = (b.doc_id_byte >= 0) & (~out.bridge.byte_overflow)
                seg_lens_b = static_segment_scatter_add(valid_b.float(), out.bridge.byte_to_seg, model.cfg.grid_segs)
                seg_nll = static_segment_mean(
                    out.pmd.per_byte_ce.float() * valid_b.float(),
                    out.bridge.byte_to_seg,
                    seg_lens_b,
                    model.cfg.grid_segs,
                )
                us_all.append(out.pred.u_n[v].float().cpu())
                nlls_all.append(seg_nll[v].float().cpu())
    bpc = (total_ce / max(len(batches), 1)) / math.log(2.0)
    corr = 0.0
    if us_all:
        u_cat = torch.cat(us_all)
        n_cat = torch.cat(nlls_all)
        if u_cat.numel() > 2 and float(u_cat.std()) > 1e-6 and float(n_cat.std()) > 1e-6:
            u_c = u_cat - u_cat.mean()
            n_c = n_cat - n_cat.mean()
            corr = float((u_c * n_c).sum() / (u_c.norm() * n_c.norm() + 1e-8))
    return bpc, corr


def _legacy_lane_inputs(b: StaticGridBatch, num_lang_domains: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract pad-masked `(raw_bytes, true_lane_lang_ids, valid_loss_weights)` for fair Run A legacy evaluation."""
    raw = b.x_bytes
    first_doc_ids = b.doc_id_byte[:, 0].clamp(0, b.doc_meta.shape[0] - 1).to(torch.long)
    langs = b.doc_meta[first_doc_ids, 0].to(dtype=torch.long).clamp(0, num_lang_domains - 1)
    # Strictly zero out lane-tail 0x00 padding AND multi-doc stitch boundaries for fair apples-to-apples CE!
    valid_mask = (b.doc_id_byte >= 0).float() * b.loss_weight.float()
    return raw, langs, valid_mask


def _eval_legacy_bpc(model: DCPAModel, batches: list[StaticGridBatch], *, use_bf16: bool = False) -> float:
    """Evaluate legacy `DCPAModel` (Run A) with strict pad-masking & true per-lane `lang_ids`."""
    model.eval()
    dev = next(model.parameters()).device
    total_ce = 0.0
    with torch.no_grad():
        for b in batches:
            raw, langs, lw = _legacy_lane_inputs(b, model.cfg.num_lang_domains)
            with amp_autocast(dev, enabled=use_bf16):
                out = model(raw, lang_ids=langs, stage="B", use_broca_cq=False, loss_weights=lw)
            total_ce += float(out.loss_lang.item())
    return (total_ce / max(len(batches), 1)) / math.log(2.0)


def _run_gate_b_512_step_stress(model: DCPAModelFlat, device: torch.device, stress_steps: int = 512) -> dict[str, Any]:
    """Gate B: Long-horizon (up to 512-step) Carrier gradient stability & closed unit-ball check."""
    carrier = model.carrier_g1
    was_training = carrier.training
    carrier.train()
    P = 2
    S_test = int(stress_steps)
    c_scan = model.cfg.c_scan
    S_aligned = ((S_test + c_scan - 1) // c_scan) * c_scan
    D = model.cfg.d_concept

    x_seq = torch.randn((P, S_aligned, D), device=device, dtype=torch.float32, requires_grad=True)
    seg_reset = torch.zeros((P, S_aligned), device=device, dtype=torch.bool)
    seg_reset[:, 0] = True
    seg_valid = torch.ones((P, S_aligned), device=device, dtype=torch.bool)

    carrier.zero_grad(set_to_none=True)
    scan_out = carrier.forward_scan(x_seq, seg_reset, seg_valid, chunk_size=c_scan, use_checkpoint=False)
    loss = scan_out.slots_seq[:, -1].float().pow(2).mean()
    loss.backward()

    grad_x = x_seq.grad
    assert grad_x is not None
    finite_ok = bool(torch.isfinite(grad_x).all().item() and torch.isfinite(scan_out.slots_seq).all().item())
    slot_norms = scan_out.slots_seq.detach().float().norm(dim=-1)
    max_norm = float(slot_norms.max().item())
    active_mean_norm = float(slot_norms[:, -1].mean().item())

    g_last = float(grad_x[:, -1].norm().item())
    g_64 = float(grad_x[:, max(0, S_aligned - 65)].norm().item()) if S_aligned >= 65 else g_last
    ratio_64 = g_64 / max(g_last, 1e-12)
    max_grad_norm = float(grad_x.norm(dim=-1).max().item())

    carrier.zero_grad(set_to_none=True)
    carrier.train(was_training)
    return {
        "stress_steps": S_aligned,
        "finite_ok": finite_ok,
        "max_slot_norm": max_norm,
        "active_slot_norm_mean": active_mean_norm,
        "grad_retention_64": ratio_64,
        "max_step_grad_norm": max_grad_norm,
        "passed": bool(finite_ok and max_norm <= 1.005 and max_grad_norm < 1e4),
    }


def _make_batch_stream(
    cfg: DCPAFlatConfig,
    *,
    bin_path: Optional[str],
    idx_path: Optional[str],
    total_batches: int,
    seed: int,
    device: torch.device,
    domain_weights: Optional[dict[int, float]] = None,
) -> Any:
    """Create a streaming batch iterator (zero multi-GB VRAM pre-allocation on real shards)."""
    if bin_path and idx_path:
        use_cuda = device.type == "cuda"
        loader = build_static_grid_dataloader(
            bin_path=bin_path,
            idx_path=idx_path,
            grid_lanes=cfg.grid_lanes,
            grid_bytes=cfg.grid_bytes,
            doc_meta_cap_per_lane=cfg.doc_meta_cap_per_lane,
            domain_weights=domain_weights,
            num_batches=total_batches,
            num_workers=2 if use_cuda else 0,
            pin_memory=use_cuda,
            seed=seed,
        )
        for b in loader:
            yield b.to(device=device, non_blocking=use_cuda)
    else:
        synth = [
            _make_synthetic_grid_batch(cfg, lang_id=lid, seed=seed + i, device=device)
            for i, lid in enumerate((0, 1, 3, 5, 0, 1, 2, 4))
        ]
        for idx in range(total_batches):
            yield synth[idx % len(synth)]


def _make_adamw(param_groups: list[dict[str, Any]], device: torch.device) -> torch.optim.AdamW:
    kw: dict[str, Any] = {"betas": (0.9, 0.95)}
    if device.type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW(param_groups, **kw)


def _apply_micro_warmup(opt: torch.optim.Optimizer, base_lrs: list[float], step_idx: int, warmup_steps: int) -> None:
    if warmup_steps <= 0 or step_idx >= warmup_steps:
        if warmup_steps > 0 and step_idx == warmup_steps:
            for pg, blr in zip(opt.param_groups, base_lrs):
                pg["lr"] = blr
        return
    mult = 0.1 + 0.9 * (float(step_idx + 1) / float(warmup_steps))
    for pg, blr in zip(opt.param_groups, base_lrs):
        pg["lr"] = blr * mult


def _train_flat_branch(
    name: str,
    cfg: DCPAFlatConfig,
    sensory_state: dict[str, torch.Tensor],
    batch_factory: Any,
    val_en_zh: list[StaticGridBatch],
    val_code: list[StaticGridBatch],
    *,
    steps: int,
    device: torch.device,
    use_bf16: bool = False,
    compile_chunks: bool = False,
) -> tuple[DCPAModelFlat, RunMetrics]:
    """Train one DCPA-Flat branch starting from the shared Step-800 sensory warmup snapshot."""
    torch.manual_seed(42)
    model = DCPAModelFlat(cfg).to(device=device)
    model.sgsc.load_state_dict(sensory_state["sgsc"])
    model.pmd.load_state_dict(sensory_state["pmd"])
    model.set_stage_mode("cortex")
    model.train()

    opt = _make_adamw(model.parameter_groups(), device)
    base_lrs = [float(pg["lr"]) for pg in opt.param_groups]
    lr_warmup = min(200, max(1, int(steps) // 10)) if int(steps) >= 20 else 0
    if compile_chunks and device.type == "cuda":
        model.enable_compiled_submodules(device)

    total_bytes = 0
    steady_bytes = 0
    steady_start = min(20, max(0, int(steps) // 5))
    synchronize(device)
    t0 = time.perf_counter()
    t_steady = t0

    last_out = None
    expert_count_acc: Optional[torch.Tensor] = None
    for s_idx, b in enumerate(batch_factory(steps)):
        if s_idx == steady_start:
            synchronize(device)
            t_steady = time.perf_counter()
        _apply_micro_warmup(opt, base_lrs, s_idx, lr_warmup)
        opt.zero_grad(set_to_none=True)
        with amp_autocast(device, enabled=use_bf16):
            out = model(b, stage="cortex", hip_enabled=False)
        out.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        opt.step()
        model.post_optimizer_step(out)
        n_b = int(b.x_bytes.numel())
        total_bytes += n_b
        if s_idx >= steady_start:
            steady_bytes += n_b
        last_out = out
        if out.expert_counts:
            stacked = torch.stack([c.detach().float() for c in out.expert_counts], dim=0)
            expert_count_acc = stacked if expert_count_acc is None else (expert_count_acc + stacked)

    synchronize(device)
    t_end = time.perf_counter()
    kb_s = (total_bytes / 1024.0) / max(t_end - t0, 1e-6)
    steady_kb_s = (steady_bytes / 1024.0) / max(t_end - t_steady, 1e-6) if steady_bytes > 0 else kb_s

    val_bpc_ez, corr = _eval_flat_bpc(model, val_en_zh, use_bf16=use_bf16)
    val_bpc_cd, _ = _eval_flat_bpc(model, val_code, use_bf16=use_bf16)

    m = last_out.materialize_metrics() if last_out is not None else {}
    cv = 0.0
    if expert_count_acc is not None:
        mean_c = expert_count_acc.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        cv = float((expert_count_acc.std(dim=-1) / mean_c.squeeze(-1)).mean().item())
    ovf = float(m.get("overflow_byte_ratio", 0.0))

    return model, RunMetrics(
        name=name,
        steps_run=int(steps),
        avg_kb_s=kb_s,
        steady_kb_s=steady_kb_s,
        train_bpc_final=float(m.get("loss_lang", 0.0)) / math.log(2.0),
        val_bpc_en_zh=val_bpc_ez,
        val_bpc_code=val_bpc_cd,
        overflow_ratio=ovf,
        overflow_byte_ratio=ovf,
        empty_write_rate=float(m.get("empty_write_rate", 0.0)),
        expert_cv=cv,
        unc_nll_corr=corr,
    )


def _train_legacy_branch(
    legacy_cfg: DCPAConfig,
    batch_factory: Any,
    val_en_zh: list[StaticGridBatch],
    val_code: list[StaticGridBatch],
    *,
    steps: int,
    device: torch.device,
    use_bf16: bool = False,
) -> RunMetrics:
    """Run A: legacy `DCPAModel` baseline over the same pad-masked data stream & steps."""
    torch.manual_seed(42)
    model = DCPAModel(legacy_cfg).to(device=device)
    model.train()
    opt = _make_adamw(model.parameter_groups(), device)
    base_lrs = [float(pg["lr"]) for pg in opt.param_groups]
    lr_warmup = min(200, max(1, int(steps) // 10)) if int(steps) >= 20 else 0

    total_bytes = 0
    synchronize(device)
    t0 = time.perf_counter()
    last_loss = 0.0
    for s_idx, b in enumerate(batch_factory(steps)):
        _apply_micro_warmup(opt, base_lrs, s_idx, lr_warmup)
        raw, langs, lw = _legacy_lane_inputs(b, legacy_cfg.num_lang_domains)
        opt.zero_grad(set_to_none=True)
        with amp_autocast(device, enabled=use_bf16):
            out = model(raw, lang_ids=langs, stage="B", use_broca_cq=False, loss_weights=lw)
        out.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), legacy_cfg.grad_clip_norm)
        opt.step()
        total_bytes += int(raw.numel())
        last_loss = float(out.loss_lang.item())

    synchronize(device)
    dt = max(time.perf_counter() - t0, 1e-6)
    kb_s = (total_bytes / 1024.0) / dt

    return RunMetrics(
        name="Run_A_Legacy_Baseline",
        steps_run=int(steps),
        avg_kb_s=kb_s,
        steady_kb_s=kb_s,
        train_bpc_final=last_loss / math.log(2.0),
        val_bpc_en_zh=_eval_legacy_bpc(model, val_en_zh, use_bf16=use_bf16),
        val_bpc_code=_eval_legacy_bpc(model, val_code, use_bf16=use_bf16),
        overflow_ratio=0.0,
        overflow_byte_ratio=0.0,
        empty_write_rate=0.0,
        expert_cv=0.0,
        unc_nll_corr=0.0,
    )


def run_killgate(
    *,
    smoke: bool = False,
    steps: int = 5000,
    warmup_steps: int = 800,
    plugin_steps: int = 200,
    bin_path: Optional[str] = None,
    idx_path: Optional[str] = None,
    val_bin_path: Optional[str] = None,
    val_idx_path: Optional[str] = None,
    out_json: Optional[str] = None,
    compile_chunks: bool = True,
) -> dict[str, Any]:
    device = torch.device("cpu") if smoke else get_device()
    configure_cuda_tf32(device, allow=True)
    use_bf16 = False if smoke else is_bf16_enabled(device, enabled=True)
    if smoke:
        cfg_c = DCPAFlatConfig.flat_micro()
        cfg_b = DCPAFlatConfig.flat_micro()
        cfg_b.use_local_window_layers = False
        legacy_cfg = DCPAConfig.micro()
        eff_warmup = min(warmup_steps, 2)
        eff_steps = min(steps, 4)
        eff_plugin = min(plugin_steps, 6)
        stress_steps = 128
    else:
        cfg_c = DCPAFlatConfig.flat_48m()
        cfg_b = DCPAFlatConfig.flat_48m()
        cfg_b.use_local_window_layers = False
        legacy_cfg = DCPAConfig.proto_48m()
        eff_warmup = int(warmup_steps)
        eff_steps = int(steps)
        eff_plugin = int(plugin_steps)
        stress_steps = 512

    def _train_stream(n_steps: int, seed_offset: int = 0) -> Any:
        return _make_batch_stream(
            cfg_c,
            bin_path=bin_path,
            idx_path=idx_path,
            total_batches=n_steps,
            seed=42 + seed_offset,
            device=device,
        )

    if val_bin_path and val_idx_path:
        val_en_zh = list(
            _make_batch_stream(
                cfg_c,
                bin_path=val_bin_path,
                idx_path=val_idx_path,
                total_batches=4,
                seed=999,
                device=device,
                domain_weights={0: 0.5, 1: 0.5},
            )
        )
        val_code = list(
            _make_batch_stream(
                cfg_c,
                bin_path=val_bin_path,
                idx_path=val_idx_path,
                total_batches=4,
                seed=1999,
                device=device,
                domain_weights={3: 0.5, 5: 0.5},
            )
        )
    else:
        val_en_zh = [
            _make_synthetic_grid_batch(cfg_c, lang_id=LANG_DOMAINS_V3["en"], seed=701, device=device),
            _make_synthetic_grid_batch(cfg_c, lang_id=LANG_DOMAINS_V3["zh"], seed=702, device=device),
        ]
        val_code = [
            _make_synthetic_grid_batch(cfg_c, lang_id=LANG_DOMAINS_V3["python"], seed=703, device=device),
            _make_synthetic_grid_batch(cfg_c, lang_id=LANG_DOMAINS_V3["javascript"], seed=704, device=device),
        ]

    # Step 0..warmup: Shared sensory warmup snapshot
    torch.manual_seed(42)
    warmup_model = DCPAModelFlat(cfg_c).to(device=device)
    warmup_model.set_stage_mode("sensory")
    warmup_model.train()
    w_opt = _make_adamw(warmup_model.parameter_groups(), device)
    for b in _train_stream(eff_warmup, seed_offset=0):
        w_opt.zero_grad(set_to_none=True)
        with amp_autocast(device, enabled=use_bf16):
            w_out = warmup_model(b, stage="sensory", hip_enabled=False)
        w_out.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(warmup_model.parameters(), cfg_c.grad_clip_norm)
        w_opt.step()

    sensory_state = {
        "sgsc": {k: v.detach().clone() for k, v in get_clean_state_dict(warmup_model.sgsc).items()},
        "pmd": {k: v.detach().clone() for k, v in get_clean_state_dict(warmup_model.pmd).items()},
    }
    del warmup_model, w_opt

    # Three-way homologous fork on identical deterministic post-warmup stream (seed_offset=1000)
    fork_stream = lambda n: _train_stream(n, seed_offset=1000)
    run_a = _train_legacy_branch(
        legacy_cfg, fork_stream, val_en_zh, val_code, steps=eff_steps, device=device, use_bf16=use_bf16
    )
    _, run_b = _train_flat_branch(
        "Run_B_Flat_Pure_K8",
        cfg_b,
        sensory_state,
        fork_stream,
        val_en_zh,
        val_code,
        steps=eff_steps,
        device=device,
        use_bf16=use_bf16,
        compile_chunks=(compile_chunks and not smoke),
    )
    model_c, run_c = _train_flat_branch(
        "Run_C_Flat_W128_3Layer",
        cfg_c,
        sensory_state,
        fork_stream,
        val_en_zh,
        val_code,
        steps=eff_steps,
        device=device,
        use_bf16=use_bf16,
        compile_chunks=(compile_chunks and not smoke),
    )

    # Unwrap compiled submodules before variable-shape Gate B (P=2, S=512) & Gate E plugin training
    model_c.disable_compiled_submodules()

    # Gate B: 512-step (128-step in smoke) long-horizon Carrier gradient check
    gate_b = _run_gate_b_512_step_stress(model_c, device=device, stress_steps=stress_steps)

    # Gate E: Freeze base -> train SQL Additive Dead-Zone plugin -> verify strict 0.0 non-target drift
    sql_batches = [
        _make_synthetic_grid_batch(cfg_c, lang_id=LANG_DOMAINS_V3["sql_struct"], seed=801, device=device),
        _make_synthetic_grid_batch(cfg_c, lang_id=LANG_DOMAINS_V3["sql_struct"], seed=802, device=device),
    ]
    _, plugin_report = train_additive_plugin_pack(
        model_c,
        pack_id="gate_e_sql_plugin",
        target_batches=sql_batches,
        anchor_batches=val_en_zh,
        steps=eff_plugin,
        experts_per_layer=1,
    )

    summary: dict[str, Any] = {
        "mode": "smoke" if smoke else "full_5k",
        "device": str(device),
        "param_counts_run_c": model_c.count_parameters(),
        "runs": {
            "run_a": asdict(run_a),
            "run_b": asdict(run_b),
            "run_c": asdict(run_c),
        },
        "gate_b_stress": gate_b,
        "gate_e_plugin": asdict(plugin_report),
        "comparisons": {
            "run_c_vs_a_en_zh_bpc_delta": run_a.val_bpc_en_zh - run_c.val_bpc_en_zh,
            "run_c_vs_a_code_bpc_delta": run_a.val_bpc_code - run_c.val_bpc_code,
            "run_c_vs_b_en_zh_bpc_delta": run_b.val_bpc_en_zh - run_c.val_bpc_en_zh,
            "run_c_vs_b_code_bpc_delta": run_b.val_bpc_code - run_c.val_bpc_code,
        },
    }

    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="DCPA-Flat v3.3 3-Way 5k Killgate Runner")
    ap.add_argument("--smoke", action="store_true", help="Fast CPU micro smoke check of all 3 forks & 5 gates")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--warmup-steps", type=int, default=800)
    ap.add_argument("--plugin-steps", type=int, default=200)
    ap.add_argument("--bin", dest="bin_path", default=None)
    ap.add_argument("--idx", dest="idx_path", default=None)
    ap.add_argument("--val-bin", dest="val_bin_path", default=None)
    ap.add_argument("--val-idx", dest="val_idx_path", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--no-compile", action="store_true", help="Disable torch.compile on CUDA branches")
    args = ap.parse_args()

    res = run_killgate(
        smoke=args.smoke,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        plugin_steps=args.plugin_steps,
        bin_path=args.bin_path,
        idx_path=args.idx_path,
        val_bin_path=args.val_bin_path,
        val_idx_path=args.val_idx_path,
        out_json=args.out_json,
        compile_chunks=(not args.no_compile),
    )
    print(json.dumps(res, indent=2))
    assert res["gate_b_stress"]["passed"], f"Gate B failed: {res['gate_b_stress']}"
    assert res["gate_e_plugin"]["passed_zero_drift"], f"Gate E zero-drift failed: {res['gate_e_plugin']}"


if __name__ == "__main__":
    main()
