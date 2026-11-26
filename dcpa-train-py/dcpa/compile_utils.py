"""Selective static-shape `torch.compile` orchestration and `_orig_mod.` state-dict sanitizer for DCPA.

Implements Performance Audit Task 2:
  1. NEVER wraps outer dynamic-length `DCPAModel` root in `torch.compile(..., fullgraph=True)`.
  2. Compiles ONLY the two static-shape hot kernels with `dynamic=False`:
       - Hotspot A: `model.bcg_pc` single-step concept graph kernel (`BCGPC.forward`, bucketed
         along the document dimension to multiples of 16 via `active_mask`).
       - Hotspot B: `model.sgsc._byte_scan_impl` & `model.pmd._motor_scan_impl` per-byte gated
         projections + C=256 chunked associative scans (aligned to multiples of 256 bytes).
  3. Sets `torch._dynamo.config.cache_size_limit = 64` to eliminate mid-run recompilation stalls.
  4. Automatically falls back to `"none"` on `cpu` / `mps` unless explicitly enabled for portability
     tests (`allow_cpu_compile=True`).
  5. Guarantees 100% `_orig_mod.`-free Checkpoint and `.safetensors` keys via
     `strip_orig_mod_prefix(...)` and `get_clean_state_dict(model)` so compiled CUDA weights
     always load with `strict=True` on uncompiled local M3 Pro models.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


VALID_COMPILE_MODES: tuple[str, ...] = ("none", "auto", "reduce-overhead")
DEFAULT_DOC_BUCKET_SIZE: int = 16
DEFAULT_BYTE_ALIGN_CHUNK: int = 256


@dataclass(frozen=True)
class CompileReport:
    requested_mode: str
    effective_mode: str
    backend: str
    compiled_hotspots: tuple[str, ...]
    doc_bucket_size: int
    cache_size_limit: int
    fallback_reason: str = ""


def strip_orig_mod_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively remove all `_orig_mod.` wrappers from any module state_dict keys."""
    clean: dict[str, Any] = {}
    for key, val in state_dict.items():
        parts = [seg for seg in str(key).split(".") if seg != "_orig_mod"]
        new_key = ".".join(parts)
        clean[new_key] = val
    return clean


def migrate_legacy_moe_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Convert legacy per-expert `*.moe.experts.{e}.w_*` keys into native 3D stacked tensors.

    Idempotent: returns a shallow copy unchanged if the state_dict already uses `w_gu1`/`w_d1`/`w_gu2`/`w_d2`.
    """
    out: dict[str, Any] = dict(state_dict)
    legacy_prefixes: set[str] = set()
    for k in out:
        if k.endswith("experts.0.w_gate1.weight"):
            legacy_prefixes.add(k[: -len("experts.0.w_gate1.weight")])

    for pref in sorted(legacy_prefixes):
        if f"{pref}w_gu1" in out:
            continue
        n_exp = 0
        while f"{pref}experts.{n_exp}.w_gate1.weight" in out:
            n_exp += 1
        if n_exp == 0:
            continue
        g1_l, u1_l, d1_l, g2_l, u2_l, d2_l = [], [], [], [], [], []
        ok = True
        for e in range(n_exp):
            ep = f"{pref}experts.{e}."
            ks = (
                f"{ep}w_gate1.weight",
                f"{ep}w_up1.weight",
                f"{ep}w_down1.weight",
                f"{ep}w_gate2.weight",
                f"{ep}w_up2.weight",
                f"{ep}w_down2.weight",
            )
            if not all(k in out for k in ks):
                ok = False
                break
            g1_l.append(out.pop(ks[0]))
            u1_l.append(out.pop(ks[1]))
            d1_l.append(out.pop(ks[2]))
            g2_l.append(out.pop(ks[3]))
            u2_l.append(out.pop(ks[4]))
            d2_l.append(out.pop(ks[5]))
        if ok:
            out[f"{pref}w_gu1"] = torch.stack(
                [torch.cat([g, u], dim=0) for g, u in zip(g1_l, u1_l)], dim=0
            ).contiguous()
            out[f"{pref}w_d1"] = torch.stack(d1_l, dim=0).contiguous()
            out[f"{pref}w_gu2"] = torch.stack(
                [torch.cat([g, u], dim=0) for g, u in zip(g2_l, u2_l)], dim=0
            ).contiguous()
            out[f"{pref}w_d2"] = torch.stack(d2_l, dim=0).contiguous()
    legacy_1blk_prefixes: set[str] = set()
    for k in list(out.keys()):
        if k.endswith("experts.0.w_gate.weight"):
            legacy_1blk_prefixes.add(k[: -len("experts.0.w_gate.weight")])

    for pref in sorted(legacy_1blk_prefixes):
        if f"{pref}w_gu" in out:
            continue
        n_exp = 0
        while f"{pref}experts.{n_exp}.w_gate.weight" in out:
            n_exp += 1
        if n_exp == 0:
            continue
        g_l, u_l, d_l = [], [], []
        ok = True
        for e in range(n_exp):
            ep = f"{pref}experts.{e}."
            ks = (f"{ep}w_gate.weight", f"{ep}w_up.weight", f"{ep}w_down.weight")
            if not all(k in out for k in ks):
                ok = False
                break
            g_l.append(out.pop(ks[0]))
            u_l.append(out.pop(ks[1]))
            d_l.append(out.pop(ks[2]))
        if ok:
            out[f"{pref}w_gu"] = torch.stack(
                [torch.cat([g, u], dim=0) for g, u in zip(g_l, u_l)], dim=0
            ).contiguous()
            out[f"{pref}w_d"] = torch.stack(d_l, dim=0).contiguous()
    return out


def migrate_legacy_moe_optimizer_state_dict(
    opt_sd: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Losslessly migrate AdamW momentum buffers (`exp_avg`, `exp_avg_sq`, `step`) from legacy
    per-expert `experts.{e}.w_*` parameter layout into native 3D stacked `w_gu1`/`w_d1`/`w_gu2`/`w_d2`.

    Uses the `"param_names"` metadata persisted in every DCPA optimizer parameter group so in-flight
    training checkpoints (e.g. step 75,000) resume with 100% preserved momentum states!
    """
    old_groups = list(opt_sd.get("param_groups", []))
    old_state = dict(opt_sd.get("state", {}))

    has_legacy_opt = any(
        any(str(pn).endswith(".experts.0.w_gate1.weight") for pn in (g.get("param_names") or ()))
        for g in old_groups
    )
    has_stacked_live = any(
        any(str(pn).endswith(".w_gu1") for pn in (g.get("param_names") or ()))
        for g in optimizer.param_groups
    )
    if not (has_legacy_opt and has_stacked_live):
        return dict(opt_sd)

    old_name_to_state: dict[str, dict[str, Any]] = {}
    old_group_by_name: dict[str, Mapping[str, Any]] = {}
    for og in old_groups:
        gname = og.get("name")
        if gname is not None:
            old_group_by_name[str(gname)] = og
        pnames = og.get("param_names")
        pids = og.get("params", [])
        if pnames is not None and len(pnames) == len(pids):
            for pn, pid in zip(pnames, pids):
                if pid in old_state:
                    old_name_to_state[str(pn)] = old_state[pid]

    def _clone_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in entry.items()
        }

    def _pack_expert_moment(
        moe_prefix: str,
        param_shape: tuple[int, ...],
        kind: str,
    ) -> Optional[dict[str, Any]]:
        n_exp = int(param_shape[0])
        if kind in ("w_gu1", "w_gu2"):
            suffix_g = "w_gate1.weight" if kind == "w_gu1" else "w_gate2.weight"
            suffix_u = "w_up1.weight" if kind == "w_gu1" else "w_up2.weight"
            h_dim = int(param_shape[1]) // 2
            d_dim = int(param_shape[2])
            g_entries = [old_name_to_state.get(f"{moe_prefix}.experts.{e}.{suffix_g}") for e in range(n_exp)]
            u_entries = [old_name_to_state.get(f"{moe_prefix}.experts.{e}.{suffix_u}") for e in range(n_exp)]
            present = [ent for ent in (*g_entries, *u_entries) if ent is not None]
            if not present:
                return None
            ref = present[0]
            out_ent: dict[str, Any] = {}
            if "step" in ref:
                steps = [ent["step"] for ent in present if "step" in ent]
                max_step = max(steps, key=lambda s: float(s.item()) if isinstance(s, torch.Tensor) else float(s))
                out_ent["step"] = max_step.clone() if isinstance(max_step, torch.Tensor) else max_step
            for buf_key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if buf_key not in ref:
                    continue
                ref_t = ref[buf_key]
                stacked_rows = []
                for g_ent, u_ent in zip(g_entries, u_entries):
                    gt = (
                        g_ent[buf_key]
                        if (g_ent is not None and buf_key in g_ent)
                        else torch.zeros((h_dim, d_dim), dtype=ref_t.dtype, device=ref_t.device)
                    )
                    ut = (
                        u_ent[buf_key]
                        if (u_ent is not None and buf_key in u_ent)
                        else torch.zeros((h_dim, d_dim), dtype=ref_t.dtype, device=ref_t.device)
                    )
                    stacked_rows.append(torch.cat([gt, ut], dim=0))
                out_ent[buf_key] = torch.stack(stacked_rows, dim=0).contiguous()
            return out_ent

        suffix_d = "w_down1.weight" if kind == "w_d1" else "w_down2.weight"
        out_dim = int(param_shape[1])
        in_dim = int(param_shape[2])
        d_entries = [old_name_to_state.get(f"{moe_prefix}.experts.{e}.{suffix_d}") for e in range(n_exp)]
        present = [ent for ent in d_entries if ent is not None]
        if not present:
            return None
        ref = present[0]
        out_ent = {}
        if "step" in ref:
            steps = [ent["step"] for ent in present if "step" in ent]
            max_step = max(steps, key=lambda s: float(s.item()) if isinstance(s, torch.Tensor) else float(s))
            out_ent["step"] = max_step.clone() if isinstance(max_step, torch.Tensor) else max_step
        for buf_key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if buf_key not in ref:
                continue
            ref_t = ref[buf_key]
            rows = [
                (
                    ent[buf_key]
                    if (ent is not None and buf_key in ent)
                    else torch.zeros((out_dim, in_dim), dtype=ref_t.dtype, device=ref_t.device)
                )
                for ent in d_entries
            ]
            out_ent[buf_key] = torch.stack(rows, dim=0).contiguous()
        return out_ent

    base_live_sd = optimizer.state_dict()
    new_groups: list[dict[str, Any]] = []
    new_state: dict[int, dict[str, Any]] = {}

    for g_idx, (live_group, live_sd_group) in enumerate(
        zip(optimizer.param_groups, base_live_sd["param_groups"])
    ):
        ng = dict(live_sd_group)
        gname = ng.get("name")
        if gname is not None and str(gname) in old_group_by_name:
            og = old_group_by_name[str(gname)]
            for k, v in og.items():
                if k not in ("params", "param_names", "fused", "foreach"):
                    ng[k] = v
        new_groups.append(ng)

        pnames = live_group.get("param_names")
        live_params = live_group["params"]
        live_pids = ng["params"]
        if pnames is None or len(pnames) != len(live_pids):
            continue
        for pn_raw, live_p, new_pid in zip(pnames, live_params, live_pids):
            pn = str(pn_raw)
            if pn in old_name_to_state:
                new_state[new_pid] = _clone_entry(old_name_to_state[pn])
            else:
                for kind in ("w_gu1", "w_d1", "w_gu2", "w_d2"):
                    if pn.endswith(f".{kind}"):
                        moe_pref = pn[: -(len(kind) + 1)]
                        packed = _pack_expert_moment(moe_pref, tuple(live_p.shape), kind)
                        if packed is not None:
                            new_state[new_pid] = packed
                        break

    return {"state": new_state, "param_groups": new_groups}


def unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Follow `_orig_mod` chain to recover the underlying unwrapped `nn.Module`."""
    cur = module
    while hasattr(cur, "_orig_mod") and isinstance(getattr(cur, "_orig_mod"), nn.Module):
        cur = getattr(cur, "_orig_mod")
    return cur


def get_clean_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract a canonical state_dict strictly free of `_orig_mod.` wrapper prefixes.

    Works identically whether zero, some, or all submodules are wrapped by `torch.compile`.
    """
    raw = unwrap_compiled_module(model).state_dict()
    cleaned = strip_orig_mod_prefix(raw)
    for k in cleaned:
        if "_orig_mod" in k:
            raise RuntimeError(f"Checkpoint purity violation: residual '_orig_mod' found in key {k!r}")
    return cleaned


def resolve_compile_mode(
    requested_mode: str,
    device: torch.device,
    *,
    allow_cpu_compile: bool = False,
) -> tuple[str, str]:
    """Resolve CLI/YAML `--compile-mode` safely for CUDA, Apple MPS, and CPU."""
    mode = str(requested_mode or "none").strip().lower()
    if mode not in VALID_COMPILE_MODES:
        raise ValueError(f"Unsupported compile_mode={requested_mode!r}; expected one of {VALID_COMPILE_MODES}")
    if mode == "none":
        return "none", ""
    if device.type == "cuda":
        return mode, ""
    if allow_cpu_compile and device.type == "cpu":
        return mode, "cpu_test_compile_enabled"
    return "none", f"auto-fallback to 'none' on non-CUDA device ({device.type})"


def configure_model_compile(
    model: nn.Module,
    compile_mode: str,
    device: torch.device,
    *,
    doc_bucket_size: int = DEFAULT_DOC_BUCKET_SIZE,
    cache_size_limit: int = 128,
    allow_cpu_compile: bool = False,
    cpu_backend: str = "aot_eager",
    cache_dir: Optional[str | Path] = None,
    bcg_cudagraphs: bool = True,
) -> CompileReport:
    """Selectively compile ONLY static-shape Hotspot A (BCG step) & Hotspot B (SGSC/PMD byte scans).

    NEVER wraps the outer `DCPAModel` root in `fullgraph=True`.
    `bcg_cudagraphs=True` (CUDA only) additionally captures the static-shape BCG-PC clause-chunk kernel
    with CUDA Graphs (`mode="reduce-overhead"`), removing per-chunk launch overhead of the serial loop.
    """
    eff_mode, reason = resolve_compile_mode(compile_mode, device, allow_cpu_compile=allow_cpu_compile)

    # Always disable bucketing/compiled hooks cleanly first
    if hasattr(model, "enable_doc_bucketing"):
        setattr(model, "enable_doc_bucketing", False)
    if hasattr(model, "doc_bucket_size"):
        setattr(model, "doc_bucket_size", int(doc_bucket_size))
    if hasattr(model, "bcg_pc"):
        bcg_unwrapped = unwrap_compiled_module(getattr(model, "bcg_pc"))
        if hasattr(bcg_unwrapped, "disable_compiled_subkernels"):
            bcg_unwrapped.disable_compiled_subkernels()
    if hasattr(model, "cortical"):
        cort_unwrapped = unwrap_compiled_module(getattr(model, "cortical"))
        if hasattr(cort_unwrapped, "disable_compiled_kernels"):
            cort_unwrapped.disable_compiled_kernels()

    if eff_mode == "none":
        return CompileReport(
            requested_mode=compile_mode,
            effective_mode="none",
            backend="none",
            compiled_hotspots=(),
            doc_bucket_size=int(doc_bucket_size),
            cache_size_limit=int(cache_size_limit),
            fallback_reason=reason,
        )

    # Guard against mid-run recompilation stalls across fixed doc buckets (Task 2.3)
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.cache_size_limit = max(int(cache_size_limit), 64)
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = max(int(cache_size_limit) * 4, 256)

    # Persist TorchInductor & Triton disk caches on local NVMe (`cache_dir`, isolated per rank under DDP)
    # so single-GPU restarts and Stage-5 SFT skip cold compilation stalls completely!
    world_sz = int(os.environ.get("WORLD_SIZE", "1") or "1")
    local_rk = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    if device.type == "cuda":
        default_root = (
            str(Path(cache_dir).resolve())
            if cache_dir is not None
            else str(Path(tempfile.gettempdir()) / "dcpa_torchinductor")
        )
        base_inductor_dir = os.environ.get("DCPA_INDUCTOR_CACHE_ROOT", default_root)
        if world_sz > 1:
            rank_cache = Path(base_inductor_dir) / f"rank_{local_rk}"
            triton_cache = Path(base_inductor_dir) / f"triton_rank_{local_rk}"
            rank_cache.mkdir(parents=True, exist_ok=True)
            triton_cache.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(rank_cache)
            os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        elif cache_dir is not None:
            single_ind = Path(base_inductor_dir) / "inductor"
            single_tri = Path(base_inductor_dir) / "triton"
            single_ind.mkdir(parents=True, exist_ok=True)
            single_tri.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(single_ind))
            os.environ.setdefault("TRITON_CACHE_DIR", str(single_tri))

    use_cuda_inductor = device.type == "cuda"
    if use_cuda_inductor and hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
        n_cpus = max(1, int(os.cpu_count() or 1))
        torch._inductor.config.compile_threads = min(32, n_cpus)
    backend = "inductor" if use_cuda_inductor else cpu_backend
    compile_kwargs: dict[str, Any] = {"dynamic": False}
    if use_cuda_inductor:
        compile_kwargs["mode"] = "reduce-overhead" if eff_mode == "reduce-overhead" else "default"
    else:
        compile_kwargs["backend"] = backend

    hotspots: list[str] = []

    # --- Hotspot A: BCG-PC single-step & multi-step chunked concept graph kernels ---
    if hasattr(model, "bcg_pc"):
        bcg_raw = unwrap_compiled_module(getattr(model, "bcg_pc"))
        if hasattr(bcg_raw, "set_static_kernel_mode"):
            bcg_raw.set_static_kernel_mode(True)
        if hasattr(bcg_raw, "enable_compiled_chunk_kernel"):
            bcg_raw.enable_compiled_chunk_kernel(**compile_kwargs)
            hotspots.append("bcg_chunk_kernel")
        if bcg_cudagraphs and use_cuda_inductor and hasattr(bcg_raw, "enable_cudagraph_chunk_kernel"):
            bcg_raw.enable_cudagraph_chunk_kernel(**compile_kwargs)
            hotspots.append("bcg_chunk_cudagraph")
        if hasattr(bcg_raw, "enable_compiled_ponder_parallel"):
            bcg_raw.enable_compiled_ponder_parallel(**compile_kwargs)
            hotspots.append("bcg_ponder_parallel_kernel")
        compiled_bcg = torch.compile(bcg_raw, **compile_kwargs)
        setattr(model, "bcg_pc", compiled_bcg)
        if hasattr(model, "enable_doc_bucketing"):
            setattr(model, "enable_doc_bucketing", True)
        hotspots.append("bcg_step_kernel")

    # --- Hotspot C: Parallel Cortical Sandwich (Tiled 2D Concept-RoPE SWA + Shared/Routed MoE) ---
    if hasattr(model, "cortical"):
        cort_mod = unwrap_compiled_module(getattr(model, "cortical"))
        if hasattr(cort_mod, "enable_compiled_kernels"):
            cort_mod.enable_compiled_kernels(**compile_kwargs)
            hotspots.append("cortical_sandwich_kernel")

    # --- Hotspot B1: SGSC per-byte gated projections & chunked diagonal scan ---
    if hasattr(model, "sgsc"):
        sgsc_mod = unwrap_compiled_module(getattr(model, "sgsc"))
        if hasattr(sgsc_mod, "enable_compiled_byte_scan"):
            sgsc_mod.enable_compiled_byte_scan(**compile_kwargs)
            hotspots.append("sgsc_byte_scan_kernel")

    # --- Hotspot B2: PMD 128B chunked stack scan + motor diagonal scan + byte head ---
    if hasattr(model, "pmd"):
        pmd_mod = unwrap_compiled_module(getattr(model, "pmd"))
        if hasattr(pmd_mod, "enable_compiled_motor_scan"):
            pmd_mod.enable_compiled_motor_scan(**compile_kwargs)
            hotspots.append("pmd_motor_scan_kernel")

    return CompileReport(
        requested_mode=compile_mode,
        effective_mode=eff_mode,
        backend=backend,
        compiled_hotspots=tuple(hotspots),
        doc_bucket_size=int(doc_bucket_size),
        cache_size_limit=int(cache_size_limit),
        fallback_reason=reason,
    )
