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

from dataclasses import dataclass
from typing import Any, Mapping

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
) -> CompileReport:
    """Selectively compile ONLY static-shape Hotspot A (BCG step) & Hotspot B (SGSC/PMD byte scans).

    NEVER wraps the outer `DCPAModel` root in `fullgraph=True`.
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

    use_cuda_inductor = device.type == "cuda"
    backend = "inductor" if use_cuda_inductor else cpu_backend
    compile_kwargs: dict[str, Any] = {"dynamic": False}
    if use_cuda_inductor:
        compile_kwargs["mode"] = "reduce-overhead" if eff_mode == "reduce-overhead" else "default"
    else:
        compile_kwargs["backend"] = backend

    hotspots: list[str] = []

    # --- Hotspot A: BCG-PC single-step concept graph kernel + compacted Pondering/MoE sub-kernels ---
    if hasattr(model, "bcg_pc"):
        bcg_raw = unwrap_compiled_module(getattr(model, "bcg_pc"))
        if hasattr(bcg_raw, "set_static_kernel_mode"):
            bcg_raw.set_static_kernel_mode(True)
        if hasattr(bcg_raw, "enable_compiled_subkernels"):
            bcg_raw.enable_compiled_subkernels(**compile_kwargs)
        compiled_bcg = torch.compile(bcg_raw, **compile_kwargs)
        setattr(model, "bcg_pc", compiled_bcg)
        if hasattr(model, "enable_doc_bucketing"):
            setattr(model, "enable_doc_bucketing", True)
        hotspots.append("bcg_step_kernel")

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
