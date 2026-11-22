"""Cross-platform device abstraction layer for DCPA (Apple Silicon M3 Pro + Google Colab CUDA).

Implements Iron Law #1:
  1. Single source of truth for device selection, AMP autocast, synchronization, RNG state
     checkpointing, and 4-tier hardware micro-batch / gradient-accumulation sharding.
  2. Deterministic unit tests (`tests/test_sanity.py`) force pure `cpu` + `float32` whenever
     CUDA is unavailable, eliminating Apple MPS floating-point rounding nondeterminism.
  3. Smoke & lifecycle scripts (`smoke_overfit_64.py`, `tests/test_trainer_lifecycle.py`) auto-pick
     `cuda -> mps -> cpu`, set `PYTORCH_ENABLE_MPS_FALLBACK=1` on MPS, and ONLY enable bfloat16
     autocast when `device.type == "cuda"` (safe pure float32 fallback on MPS/CPU).
  4. Cross-platform RNG state save/restore never fails when moving checkpoints between Mac and Colab.
  5. Four-tier hardware auto-sharding contract:
       - H100 / A100 80GB (VRAM >= 75 GB)      -> micro_batch_bytes=131072, grad_accum_steps=1
       - A100 40GB        (38 <= VRAM < 75 GB) -> micro_batch_bytes=65536,  grad_accum_steps=2
       - L4 / RTX 4090    (20 <= VRAM < 38 GB) -> micro_batch_bytes=32768,  grad_accum_steps=4
       - Apple MPS / CPU  (local debug)        -> default target_tokens_bytes_per_step=8192
"""

from __future__ import annotations

import contextlib
import math
import os
import platform
import random
from dataclasses import asdict, dataclass
from typing import Any, ContextManager, Optional

# Must be configured BEFORE importing torch so PyTorch's C++ MPS dispatcher reads it on macOS.
if platform.system() == "Darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch


DEFAULT_PROD_STEP_BYTES: int = 131_072
DEFAULT_LOCAL_STEP_BYTES: int = 8_192


@dataclass(frozen=True)
class HardwareProfile:
    """Resolved execution hardware tier and micro-batch / accumulation contract."""

    device_type: str  # "cuda" | "mps" | "cpu"
    device_name: str
    vram_gb: float
    tier: str
    micro_batch_bytes: int
    grad_accum_steps: int
    target_tokens_bytes_per_step: int
    use_bf16: bool

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def is_mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def configure_mps_fallback(device: torch.device) -> None:
    """Enable CPU fallback for unsupported MPS ops when executing on Apple Silicon GPU."""
    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def get_device(preferred: str = "auto", deterministic: bool = False) -> torch.device:
    """Resolve compute device safely across CUDA, Apple MPS, and CPU.

    Args:
      preferred: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"`.
      deterministic: When True, never select MPS; use `"cuda"` if available (and requested/auto),
        otherwise force pure `"cpu"` to guarantee exact float32 reproducibility.
    """
    pref = preferred.strip().lower()
    if pref not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"Unsupported device preference: {preferred!r}")

    if deterministic:
        if pref in ("auto", "cuda") and is_cuda_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if pref == "auto":
        if is_cuda_available():
            dev = torch.device("cuda")
        elif is_mps_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    elif pref == "cuda":
        if not is_cuda_available():
            raise RuntimeError("CUDA device explicitly requested, but torch.cuda.is_available() is False.")
        dev = torch.device("cuda")
    elif pref == "mps":
        if not is_mps_available():
            raise RuntimeError("MPS device explicitly requested, but torch.backends.mps.is_available() is False.")
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    configure_mps_fallback(dev)
    return dev


def get_deterministic_test_device() -> torch.device:
    """Device for deterministic unit tests (`tests/test_sanity.py`): CUDA if present, else pure CPU."""
    return get_device(preferred="auto", deterministic=True)


def is_bf16_enabled(device: torch.device, enabled: bool = True) -> bool:
    """Only enable bfloat16 autocast on CUDA devices supporting bf16; pure float32 on MPS/CPU."""
    if not enabled or device.type != "cuda" or not is_cuda_available():
        return False
    return bool(torch.cuda.is_bf16_supported())


def amp_autocast(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    enabled: bool = True,
) -> ContextManager[Any]:
    """Cross-platform AMP autocast context manager.

    Strictly activates `torch.amp.autocast('cuda', ...)` only when `device.type == 'cuda'`.
    When `enabled=False` on CUDA, explicitly returns `torch.amp.autocast(device_type='cuda', enabled=False)`
    so nested float32 precision islands genuinely override an outer bfloat16 autocast block.
    Safely falls back to `contextlib.nullcontext()` (pure float32) on MPS and CPU.
    """
    if device.type != "cuda" or not is_cuda_available():
        return contextlib.nullcontext()
    if not enabled:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)


def fp32_island(device: torch.device) -> ContextManager[Any]:
    """Explicit float32 precision island context manager overriding outer CUDA bfloat16 autocast."""
    return amp_autocast(device, enabled=False)


def configure_cuda_tf32(device: torch.device, allow: bool = True) -> None:
    """Safely configure CUDA TF32 matmul/cuDNN flags without touching CUDA APIs on non-CUDA hosts."""
    if device.type == "cuda" and is_cuda_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow)
        torch.backends.cudnn.allow_tf32 = bool(allow)


def reset_peak_memory_stats(device: torch.device) -> None:
    """Reset peak device memory tracking statistics safely across CUDA, MPS, and CPU."""
    if device.type == "cuda" and is_cuda_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_gb(device: torch.device) -> float:
    """Return peak allocated accelerator memory in GB (or host process RSS GB on CPU/MPS fallback)."""
    if device.type == "cuda" and is_cuda_available():
        return float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
    if device.type == "mps" and is_mps_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
            try:
                val = float(torch.mps.driver_allocated_memory()) / float(1024**3)
                if val > 0.0:
                    return val
            except Exception:
                pass
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                val = float(torch.mps.current_allocated_memory()) / float(1024**3)
                if val > 0.0:
                    return val
            except Exception:
                pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS ru_maxrss is in bytes; Linux ru_maxrss is in KiB.
        if platform.system() == "Darwin":
            return rss / float(1024**3)
        return rss / float(1024**2)
    except Exception:
        return 0.0


def synchronize(device: torch.device) -> None:
    """Synchronize asynchronous device kernels safely across CUDA, MPS, and CPU."""
    if device.type == "cuda" and is_cuda_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and is_mps_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def select_hardware_tier(
    device_type: str,
    vram_gb: float,
    device_name: str = "",
    target_tokens_bytes_per_step: Optional[int] = None,
    bf16_supported: Optional[bool] = None,
) -> HardwareProfile:
    """Pure hardware tier classifier implementing the 4-tier auto-sharding contract.

    Separating classification from live CUDA queries keeps all 4 hardware tiers
    (H100/A100-80G, A100-40G, L4/4090, Apple MPS/CPU) 100% unit-testable on M3 Pro.
    """
    dtype_key = device_type.strip().lower()
    if dtype_key == "cuda":
        if vram_gb >= 75.0:
            tier = "h100_or_a100_80g"
            base_micro = 131_072
            base_accum = 1
        elif vram_gb >= 38.0:
            tier = "a100_40g"
            base_micro = 65_536
            base_accum = 2
        elif vram_gb >= 20.0:
            tier = "l4_or_4090"
            base_micro = 32_768
            base_accum = 4
        else:
            tier = "cuda_compact"
            base_micro = 16_384
            base_accum = 8

        target = DEFAULT_PROD_STEP_BYTES if target_tokens_bytes_per_step is None else int(target_tokens_bytes_per_step)
        if target <= 0:
            raise ValueError(f"target_tokens_bytes_per_step must be > 0, got {target}")

        if target == DEFAULT_PROD_STEP_BYTES:
            micro = base_micro
            accum = base_accum
        else:
            micro = min(base_micro, target)
            accum = max(1, math.ceil(target / micro))

        use_bf16 = True if bf16_supported is None else bool(bf16_supported)
        return HardwareProfile(
            device_type="cuda",
            device_name=device_name or "NVIDIA CUDA GPU",
            vram_gb=round(float(vram_gb), 2),
            tier=tier,
            micro_batch_bytes=micro,
            grad_accum_steps=accum,
            target_tokens_bytes_per_step=target,
            use_bf16=use_bf16,
        )

    # Local Apple MPS or CPU debug tier
    target = DEFAULT_LOCAL_STEP_BYTES if target_tokens_bytes_per_step is None else int(target_tokens_bytes_per_step)
    if target <= 0:
        raise ValueError(f"target_tokens_bytes_per_step must be > 0, got {target}")

    micro = min(target, DEFAULT_LOCAL_STEP_BYTES)
    accum = max(1, math.ceil(target / micro))
    tier = "apple_mps" if dtype_key == "mps" else "cpu_local"
    default_name = (
        f"Apple Silicon MPS ({platform.machine()})"
        if dtype_key == "mps"
        else f"CPU ({platform.processor() or platform.machine()})"
    )
    return HardwareProfile(
        device_type=dtype_key,
        device_name=device_name or default_name,
        vram_gb=round(float(vram_gb), 2),
        tier=tier,
        micro_batch_bytes=micro,
        grad_accum_steps=accum,
        target_tokens_bytes_per_step=target,
        use_bf16=False,
    )


def detect_hardware_profile(
    preferred: str = "auto",
    deterministic: bool = False,
    target_tokens_bytes_per_step: Optional[int] = None,
) -> HardwareProfile:
    """Inspect active hardware and return the matched `HardwareProfile`."""
    device = get_device(preferred=preferred, deterministic=deterministic)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram_gb = float(props.total_memory) / float(1024**3)
        name = str(props.name)
        # Unified-memory Grace Blackwell (e.g. DGX Spark GB10) fallback if driver reports 0 VRAM
        if vram_gb < 1.0 and hasattr(os, "sysconf"):
            try:
                vram_gb = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / float(1024**3)
            except Exception:
                pass
        bf16_ok = bool(torch.cuda.is_bf16_supported())
        return select_hardware_tier(
            device_type="cuda",
            vram_gb=vram_gb,
            device_name=name,
            target_tokens_bytes_per_step=target_tokens_bytes_per_step,
            bf16_supported=bf16_ok,
        )
    return select_hardware_tier(
        device_type=device.type,
        vram_gb=0.0,
        target_tokens_bytes_per_step=target_tokens_bytes_per_step,
        bf16_supported=False,
    )


def resolve_micro_batch_plan(
    profile: HardwareProfile,
    seq_len: int,
    target_tokens_bytes_per_step: Optional[int] = None,
    micro_batch_bytes_override: Optional[int] = None,
) -> tuple[int, int, int]:
    """Convert byte-budget contract into `(micro_batch_size, grad_accum_steps, actual_bytes_per_step)`.

    Ensures `micro_batch_size >= 1` and `grad_accum_steps >= 1` for arbitrary sequence lengths.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    target = (
        profile.target_tokens_bytes_per_step
        if target_tokens_bytes_per_step is None
        else int(target_tokens_bytes_per_step)
    )
    micro_bytes_cap = (
        profile.micro_batch_bytes
        if micro_batch_bytes_override is None
        else int(micro_batch_bytes_override)
    )
    micro_bytes_cap = max(seq_len, min(micro_bytes_cap, max(target, seq_len)))
    micro_batch_size = max(1, micro_bytes_cap // seq_len)
    bytes_per_micro = micro_batch_size * seq_len
    grad_accum_steps = max(1, math.ceil(target / bytes_per_micro))
    actual_bytes = bytes_per_micro * grad_accum_steps
    return micro_batch_size, grad_accum_steps, actual_bytes


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and optional CUDA/MPS RNG states cleanly."""
    state: dict[str, Any] = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state().clone(),
        "cuda_rng_state_all": None,
        "mps_rng_state": None,
    }
    if is_cuda_available():
        state["cuda_rng_state_all"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    if is_mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            state["mps_rng_state"] = torch.mps.get_rng_state().clone()
        except Exception:
            state["mps_rng_state"] = None
    return state


def restore_rng_state(rng_state: Optional[dict[str, Any]]) -> None:
    """Restore RNG states across heterogeneous hardware (Mac <-> Colab) without raising errors."""
    if not rng_state:
        return

    py_state = rng_state.get("python_rng_state")
    if py_state is not None:
        random.setstate(py_state)

    np_state = rng_state.get("numpy_rng_state")
    if np_state is not None:
        np.random.set_state(np_state)

    cpu_state = rng_state.get("torch_cpu_rng_state")
    if cpu_state is not None:
        if isinstance(cpu_state, torch.Tensor):
            torch.random.set_rng_state(cpu_state.to(device="cpu", dtype=torch.uint8))

    cuda_states = rng_state.get("cuda_rng_state_all")
    if cuda_states is not None and is_cuda_available():
        try:
            dev_count = torch.cuda.device_count()
            normalized = [s.to(device="cpu", dtype=torch.uint8) for s in cuda_states]
            if len(normalized) == dev_count:
                torch.cuda.set_rng_state_all(normalized)
            elif len(normalized) > 0:
                torch.cuda.set_rng_state(normalized[0], device=0)
        except Exception:
            pass

    mps_state = rng_state.get("mps_rng_state")
    if mps_state is not None and is_mps_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
            try:
                if isinstance(mps_state, torch.Tensor):
                    torch.mps.set_rng_state(mps_state.to(device="cpu", dtype=torch.uint8))
            except Exception:
                pass


def seed_everything(seed: int, device: Optional[torch.device] = None) -> random.Random:
    """Seed Python, NumPy, CPU Torch, and CUDA/MPS accelerators safely across platforms."""
    s = int(seed) & 0x7FFFFFFF
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if (device is None or device.type == "cuda") and is_cuda_available():
        torch.cuda.manual_seed_all(s)
    if (device is None or device.type == "mps") and is_mps_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            try:
                torch.mps.manual_seed(s)
            except Exception:
                pass
    return random.Random(s)


def describe_hardware_profile(profile: HardwareProfile) -> str:
    """Human-readable one-line hardware summary for Colab Cell 1 and training logs."""
    vram_str = f"{profile.vram_gb:.1f} GB VRAM" if profile.device_type == "cuda" else "Unified/Host RAM"
    amp_str = "bfloat16 autocast" if profile.use_bf16 else "float32"
    return (
        f"[Hardware] device={profile.device_type} ({profile.device_name}, {vram_str}) | "
        f"tier={profile.tier} | micro_batch_bytes={profile.micro_batch_bytes} | "
        f"grad_accum_steps={profile.grad_accum_steps} | "
        f"target_step_bytes={profile.target_tokens_bytes_per_step} | precision={amp_str}"
    )
