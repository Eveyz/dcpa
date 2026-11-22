"""Cross-platform float32 precision island helper delegating exclusively to `utils/device.py`.

Preserves Iron Law #1 (single source of truth for `torch.amp.autocast` in `utils/device.py`)
while enforcing Training Red Line #1 (explicit pure float32 precision islands for sensitive ops).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, ContextManager

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path and (_REPO_ROOT / "utils" / "device.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from utils.device import fp32_island as _device_fp32_island
except ImportError:
    _device_fp32_island = None


def fp32_island(device: torch.device) -> ContextManager[Any]:
    """Force pure float32 execution inside an outer CUDA bfloat16 autocast block."""
    if _device_fp32_island is not None:
        return _device_fp32_island(device)
    return contextlib.nullcontext()
