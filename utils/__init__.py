"""Cross-platform utilities for DCPA local (Apple Silicon) and cloud (Colab H100/A100) execution."""

from utils.device import (
    HardwareProfile,
    amp_autocast,
    capture_rng_state,
    configure_mps_fallback,
    describe_hardware_profile,
    detect_hardware_profile,
    get_deterministic_test_device,
    get_device,
    is_bf16_enabled,
    resolve_micro_batch_plan,
    restore_rng_state,
    select_hardware_tier,
    synchronize,
)

__all__ = [
    "HardwareProfile",
    "amp_autocast",
    "capture_rng_state",
    "configure_mps_fallback",
    "describe_hardware_profile",
    "detect_hardware_profile",
    "get_deterministic_test_device",
    "get_device",
    "is_bf16_enabled",
    "resolve_micro_batch_plan",
    "restore_rng_state",
    "select_hardware_tier",
    "synchronize",
]
