"""Low-level parallel, ragged, RoPE, and normalization operators for DCPA."""

from dcpa.ops.ragged import (
    build_cu_seqlens,
    ragged_local_offsets,
    ragged_segment_cumsum,
    ragged_segment_mean,
    ragged_segment_softmax,
    ragged_segment_sum,
)
from dcpa.ops.rmsnorm import RMSNorm
from dcpa.ops.rope import apply_holographic_rope, build_rope_freqs
from dcpa.ops.scan import (
    lif_membrane_scan,
    parallel_diag_scan,
    parallel_stack_scan,
    sequential_diag_scan_ref,
)

__all__ = [
    "RMSNorm",
    "build_rope_freqs",
    "apply_holographic_rope",
    "build_cu_seqlens",
    "ragged_local_offsets",
    "ragged_segment_softmax",
    "ragged_segment_sum",
    "ragged_segment_mean",
    "ragged_segment_cumsum",
    "parallel_diag_scan",
    "sequential_diag_scan_ref",
    "lif_membrane_scan",
    "parallel_stack_scan",
]
