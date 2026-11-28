"""Low-level parallel, ragged, static-grid, RoPE, and normalization operators for DCPA."""

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
from dcpa.ops.static_grid import (
    StaticSegmentMap,
    build_static_segment_map,
    compute_intra_seg_pos,
    compute_static_seg_depths_and_triggers,
    gather_seg_to_byte,
    static_lif_scan,
    static_segment_mean,
    static_segment_scatter_add,
    static_segment_softmax,
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
    "StaticSegmentMap",
    "build_static_segment_map",
    "compute_intra_seg_pos",
    "compute_static_seg_depths_and_triggers",
    "gather_seg_to_byte",
    "static_lif_scan",
    "static_segment_mean",
    "static_segment_scatter_add",
    "static_segment_softmax",
]
