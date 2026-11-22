"""Developmental training data loaders & online synthetic generators."""

from dcpa.data.shard_loader import BinaryShardDataset
from dcpa.data.synthetic import (
    FastMappingEpisode,
    generate_dyck_sequence,
    generate_fast_mapping_episode,
    generate_grid_physics_episode,
    generate_pseudoword,
)

__all__ = [
    "BinaryShardDataset",
    "FastMappingEpisode",
    "generate_grid_physics_episode",
    "generate_dyck_sequence",
    "generate_pseudoword",
    "generate_fast_mapping_episode",
]
