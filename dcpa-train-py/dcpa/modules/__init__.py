"""Decoupled core modules of DCPA-Turbo and DCPA-Flat v3.3."""

from dcpa.modules.bcg_pc import BCGPC, BCGPCStepOutput, ClauseAccumulatorState, ConceptGraphState
from dcpa.modules.copy_ring import HolographicCopyRing
from dcpa.modules.cortical import (
    CorticalCarryState,
    CorticalDecoderLayer,
    CorticalEncoderLayer,
    CorticalSandwich,
    SharedAndRoutedMoE,
)
from dcpa.modules.cortical_moe import CorticalPredOut, CorticalStackFlat, DecoupledPluggableMoE
from dcpa.modules.graph_carrier import ContractiveGraphCarrier
from dcpa.modules.hippocampus import (
    ColumnMajorFastWeights,
    CompactHouseholderState,
    EligibilityState,
    HippocampusDualMemory,
    HippocampusReadOutput,
)
from dcpa.modules.hippocampus_flat import (
    DualMorphemeWhitenedDG,
    HippocampusFlat,
    HippocampusInterleavedOut,
    HippocampusWriteEvents,
)
from dcpa.modules.pmd import PMD, BrocaStepOutput, PMDDecodeOutput, PMDFlat, PMDFlatOutput
from dcpa.modules.sgsc import SGSC, SGSCFlat, SGSCOutput

__all__ = [
    "SGSC",
    "SGSCOutput",
    "SGSCFlat",
    "CorticalSandwich",
    "CorticalEncoderLayer",
    "CorticalDecoderLayer",
    "CorticalCarryState",
    "SharedAndRoutedMoE",
    "BCGPC",
    "BCGPCStepOutput",
    "ConceptGraphState",
    "ClauseAccumulatorState",
    "HippocampusDualMemory",
    "HippocampusReadOutput",
    "EligibilityState",
    "CompactHouseholderState",
    "ColumnMajorFastWeights",
    "PMD",
    "BrocaStepOutput",
    "PMDDecodeOutput",
    "PMDFlat",
    "PMDFlatOutput",
    "HolographicCopyRing",
    "ContractiveGraphCarrier",
    "DecoupledPluggableMoE",
    "CorticalStackFlat",
    "CorticalPredOut",
    "DualMorphemeWhitenedDG",
    "HippocampusFlat",
    "HippocampusInterleavedOut",
    "HippocampusWriteEvents",
]
