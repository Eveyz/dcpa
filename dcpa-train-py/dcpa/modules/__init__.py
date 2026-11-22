"""Four decoupled core modules of DCPA: SGSC, BCG-PC, Hippocampus, and PMD."""

from dcpa.modules.bcg_pc import BCGPC, BCGPCStepOutput, ConceptGraphState
from dcpa.modules.hippocampus import (
    ColumnMajorFastWeights,
    CompactHouseholderState,
    EligibilityState,
    HippocampusDualMemory,
    HippocampusReadOutput,
)
from dcpa.modules.pmd import PMD, BrocaStepOutput, PMDDecodeOutput
from dcpa.modules.sgsc import SGSC, SGSCOutput

__all__ = [
    "SGSC",
    "SGSCOutput",
    "BCGPC",
    "BCGPCStepOutput",
    "ConceptGraphState",
    "HippocampusDualMemory",
    "HippocampusReadOutput",
    "EligibilityState",
    "CompactHouseholderState",
    "ColumnMajorFastWeights",
    "PMD",
    "BrocaStepOutput",
    "PMDDecodeOutput",
]
