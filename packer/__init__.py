"""
packer — Packer 2019 → C. elegans Wiring Atlas pipeline package.

Public API — import from here rather than from submodules directly.
"""

from .loader import PackerDataset, DatasetSummary
from .annotation import (
    NeuronMapper,
    MappingReport,
    SENSORY_PREFIXES,
    MOTOR_PREFIXES,
    INTERNEURON_PREFIXES,
)
from .expression import ExpressionMatrix, ExpressionMatrixResult, MarkerCheckResult
from .genes import GeneSelector, GeneSetResult, GuidanceGene, GeneOrigin
from .preprocess import GNNPreprocessor, PreprocessorOutput

__all__ = [
    "PackerDataset",
    "DatasetSummary",
    "NeuronMapper",
    "MappingReport",
    "SENSORY_PREFIXES",
    "MOTOR_PREFIXES",
    "INTERNEURON_PREFIXES",
    "ExpressionMatrix",
    "ExpressionMatrixResult",
    "MarkerCheckResult",
    "GeneSelector",
    "GeneSetResult",
    "GuidanceGene",
    "GeneOrigin",
    "GNNPreprocessor",
    "PreprocessorOutput",
]
