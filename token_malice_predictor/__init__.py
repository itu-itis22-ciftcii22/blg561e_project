"""Token Malice Predictor - Temporal GNN for malicious token detection."""

from .dataset import (
    TemporalGraphDataset,
    TemporalGraphData,
    create_loader,
)
from .preprocessor import (
    TokenPreprocessor,
    TokenDataFrame,
)

from .classifier import (
    TokenMaliceClassifier,
)
from .trainer import Trainer

from .metrics import compute_metrics

__all__ = [
    "TokenPreprocessor",
    "TokenDataFrame",
    "TemporalGraphDataset",
    "TemporalGraphData",
    "create_loader",
    "TokenMaliceClassifier",
    "Trainer",
    "compute_metrics",
]

__version__ = "0.1.0"
