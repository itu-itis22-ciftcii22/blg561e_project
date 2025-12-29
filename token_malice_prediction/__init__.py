"""
Token Malice Prediction Package

GNN-based malicious token detection using transaction graph analysis.

This package contains modules for:
- Data preprocessing and graph construction
- GAT-based GNN model for graph classification
- Training utilities and evaluation
"""

__version__ = "0.1.0"
__author__ = "BLG561E Project Team"

from .data import TokenPreprocessor, TransactionGraphBuilder, TokenGraphDataset
from .models import TokenGATClassifier, create_model
from .training import Trainer, compute_metrics
from .utils import load_config, setup_logging
