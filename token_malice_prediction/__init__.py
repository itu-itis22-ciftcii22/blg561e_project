"""
Token Malice Prediction Package

GNN-based malicious token detection using transaction graph analysis.

This package contains modules for:
- Data preprocessing and graph construction
- Temporal GNN model for graph classification
- Training utilities and evaluation
"""

__version__ = "0.1.0"
__author__ = "BLG561E Project Team"

from .data import (
    TokenPreprocessor,
    TemporalGraphBuilder,
    TemporalGraphData,
    TokenGraphDataset,
    TokenGraphDatasetList,
    create_data_loaders,
    build_temporal_graphs_from_processed_data,
    save_dataset,
    load_dataset
)
from .models import HeteroGINEEvolveGCN, create_temporal_model
from .training import TemporalTrainer, compute_metrics, print_model_summary
from .utils import Config, load_config, save_config, setup_logging, set_seed, get_device
