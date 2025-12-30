"""
Training Module

Contains training utilities:
- TemporalTrainer class for temporal GNN models with enhanced logging
- Metric computation utilities
"""

from .temporal_trainer import (
    TemporalTrainer,
    compute_metrics,
    print_model_summary
)

__all__ = [
    'TemporalTrainer',
    'compute_metrics',
    'print_model_summary'
]
