"""
Model Architectures Module

Contains implementations of:
- Temporal GNN models with GINEConv + EvolveGCN-O + Global Attention
- Graph classification architectures
"""

from .temporal_gnn import HeteroGINEEvolveGCN, create_temporal_model

__all__ = [
    'HeteroGINEEvolveGCN',
    'create_temporal_model'
]
