"""
Model Architectures Module

Contains implementations of:
- GAT-based GNN models using GATv2Conv
- Graph classification architectures
"""

from .gnn import TokenGATClassifier, TokenGATClassifierSimple, create_model
