"""
GNN-based Models

Implements Graph Neural Network architectures for token malice classification.
Uses GAT2Conv (GATv2) layers from PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    BatchNorm
)
from torch_geometric.data import Data, Batch


class TokenGATClassifier(nn.Module):
    """
    GAT-based classifier for token malice prediction.
    
    Uses GATv2Conv (improved Graph Attention Networks) for message passing.
    Supports edge features through edge_attr.
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        pooling: str = 'mean',
        use_edge_features: bool = True
    ):
        """
        Initialize the GAT classifier.
        
        Args:
            node_input_dim: Node feature dimension
            edge_input_dim: Edge feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (2 for binary)
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            pooling: Graph pooling method ('mean', 'max', 'add', 'concat')
            use_edge_features: Whether to use edge features in attention
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.pooling_type = pooling
        
        # Input projection
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        
        # Edge feature projection (if using edge features)
        if use_edge_features:
            self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            # GATv2Conv with edge_dim support
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim if use_edge_features else None,
                concat=True,  # Concatenate head outputs
                add_self_loops=True,
                share_weights=False
            )
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden_dim))
        
        # Pooling
        if pooling == 'concat':
            # Concatenate mean and max pooling
            classifier_input_dim = hidden_dim * 2
        else:
            classifier_input_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def _get_pooled_representation(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Apply graph-level pooling."""
        if self.pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling_type == 'max':
            return global_max_pool(x, batch)
        elif self.pooling_type == 'add':
            return global_add_pool(x, batch)
        elif self.pooling_type == 'concat':
            return torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=-1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, node_input_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_input_dim)
            batch: Batch assignment for each node
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Project edge features
        edge_feat = None
        if self.use_edge_features and edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
        
        # Apply GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_res = x  # Residual connection
            x = conv(x, edge_index, edge_attr=edge_feat)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection (skip first layer to match dimensions)
            if i > 0:
                x = x + x_res
        
        # Pool to graph-level representation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self._get_pooled_representation(x, batch)
        
        # Classify
        return self.classifier(x)
    
    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get node-level embeddings (before pooling).
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            
        Returns:
            Node embeddings
        """
        x = self.input_proj(x)
        x = F.relu(x)
        
        edge_feat = None
        if self.use_edge_features and edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_res = x
            x = conv(x, edge_index, edge_attr=edge_feat)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                if i > 0:
                    x = x + x_res
        
        return x
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights from a specific layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            layer_idx: Which layer to get attention from (-1 for last)
            
        Returns:
            Tuple of (edge_index, attention_weights)
        """
        x = self.input_proj(x)
        x = F.relu(x)
        
        edge_feat = None
        if self.use_edge_features and edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
        
        target_layer = layer_idx if layer_idx >= 0 else self.num_layers + layer_idx
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == target_layer:
                # Return attention weights from this layer
                x, (edge_index_out, attention) = conv(
                    x, edge_index, edge_attr=edge_feat, 
                    return_attention_weights=True
                )
                return edge_index_out, attention
            
            x = conv(x, edge_index, edge_attr=edge_feat)
            x = bn(x)
            x = F.relu(x)
        
        raise ValueError(f"Layer index {layer_idx} out of range")


class TokenGATClassifierSimple(nn.Module):
    """
    Simplified GAT classifier without edge features.
    
    Use this if edge feature processing causes memory issues
    or for simpler graph structures.
    """
    
    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        pooling: str = 'mean'
    ):
        """
        Initialize the simple GAT classifier.
        
        Args:
            node_input_dim: Node feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            pooling: Graph pooling method
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = node_input_dim if i == 0 else hidden_dim
            conv = GATv2Conv(
                in_channels=in_channels,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True,
                add_self_loops=True
            )
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden_dim))
        
        # Pooling
        self.pool = {
            'mean': global_mean_pool,
            'max': global_max_pool,
            'add': global_add_pool
        }.get(pooling, global_mean_pool)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self.pool(x, batch)
        return self.classifier(x)


def create_model(
    node_dim: int = 11,
    edge_dim: int = 10,
    hidden_dim: int = 128,
    num_classes: int = 2,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.3,
    pooling: str = 'concat',
    use_edge_features: bool = True
) -> nn.Module:
    """
    Factory function to create a token classification model.
    
    Args:
        node_dim: Node feature dimension
        edge_dim: Edge feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        pooling: Pooling method
        use_edge_features: Whether to use edge features
        
    Returns:
        Configured model
    """
    if use_edge_features:
        return TokenGATClassifier(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pooling=pooling,
            use_edge_features=True
        )
    else:
        return TokenGATClassifierSimple(
            node_input_dim=node_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pooling=pooling
        )
