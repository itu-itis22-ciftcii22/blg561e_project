"""Token Malice Classifier - Composite temporal graph model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch_geometric_temporal.nn.recurrent import EvolveGCNO

from token_malice_predictor.dataset import TemporalGraphData

class ClassifierError(Exception):
    """Raised when classifier operations fail."""
    pass

class TokenMaliceClassifier(nn.Module):
    """Composite model for token malice classification."""
    
    # Node feature dim: 4 windowed + 4 cumulative = 8
    NODE_FEATURE_DIM = 8
    
    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 1,
        num_heads: int = 2,
        dropout: float = 0.2,
        pooling: str = 'concat',
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        
        # Node Encoder Layers
        self.conv = TransformerConv(
            in_channels=self.NODE_FEATURE_DIM,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        
        # With concat=True, output dim is hidden_dim * num_heads
        self.conv_out_dim = hidden_dim * num_heads
        
        self.node_norm = nn.LayerNorm(self.conv_out_dim)
        self.node_dropout = nn.Dropout(dropout)
        
        self.evolver = EvolveGCNO(
            in_channels=self.conv_out_dim,
        )
        
        self.evolve_norm = nn.LayerNorm(self.conv_out_dim)
        self.evolve_dropout = nn.Dropout(dropout)
        
        if pooling == 'concat':
            classifier_input_dim = self.conv_out_dim * 2
        else:
            classifier_input_dim = self.conv_out_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def _pool_nodes(self, x: Tensor, batch: Tensor):
        """Pool node features to graph-level representation."""
        if self.pooling == 'mean':
            pooled: Tensor = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            pooled: Tensor = global_max_pool(x, batch)
        elif self.pooling == 'concat':
            pooled: Tensor = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ], dim=-1)
        else:
            raise ClassifierError(f"Unknown pooling: {self.pooling}")
        return pooled
    
    def _encode_nodes(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        """Encode nodes using GNN with edge features."""
        if edge_index.size(1) == 0:
            h: Tensor = torch.zeros(
                x.size(0), self.conv_out_dim,
                device=x.device, dtype=x.dtype,
            )
            return h
        
        h: Tensor = self.conv(x, edge_index, edge_attr)
        h = self.node_norm(h)
        h = F.relu(h)
        h = self.node_dropout(h)
        return h

    def forward_temporal(self, temporal_data: TemporalGraphData):
        """Forward pass using precomputed features from dataset."""
        snapshots = temporal_data.snapshots
        num_nodes = temporal_data.num_nodes
        
        if not snapshots:
            raise ClassifierError(
                f"No snapshots in temporal data: {temporal_data.token_name}"
            )
        
        device = temporal_data.device
        
        final_embeddings: Tensor | None = None
        
        # Reset evolver state for independent sequence processing
        # This prevents "Trying to backward through the graph a second time"
        # by forcing EvolveGCNO to restart from initial_weight (new graph).
        self.evolver.weight = None

        for snap_idx, snapshot in enumerate(snapshots):
            edge_index = snapshot.edge_index
            edge_attr = snapshot.edge_attr
            
            assert edge_index is not None
            assert edge_attr is not None
            
            # Use precomputed features (already log1p transformed in dataset)
            x = torch.cat([snapshot.x_cumulative, snapshot.x_windowed], dim=1)
            
            # NaN check: input features
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ClassifierError(
                    f"NaN/Inf in input features! Token: {temporal_data.token_name}, "
                    f"snapshot: {snap_idx}, x stats: min={x.min()}, max={x.max()}"
                )
            
            # Encode nodes
            x_emb = self._encode_nodes(x, edge_index, edge_attr)
            
            # NaN check: after encoding
            if torch.isnan(x_emb).any() or torch.isinf(x_emb).any():
                raise ClassifierError(
                    f"NaN/Inf after node encoding! Token: {temporal_data.token_name}, "
                    f"snapshot: {snap_idx}, x_emb stats: min={x_emb.min()}, max={x_emb.max()}"
                )
            
            # EvolveGCN-O step
            # Note: library EvolveGCNO processes one step and updates internal weights
            x_evolved = self.evolver(
                x_emb,
                snapshot.simple_edge_index,
                snapshot.simple_edge_weight,
            )
            
            # NaN check: after EvolveGCN
            if torch.isnan(x_evolved).any() or torch.isinf(x_evolved).any():
                raise ClassifierError(
                    f"NaN/Inf after EvolveGCN! Token: {temporal_data.token_name}, "
                    f"snapshot: {snap_idx}, x_evolved stats: min={x_evolved.min()}, max={x_evolved.max()}"
                )
            
            # Apply same post-processing as custom layer had
            x_evolved = self.evolve_norm(x_evolved)
            x_evolved = F.relu(x_evolved)
            x_evolved = self.evolve_dropout(x_evolved)
            
            final_embeddings = x_evolved
        
        assert final_embeddings is not None
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        graph_emb = self._pool_nodes(final_embeddings, batch)
        
        logits: Tensor = self.classifier(graph_emb)
        return logits
    
    def forward(self, batch: list[TemporalGraphData]):
        """Forward pass for a batch of temporal graphs."""
        if not batch:
            raise ClassifierError("Empty batch")
        
        batch_logits = []
        for temporal_data in batch:
            logits = self.forward_temporal(temporal_data)
            batch_logits.append(logits)
        
        result: Tensor = torch.cat(batch_logits, dim=0)
        return result
    
    def get_num_parameters(self):
        """Get parameter counts per component."""
        counts: dict[str, int] = {}
        
        # Calculate params for node encoder parts explicitly
        encoder_params = 0
        encoder_params += sum(p.numel() for p in self.conv.parameters())
        encoder_params += sum(p.numel() for p in self.node_norm.parameters())
        
        counts['encoder'] = encoder_params
        counts['evolver'] = sum(p.numel() for p in self.evolver.parameters())
        counts['classifier'] = sum(p.numel() for p in self.classifier.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
