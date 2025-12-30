"""
Hetero-GINE-EvolveGCN Model Architecture

Implements the proposed architecture for token malice detection:
1. GINEConv for spatial aggregation (captures transaction graph structure)
2. EvolveGCN-O for temporal evolution (captures network dynamics over time)
3. Global Attention for hierarchical readout (graph-level representation)

Based on the design document: "Architectural Frameworks for Malicious Token 
Detection via Temporal Graph Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    GlobalAttention,
    BatchNorm
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax


class GINEConvBlock(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINEConv) block.
    
    GINEConv is chosen because:
    - Maximally expressive among 1-WL GNNs
    - Natively handles edge features (important for transaction attributes)
    - Better at distinguishing graph structures than GCN/GAT
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.3,
        eps: float = 0.0,
        train_eps: bool = True
    ):
        super().__init__()
        
        # MLP for node update
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature projection
        self.edge_encoder = nn.Linear(edge_dim, in_channels)
        
        self.conv = GINEConv(
            nn=self.mlp,
            eps=eps,
            train_eps=train_eps,
            edge_dim=in_channels
        )
        
        self.bn = BatchNorm(out_channels)
        self.dropout = dropout
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_dim)
            
        Returns:
            Updated node features (N, out_channels)
        """
        # Project edge features to match node dimension
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class SpatialEncoder(nn.Module):
    """
    Spatial encoder using stacked GINEConv layers.
    
    Processes a single graph snapshot to extract node embeddings
    that capture the local transaction graph structure.
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GINEConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINEConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_input_dim,
                dropout=dropout
            ))
        
        # Residual weights
        self.residual_weight = nn.Parameter(torch.ones(num_layers) * 0.5)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through spatial encoder.
        
        Args:
            x: Node features (N, node_input_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, edge_input_dim)
            
        Returns:
            Node embeddings (N, hidden_dim)
        """
        x = self.input_proj(x)
        
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr)
            # Weighted residual connection
            alpha = torch.sigmoid(self.residual_weight[i])
            x = alpha * x_new + (1 - alpha) * x
        
        return x


class EvolveGCNO(nn.Module):
    """
    EvolveGCN-O: Temporal evolution using GRU to evolve GCN weights.
    
    The key insight is that the GCN weight matrix itself evolves over time,
    capturing how the importance of different features changes.
    
    Unlike TGN which requires node memory, EvolveGCN-O is more suitable
    for our graph-level classification task.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU to evolve the weight matrix
        # Input: flattened weight matrix from previous step
        # Output: updated weight matrix
        self.weight_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Weight matrix initialization
        self.init_weight = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim) * 0.01
        )
        
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_embeddings_seq: List[torch.Tensor],
        edge_indices_seq: List[torch.Tensor],
        num_nodes: int
    ) -> torch.Tensor:
        """
        Process sequence of node embeddings through temporal GRU.
        
        Args:
            node_embeddings_seq: List of (N, hidden_dim) tensors per snapshot
            edge_indices_seq: List of edge indices per snapshot
            num_nodes: Number of nodes in the graph
            
        Returns:
            Final node embeddings (N, hidden_dim)
        """
        num_snapshots = len(node_embeddings_seq)
        device = node_embeddings_seq[0].device
        
        # Initialize weight matrix
        W = self.init_weight.unsqueeze(0)  # (1, hidden_dim, hidden_dim)
        
        # Process each snapshot
        evolved_embeddings = []
        
        for t in range(num_snapshots):
            x_t = node_embeddings_seq[t]  # (N, hidden_dim)
            
            # Apply current weight matrix (graph convolution-like)
            W_t = W.squeeze(0)  # (hidden_dim, hidden_dim)
            x_transformed = torch.mm(x_t, W_t)  # (N, hidden_dim)
            
            # Apply simple aggregation using edge structure
            edge_index = edge_indices_seq[t]
            if edge_index.size(1) > 0:
                # Message passing aggregation
                row, col = edge_index
                x_agg = torch.zeros_like(x_transformed)
                x_agg.index_add_(0, row, x_transformed[col])
                # Normalize by degree
                deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1)
                x_agg = x_agg / deg.unsqueeze(-1)
                x_transformed = x_transformed + x_agg
            
            x_transformed = self.bn(x_transformed)
            x_transformed = F.relu(x_transformed)
            x_transformed = self.dropout(x_transformed)
            
            evolved_embeddings.append(x_transformed)
            
            # Evolve weight matrix using GRU
            # Use mean node embedding as context for weight evolution
            context = x_transformed.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            context = context.unsqueeze(0)  # (1, 1, hidden_dim)
            
            # Update each row of weight matrix using GRU
            W_input = W.squeeze(0).unsqueeze(0)  # (1, hidden_dim, hidden_dim)
            W_input = W_input.permute(0, 2, 1)  # (1, hidden_dim, hidden_dim)
            
            # Process each column of W through GRU conditioned on context
            W_evolved, _ = self.weight_gru(
                W_input.reshape(1, self.hidden_dim, self.hidden_dim),
                context.expand(self.num_layers, -1, -1)
            )
            W = W_evolved.reshape(1, self.hidden_dim, self.hidden_dim)
        
        # Return final evolved embeddings
        return evolved_embeddings[-1]


class TemporalEncoder(nn.Module):
    """
    Simplified temporal encoder using LSTM over graph-level representations.
    
    This is a more practical alternative to EvolveGCN-O for our case,
    processing the sequence of graph representations directly.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        snapshot_embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process sequence of snapshot embeddings.
        
        Args:
            snapshot_embeddings: (batch, num_snapshots, hidden_dim)
            lengths: Optional sequence lengths for padding
            
        Returns:
            Temporal embedding (batch, hidden_dim)
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(snapshot_embeddings)
        
        # Take the last output
        if self.bidirectional:
            # Concatenate forward and backward final states
            temporal_repr = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            temporal_repr = h_n[-1]
        
        return self.output_proj(temporal_repr)


class GlobalAttentionReadout(nn.Module):
    """
    Global attention pooling for graph-level readout.
    
    Learns to weight different nodes based on their importance
    for the classification task, providing interpretable attention weights.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention gates
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads)
        )
        
        # Value projection
        self.nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Alternative: use PyG's GlobalAttention
        self.global_attn = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        )
        
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Global attention pooling.
        
        Args:
            x: Node features (N, hidden_dim)
            batch: Batch assignment (N,)
            
        Returns:
            Tuple of (graph_embedding, attention_weights)
        """
        # Use PyG's GlobalAttention
        graph_embedding = self.global_attn(x, batch)
        
        # Compute attention weights for interpretability
        gate = self.gate_nn(x)  # (N, num_heads)
        attention = softmax(gate.mean(dim=-1), batch)  # (N,)
        
        return graph_embedding, attention


class HeteroGINEEvolveGCN(nn.Module):
    """
    Main model: Hetero-GINE-EvolveGCN for token malice detection.
    
    Architecture:
    1. Spatial Encoder (GINEConv): Processes each snapshot
    2. Temporal Encoder (LSTM/EvolveGCN-O): Captures evolution
    3. Global Attention Readout: Graph-level classification
    
    This model handles temporal graph sequences and produces
    a binary classification for malicious token detection.
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_gine_layers: int = 3,
        num_temporal_layers: int = 2,
        num_snapshots: int = 6,
        dropout: float = 0.3,
        use_evolve_gcn: bool = False,
        pooling: str = 'attention'
    ):
        """
        Initialize the model.
        
        Args:
            node_input_dim: Input node feature dimension
            edge_input_dim: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_gine_layers: Number of GINEConv layers
            num_temporal_layers: Number of LSTM/GRU layers
            num_snapshots: Number of temporal snapshots
            dropout: Dropout rate
            use_evolve_gcn: Use EvolveGCN-O instead of LSTM
            pooling: Pooling method ('attention', 'mean', 'max', 'concat')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_snapshots = num_snapshots
        self.use_evolve_gcn = use_evolve_gcn
        self.pooling = pooling
        
        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gine_layers,
            dropout=dropout
        )
        
        # Temporal encoder
        if use_evolve_gcn:
            self.temporal_encoder = EvolveGCNO(
                hidden_dim=hidden_dim,
                num_layers=num_temporal_layers,
                dropout=dropout
            )
        else:
            self.temporal_encoder = TemporalEncoder(
                hidden_dim=hidden_dim,
                num_layers=num_temporal_layers,
                dropout=dropout,
                bidirectional=True
            )
        
        # Graph-level readout
        if pooling == 'attention':
            self.readout = GlobalAttentionReadout(
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        # Snapshot pooling
        self.snapshot_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        if pooling == 'concat':
            classifier_input_dim = hidden_dim * 2
        else:
            classifier_input_dim = hidden_dim
        
        # Add temporal embedding dimension
        classifier_input_dim += hidden_dim
        
        # Use LayerNorm instead of BatchNorm to support batch size 1
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def _pool_snapshot(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool node features to graph-level representation."""
        if self.pooling == 'attention':
            pooled, _ = self.readout(x, batch)
            return pooled
        elif self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        elif self.pooling == 'concat':
            return torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=-1)
        elif self.pooling == 'add':
            return global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward_temporal(
        self,
        snapshots: List[Data],
        num_nodes: int
    ) -> torch.Tensor:
        """
        Forward pass for a single temporal graph.
        
        Args:
            snapshots: List of Data objects (one per snapshot)
            num_nodes: Total number of nodes
            
        Returns:
            Logits (1, num_classes)
        """
        device = snapshots[0].x.device
        
        # Process each snapshot through spatial encoder
        snapshot_embeddings = []
        node_embeddings_seq = []
        edge_indices_seq = []
        
        for snapshot in snapshots:
            # Spatial encoding
            node_emb = self.spatial_encoder(
                x=snapshot.x,
                edge_index=snapshot.edge_index,
                edge_attr=snapshot.edge_attr
            )
            node_embeddings_seq.append(node_emb)
            edge_indices_seq.append(snapshot.edge_index)
            
            # Pool to graph-level
            batch = torch.zeros(snapshot.num_nodes, dtype=torch.long, device=device)
            graph_emb = self._pool_snapshot(node_emb, batch)
            graph_emb = self.snapshot_pool(graph_emb)
            snapshot_embeddings.append(graph_emb)
        
        # Stack snapshot embeddings: (num_snapshots, hidden_dim)
        snapshot_embeddings = torch.stack(snapshot_embeddings, dim=0)
        snapshot_embeddings = snapshot_embeddings.unsqueeze(0)  # (1, num_snapshots, hidden_dim)
        
        # Temporal encoding
        if self.use_evolve_gcn:
            final_node_emb = self.temporal_encoder(
                node_embeddings_seq, edge_indices_seq, num_nodes
            )
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
            temporal_emb = self._pool_snapshot(final_node_emb, batch)
        else:
            temporal_emb = self.temporal_encoder(snapshot_embeddings)
        
        # Final graph embedding: combine last snapshot + temporal
        final_snapshot_emb = snapshot_embeddings[:, -1, :]  # (1, hidden_dim)
        
        # Normalize dimensions - ensure both are exactly 2D: (1, hidden_dim)
        while temporal_emb.dim() > 2:
            temporal_emb = temporal_emb.squeeze(0)
        while temporal_emb.dim() < 2:
            temporal_emb = temporal_emb.unsqueeze(0)
            
        while final_snapshot_emb.dim() > 2:
            final_snapshot_emb = final_snapshot_emb.squeeze(0)
        while final_snapshot_emb.dim() < 2:
            final_snapshot_emb = final_snapshot_emb.unsqueeze(0)
        
        if self.pooling == 'concat':
            # Already concatenated in _pool_snapshot
            combined = torch.cat([
                final_snapshot_emb.view(1, -1)[:, :self.hidden_dim],
                temporal_emb
            ], dim=-1)
        else:
            combined = torch.cat([final_snapshot_emb, temporal_emb], dim=-1)
        
        # Classify
        return self.classifier(combined)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        snapshots: Optional[List[List[Data]]] = None,
        num_nodes_list: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Forward pass supporting both static and temporal graphs.
        
        For static graphs (single snapshot):
            Uses only spatial encoder
            
        For temporal graphs:
            Processes snapshot sequence
            
        Args:
            x: Node features (for static graph)
            edge_index: Edge indices (for static graph)
            edge_attr: Edge features
            batch: Batch assignment
            snapshots: List of snapshot lists (for temporal)
            num_nodes_list: Number of nodes per graph (for temporal)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Handle temporal graphs
        if snapshots is not None:
            batch_logits = []
            for i, (graph_snapshots, num_nodes) in enumerate(zip(snapshots, num_nodes_list)):
                logits = self.forward_temporal(graph_snapshots, num_nodes)
                batch_logits.append(logits)
            return torch.cat(batch_logits, dim=0)
        
        # Handle static graph (single snapshot)
        x = self.spatial_encoder(x, edge_index, edge_attr)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Pool to graph level
        graph_emb = self._pool_snapshot(x, batch)
        
        # For static, duplicate temporal embedding as itself
        combined = torch.cat([graph_emb, graph_emb], dim=-1)
        if self.pooling == 'concat':
            combined = combined[:, :self.hidden_dim * 2]
            combined = torch.cat([combined, graph_emb[:, :self.hidden_dim]], dim=-1)
        
        return self.classifier(combined)
    
    def get_attention_weights(
        self,
        snapshots: List[Data],
        num_nodes: int
    ) -> List[torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Args:
            snapshots: List of snapshot Data objects
            num_nodes: Number of nodes
            
        Returns:
            List of attention weight tensors per snapshot
        """
        self.eval()
        attention_weights = []
        
        with torch.no_grad():
            for snapshot in snapshots:
                device = snapshot.x.device
                node_emb = self.spatial_encoder(
                    x=snapshot.x,
                    edge_index=snapshot.edge_index,
                    edge_attr=snapshot.edge_attr
                )
                batch = torch.zeros(snapshot.num_nodes, dtype=torch.long, device=device)
                _, attn = self.readout(node_emb, batch)
                attention_weights.append(attn)
        
        return attention_weights


def create_temporal_model(
    node_dim: int,
    edge_dim: int,
    hidden_dim: int = 128,
    num_classes: int = 2,
    num_gine_layers: int = 3,
    num_temporal_layers: int = 2,
    num_snapshots: int = 6,
    dropout: float = 0.3,
    use_evolve_gcn: bool = False,
    pooling: str = 'attention'
) -> HeteroGINEEvolveGCN:
    """
    Factory function to create the temporal model.
    
    Args:
        node_dim: Node feature dimension
        edge_dim: Edge feature dimension
        hidden_dim: Hidden dimension
        num_classes: Number of output classes
        num_gine_layers: Number of GINEConv layers
        num_temporal_layers: Number of temporal layers
        num_snapshots: Number of temporal snapshots
        dropout: Dropout rate
        use_evolve_gcn: Use EvolveGCN-O vs LSTM
        pooling: Pooling strategy
        
    Returns:
        HeteroGINEEvolveGCN model
    """
    return HeteroGINEEvolveGCN(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_gine_layers=num_gine_layers,
        num_temporal_layers=num_temporal_layers,
        num_snapshots=num_snapshots,
        dropout=dropout,
        use_evolve_gcn=use_evolve_gcn,
        pooling=pooling
    )
