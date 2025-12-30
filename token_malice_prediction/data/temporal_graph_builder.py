"""
Temporal Graph Construction Module

Builds temporal transaction graphs with time-based snapshots for EvolveGCN.
Creates sequences of graph snapshots that capture the evolution of the transaction network.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class TemporalGraphBuilder:
    """
    Build temporal transaction graphs with time-based snapshots.
    
    Creates a sequence of graph snapshots where:
    - Each snapshot covers a fixed time window
    - Nodes: Unique wallet addresses (consistent across snapshots)
    - Edges: Transaction relationships within each time window
    - Edge features: amount, value, time features, edge type
    
    This supports the EvolveGCN architecture which processes
    graph sequences to capture temporal dynamics.
    """
    
    # Edge types
    EDGE_TYPE_SENT = 0
    EDGE_TYPE_RECEIVED = 1
    EDGE_TYPE_MINT = 2
    
    # Time constants (in seconds)
    SIX_WEEKS_SECONDS = 45 * 24 * 60 * 60  # 1.5 months
    
    def __init__(
        self,
        num_snapshots: int = 6,
        overlap_ratio: float = 0.0
    ):
        """
        Initialize the temporal graph builder.
        
        Args:
            num_snapshots: Number of time snapshots to create
            overlap_ratio: Overlap between consecutive snapshots (0-0.5)
        """
        self.num_snapshots = num_snapshots
        self.overlap_ratio = overlap_ratio
        self.node_to_idx: Dict[str, int] = {}
        
    def _build_global_node_mapping(self, df: pd.DataFrame) -> int:
        """
        Build global node index mapping from all addresses.
        
        Args:
            df: Full transaction DataFrame
            
        Returns:
            Number of unique nodes
        """
        all_addresses = pd.concat([df['From'], df['To']]).unique()
        self.node_to_idx = {addr: idx for idx, addr in enumerate(all_addresses)}
        return len(all_addresses)
    
    def _get_time_windows(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[float, float]]:
        """
        Calculate time windows for snapshots.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            List of (start_time, end_time) tuples for each snapshot
        """
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        total_duration = end_time - start_time
        
        if total_duration <= 0:
            return [(start_time, end_time)] * self.num_snapshots
        
        # Calculate window size with optional overlap
        base_window = total_duration / self.num_snapshots
        step_size = base_window * (1 - self.overlap_ratio)
        window_size = base_window * (1 + self.overlap_ratio)
        
        windows = []
        for i in range(self.num_snapshots):
            window_start = start_time + i * step_size
            window_end = min(window_start + window_size, end_time)
            windows.append((window_start, window_end))
        
        return windows
    
    def _create_edge_features(
        self,
        row: pd.Series,
        edge_type: int,
        window_start: float,
        window_end: float
    ) -> List[float]:
        """
        Create edge feature vector for a transaction.
        
        Features:
        - amount (log-scaled)
        - value (log-scaled)
        - amount * value (log-scaled)
        - relative time within window (0-1)
        - global relative time (0-1)
        - time_sin (cyclical encoding)
        - time_cos (cyclical encoding)
        - edge_type (one-hot encoded, 3 values)
        
        Args:
            row: Transaction row
            edge_type: Type of edge
            window_start: Start of current window
            window_end: End of current window
            
        Returns:
            Feature vector
        """
        # Numeric features (log-scaled)
        # Use decimal-adjusted token_amount if available, otherwise use raw Amount
        token_amount = float(row.get('token_amount', row['Amount']))
        amount = np.log1p(token_amount)
        value = np.log1p(float(row['Value']))  # Value is now token price after preprocessing fix
        # transaction_value is the USD value of transaction (token_amount * price)
        transaction_value = float(row.get('transaction_value', token_amount * row['Value']))
        amount_x_value = np.log1p(transaction_value)
        
        # Time features
        timestamp = float(row['timestamp'])
        window_duration = max(window_end - window_start, 1)
        rel_time_window = (timestamp - window_start) / window_duration
        rel_time_global = float(row.get('rel_time_normalized', row.get('rel_time_6w', 0.5)))
        
        # Cyclical time encoding
        time_sin = np.sin(2 * np.pi * rel_time_global)
        time_cos = np.cos(2 * np.pi * rel_time_global)
        
        # Edge type one-hot
        edge_type_onehot = [0.0, 0.0, 0.0]
        edge_type_onehot[edge_type] = 1.0
        
        return [
            amount,
            value,
            amount_x_value,
            rel_time_window,
            rel_time_global,
            time_sin,
            time_cos,
        ] + edge_type_onehot
    
    def _build_snapshot_graph(
        self,
        df: pd.DataFrame,
        window_start: float,
        window_end: float,
        num_nodes: int,
        snapshot_idx: int
    ) -> Data:
        """
        Build a single snapshot graph from transactions in time window.
        
        Args:
            df: Transaction DataFrame
            window_start: Start timestamp of window
            window_end: End timestamp of window
            num_nodes: Total number of nodes (global)
            snapshot_idx: Index of this snapshot
            
        Returns:
            PyTorch Geometric Data object for this snapshot
        """
        # Filter transactions in this time window
        mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
        window_df = df[mask]
        
        edge_indices = []
        edge_attrs = []
        edge_types = []
        
        for _, row in window_df.iterrows():
            from_addr = row['From']
            to_addr = row['To']
            
            from_idx = self.node_to_idx[from_addr]
            to_idx = self.node_to_idx[to_addr]
            
            if from_addr == to_addr:
                # Mint transaction
                edge_indices.append((from_idx, to_idx))
                edge_attrs.append(self._create_edge_features(
                    row, self.EDGE_TYPE_MINT, window_start, window_end
                ))
                edge_types.append(self.EDGE_TYPE_MINT)
            else:
                # Bidirectional edges
                edge_indices.append((from_idx, to_idx))
                edge_attrs.append(self._create_edge_features(
                    row, self.EDGE_TYPE_SENT, window_start, window_end
                ))
                edge_types.append(self.EDGE_TYPE_SENT)
                
                edge_indices.append((to_idx, from_idx))
                edge_attrs.append(self._create_edge_features(
                    row, self.EDGE_TYPE_RECEIVED, window_start, window_end
                ))
                edge_types.append(self.EDGE_TYPE_RECEIVED)
        
        # Convert to tensors
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.get_edge_feature_dim()), dtype=torch.float)
            edge_type = torch.zeros((0,), dtype=torch.long)
        
        # Compute node features for this snapshot
        node_features = self._compute_snapshot_node_features(
            window_df, num_nodes, snapshot_idx
        )
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            num_nodes=num_nodes,
            snapshot_idx=snapshot_idx
        )
    
    def _compute_snapshot_node_features(
        self,
        df: pd.DataFrame,
        num_nodes: int,
        snapshot_idx: int
    ) -> torch.Tensor:
        """
        Compute node features for a snapshot.
        
        Features:
        - in_degree (within snapshot)
        - out_degree (within snapshot)
        - total_degree
        - total_amount_sent (log-scaled)
        - total_amount_received (log-scaled)
        - total_value_sent (log-scaled)
        - total_value_received (log-scaled)
        - is_active (1 if node has transactions in snapshot)
        - snapshot_position (normalized 0-1)
        
        Args:
            df: Transaction DataFrame for this snapshot
            num_nodes: Total number of nodes
            snapshot_idx: Index of this snapshot
            
        Returns:
            Node feature tensor
        """
        in_degree = np.zeros(num_nodes)
        out_degree = np.zeros(num_nodes)
        total_amount_sent = np.zeros(num_nodes)
        total_amount_received = np.zeros(num_nodes)
        total_value_sent = np.zeros(num_nodes)
        total_value_received = np.zeros(num_nodes)
        is_active = np.zeros(num_nodes)
        
        for _, row in df.iterrows():
            from_idx = self.node_to_idx[row['From']]
            to_idx = self.node_to_idx[row['To']]
            # Use decimal-adjusted token_amount if available
            amount = float(row.get('token_amount', row['Amount']))
            value = float(row.get('transaction_value', row['Value']))  # Use transaction USD value for aggregation
            
            is_active[from_idx] = 1
            is_active[to_idx] = 1
            
            if from_idx != to_idx:
                out_degree[from_idx] += 1
                in_degree[to_idx] += 1
                total_amount_sent[from_idx] += amount
                total_amount_received[to_idx] += amount
                total_value_sent[from_idx] += value
                total_value_received[to_idx] += value
            else:
                in_degree[from_idx] += 1
                out_degree[from_idx] += 1
                total_amount_received[from_idx] += amount
                total_value_received[from_idx] += value
        
        total_degree = in_degree + out_degree
        snapshot_position = np.full(num_nodes, snapshot_idx / max(self.num_snapshots - 1, 1))
        
        # Safely compute log features (clip to avoid -inf from log1p of negative values)
        def safe_log1p(x):
            return np.log1p(np.maximum(x, 0))
        
        node_features = np.column_stack([
            in_degree,
            out_degree,
            total_degree,
            safe_log1p(total_amount_sent),
            safe_log1p(total_amount_received),
            safe_log1p(total_value_sent),
            safe_log1p(total_value_received),
            is_active,
            snapshot_position
        ])
        
        return torch.tensor(node_features, dtype=torch.float)
    
    def build_temporal_graph(
        self,
        df: pd.DataFrame,
        label: int
    ) -> Dict[str, Any]:
        """
        Build a temporal graph with multiple snapshots.
        
        Args:
            df: Preprocessed transaction DataFrame
            label: Token label (0=benign, 1=malicious)
            
        Returns:
            Dictionary containing:
            - snapshots: List of Data objects for each time window
            - label: Graph label
            - num_nodes: Total number of nodes
            - num_snapshots: Number of snapshots
        """
        num_nodes = self._build_global_node_mapping(df)
        time_windows = self._get_time_windows(df)
        
        snapshots = []
        for i, (start, end) in enumerate(time_windows):
            snapshot = self._build_snapshot_graph(df, start, end, num_nodes, i)
            snapshots.append(snapshot)
        
        return {
            'snapshots': snapshots,
            'label': torch.tensor([label], dtype=torch.long),
            'num_nodes': num_nodes,
            'num_snapshots': self.num_snapshots
        }
    
    def build_static_graph(
        self,
        df: pd.DataFrame,
        label: int
    ) -> Data:
        """
        Build a static graph (single snapshot) for comparison.
        
        Args:
            df: Preprocessed transaction DataFrame
            label: Token label
            
        Returns:
            PyTorch Geometric Data object
        """
        num_nodes = self._build_global_node_mapping(df)
        
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        
        snapshot = self._build_snapshot_graph(df, start_time, end_time, num_nodes, 0)
        snapshot.y = torch.tensor([label], dtype=torch.long)
        
        return snapshot
    
    @staticmethod
    def get_edge_feature_dim() -> int:
        """Return the dimension of edge features."""
        return 10  # 7 numeric + 3 one-hot edge type
    
    @staticmethod
    def get_node_feature_dim() -> int:
        """Return the dimension of node features."""
        return 9  # 7 statistics + is_active + snapshot_position


class TemporalGraphData:
    """
    Container for temporal graph data with multiple snapshots.
    """
    
    def __init__(
        self,
        snapshots: List[Data],
        label: torch.Tensor,
        num_nodes: int,
        token_name: str = ""
    ):
        self.snapshots = snapshots
        self.label = label
        self.num_nodes = num_nodes
        self.token_name = token_name
        self.num_snapshots = len(snapshots)
    
    def to(self, device: str) -> 'TemporalGraphData':
        """Move all tensors to device."""
        return TemporalGraphData(
            snapshots=[s.to(device) for s in self.snapshots],
            label=self.label.to(device),
            num_nodes=self.num_nodes,
            token_name=self.token_name
        )
    
    @property
    def y(self) -> torch.Tensor:
        """Alias for label."""
        return self.label


def build_temporal_graphs_from_processed_data(
    processed_data: List[Tuple[pd.DataFrame, int, str]],
    num_snapshots: int = 6,
    overlap_ratio: float = 0.0
) -> List[Tuple[TemporalGraphData, str]]:
    """
    Build temporal graphs from all processed token data.
    
    Args:
        processed_data: List of (df, label, token_name) from preprocessor
        num_snapshots: Number of temporal snapshots
        overlap_ratio: Overlap between snapshots
        
    Returns:
        List of (TemporalGraphData, token_name) tuples
    """
    graphs = []
    builder = TemporalGraphBuilder(num_snapshots, overlap_ratio)
    
    for df, label, token_name in processed_data:
        try:
            temporal_data = builder.build_temporal_graph(df, label)
            graph = TemporalGraphData(
                snapshots=temporal_data['snapshots'],
                label=temporal_data['label'],
                num_nodes=temporal_data['num_nodes'],
                token_name=token_name
            )
            graphs.append((graph, token_name))
            logger.debug(f"Built temporal graph for {token_name}: "
                        f"{temporal_data['num_nodes']} nodes, "
                        f"{num_snapshots} snapshots")
        except Exception as e:
            logger.error(f"Error building temporal graph for {token_name}: {e}")
    
    logger.info(f"Built {len(graphs)} temporal graphs")
    return graphs
