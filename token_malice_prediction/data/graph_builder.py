"""
Graph Construction Module

Builds transaction multigraphs from preprocessed token data.
Creates PyTorch Geometric Data objects suitable for GNN processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

import torch
from torch_geometric.data import Data


class TransactionGraphBuilder:
    """
    Build transaction multigraphs from token transaction data.
    
    Creates undirected multigraphs where:
    - Nodes: Unique wallet addresses from From and To columns
    - Edges: Transaction relationships with types (sent, received, mint)
    - Edge features: amount, value, amount*value, relative time features
    
    Edge types:
    - SENT (0): From perspective of sender
    - RECEIVED (1): From perspective of receiver  
    - MINT (2): Self-connected mint transaction
    """
    
    EDGE_TYPE_SENT = 0
    EDGE_TYPE_RECEIVED = 1
    EDGE_TYPE_MINT = 2
    
    def __init__(self):
        """Initialize the graph builder."""
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        
    def _build_node_mapping(self, df: pd.DataFrame) -> int:
        """
        Build node index mapping from addresses.
        
        Args:
            df: Transaction DataFrame with From and To columns
            
        Returns:
            Number of unique nodes
        """
        # Get all unique addresses
        all_addresses = pd.concat([df['From'], df['To']]).unique()
        
        self.node_to_idx = {addr: idx for idx, addr in enumerate(all_addresses)}
        self.idx_to_node = {idx: addr for addr, idx in self.node_to_idx.items()}
        
        return len(all_addresses)
    
    def _create_edge_features(
        self,
        row: pd.Series,
        edge_type: int
    ) -> List[float]:
        """
        Create edge feature vector for a transaction.
        
        Features:
        - amount (log-scaled)
        - value (log-scaled)
        - amount * value (log-scaled)
        - relative time (6 weeks normalized)
        - period_2w (2-week cycle)
        - period_1w (1-week cycle)
        - period_1d (daily cycle)
        - edge_type (one-hot encoded, 3 values)
        
        Args:
            row: Transaction row from DataFrame
            edge_type: Type of edge (sent/received/mint)
            
        Returns:
            Feature vector as list of floats
        """
        # Numeric features (log-scaled to handle large values)
        amount = np.log1p(float(row['Amount']))
        value = np.log1p(float(row['Value']))
        amount_x_value = np.log1p(float(row['amount_x_value']))
        
        # Time features
        rel_time_6w = float(row['rel_time_6w'])
        period_2w = float(row['period_2w'])
        period_1w = float(row['period_1w'])
        period_1d = float(row['period_1d'])
        
        # Edge type one-hot encoding
        edge_type_onehot = [0.0, 0.0, 0.0]
        edge_type_onehot[edge_type] = 1.0
        
        return [
            amount,
            value,
            amount_x_value,
            rel_time_6w,
            period_2w,
            period_1w,
            period_1d,
        ] + edge_type_onehot
    
    def build_graph(
        self,
        df: pd.DataFrame,
        label: int
    ) -> Data:
        """
        Build a PyTorch Geometric Data object from transaction DataFrame.
        
        Creates an undirected multigraph with:
        - Edges for both directions (sent and received perspectives)
        - Self-loops for mint transactions
        - Edge features including transaction attributes and time features
        
        Args:
            df: Preprocessed transaction DataFrame
            label: Token label (0=benign, 1=malicious)
            
        Returns:
            PyTorch Geometric Data object
        """
        num_nodes = self._build_node_mapping(df)
        
        edge_indices = []  # List of (src, dst) tuples
        edge_attrs = []    # List of feature vectors
        edge_types = []    # List of edge type integers
        
        for _, row in df.iterrows():
            from_addr = row['From']
            to_addr = row['To']
            
            from_idx = self.node_to_idx[from_addr]
            to_idx = self.node_to_idx[to_addr]
            
            # Check if this is a mint (self-transaction)
            if from_addr == to_addr:
                # Mint: self-connected edge with MINT type
                edge_indices.append((from_idx, to_idx))
                edge_attrs.append(self._create_edge_features(row, self.EDGE_TYPE_MINT))
                edge_types.append(self.EDGE_TYPE_MINT)
            else:
                # Regular transfer: create undirected edges
                # SENT edge: from sender's perspective (from -> to)
                edge_indices.append((from_idx, to_idx))
                edge_attrs.append(self._create_edge_features(row, self.EDGE_TYPE_SENT))
                edge_types.append(self.EDGE_TYPE_SENT)
                
                # RECEIVED edge: from receiver's perspective (to -> from)
                edge_indices.append((to_idx, from_idx))
                edge_attrs.append(self._create_edge_features(row, self.EDGE_TYPE_RECEIVED))
                edge_types.append(self.EDGE_TYPE_RECEIVED)
        
        # Convert to tensors
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 10), dtype=torch.float)  # 10 = feature dim
            edge_type = torch.zeros((0,), dtype=torch.long)
        
        # Create node features based on degree statistics
        node_features = self._compute_node_features(df, num_nodes)
        
        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_type=edge_type,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=num_nodes
        )
        
        return data
    
    def _compute_node_features(
        self,
        df: pd.DataFrame,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute node features based on transaction statistics.
        
        Features per node:
        - in_degree (number of incoming transactions)
        - out_degree (number of outgoing transactions)
        - total_degree
        - total_amount_sent (log-scaled)
        - total_amount_received (log-scaled)
        - total_value_sent (log-scaled)
        - total_value_received (log-scaled)
        - avg_transaction_value_sent
        - avg_transaction_value_received
        - first_transaction_time (normalized)
        - last_transaction_time (normalized)
        
        Args:
            df: Transaction DataFrame
            num_nodes: Total number of nodes
            
        Returns:
            Node feature tensor of shape (num_nodes, num_features)
        """
        # Initialize feature arrays
        in_degree = np.zeros(num_nodes)
        out_degree = np.zeros(num_nodes)
        total_amount_sent = np.zeros(num_nodes)
        total_amount_received = np.zeros(num_nodes)
        total_value_sent = np.zeros(num_nodes)
        total_value_received = np.zeros(num_nodes)
        first_time = np.ones(num_nodes) * float('inf')
        last_time = np.zeros(num_nodes)
        
        for _, row in df.iterrows():
            from_idx = self.node_to_idx[row['From']]
            to_idx = self.node_to_idx[row['To']]
            amount = float(row['Amount'])
            value = float(row['Value'])
            time = float(row['rel_time_6w'])
            
            if from_idx != to_idx:  # Not a self-transaction
                out_degree[from_idx] += 1
                in_degree[to_idx] += 1
                total_amount_sent[from_idx] += amount
                total_amount_received[to_idx] += amount
                total_value_sent[from_idx] += value
                total_value_received[to_idx] += value
            else:  # Mint/self-transaction
                in_degree[from_idx] += 1
                out_degree[from_idx] += 1
                total_amount_received[from_idx] += amount
                total_value_received[from_idx] += value
            
            # Update time bounds
            first_time[from_idx] = min(first_time[from_idx], time)
            first_time[to_idx] = min(first_time[to_idx], time)
            last_time[from_idx] = max(last_time[from_idx], time)
            last_time[to_idx] = max(last_time[to_idx], time)
        
        # Handle nodes with no transactions (shouldn't happen, but safety check)
        first_time[first_time == float('inf')] = 0
        
        # Compute derived features
        total_degree = in_degree + out_degree
        
        # Avoid division by zero for averages
        avg_value_sent = np.divide(
            total_value_sent, 
            out_degree, 
            out=np.zeros_like(total_value_sent),
            where=out_degree > 0
        )
        avg_value_received = np.divide(
            total_value_received,
            in_degree,
            out=np.zeros_like(total_value_received),
            where=in_degree > 0
        )
        
        # Stack features
        node_features = np.column_stack([
            in_degree,
            out_degree,
            total_degree,
            np.log1p(total_amount_sent),
            np.log1p(total_amount_received),
            np.log1p(total_value_sent),
            np.log1p(total_value_received),
            np.log1p(avg_value_sent),
            np.log1p(avg_value_received),
            first_time,
            last_time
        ])
        
        return torch.tensor(node_features, dtype=torch.float)
    
    @staticmethod
    def get_edge_feature_dim() -> int:
        """Return the dimension of edge features."""
        return 10  # 7 numeric + 3 one-hot edge type
    
    @staticmethod
    def get_node_feature_dim() -> int:
        """Return the dimension of node features."""
        return 11


def build_graphs_from_processed_data(
    processed_data: List[Tuple[pd.DataFrame, int, str]]
) -> List[Tuple[Data, str]]:
    """
    Build graphs from all processed token data.
    
    Args:
        processed_data: List of (df, label, token_name) from preprocessor
        
    Returns:
        List of (Data, token_name) tuples
    """
    graphs = []
    builder = TransactionGraphBuilder()
    
    for df, label, token_name in processed_data:
        try:
            graph = builder.build_graph(df, label)
            graphs.append((graph, token_name))
            logger.debug(f"Built graph for {token_name}: "
                        f"{graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
        except Exception as e:
            logger.error(f"Error building graph for {token_name}: {e}")
    
    logger.info(f"Built {len(graphs)} graphs")
    return graphs
