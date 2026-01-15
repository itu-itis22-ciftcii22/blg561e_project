"""Temporal Graph Dataset - Builds PyG-compatible Data objects from TokenDataFrame."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.data import Data

from .preprocessor import TokenDataFrame, Cols


class DatasetError(Exception):
    """Raised when dataset operations fail."""
    pass


@dataclass
class TemporalGraphData:
    """Container for temporal graph data with multiple snapshots."""
    snapshots: list[Data]
    label: torch.Tensor
    num_nodes: int
    token_name: str
    num_transactions: int
    
    @property
    def num_snapshots(self):
        count: int = len(self.snapshots)
        return count
    
    @property
    def y(self):
        """Alias for label (PyG convention)."""
        return self.label
    
    def to(self, device: str):
        """Move all tensors to device."""
        moved: TemporalGraphData = TemporalGraphData(
            snapshots=[s.to(device) for s in self.snapshots],
            label=self.label.to(device),
            num_nodes=self.num_nodes,
            token_name=self.token_name,
            num_transactions=self.num_transactions,
        )
        return moved
    
    @property
    def device(self):
        dev: torch.device = self.label.device
        return dev


# Module-level constants for multiprocessing
_EDGE_FEATURE_DIM = 5
_WINDOWED_DIM = 4  # out_degree, out_volume, in_degree, in_volume


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int):
    """Scatter add operation using native PyTorch."""
    out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    return out.scatter_add_(0, index, src)


def _compute_windowed_delta_from_df(window_df, num_nodes: int):
    """Compute per-snapshot windowed delta features from DataFrame using Python arithmetic.
    
    Uses Python's unbounded integers for summing to prevent float32 overflow.
    Returns raw sums (not normalized) - caller should apply log1p after cumulative sum.
    """
    if window_df.empty:
        # Return zeros as Python lists for consistency
        return [[0, 0, 0, 0] for _ in range(num_nodes)]
    
    # Initialize with Python ints (unbounded)
    out_degree = [0] * num_nodes
    out_volume = [0] * num_nodes  # Python int, no overflow
    in_degree = [0] * num_nodes
    in_volume = [0] * num_nodes
    
    for _, row in window_df.iterrows():
        from_idx = int(row[Cols.FROM])
        to_idx = int(row[Cols.TO])
        # Keep as Python int to avoid overflow
        amount = int(row[Cols.AMOUNT])
        
        out_degree[from_idx] += 1
        out_volume[from_idx] += amount
        in_degree[to_idx] += 1
        in_volume[to_idx] += amount
    
    # Return as nested list of Python ints
    return [[out_degree[i], out_volume[i], in_degree[i], in_volume[i]] for i in range(num_nodes)]


def _compute_simplified_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    """Collapse multi-edges into single weighted edge per node pair."""
    from torch_geometric.utils import coalesce
    
    if edge_index.size(1) == 0:
        return edge_index, torch.zeros(0, device=edge_index.device, dtype=torch.float)
    
    raw_weights = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
    
    # Make bidirectional
    full_src = torch.cat([edge_index[0], edge_index[1]])
    full_dst = torch.cat([edge_index[1], edge_index[0]])
    full_weights = torch.cat([raw_weights, raw_weights])
    full_edge_index = torch.stack([full_src, full_dst], dim=0)
    
    simple_index, simple_weight = coalesce(full_edge_index, full_weights, reduce='sum')
    simple_weight = torch.log1p(torch.abs(simple_weight))
    
    return simple_index, simple_weight


def _get_time_windows(num_snapshots: int):
    """Calculate time windows for snapshots."""
    step = 1.0 / num_snapshots
    windows: list[tuple[float, float]] = []
    for i in range(num_snapshots):
        w_start = i * step
        w_end = (i + 1) * step
        windows.append((w_start, w_end))
    return windows


def _build_snapshot(
    df: pd.DataFrame,
    num_nodes: int,
    window_start: float,
    window_end: float,
):
    """Build single snapshot from transactions in time window.
    
    Note: x_windowed_raw contains Python ints (unbounded). Normalization
    is applied later in _build_graph after cumulative sums are computed.
    """
    mask = (df[Cols.REL_TIME] >= window_start) & (df[Cols.REL_TIME] < window_end)
    window_df = df[mask]
    
    # Compute windowed delta using Python arithmetic (no overflow)
    x_windowed_raw = _compute_windowed_delta_from_df(window_df, num_nodes)
    
    if window_df.empty:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, _EDGE_FEATURE_DIM), dtype=torch.float)
        simple_edge_index = torch.zeros((2, 0), dtype=torch.long)
        simple_edge_weight = torch.zeros(0, dtype=torch.float)
        
        snapshot: Data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
            x_windowed_raw=x_windowed_raw,  # Python list of ints
            simple_edge_index=simple_edge_index,
            simple_edge_weight=simple_edge_weight,
        )
        return snapshot
    
    window_duration = max(window_end - window_start, 1e-9)
    
    edge_indices = []
    edge_attrs = []
    
    for _, row in window_df.iterrows():
        from_idx = int(row[Cols.FROM])
        to_idx = int(row[Cols.TO])
        
        rel_time_global = row[Cols.REL_TIME]
        rel_time_window = (rel_time_global - window_start) / window_duration
        
        features = [
            np.log1p(float(row[Cols.AMOUNT])),  # Edge features still log-normalized
            rel_time_window,
            rel_time_global,
            np.sin(2 * np.pi * rel_time_global),
            np.cos(2 * np.pi * rel_time_global),
        ]
        
        edge_indices.append((from_idx, to_idx))
        edge_attrs.append(features)
        
        if from_idx != to_idx:
            edge_indices.append((to_idx, from_idx))
            edge_attrs.append(features)
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Precompute simplified edges for EvolveGCN
    simple_edge_index, simple_edge_weight = _compute_simplified_edges(edge_index, edge_attr)
    
    snapshot: Data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        x_windowed_raw=x_windowed_raw,  # Python list of ints (unbounded)
        simple_edge_index=simple_edge_index,
        simple_edge_weight=simple_edge_weight,
    )
    return snapshot


def _build_graph(token_data: TokenDataFrame, num_snapshots: int):
    """Build temporal graph from TokenDataFrame (module-level for multiprocessing).
    
    Computes cumulative sums using Python's unbounded integers, then applies
    log1p normalization at the end to prevent float32 overflow.
    """
    df = token_data.df
    num_nodes = token_data.num_nodes
    
    windows = _get_time_windows(num_snapshots)
    
    snapshots = []
    for start, end in windows:
        snapshot = _build_snapshot(df, num_nodes, start, end)
        snapshots.append(snapshot)
    
    # Compute cumulative prefix sums using Python ints (no overflow)
    # cumulative_raw[i] = [out_degree, out_volume, in_degree, in_volume] as Python ints
    cumulative_raw = [[0, 0, 0, 0] for _ in range(num_nodes)]
    
    for snapshot in snapshots:
        windowed_raw = snapshot.x_windowed_raw  # Python list of ints
        
        # Accumulate using Python arithmetic (unbounded)
        for i in range(num_nodes):
            for j in range(4):
                cumulative_raw[i][j] += windowed_raw[i][j]
        
        # Now apply log1p and convert to tensor
        # Use numpy for log1p on Python ints (handles arbitrary precision)
        cumulative_normalized = np.log1p(np.array(cumulative_raw, dtype=np.float64))
        windowed_normalized = np.log1p(np.array(windowed_raw, dtype=np.float64))
        
        snapshot.x_cumulative = torch.tensor(cumulative_normalized, dtype=torch.float32)
        snapshot.x_windowed = torch.tensor(windowed_normalized, dtype=torch.float32)
        
        # Clean up raw data (no longer needed)
        del snapshot.x_windowed_raw
    
    graph: TemporalGraphData = TemporalGraphData(
        snapshots=snapshots,
        label=torch.tensor([token_data.label], dtype=torch.long),
        num_nodes=num_nodes,
        token_name=token_data.token_name,
        num_transactions=token_data.num_transactions,
    )
    return graph


class TemporalGraphDataset(Dataset):
    """Dataset for temporal graph sequences."""
    
    EDGE_FEATURE_DIM = _EDGE_FEATURE_DIM
    
    def __init__(
        self,
        graphs: list[TemporalGraphData],
        transform: Optional[Callable[[TemporalGraphData], TemporalGraphData]] = None,
    ):
        """Initialize dataset with pre-built graphs.
        
        Args:
            graphs: List of TemporalGraphData objects.
            transform: Optional transform to apply to each sample.
        """
        self._graphs = graphs
        self.transform = transform
        
    @classmethod
    def from_token_data(
        cls,
        token_data_list: list[TokenDataFrame],
        num_snapshots: int = 6,
        transform: Optional[Callable[[TemporalGraphData], TemporalGraphData]] = None,
        num_workers: Optional[int] = None,
    ) -> TemporalGraphDataset:
        """Create dataset from token data with parallel processing.
        
        Args:
            token_data_list: List of TokenDataFrame objects.
            num_snapshots: Number of snapshots per graph.
            transform: Optional transform.
            num_workers: Number of workers for parallel processing.
        """
        if not token_data_list:
             raise DatasetError("Cannot create dataset from empty list")

        # Determine number of workers
        if num_workers is None:
            num_workers = cpu_count()
        
        if num_workers == 1:
            # Sequential processing
            graphs = [
                _build_graph(token_data, num_snapshots)
                for token_data in token_data_list
            ]
        else:
            # Parallel processing
            build_fn = partial(_build_graph, num_snapshots=num_snapshots)
            with Pool(processes=num_workers) as pool:
                graphs = pool.map(build_fn, token_data_list)
                
        return cls(graphs=graphs, transform=transform)
    
    def __len__(self):
        length: int = len(self._graphs)
        return length
    
    def get(self, index: int):
        """Get raw sample without transforms (PyG convention)."""
        sample: TemporalGraphData = self._graphs[index]
        return sample
    
    def __getitem__(self, index: int):
        """Get sample with transforms applied."""
        data = self.get(index)
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    @property
    def labels(self):
        """All labels as a tensor."""
        tensor: torch.Tensor = torch.tensor(
            [g.label.item() for g in self._graphs], dtype=torch.long
        )
        return tensor
    
    @property
    def transaction_counts(self):
        """Transaction counts for stratification."""
        counts: np.ndarray = np.array([g.num_transactions for g in self._graphs])
        return counts
    
    def get_stratification_key(self, n_bins: int = 5):
        """Composite stratification key: label * n_bins + transaction_size_bin."""
        labels = self.labels.numpy()
        
        try:
            size_bins = pd.qcut(
                self.transaction_counts,
                q=n_bins,
                labels=False,
                duplicates='drop'
            )
            size_bins = np.asarray(size_bins)
        except ValueError:
            size_bins = np.full(len(self._graphs), n_bins // 2, dtype=int)
        
        key: np.ndarray = labels * n_bins + size_bins
        return key
    
    def get_pos_weight(self):
        """Compute positive class weight for imbalanced binary data."""
        # pos_weight = negative_samples / positive_samples
        counts = torch.bincount(self.labels).float()
        if (counts == 0).any():
            raise DatasetError("Some classes have zero samples")
        
        # counts[0] is benign (neg), counts[1] is malicious (pos)
        pos_weight = counts[0] / counts[1]
        return pos_weight
    
    def get_statistics(self):
        """Get dataset statistics."""
        labels = self.labels
        stats: dict[str, float] = {
            "total_samples": len(self._graphs),
            "num_malicious": (labels == 1).sum().item(),
            "num_benign": (labels == 0).sum().item(),
            "malicious_ratio": (labels == 1).sum().item() / len(labels),
            "mean_transactions": self.transaction_counts.mean(),
            "median_transactions": float(np.median(self.transaction_counts)),
        }
        return stats
    
    @classmethod
    def get_edge_feature_dim(cls):
        """Return edge feature dimension."""
        dim: int = cls.EDGE_FEATURE_DIM
        return dim


def temporal_collate_fn(batch: list[TemporalGraphData]):
    """Identity collation (temporal graphs don't batch naturally)."""
    return batch


def create_loader(
    dataset: TemporalGraphDataset,
    indices: list[int],
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
):
    """Create DataLoader from dataset and indices."""
    subset = Subset(dataset, indices)
    loader: DataLoader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=temporal_collate_fn,
    )
    return loader

