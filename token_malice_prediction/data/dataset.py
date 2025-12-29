"""
Dataset Classes

PyTorch Geometric Dataset implementations for token malice prediction.
"""

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional, Callable
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class TokenGraphDataset(InMemoryDataset):
    """
    In-memory dataset for token transaction graphs.
    
    Each sample is a graph representing a token's transaction history
    with a binary label (malicious or benign).
    """
    
    def __init__(
        self,
        graphs: Optional[List[Data]] = None,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            graphs: List of PyTorch Geometric Data objects
            root: Root directory for saving/loading processed data
            transform: Transform to apply to each sample
            pre_transform: Transform to apply before saving
            pre_filter: Filter to apply before saving
        """
        self._graphs = graphs
        
        if root is not None:
            super().__init__(root, transform, pre_transform, pre_filter)
            self.load(self.processed_paths[0])
        else:
            super().__init__(None, transform, pre_transform, pre_filter)
            if graphs is not None:
                self.data, self.slices = self.collate(graphs)
    
    @property
    def raw_file_names(self) -> List[str]:
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        if self._graphs is not None:
            data_list = self._graphs
            
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            
            self.save(data_list, self.processed_paths[0])
    
    def get_labels(self) -> torch.Tensor:
        """Get all labels in the dataset."""
        labels = []
        for i in range(len(self)):
            data = self.get(i)
            labels.append(data.y.item())
        return torch.tensor(labels)
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.
        
        Returns:
            Tensor of class weights (inverse frequency)
        """
        labels = self.get_labels()
        class_counts = torch.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts.float())
        return weights


class TokenGraphDatasetList(Dataset):
    """
    List-based dataset for token transaction graphs.
    
    More memory efficient for very large datasets as it doesn't
    store all graphs in memory at once.
    """
    
    def __init__(
        self,
        graphs: List[Data],
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            graphs: List of PyTorch Geometric Data objects
            transform: Transform to apply to each sample
        """
        super().__init__(None, transform)
        self.graphs = graphs
    
    def len(self) -> int:
        return len(self.graphs)
    
    def get(self, idx: int) -> Data:
        data = self.graphs[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def get_labels(self) -> torch.Tensor:
        """Get all labels in the dataset."""
        return torch.tensor([g.y.item() for g in self.graphs])
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        labels = self.get_labels()
        class_counts = torch.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts.float())
        return weights


def create_data_loaders(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders with stratified splitting.
    
    Args:
        dataset: Token graph dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        num_workers: Number of data loading workers
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Get labels for stratification
    labels = dataset.get_labels().numpy()
    indices = list(range(len(dataset)))
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_seed
    )
    
    # Second split: train vs val
    train_val_labels = labels[train_val_idx]
    val_size = val_ratio / (train_ratio + val_ratio)
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=random_seed
    )
    
    # Create subset datasets
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"Created data loaders: "
               f"train={len(train_dataset)}, "
               f"val={len(val_dataset)}, "
               f"test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def save_dataset(dataset: Dataset, filepath: str):
    """Save dataset to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(dataset, 'graphs'):
        # TokenGraphDatasetList
        graphs = dataset.graphs
    else:
        # TokenGraphDataset (InMemoryDataset)
        graphs = [dataset[i] for i in range(len(dataset))]
    
    with open(filepath, 'wb') as f:
        pickle.dump(graphs, f)
    
    logger.info(f"Saved {len(graphs)} graphs to {filepath}")


def load_dataset(
    filepath: str,
    transform: Optional[Callable] = None
) -> TokenGraphDatasetList:
    """Load dataset from disk."""
    with open(filepath, 'rb') as f:
        graphs = pickle.load(f)
    
    logger.info(f"Loaded {len(graphs)} graphs from {filepath}")
    return TokenGraphDatasetList(graphs, transform=transform)
