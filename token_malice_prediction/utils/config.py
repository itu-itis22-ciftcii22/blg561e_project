"""
Configuration Management

Handles configuration loading, validation, and management.
"""

import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = ""  # Directory containing token CSV files
    batch_size: int = 32
    num_workers: int = 0
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    malice_threshold: float = 0.9  # Threshold for malicious classification
    min_transactions: int = 10  # Minimum transactions per token


@dataclass
class ModelConfig:
    """Model configuration."""
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.3
    num_classes: int = 2
    pooling: str = 'concat'  # 'mean', 'max', 'add', 'concat'
    use_edge_features: bool = True
    
    # Temporal model settings
    model_type: str = 'temporal'  # 'gat' for old model, 'temporal' for new
    num_snapshots: int = 5  # Number of temporal snapshots
    num_gine_layers: int = 3  # Number of GINEConv layers per snapshot
    num_temporal_layers: int = 2  # Number of LSTM/EvolveGCN layers
    use_evolve_gcn: bool = True  # Use EvolveGCN-O vs LSTM for temporal
    overlap_ratio: float = 0.2  # Overlap between temporal windows
    attention_heads: int = 4  # Heads for GlobalAttention readout


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 100
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    use_class_weights: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./outputs"
    experiment_name: str = "token_malice_prediction"
    seed: int = 42
    device: str = "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            output_dir=config_dict.get('output_dir', './outputs'),
            experiment_name=config_dict.get('experiment_name', 'token_malice_prediction'),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda')
        )


def load_config(config_path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file (YAML or JSON)
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return Config()
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    logger.info(f"Loaded config from {config_path}")
    return Config.from_dict(config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Config object
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(save_path, 'w') as f:
        if save_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False)
        elif save_path.suffix == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path.suffix}")
    
    logger.info(f"Saved config to {save_path}")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
