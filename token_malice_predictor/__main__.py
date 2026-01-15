"""Token Malice Prediction - Ablation Study Runner."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .preprocessor import TokenPreprocessor
from .dataset import TemporalGraphDataset, create_loader
from .classifier import TokenMaliceClassifier
from .trainer import Trainer, EpochMetrics


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_dir: str
    num_snapshots: int = 6
    cache_path: str | None = None


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_heads: int = 2
    dropout: float = 0.2
    pooling: str = "concat"


@dataclass
class TrainingConfig:
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 100
    gradient_clip: float = 1.0


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig | None = None  # If None, use base training config


@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments."""
    base_data: DataConfig
    base_training: TrainingConfig
    runs: list[RunConfig]
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "cuda"
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load experiment config from YAML file."""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        
        base = raw.get('base', {})
        data_cfg = DataConfig(**base.get('data', {}))
        training_cfg = TrainingConfig(**base.get('training', {}))
        
        runs = []
        for run_raw in raw.get('runs', []):
            model_cfg = ModelConfig(**run_raw.get('model', {}))
            
            # Override training config if specified
            run_training = None
            if 'training' in run_raw:
                run_training = TrainingConfig(**run_raw['training'])
            
            runs.append(RunConfig(
                name=run_raw['name'],
                model=model_cfg,
                training=run_training,
            ))
        
        return cls(
            base_data=data_cfg,
            base_training=training_cfg,
            runs=runs,
            output_dir=base.get('output_dir', './outputs'),
            seed=base.get('seed', 42),
            device=base.get('device', 'cuda'),
        )


def train_model(
    trainer: Trainer,
    train_loader,
    val_loader,
    num_epochs: int,
    start_epoch: int,
    output_dir: Path,
    run_name: str,
) -> dict[str, list[dict[str, Any]]]:
    """Train model with checkpoint saving. Returns per-epoch metrics."""
    latest_path = output_dir / "latest_checkpoint.pt"
    best_path = output_dir / "best_checkpoint.pt"
    
    epoch_history: dict[str, list[dict[str, Any]]] = {'train': [], 'val': []}
    
    pbar = tqdm(
        range(start_epoch + 1, num_epochs + 1),
        desc=f"[{run_name}]",
        initial=start_epoch,
        total=num_epochs,
    )
    
    for epoch in pbar:
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        # Record metrics for this epoch
        epoch_history['train'].append({
            'epoch': epoch,
            'loss': train_metrics.loss,
            'accuracy': train_metrics.accuracy,
        })
        epoch_history['val'].append({
            'epoch': epoch,
            'loss': val_metrics.loss,
            'accuracy': val_metrics.accuracy,
        })
        
        trainer.step_scheduler(val_metrics.loss)
        
        pbar.set_postfix({
            'train_loss': f'{train_metrics.loss:.4f}',
            'val_loss': f'{val_metrics.loss:.4f}',
            'val_acc': f'{val_metrics.accuracy:.2%}',
            'best': f'{trainer.best_val_loss:.4f}',
            'lr': f'{trainer.get_lr():.2e}',
        })
        
        trainer.save_checkpoint(latest_path, epoch, val_metrics)
        
        if trainer.update_best(val_metrics.loss):
            trainer.save_checkpoint(best_path, epoch, val_metrics)
            tqdm.write(f"[{run_name}] New best model saved (val_loss={val_metrics.loss:.4f})")
    
    return epoch_history


def run_single_experiment(
    run: RunConfig,
    dataset: TemporalGraphDataset,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    base_training: TrainingConfig,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Run a single experiment and return results."""
    
    training = run.training if run.training else base_training
    run_output_dir = output_dir / run.name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Starting run: {run.name}")
    
    train_loader = create_loader(
        dataset, train_idx,
        batch_size=training.batch_size, shuffle=True,
    )
    val_loader = create_loader(
        dataset, val_idx,
        batch_size=training.batch_size, shuffle=False,
    )
    test_loader = create_loader(
        dataset, test_idx,
        batch_size=training.batch_size, shuffle=False,
    )
    
    model = TokenMaliceClassifier(
        edge_dim=TemporalGraphDataset.get_edge_feature_dim(),
        hidden_dim=run.model.hidden_dim,
        num_classes=1,
        num_heads=run.model.num_heads,
        dropout=run.model.dropout,
        pooling=run.model.pooling,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )
    
    pos_weight = dataset.get_pos_weight()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        pos_weight=pos_weight,
        scheduler=scheduler,
        gradient_clip=training.gradient_clip,
    )
    
    # Try to resume from latest checkpoint
    latest_path = run_output_dir / "latest_checkpoint.pt"
    start_epoch = 0
    
    if latest_path.exists():
        start_epoch = trainer.load_checkpoint(latest_path)
        log.info(f"[{run.name}] Resumed from epoch {start_epoch}")
    
    epoch_history = train_model(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training.num_epochs,
        start_epoch=start_epoch,
        output_dir=run_output_dir,
        run_name=run.name,
    )
    
    # Load best model for final evaluation
    best_path = run_output_dir / "best_checkpoint.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path)
    else:
        trainer.restore_best()
    
    test_metrics: EpochMetrics = trainer.evaluate(test_loader)
    
    # Build result dict with run identity and metrics
    result = {
        'run_name': run.name,
        # Model config
        'hidden_dim': run.model.hidden_dim,
        'num_heads': run.model.num_heads,
        'dropout': run.model.dropout,
        'pooling': run.model.pooling,
        # Training config
        'learning_rate': training.learning_rate,
        'weight_decay': training.weight_decay,
        'num_epochs': training.num_epochs,
        # Test metrics
        'test_loss': test_metrics.loss,
        'test_accuracy': test_metrics.accuracy,
        'predictions': test_metrics.predictions,
        'labels': test_metrics.labels,
        'probabilities': test_metrics.probabilities,
        # Per-epoch training history for plotting
        'epoch_history': epoch_history,
    }
    
    return result


def run_all_experiments(cfg: ExperimentConfig):
    """Run all experiments and save results to JSONL."""
    device = 'cuda' if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu'
    log.info(f"Using device: {device}")
    
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "results.jsonl"
    
    # Load or build dataset (shared across all runs)
    cache_path = Path(cfg.base_data.cache_path) if cfg.base_data.cache_path else None
    
    log.info("Building dataset...")
    
    if cache_path and cache_path.exists():
        log.info(f"Loading cached graphs from {cache_path}")
        graphs = torch.load(cache_path, weights_only=False)
        dataset = TemporalGraphDataset(graphs=graphs)
    else:
        log.info(f"Loading data from {cfg.base_data.data_dir}")
        token_data_list, errors = TokenPreprocessor().load_directory(cfg.base_data.data_dir)
        log.info(f"Loaded {len(token_data_list)} tokens, {errors} errors")
        
        dataset = TemporalGraphDataset.from_token_data(
            token_data_list,
            num_snapshots=cfg.base_data.num_snapshots,
        )
        
        if cache_path:
            log.info(f"Saving graphs to cache: {cache_path}")
            torch.save(dataset._graphs, cache_path)
    
    stats = dataset.get_statistics()
    log.info(
        f"Dataset: {stats['total_samples']} samples, "
        f"{stats['num_malicious']} malicious ({stats['malicious_ratio']:.1%}), "
        f"{stats['num_benign']} benign"
    )
    
    # Stratified train/val/test split (same for all runs)
    strat_key = dataset.get_stratification_key(n_bins=5)
    indices = list(range(len(dataset)))
    
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=cfg.base_training.test_ratio,
        stratify=strat_key,
        random_state=cfg.seed,
    )
    
    train_val_strat = [strat_key[i] for i in train_val_idx]
    val_size = cfg.base_training.val_ratio / (1 - cfg.base_training.test_ratio)
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=train_val_strat,
        random_state=cfg.seed,
    )
    
    log.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    log.info(f"Running {len(cfg.runs)} experiment(s)")
    
    # Run each experiment
    for run in cfg.runs:
        result = run_single_experiment(
            run=run,
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            base_training=cfg.base_training,
            output_dir=output_dir,
            device=device,
        )
        
        # Append result to JSONL
        with open(results_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        log.info(f"[{run.name}] Completed - test_loss={result['test_loss']:.4f}, test_acc={result['test_accuracy']:.2%}")
    
    log.info(f"All experiments completed. Results saved to {results_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Token Malice Prediction - Ablation Study')
    parser.add_argument('config', type=str, help='Path to experiment config.yaml file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    run_all_experiments(cfg)
