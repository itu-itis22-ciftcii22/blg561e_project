"""
Token Malice Prediction Training Task

Main script for training the GNN-based token malice classifier.

Usage:
    python task.py --config config.yaml
    python task.py --data-dir ./solana1 --output-dir ./outputs
"""

import argparse
import logging
from pathlib import Path
import json

import torch
import torch.optim as optim

from token_malice_prediction.data import (
    TokenPreprocessor,
    TransactionGraphBuilder,
    build_graphs_from_processed_data,
    TokenGraphDatasetList,
    create_data_loaders
)
from token_malice_prediction.models import create_model
from token_malice_prediction.training import Trainer, compute_metrics
from token_malice_prediction.utils import (
    load_config,
    save_config,
    setup_logging,
    set_seed,
    get_device
)
from token_malice_prediction.data.graph_builder import TransactionGraphBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Token Malice Prediction Training'
    )
    
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-dir', type=str,
        help='Directory containing token CSV files (overrides config)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--device', type=str, choices=['cuda', 'cpu'],
        help='Device to use (overrides config)'
    )
    parser.add_argument(
        '--epochs', type=int,
        help='Number of epochs (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    setup_logging(log_file=str(output_dir / 'training.log'))
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(prefer_gpu=(config.device == 'cuda'))
    
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Save configuration
    save_config(config, str(output_dir / 'config.yaml'))
    
    # =====================
    # Data Processing
    # =====================
    logger.info("=" * 50)
    logger.info("Starting data preprocessing...")
    
    preprocessor = TokenPreprocessor(
        data_dir=config.data.data_dir,
        malice_threshold=config.data.malice_threshold,
        min_transactions=config.data.min_transactions
    )
    
    processed_data = preprocessor.process_directory()
    
    if len(processed_data) == 0:
        logger.error("No valid tokens found. Exiting.")
        return
    
    logger.info(f"Processed {len(processed_data)} tokens")
    
    # =====================
    # Graph Construction
    # =====================
    logger.info("=" * 50)
    logger.info("Building transaction graphs...")
    
    graphs_with_names = build_graphs_from_processed_data(processed_data)
    graphs = [g for g, _ in graphs_with_names]
    
    logger.info(f"Built {len(graphs)} graphs")
    
    # Log graph statistics
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.edge_index.shape[1] for g in graphs]
    labels = [g.y.item() for g in graphs]
    
    logger.info(f"Graph statistics:")
    logger.info(f"  Nodes: min={min(num_nodes)}, max={max(num_nodes)}, avg={sum(num_nodes)/len(num_nodes):.1f}")
    logger.info(f"  Edges: min={min(num_edges)}, max={max(num_edges)}, avg={sum(num_edges)/len(num_edges):.1f}")
    logger.info(f"  Labels: {sum(labels)} malicious, {len(labels)-sum(labels)} benign")
    
    # =====================
    # Dataset Creation
    # =====================
    logger.info("=" * 50)
    logger.info("Creating datasets and data loaders...")
    
    dataset = TokenGraphDatasetList(graphs)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        random_seed=config.seed
    )
    
    # =====================
    # Model Creation
    # =====================
    logger.info("=" * 50)
    logger.info("Creating model...")
    
    model = create_model(
        node_dim=TransactionGraphBuilder.get_node_feature_dim(),
        edge_dim=TransactionGraphBuilder.get_edge_feature_dim(),
        hidden_dim=config.model.hidden_dim,
        num_classes=config.model.num_classes,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        pooling=config.model.pooling,
        use_edge_features=config.model.use_edge_features
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # =====================
    # Training Setup
    # =====================
    logger.info("=" * 50)
    logger.info("Setting up training...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Class weights for imbalanced data
    class_weights = None
    if config.training.use_class_weights:
        class_weights = dataset.get_class_weights()
        logger.info(f"Class weights: {class_weights.tolist()}")
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=str(device),
        scheduler=scheduler,
        class_weights=class_weights,
        gradient_clip=config.training.gradient_clip
    )
    
    # =====================
    # Training
    # =====================
    logger.info("=" * 50)
    logger.info("Starting training...")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_path=str(output_dir / 'best_model.pt')
    )
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # =====================
    # Evaluation
    # =====================
    logger.info("=" * 50)
    logger.info("Evaluating on test set...")
    
    test_results = trainer.evaluate(test_loader, desc='Test')
    
    metrics = compute_metrics(
        predictions=test_results['predictions'],
        labels=test_results['labels'],
        probabilities=test_results['probabilities']
    )
    
    logger.info("Test Results:")
    logger.info(f"  Loss: {test_results['loss']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
    
    # Save test metrics
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("=" * 50)
    logger.info(f"Training complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
