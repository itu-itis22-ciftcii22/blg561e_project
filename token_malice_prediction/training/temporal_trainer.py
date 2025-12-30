"""
Enhanced Training Module for Temporal GNN

Implements training procedures with detailed progress logging
suitable for Google Colab notebooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
from tqdm.auto import tqdm
import time
from datetime import timedelta

logger = logging.getLogger(__name__)


class ProgressLogger:
    """
    Rich progress logging for training visualization in notebooks.
    """
    
    def __init__(self, total_epochs: int, log_interval: int = 1):
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        self.start_time = None
        self.epoch_times = []
        
    def start(self):
        """Start timing."""
        self.start_time = time.time()
        self.epoch_times = []
        
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float
    ):
        """Log epoch results with timing and ETA."""
        epoch_time = time.time()
        if self.epoch_times:
            epoch_duration = epoch_time - self.epoch_times[-1]
        else:
            epoch_duration = epoch_time - self.start_time
        self.epoch_times.append(epoch_time)
        
        # Calculate ETA
        avg_epoch_time = (epoch_time - self.start_time) / epoch
        remaining_epochs = self.total_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Format metrics
        elapsed = str(timedelta(seconds=int(epoch_time - self.start_time)))
        
        if epoch % self.log_interval == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{self.total_epochs} | "
                  f"Time: {elapsed} | ETA: {eta_str} | LR: {lr:.2e}")
            print(f"{'='*70}")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, "
                  f"Acc={train_metrics['accuracy']:.4f}")
            print(f"  Val:   Loss={val_metrics['loss']:.4f}, "
                  f"Acc={val_metrics['accuracy']:.4f}")
            
            # Progress bar visual
            progress = epoch / self.total_epochs
            bar_width = 50
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"  Progress: [{bar}] {progress*100:.1f}%")
    
    def log_best_model(self, epoch: int, val_loss: float, val_acc: float):
        """Log when best model is saved."""
        print(f"\n  ★ New best model at epoch {epoch}! "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping."""
        print(f"\n  ⚠ Early stopping at epoch {epoch} "
              f"(no improvement for {patience} epochs)")
    
    def log_final(self, history: Dict[str, List[float]]):
        """Log final training summary."""
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        
        total_time = str(timedelta(seconds=int(time.time() - self.start_time)))
        best_val_loss = min(history['val_loss'])
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_loss'].index(best_val_loss) + 1
        
        print(f"  Total Time: {total_time}")
        print(f"  Total Epochs: {len(history['train_loss'])}")
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f}")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Train Acc: {history['train_acc'][-1]:.4f}")


class TemporalTrainer:
    """
    Enhanced trainer for temporal GNN models with rich progress logging.
    
    Supports both static graphs and temporal graph sequences.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        scheduler: Optional[Any] = None,
        class_weights: Optional[torch.Tensor] = None,
        gradient_clip: float = 1.0,
        use_temporal: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer instance
            device: Device to use ('cuda' or 'cpu')
            scheduler: Learning rate scheduler
            class_weights: Class weights for imbalanced data
            gradient_clip: Gradient clipping value
            use_temporal: Whether using temporal graph sequences
        """
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.use_temporal = use_temporal
        
        # Loss function with optional class weights and label smoothing for numerical stability
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def _forward_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass handling both static and temporal data.
        
        Returns:
            Tuple of (output logits, labels)
        """
        # Check if batch is a list of temporal graphs
        is_temporal_batch = (
            self.use_temporal and 
            isinstance(batch, list) and 
            len(batch) > 0 and 
            hasattr(batch[0], 'snapshots')
        )
        
        if is_temporal_batch:
            # Temporal batch - list of temporal graphs
            out = self.model(
                x=None,
                edge_index=None,
                snapshots=[b.snapshots for b in batch],
                num_nodes_list=[b.num_nodes for b in batch]
            )
            labels = torch.stack([b.label for b in batch]).view(-1)
        else:
            # Static batch (standard PyG batch)
            out = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                batch=batch.batch
            )
            labels = batch.y.view(-1)
        
        return out, labels
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        progress_bar: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch} [Train]',
            disable=not progress_bar,
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            if self.use_temporal:
                # Move temporal batch to device
                batch = [b.to(self.device) for b in batch]
            else:
                batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            out, labels = self._forward_batch(batch)
            labels = labels.to(self.device)
            
            loss = self.criterion(out, labels)
            
            # Check for NaN loss and skip batch if so
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected, skipping batch")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN/Inf gradients detected, skipping batch")
                self.optimizer.zero_grad()
                continue
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            batch_size = len(batch) if self.use_temporal else batch.num_graphs
            total_loss += loss.item() * batch_size
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += batch_size
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        desc: str = 'Eval',
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            data_loader: Data loader for evaluation
            desc: Description for progress bar
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(data_loader, desc=desc, disable=not progress_bar, leave=False)
        
        for batch in pbar:
            if self.use_temporal:
                batch = [b.to(self.device) for b in batch]
            else:
                batch = batch.to(self.device)
            
            out, labels = self._forward_batch(batch)
            labels = labels.to(self.device)
            
            loss = self.criterion(out, labels)
            
            # Skip batches with NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            batch_size = len(batch) if self.use_temporal else batch.num_graphs
            total_loss += loss.item() * batch_size
            pred = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)
            
            # Handle NaN in probabilities
            probs = torch.nan_to_num(probs, nan=0.5)
            
            correct += (pred == labels).sum().item()
            total += batch_size
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
        verbose: bool = True,
        log_interval: int = 1
    ) -> Dict[str, List[float]]:
        """
        Full training loop with rich progress logging.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
            save_path: Path to save best model
            verbose: Whether to print progress
            log_interval: Epochs between detailed logs
            
        Returns:
            Training history
        """
        patience_counter = 0
        
        # Initialize progress logger
        progress = ProgressLogger(num_epochs, log_interval)
        if verbose:
            progress.start()
            print(f"\n{'='*70}")
            print("TRAINING STARTED")
            print(f"{'='*70}")
            print(f"  Model: {self.model.__class__.__name__}")
            print(f"  Device: {self.device}")
            print(f"  Epochs: {num_epochs}")
            print(f"  Early Stopping Patience: {early_stopping_patience}")
            print(f"  Train Batches: {len(train_loader)}")
            print(f"  Val Batches: {len(val_loader)}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(
                train_loader, epoch, progress_bar=verbose
            )
            
            # Validate
            val_metrics = self.evaluate(
                val_loader, desc=f'Epoch {epoch} [Val]', progress_bar=verbose
            )
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # Log progress
            if verbose:
                progress.log_epoch(epoch, train_metrics, val_metrics, current_lr)
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}"
            )
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_metrics)
                    logger.info(f"Saved best model to {save_path}")
                
                if verbose:
                    progress.log_best_model(
                        epoch, val_metrics['loss'], val_metrics['accuracy']
                    )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    progress.log_early_stopping(epoch, early_stopping_patience)
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        if verbose:
            progress.log_final(self.history)
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': {k: v for k, v in metrics.items() 
                       if not isinstance(v, list)},
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint


def compute_metrics(
    predictions: List[int],
    labels: List[int],
    probabilities: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        probabilities: Predicted probabilities for positive class
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0)
    }
    
    if probabilities is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(labels, probabilities)
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    
    return metrics


def print_model_summary(model: nn.Module, input_shapes: Optional[Dict] = None):
    """
    Print model summary with parameter counts.
    
    Args:
        model: PyTorch model
        input_shapes: Optional dict of input tensor shapes
    """
    print(f"\n{'='*70}")
    print("MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"\nModel: {model.__class__.__name__}")
    print(f"\nArchitecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Memory estimate
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    print(f"\nEstimated Model Size: {total_size / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")
