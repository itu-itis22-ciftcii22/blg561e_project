"""
Training Module

Implements training procedures for token malice prediction GNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for GNN-based token classification models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        scheduler: Optional[Any] = None,
        class_weights: Optional[torch.Tensor] = None,
        gradient_clip: float = 1.0
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
        """
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        
        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
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
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            out = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                batch=batch.batch
            )
            
            loss = self.criterion(out, batch.y.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.num_graphs
            
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
        desc: str = 'Eval'
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            data_loader: Data loader for evaluation
            desc: Description for progress bar
            
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
        
        pbar = tqdm(data_loader, desc=desc)
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            out = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                batch=batch.batch
            )
            
            loss = self.criterion(out, batch.y.view(-1))
            
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)
            
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.num_graphs
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
        
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
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader, desc=f'Epoch {epoch} [Val]')
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log progress
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
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
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
            'metrics': metrics,
            'history': self.history
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
