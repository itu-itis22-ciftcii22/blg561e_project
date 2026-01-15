"""Trainer - Single-epoch training/evaluation utilities with checkpointing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .dataset import TemporalGraphData

log = logging.getLogger(__name__)


class TrainerError(Exception):
    """Raised when training fails."""
    pass


@dataclass
class EpochMetrics:
    """Metrics from a single epoch."""
    loss: float
    accuracy: float
    predictions: list[int]
    labels: list[int]
    probabilities: list[float]


class Trainer:
    """Utility class for single-epoch training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str,
        pos_weight: torch.Tensor,
        scheduler: Optional[LRScheduler] = None,
        gradient_clip: float = 1.0,
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        
        self.best_val_loss = float('inf')
        self.best_model_state: dict[str, torch.Tensor] | None = None
    
    def _move_batch(self, batch: list[TemporalGraphData]):
        """Move batch to target device."""
        return [b.to(self.device) for b in batch]
    
    def train_epoch(self, loader: DataLoader):
        """Train for one epoch. Returns EpochMetrics."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[float] = []
        
        for batch in loader:
            batch = self._move_batch(batch)
            
            self.optimizer.zero_grad()
            
            logits = self.model(batch)
            labels = torch.cat([b.label for b in batch]).to(self.device).float()
            logits = logits.view(-1)  # Ensure [batch_size] shape
            
            # NaN debugging
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                token_names = [b.token_name for b in batch]
                log.warning(
                    f"NaN/Inf in logits! Tokens: {token_names}, "
                    f"logits: {logits.tolist()}, labels: {labels.tolist()}"
                )
                continue  # Skip this batch
            
            loss = self.criterion(logits, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                token_names = [b.token_name for b in batch]
                log.warning(
                    f"NaN/Inf loss! Tokens: {token_names}, "
                    f"logits: {logits.tolist()}, labels: {labels.tolist()}, "
                    f"loss: {loss.item()}"
                )
                continue  # Skip this batch
            
            probs = torch.sigmoid(logits)
            loss.backward()
            
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )
            
            self.optimizer.step()
            
            batch_size = len(batch)
            total_loss += loss.item() * batch_size
            pred = (probs > 0.5).long()
            correct += int((pred == labels.long()).sum().item())
            total += batch_size

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
        
        return EpochMetrics(
            loss=total_loss / total,
            accuracy=correct / total,
            predictions=all_preds,
            labels=all_labels,
            probabilities=all_probs,
        )
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        """Evaluate model. Returns EpochMetrics with predictions."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[float] = []
        
        for batch in loader:
            batch = self._move_batch(batch)
            
            logits = self.model(batch)
            labels = torch.cat([b.label for b in batch]).to(self.device).float()
            logits = logits.view(-1)
            
            # NaN debugging
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                token_names = [b.token_name for b in batch]
                log.warning(
                    f"[EVAL] NaN/Inf in logits! Tokens: {token_names}, "
                    f"logits: {logits.tolist()}"
                )
                continue
            
            loss = self.criterion(logits, labels)
            probs = torch.sigmoid(logits)
            
            batch_size = len(batch)
            total_loss += loss.item() * batch_size
            pred = (probs > 0.5).long()
            correct += int((pred == labels.long()).sum().item())
            total += batch_size
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
        
        return EpochMetrics(
            loss=total_loss / total,
            accuracy=correct / total,
            predictions=all_preds,
            labels=all_labels,
            probabilities=all_probs,
        )
    
    def step_scheduler(self, val_loss: float):
        """Step the learning rate scheduler."""
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def get_lr(self):
        """Get current learning rate."""
        lr: float = self.optimizer.param_groups[0]['lr']
        return lr
    
    def update_best(self, val_loss: float):
        """Update best model if val_loss improves. Returns True if improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = {
                k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            return True
        return False
    
    def restore_best(self):
        """Restore best model state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
    
    def save_checkpoint(self, path: str | Path, epoch: int, metrics: EpochMetrics):
        """Save model checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': {
                'loss': metrics.loss,
                'accuracy': metrics.accuracy,
            },
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, path: str | Path):
        """Load model checkpoint. Returns epoch number."""
        checkpoint: dict = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        epoch: int = checkpoint.get('epoch', 0)
        return epoch
