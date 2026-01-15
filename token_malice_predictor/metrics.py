"""Evaluation Metrics - Classification metrics computation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


class MetricsError(Exception):
    """Raised when metrics computation fails."""
    pass


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_prob: np.ndarray | list | None = None,
):
    """Compute classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        raise MetricsError("Empty y_true array")
    if len(y_true) != len(y_pred):
        raise MetricsError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    metrics: dict[str, float] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if len(y_prob) != len(y_true):
            raise MetricsError(
                f"Length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}"
            )
        
        unique_true = np.unique(y_true)
        if len(unique_true) == 2:
            metrics['auroc'] = roc_auc_score(y_true, y_prob)
            metrics['auprc'] = average_precision_score(y_true, y_prob)
        else:
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
    
    return metrics


def format_metrics(metrics: dict[str, float]):
    """Format metrics dict as readable string."""
    lines: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    result: str = "\n".join(lines)
    return result
