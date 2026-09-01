"""
MentalScope — Mental Health Text Classification
Custom loss functions for class imbalance handling.

Implements:
  1. WeightedCrossEntropyLoss  — inverse frequency weighting
  2. FocalLoss                 — dynamically downweights easy examples
  3. FocalLossWithLabelSmoothing — Focal Loss + label smoothing regularization

These are standalone nn.Module classes, injected into a custom HuggingFace
Trainer via the compute_loss() override.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy with static inverse-frequency class weights.

    Args:
        class_weights: Tensor of shape (num_classes,). Use
            `compute_class_weights()` from preprocessing.py.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) raw model outputs.
            labels: (batch_size,) integer class indices.
        """
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return F.cross_entropy(logits, labels, weight=weight)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for multi-class classification.

    Dynamically down-weights well-classified samples, forcing the model
    to focus training on hard, misclassified examples — critical for
    rare mental health categories like Bipolar or Personality Disorder.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher gamma → more focus on hard samples.
               0.0 = standard CE. Typical range: [0.5, 5]. Default: 2.0
        alpha: Per-class weighting tensor (shape: num_classes). Can be
               inverse-frequency weights. If None, all classes equal.
        reduction: 'mean' (default) | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) raw model outputs.
            labels: (batch_size,) integer class indices.
        """
        # Compute softmax probabilities and log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)           # (B, C)
        probs = torch.exp(log_probs)                         # (B, C)

        # Gather the probability for the true class of each sample
        log_pt = log_probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)  # (B,)
        pt = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)          # (B,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt).pow(self.gamma)              # (B,)

        # Alpha weighting (class-specific)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, labels)               # (B,)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt                        # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Combines Focal Loss with Label Smoothing.

    Label smoothing (Szegedy et al., 2016) prevents overconfident predictions
    by mixing the true label distribution with a uniform distribution.
    Combined with Focal Loss, this improves calibration on mental health data
    where class boundaries can be ambiguous (e.g., Depression vs. Stress).

    Args:
        gamma: Focal Loss focusing parameter. Default: 2.0
        smoothing: Label smoothing factor in [0, 1). Default: 0.1
        alpha: Per-class weighting tensor (optional).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        alpha: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        # Smooth targets: (1 - ε) * one_hot + ε / C
        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(self.smoothing / num_classes)
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # p_t from the true (non-smoothed) class for the focal weight
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, labels)
            focal_weight = alpha_t * focal_weight

        # Cross-entropy with smooth labels
        ce_loss = -(smooth_labels * log_probs).sum(dim=-1)  # (B,)
        loss = focal_weight * ce_loss

        return loss.mean()


# ── Loss factory ─────────────────────────────────────────────────────────────

def get_loss_fn(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    smoothing: float = 0.1,
) -> nn.Module:
    """
    Factory function for loss selection via config.

    Args:
        loss_type: One of 'ce', 'weighted_ce', 'focal', 'focal_smooth'
        class_weights: Class weight tensor for 'weighted_ce' and 'focal'
        gamma: Focal Loss gamma parameter
        smoothing: Label smoothing epsilon

    Returns:
        Configured loss nn.Module.

    Raises:
        ValueError: If loss_type is unrecognized.
    """
    loss_type = loss_type.lower()

    if loss_type == "ce":
        return WeightedCrossEntropyLoss(class_weights=None)

    elif loss_type == "weighted_ce":
        return WeightedCrossEntropyLoss(class_weights=class_weights)

    elif loss_type == "focal":
        return FocalLoss(gamma=gamma, alpha=class_weights)

    elif loss_type == "focal_smooth":
        return FocalLossWithLabelSmoothing(gamma=gamma, smoothing=smoothing, alpha=class_weights)

    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. "
            "Choose from: 'ce', 'weighted_ce', 'focal', 'focal_smooth'"
        )
