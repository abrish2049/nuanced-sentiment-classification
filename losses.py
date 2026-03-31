"""
Custom loss functions for the sentiment pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss — down-weights easy examples to focus training on hard ones.

    Parameters
    ----------
    gamma : float
        Focusing parameter. 0 = standard cross-entropy. 2 is the value
        used in the original paper and works well in practice.
    weight : torch.Tensor or None
        Per-class weights (same tensor you pass to CrossEntropyLoss).
        Can be combined with focal loss for doubly-targeted imbalance handling.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        # compute standard CE first (per sample, unreduced)
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()