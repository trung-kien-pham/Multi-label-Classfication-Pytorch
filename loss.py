import torch
import torch.nn as nn


def BCEWithLogitsLoss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    """
    BCEWithLogitsLoss for multi-label classification.

    Args:
        pos_weight: optional tensor of shape [num_classes]
    """
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return nn.BCEWithLogitsLoss()