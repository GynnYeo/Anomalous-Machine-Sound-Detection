"""Loss helpers for the baseline binary classification pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_baseline_loss(
    pos_weight: float | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Return the baseline loss for one-logit binary classification."""
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    if pos_weight <= 0:
        raise ValueError(f"pos_weight must be positive, got {pos_weight}.")

    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
