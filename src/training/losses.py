"""Loss helpers for the baseline binary classification pipeline."""

from __future__ import annotations

import torch.nn as nn


def build_baseline_loss() -> nn.Module:
    """Return the baseline loss for one-logit binary classification."""
    return nn.BCEWithLogitsLoss()
