"""Model factory helpers for selecting available baseline architectures."""

from __future__ import annotations

from typing import Final

from torch import nn

from src.models.cnn_baseline import BaselineCNN
from src.models.cnn_deeper import DeeperCNN
from src.models.cnn_wider import WiderCNN

AVAILABLE_MODELS: Final[dict[str, type[nn.Module]]] = {
    "baseline_cnn": BaselineCNN,
    "deeper_cnn": DeeperCNN,
    "wider_cnn": WiderCNN,
}


def build_model(model_name: str) -> nn.Module:
    """Instantiate a known model by name."""
    try:
        model_class = AVAILABLE_MODELS[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(AVAILABLE_MODELS))
        raise ValueError(
            f"Unknown model_name '{model_name}'. Available models: {available}."
        ) from exc

    return model_class()
