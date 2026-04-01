"""Checkpoint and history helpers for baseline training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(checkpoint: dict[str, Any], output_path: str | Path) -> Path:
    """Save a PyTorch checkpoint dictionary to disk."""
    checkpoint_path = Path(output_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a PyTorch checkpoint dictionary from disk."""
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: '{resolved_path}'.")
    return torch.load(resolved_path, map_location=map_location)


def save_history(history: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a training history dictionary as JSON."""
    history_path = Path(output_path).expanduser().resolve()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    return history_path


def load_history(history_path: str | Path) -> dict[str, Any]:
    """Load a saved training history JSON file."""
    resolved_path = Path(history_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"History file does not exist: '{resolved_path}'.")
    with resolved_path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    if not isinstance(history, dict):
        raise ValueError(f"History file '{resolved_path}' did not contain a JSON object.")
    return history
