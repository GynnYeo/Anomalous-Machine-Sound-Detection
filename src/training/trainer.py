"""Reusable epoch-level training and validation routines."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch and return aggregate loss."""
    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        inputs, labels = _prepare_batch(batch, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader produced zero samples.")

    return {
        "loss": running_loss / total_samples,
    }


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation epoch and return aggregate loss and accuracy."""
    model.eval()

    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = _prepare_batch(batch, device=device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).to(dtype=torch.float32)

            batch_size = inputs.shape[0]
            running_loss += loss.item() * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation dataloader produced zero samples.")

    return {
        "loss": running_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def _prepare_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a collated batch to device and cast labels for BCEWithLogitsLoss."""
    if "input" not in batch:
        raise KeyError("Batch is missing required key 'input'.")
    if "label" not in batch:
        raise KeyError("Batch is missing required key 'label'.")

    inputs = batch["input"].to(device=device, dtype=torch.float32)
    labels = batch["label"].to(device=device, dtype=torch.float32)
    return inputs, labels
