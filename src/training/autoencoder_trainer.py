from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_autoencoder_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        inputs = batch["input"].to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, inputs)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader produced zero samples.")

    return {"loss": running_loss / total_samples}


def validate_autoencoder_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device=device, dtype=torch.float32)

            reconstructions = model(inputs)
            loss = criterion(reconstructions, inputs)

            batch_size = inputs.shape[0]
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation dataloader produced zero samples.")

    return {"loss": running_loss / total_samples}