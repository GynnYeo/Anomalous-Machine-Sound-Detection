"""Convolutional autoencoder for log-mel spectrogram reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """Reconstruct log-mel inputs with shape [B, 1, 64, 313]."""

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, 1, 64, 313], got {tuple(x.shape)}")

        original_size = x.shape[-2:]

        z = self.encoder(x)
        reconstruction = self.decoder(z)

        reconstruction = F.interpolate(
            reconstruction,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )

        return reconstruction