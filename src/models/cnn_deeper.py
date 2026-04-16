"""Deeper CNN variant for binary classification on log-mel spectrogram inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeeperCNN(nn.Module):
    """A deeper 4-block CNN variant based on the baseline architecture.

    Expected input shape:
        [batch_size, 1, 64, 313]

    Output shape:
        [batch_size]
        One logit per sample for binary classification.
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.dropout1 = nn.Dropout2d(0.10)
        self.dropout2 = nn.Dropout2d(0.15)
        self.dropout3 = nn.Dropout2d(0.20)
        self.dropout4 = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.35)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per sample."""
        if x.ndim != 4:
            raise ValueError(
                "Expected input with shape [batch_size, channels, n_mels, time], "
                f"got shape {tuple(x.shape)}."
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"Expected input with 1 channel, got {x.shape[1]} channels."
            )

        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = self.pool(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        x = x.squeeze(dim=1)

        return x
