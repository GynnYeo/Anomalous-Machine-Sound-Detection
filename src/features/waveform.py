"""Waveform utilities for deterministic baseline preprocessing."""

from __future__ import annotations

import torch


def average_channels_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Average a multichannel waveform across channels to produce mono audio."""
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(
            f"waveform must be a torch.Tensor, got {type(waveform).__name__}."
        )
    if waveform.ndim != 2:
        raise ValueError(
            "Expected waveform with shape [channels, num_samples], "
            f"got shape {tuple(waveform.shape)}."
        )
    if waveform.shape[0] < 1:
        raise ValueError("Waveform must contain at least one channel.")
    if waveform.shape[1] < 1:
        raise ValueError("Waveform must contain at least one sample.")

    return waveform.mean(dim=0)
