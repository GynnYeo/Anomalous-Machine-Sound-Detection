"""Reusable spectrogram feature extraction for the baseline pipeline."""

from __future__ import annotations

from typing import Final

import torch
import torchaudio
from torch import nn

DEFAULT_SAMPLE_RATE: Final[int] = 16000


class LogMelSpectrogramExtractor(nn.Module):
    """Convert a mono waveform into a deterministic log-mel spectrogram."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # The mel spectrogram transform computes the power spectrogram and applies the mel filterbank.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

        # The amplitude-to-decibel transform converts the power spectrogram to a log-mel spectrogram.
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=None,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """Return a log-mel spectrogram with shape [n_mels, time]."""
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Unexpected sample rate {sample_rate}. Expected {self.sample_rate} Hz."
            )
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(
                f"waveform must be a torch.Tensor, got {type(waveform).__name__}."
            )
        if waveform.ndim != 1:
            raise ValueError(
                "Expected mono waveform with shape [num_samples], "
                f"got shape {tuple(waveform.shape)}."
            )
        if waveform.shape[0] < 1:
            raise ValueError("Waveform must contain at least one sample.")

        mel_spectrogram = self.mel_transform(waveform)
        log_mel = self.db_transform(mel_spectrogram)
        if log_mel.ndim != 2:
            raise ValueError(
                "Expected log-mel spectrogram with shape [n_mels, time], "
                f"got shape {tuple(log_mel.shape)}."
            )
        return log_mel.to(dtype=torch.float32)
