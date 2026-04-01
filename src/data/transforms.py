"""Dataset-facing deterministic transforms for baseline preprocessing."""

from __future__ import annotations

from typing import Any

import torch

from src.features.spectrogram import DEFAULT_SAMPLE_RATE, LogMelSpectrogramExtractor
from src.features.waveform import average_channels_to_mono


class BaselineLogMelTransform:
    """Convert raw dataset samples into baseline log-mel model inputs."""

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
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.feature_extractor = LogMelSpectrogramExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Transform a raw dataset sample into a model-ready input dictionary."""
        waveform = sample.get("waveform")
        sample_rate = sample.get("sample_rate")
        label = sample.get("label")

        if waveform is None:
            raise KeyError("Sample is missing required key 'waveform'.")
        if sample_rate is None:
            raise KeyError("Sample is missing required key 'sample_rate'.")
        if label is None:
            raise KeyError("Sample is missing required key 'label'.")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Unexpected sample rate {sample_rate}. Expected {self.sample_rate} Hz."
            )

        mono_waveform = average_channels_to_mono(waveform)
        log_mel = self.feature_extractor(mono_waveform, sample_rate=sample_rate)
        model_input = log_mel.unsqueeze(0)
        if model_input.ndim != 3:
            raise ValueError(
                "Expected model input with shape [1, n_mels, time], "
                f"got shape {tuple(model_input.shape)}."
            )

        transformed_sample: dict[str, Any] = {
            "input": model_input.to(dtype=torch.float32),
            "label": label,
        }
        if "metadata" in sample:
            transformed_sample["metadata"] = sample["metadata"]

        return transformed_sample
