"""PyTorch dataset for loading raw MIMII audio from saved split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Final

import torch
import torchaudio
from torch.utils.data import Dataset

from src.data.split import SPLIT_COLUMNS, VALID_LABELS, VALID_SPLITS, load_split_manifest
from src.utils.io import resolve_path

LABEL_TO_INDEX: Final[dict[str, int]] = {
    "normal": 0,
    "abnormal": 1,
}
EXPECTED_SAMPLE_RATE: Final[int] = 16000


class MIMIIDataset(Dataset[dict[str, Any]]):
    """Load one manifest split of raw MIMII WAV files as PyTorch samples."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.return_metadata = return_metadata

        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {VALID_SPLITS}."
            )

        manifest_df = load_split_manifest(self.manifest_path)
        split_df = manifest_df.loc[manifest_df["split"] == split, SPLIT_COLUMNS].copy()
        if split_df.empty:
            raise ValueError(
                f"No rows found for split='{split}' in manifest '{self.manifest_path}'."
            )

        invalid_labels = sorted(set(split_df["label"]) - VALID_LABELS)
        if invalid_labels:
            raise ValueError(
                "Split manifest contains invalid labels: "
                f"{', '.join(invalid_labels)}."
            )

        self._records = split_df.to_dict(orient="records")
        self._validate_audio_filepaths()

    def __len__(self) -> int:
        """Return the number of rows in the selected split."""
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one WAV file and return a sample dictionary."""
        record = self._records[index]
        audio_path = resolve_path(record["filepath"])

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as exc:  # pragma: no cover - depends on external file state
            raise RuntimeError(
                f"Failed to read audio file '{audio_path}'. The file may be missing or corrupted."
            ) from exc

        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"Unexpected sample rate {sample_rate} for '{audio_path}'. "
                f"Expected {EXPECTED_SAMPLE_RATE} Hz."
            )

        sample: dict[str, Any] = {
            "waveform": waveform.to(dtype=torch.float32),
            "sample_rate": sample_rate,
            "label": LABEL_TO_INDEX[record["label"]],
        }
        if self.return_metadata:
            sample["metadata"] = {
                "filepath": record["filepath"],
                "machine_type": record["machine_type"],
                "machine_id": record["machine_id"],
                "snr_db": record["snr_db"],
                "group_id": record["group_id"],
                "split": record["split"],
            }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _validate_audio_filepaths(self) -> None:
        """Ensure every referenced audio file exists before iteration starts."""
        missing_files = [
            record["filepath"]
            for record in self._records
            if not resolve_path(record["filepath"]).is_file()
        ]
        if missing_files:
            preview = ", ".join(missing_files[:5])
            suffix = " ..." if len(missing_files) > 5 else ""
            raise FileNotFoundError(
                "Split manifest references missing audio files: "
                f"{preview}{suffix}"
            )
