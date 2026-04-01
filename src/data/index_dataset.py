"""Dataset indexing utilities for the MIMII raw audio layout."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from src.utils.io import PROJECT_ROOT, to_portable_path

INDEX_COLUMNS: Final[list[str]] = [
    "filepath",
    "machine_type",
    "snr_db",
    "machine_id",
    "label",
]
VALID_LABELS: Final[set[str]] = {"normal", "abnormal"}


def _parse_audio_path(audio_path: Path, data_root: Path) -> dict[str, str]:
    """Parse metadata from a WAV path under the expected raw dataset layout."""
    try:
        relative_parts = audio_path.relative_to(data_root).parts
    except ValueError as exc:
        raise ValueError(
            f"File '{audio_path}' is not located under data root '{data_root}'."
        ) from exc

    if len(relative_parts) != 4:
        raise ValueError(
            "Unexpected folder structure for "
            f"'{audio_path}'. Expected "
            "'{snr_db}_{machine_type}/{machine_id}/{label}/*.wav'."
        )

    scope_dir, machine_id, label, filename = relative_parts
    if not filename.lower().endswith(".wav"):
        raise ValueError(f"Expected a WAV file, got '{audio_path}'.")

    if "_" not in scope_dir:
        raise ValueError(
            f"Could not parse '{scope_dir}' into '{{snr_db}}_{{machine_type}}'."
        )

    snr_db, machine_type = scope_dir.rsplit("_", maxsplit=1)
    if not snr_db:
        raise ValueError(f"Missing SNR value in '{scope_dir}'.")
    if not machine_type:
        raise ValueError(f"Missing machine type in '{scope_dir}'.")
    if not machine_id:
        raise ValueError(f"Missing machine_id for '{audio_path}'.")
    if label not in VALID_LABELS:
        raise ValueError(
            f"Invalid label '{label}' for '{audio_path}'. "
            f"Expected one of {sorted(VALID_LABELS)}."
        )

    return {
        "filepath": to_portable_path(audio_path, base_dir=PROJECT_ROOT),
        "machine_type": machine_type,
        "snr_db": snr_db,
        "machine_id": machine_id,
        "label": label,
    }


def build_master_index(data_root: str | Path) -> pd.DataFrame:
    """Scan a raw dataset directory and return a validated master index."""
    root_path = Path(data_root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Data root does not exist: '{root_path}'.")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: '{root_path}'.")

    wav_paths = sorted(path for path in root_path.rglob("*.wav") if path.is_file())
    records = [_parse_audio_path(audio_path=wav_path, data_root=root_path) for wav_path in wav_paths]

    index_df = pd.DataFrame(records, columns=INDEX_COLUMNS)
    if index_df.empty:
        return index_df

    return index_df.sort_values(INDEX_COLUMNS).reset_index(drop=True)


def save_master_index(index_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist a master index CSV to disk."""
    missing_columns = [column for column in INDEX_COLUMNS if column not in index_df.columns]
    if missing_columns:
        raise ValueError(
            "Master index is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    index_df.loc[:, INDEX_COLUMNS].to_csv(output_file, index=False)
    return output_file
