"""Plot reconstruction error distributions for autoencoder evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_reconstruction_error_distribution(
    errors_json_path: str | Path,
    output_path: str | Path,
    split: str = "test",
) -> Path:
    errors_json_path = Path(errors_json_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    with errors_json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if split not in payload:
        raise KeyError(f"Split '{split}' not found in {errors_json_path}.")

    records = payload[split]
    normal_errors = [
        record["reconstruction_error"]
        for record in records
        if record["label"] == 0
    ]
    abnormal_errors = [
        record["reconstruction_error"]
        for record in records
        if record["label"] == 1
    ]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(normal_errors, bins=80, alpha=0.6, label="normal")
    axis.hist(abnormal_errors, bins=80, alpha=0.6, label="abnormal")
    axis.set_title(f"Autoencoder Reconstruction Error Distribution ({split})")
    axis.set_xlabel("Reconstruction Error")
    axis.set_ylabel("Count")
    axis.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return output_path