"""Plotting helpers for training, evaluation, and experiment-summary artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_loss_curves(history: dict[str, Any], output_path: str | Path) -> Path:
    """Plot training and validation loss curves over epochs."""
    epoch_records = _extract_epoch_records(history)
    epochs = [record["epoch"] for record in epoch_records]
    train_loss = [record["train_loss"] for record in epoch_records]
    val_loss = [record["val_loss"] for record in epoch_records]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_loss, marker="o", label="train_loss")
    axis.plot(epochs, val_loss, marker="o", label="val_loss")
    axis.set_title("Training and Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_validation_accuracy(history: dict[str, Any], output_path: str | Path) -> Path:
    """Plot validation accuracy over epochs."""
    epoch_records = _extract_epoch_records(history)
    epochs = [record["epoch"] for record in epoch_records]
    val_accuracy = [record["val_accuracy"] for record in epoch_records]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, val_accuracy, marker="o")
    axis.set_title("Validation Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_confusion_matrix(
    confusion_matrix: list[list[int]],
    output_path: str | Path,
) -> Path:
    """Plot a simple 2x2 confusion matrix image for normal vs abnormal."""
    _validate_confusion_matrix(confusion_matrix)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion_matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    class_labels = ["normal", "abnormal"]
    axis.set_title("Test Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks([0, 1], class_labels)
    axis.set_yticks([0, 1], class_labels)

    max_value = max(max(row) for row in confusion_matrix)
    text_threshold = max_value / 2.0 if max_value > 0 else 0.0
    for row_index, row in enumerate(confusion_matrix):
        for col_index, value in enumerate(row):
            text_color = "white" if value > text_threshold else "black"
            axis.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_grid_search_metric(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    metric: str = "recall",
    top_k: int = 8,
) -> Path:
    """Plot a horizontal bar chart comparing the top grid-search runs by one metric."""
    if not rows:
        raise ValueError("Grid-search summary must contain at least one row.")

    filtered_rows = [row for row in rows if metric in row]
    if not filtered_rows:
        raise KeyError(f"Grid-search summary rows do not contain metric '{metric}'.")

    ranked_rows = sorted(
        filtered_rows,
        key=lambda row: float(row[metric]),
        reverse=True,
    )[:top_k]
    ranked_rows.reverse()

    labels = [row.get("run_name", f"run_{index}") for index, row in enumerate(ranked_rows)]
    values = [float(row[metric]) for row in ranked_rows]

    figure_height = max(4.5, 0.6 * len(ranked_rows) + 1.5)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    bars = axis.barh(labels, values, color="steelblue")
    axis.set_title(f"Top Grid-Search Runs by {metric}")
    axis.set_xlabel(metric)
    axis.set_xlim(0.0, min(1.05, max(values) + 0.05))
    axis.grid(True, axis="x", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        axis.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )

    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_threshold_tradeoff(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    selected_threshold: float | None = None,
) -> Path:
    """Plot precision, recall, and F1 across a threshold sweep."""
    if not rows:
        raise ValueError("Threshold summary must contain at least one row.")

    required_keys = {"threshold", "precision", "recall", "f1"}
    for index, row in enumerate(rows):
        missing_keys = required_keys.difference(row)
        if missing_keys:
            raise ValueError(
                f"Threshold summary row {index} is missing required keys: "
                f"{', '.join(sorted(missing_keys))}."
            )

    sorted_rows = sorted(rows, key=lambda row: float(row["threshold"]))
    thresholds = [float(row["threshold"]) for row in sorted_rows]
    precision = [float(row["precision"]) for row in sorted_rows]
    recall = [float(row["recall"]) for row in sorted_rows]
    f1 = [float(row["f1"]) for row in sorted_rows]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(thresholds, precision, marker="o", label="precision")
    axis.plot(thresholds, recall, marker="o", label="recall")
    axis.plot(thresholds, f1, marker="o", label="f1")

    if selected_threshold is not None:
        axis.axvline(
            selected_threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"selected threshold = {selected_threshold:.2f}",
        )

    axis.set_title("Threshold Sweep Trade-Off")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    return _save_figure(figure, output_path)


def load_json(path: str | Path) -> Any:
    """Load a JSON value from disk."""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: '{resolved_path}'.")
    with resolved_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _extract_epoch_records(history: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated epoch records from a saved history dictionary."""
    epoch_records = history.get("epochs")
    if not isinstance(epoch_records, list) or not epoch_records:
        raise ValueError("History must contain a non-empty 'epochs' list.")

    required_keys = {"epoch", "train_loss", "val_loss", "val_accuracy"}
    for index, record in enumerate(epoch_records):
        if not isinstance(record, dict):
            raise ValueError(f"Epoch record at index {index} is not a dictionary.")
        missing_keys = required_keys.difference(record)
        if missing_keys:
            raise ValueError(
                "Epoch record is missing required keys: "
                f"{', '.join(sorted(missing_keys))}."
            )
    return epoch_records


def _validate_confusion_matrix(confusion_matrix: list[list[int]]) -> None:
    """Ensure confusion matrix matches the expected 2x2 layout."""
    if len(confusion_matrix) != 2 or any(len(row) != 2 for row in confusion_matrix):
        raise ValueError(
            "Confusion matrix must have shape 2x2 in the form [[tn, fp], [fn, tp]]."
        )


def _save_figure(figure: plt.Figure, output_path: str | Path) -> Path:
    """Persist a matplotlib figure to disk and close it."""
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return resolved_path
