"""Evaluate a trained autoencoder using reconstruction error."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MIMIIDataset
from src.data.transforms import BaselineLogMelTransform
from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.conv_autoencoder import ConvAutoencoder
from src.training.callbacks import load_checkpoint, save_json
from src.training.train import select_device
from src.utils.io import to_portable_path


def run_autoencoder_evaluation(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    run_name: str = "autoencoder",
    metrics_dir: str | Path = "artifacts/metrics",
) -> dict[str, Any]:
    device = select_device()
    transform = BaselineLogMelTransform()

    model = ConvAutoencoder().to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_labels, val_errors = _collect_reconstruction_errors(
        model=model,
        manifest_path=manifest_path,
        split="val",
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    threshold = _find_best_f1_threshold(val_labels, val_errors)

    test_labels, test_errors = _collect_reconstruction_errors(
        model=model,
        manifest_path=manifest_path,
        split="test",
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    test_predictions = [int(error >= threshold) for error in test_errors]

    metrics = compute_binary_classification_metrics(
        labels=test_labels,
        predictions=test_predictions,
        probabilities=test_errors,
    )

    results: dict[str, Any] = {
        "run_name": run_name,
        "manifest_path": to_portable_path(manifest_path),
        "checkpoint_path": to_portable_path(checkpoint_path),
        "threshold_source": "validation_best_f1",
        "threshold": float(threshold),
        "split": "test",
        "num_samples": len(test_labels),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    if "roc_auc" in metrics:
        results["roc_auc"] = metrics["roc_auc"]

    metrics_path = Path(metrics_dir).expanduser().resolve() / run_name / "test_metrics.json"
    results["metrics_path"] = to_portable_path(metrics_path)
    save_json(results, metrics_path)

    _print_summary(results)
    return results


def _collect_reconstruction_errors(
    model: torch.nn.Module,
    manifest_path: str | Path,
    split: str,
    transform: BaselineLogMelTransform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    dataset = MIMIIDataset(
        manifest_path=manifest_path,
        split=split,
        transform=transform,
        return_metadata=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    all_labels: list[int] = []
    all_errors: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device=device, dtype=torch.float32)
            labels = batch["label"]

            reconstructions = model(inputs)

            errors = torch.mean(
                (inputs - reconstructions) ** 2,
                dim=(1, 2, 3),
            )

            all_errors.extend(errors.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().to(dtype=torch.int64).tolist())

    return all_labels, all_errors


def _find_best_f1_threshold(
    labels: list[int],
    errors: list[float],
) -> float:
    if len(labels) != len(errors):
        raise ValueError("labels and errors must have the same length.")
    if not labels:
        raise ValueError("Cannot choose threshold from empty validation results.")

    candidate_thresholds = sorted(set(errors))

    best_threshold = candidate_thresholds[0]
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        predictions = [int(error >= threshold) for error in errors]
        metrics = compute_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
        f1 = float(metrics["f1"])

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold)


def _print_summary(results: dict[str, Any]) -> None:
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Threshold: {results['threshold']:.6f}")
    print(f"Test samples: {results['num_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    if "roc_auc" in results:
        print(f"ROC-AUC: {results['roc_auc']:.4f}")