"""Reusable baseline evaluation pipeline for the saved test split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MIMIIDataset
from src.data.transforms import BaselineLogMelTransform
from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.registry import build_model
from src.training.callbacks import load_checkpoint, save_json
from src.training.losses import build_baseline_loss
from src.training.train import select_device
from src.utils.io import to_portable_path


def run_evaluation(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    run_name: str | None = None,
    metrics_dir: str | Path = "artifacts/metrics",
    model_name: str = "baseline_cnn",
) -> dict[str, Any]:
    """Evaluate the best checkpoint on the saved test split and persist metrics."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")

    device = select_device()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    resolved_run_name = _infer_run_name(checkpoint_file, run_name=run_name)
    metrics_root = Path(metrics_dir).expanduser().resolve()
    metrics_run_dir = metrics_root / resolved_run_name
    metrics_path = metrics_run_dir / "test_metrics.json"

    transform = BaselineLogMelTransform()
    test_dataset = MIMIIDataset(
        manifest_path=manifest_path,
        split="test",
        transform=transform,
        return_metadata=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_name).to(device)
    criterion = build_baseline_loss()

    checkpoint = load_checkpoint(checkpoint_file, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{checkpoint_file}' is missing required key 'model_state_dict'."
        )
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    all_predictions: list[int] = []
    all_logits: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            if "input" not in batch or "label" not in batch:
                raise KeyError("Evaluation batch must contain 'input' and 'label'.")

            inputs = batch["input"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.float32)

            logits = model(inputs)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).to(dtype=torch.int64)

            batch_size_actual = inputs.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

            all_logits.extend(logits.detach().cpu().tolist())
            all_probabilities.extend(probabilities.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().to(dtype=torch.int64).tolist())

    if total_samples == 0:
        raise ValueError("Test dataloader produced zero samples.")

    metrics = compute_binary_classification_metrics(
        labels=all_labels,
        predictions=all_predictions,
        probabilities=all_probabilities,
    )
    results: dict[str, Any] = {
        "run_name": resolved_run_name,
        "manifest_path": to_portable_path(manifest_path),
        "checkpoint_path": to_portable_path(checkpoint_file),
        "split": "test",
        "device": device.type,
        "num_samples": total_samples,
        "model_name": model_name,
        "loss": total_loss / total_samples,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "metrics_path": to_portable_path(metrics_path),
    }
    if "roc_auc" in metrics:
        results["roc_auc"] = metrics["roc_auc"]

    save_json(results, metrics_path)
    _print_summary(results)
    return results


def _infer_run_name(checkpoint_path: Path, run_name: str | None) -> str:
    """Infer the run folder name from the checkpoint path when not provided."""
    if run_name is not None:
        return run_name

    parent_name = checkpoint_path.parent.name
    if checkpoint_path.name in {"best.pt", "last.pt"} and parent_name:
        return parent_name
    return checkpoint_path.stem


def _print_summary(results: dict[str, Any]) -> None:
    """Print a short human-readable evaluation summary."""
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Split: {results['split']}")
    print(f"Test samples: {results['num_samples']}")
    print(f"Test loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    if "roc_auc" in results:
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
