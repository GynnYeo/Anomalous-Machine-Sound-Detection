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

    best_f1_threshold = _find_best_f1_threshold(val_labels, val_errors)
    p95_threshold = _find_normal_percentile_threshold(
        val_labels,
        val_errors,
        percentile=95,
    )
    p99_threshold = _find_normal_percentile_threshold(
        val_labels,
        val_errors,
        percentile=99,
    )

    test_labels, test_errors = _collect_reconstruction_errors(
        model=model,
        manifest_path=manifest_path,
        split="test",
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    threshold_results: dict[str, dict[str, Any]] = {}

    for threshold_name, candidate_threshold in {
        "validation_best_f1": best_f1_threshold,
        "normal_val_p95": p95_threshold,
        "normal_val_p99": p99_threshold,
    }.items():
        predictions = [int(error >= candidate_threshold) for error in test_errors]
        threshold_metrics = compute_binary_classification_metrics(
            labels=test_labels,
            predictions=predictions,
            probabilities=test_errors,
        )

        threshold_result: dict[str, Any] = {
            "threshold": float(candidate_threshold),
            "accuracy": threshold_metrics["accuracy"],
            "precision": threshold_metrics["precision"],
            "recall": threshold_metrics["recall"],
            "f1": threshold_metrics["f1"],
            "confusion_matrix": threshold_metrics["confusion_matrix"],
        }
        if "roc_auc" in threshold_metrics:
            threshold_result["roc_auc"] = threshold_metrics["roc_auc"]

        threshold_results[threshold_name] = threshold_result

    selected_threshold = best_f1_threshold
    selected_metrics = threshold_results["validation_best_f1"]

    val_error_summary = _summarize_errors_by_class(val_labels, val_errors)
    test_error_summary = _summarize_errors_by_class(test_labels, test_errors)

    error_records = {
        "validation": _build_error_records(val_labels, val_errors),
        "test": _build_error_records(test_labels, test_errors),
    }


    results: dict[str, Any] = {
        "run_name": run_name,
        "manifest_path": to_portable_path(manifest_path),
        "checkpoint_path": to_portable_path(checkpoint_path),
        "threshold_source": "validation_best_f1",
        "threshold": float(selected_threshold),
        "split": "test",
        "num_samples": len(test_labels),
        "accuracy": selected_metrics["accuracy"],
        "precision": selected_metrics["precision"],
        "recall": selected_metrics["recall"],
        "f1": selected_metrics["f1"],
        "confusion_matrix": selected_metrics["confusion_matrix"],
        "validation_error_summary": val_error_summary,
        "test_error_summary": test_error_summary,
        "threshold_results": threshold_results,
    }

    if "roc_auc" in selected_metrics:
        results["roc_auc"] = selected_metrics["roc_auc"]


    metrics_run_dir = Path(metrics_dir).expanduser().resolve() / run_name
    metrics_path = metrics_run_dir / "test_metrics.json"
    errors_path = metrics_run_dir / "reconstruction_errors.json"

    results["metrics_path"] = to_portable_path(metrics_path)
    results["errors_path"] = to_portable_path(errors_path)

    save_json(results, metrics_path)
    save_json(error_records, errors_path)

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
    if "validation_error_summary" in results:
        print("\nValidation reconstruction error summary:")
        _print_error_summary(results["validation_error_summary"])

    if "test_error_summary" in results:
        print("\nTest reconstruction error summary:")
        _print_error_summary(results["test_error_summary"])


def _build_error_records(
    labels: list[int],
    errors: list[float],
) -> list[dict[str, float | int | str]]:
    if len(labels) != len(errors):
        raise ValueError("labels and errors must have the same length.")

    records = []
    for label, error in zip(labels, errors):
        records.append(
            {
                "label": int(label),
                "label_name": "abnormal" if int(label) == 1 else "normal",
                "reconstruction_error": float(error),
            }
        )
    return records


def _summarize_errors_by_class(
    labels: list[int],
    errors: list[float],
) -> dict[str, dict[str, float | int]]:
    if len(labels) != len(errors):
        raise ValueError("labels and errors must have the same length.")
    if not labels:
        raise ValueError("Cannot summarize empty error list.")

    normal_errors = [float(error) for label, error in zip(labels, errors) if int(label) == 0]
    abnormal_errors = [float(error) for label, error in zip(labels, errors) if int(label) == 1]

    return {
        "normal": _summarize_one_class(normal_errors),
        "abnormal": _summarize_one_class(abnormal_errors),
    }


def _summarize_one_class(errors: list[float]) -> dict[str, float | int]:
    if not errors:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }

    sorted_errors = sorted(errors)

    return {
        "count": len(sorted_errors),
        "mean": float(sum(sorted_errors) / len(sorted_errors)),
        "median": float(_percentile(sorted_errors, 50)),
        "min": float(sorted_errors[0]),
        "max": float(sorted_errors[-1]),
        "p05": float(_percentile(sorted_errors, 5)),
        "p95": float(_percentile(sorted_errors, 95)),
        "p99": float(_percentile(sorted_errors, 99)),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list.")

    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]

    position = (len(sorted_values) - 1) * percentile / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index

    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def _print_error_summary(summary: dict[str, dict[str, float | int]]) -> None:
    for class_name, values in summary.items():
        print(
            f"{class_name}: "
            f"count={values['count']} | "
            f"mean={values['mean']:.4f} | "
            f"median={values['median']:.4f} | "
            f"p95={values['p95']:.4f} | "
            f"p99={values['p99']:.4f}"
        )


def _find_normal_percentile_threshold(
    labels: list[int],
    errors: list[float],
    percentile: float,
) -> float:
    if len(labels) != len(errors):
        raise ValueError("labels and errors must have the same length.")

    normal_errors = sorted(
        float(error)
        for label, error in zip(labels, errors)
        if int(label) == 0
    )

    if not normal_errors:
        raise ValueError("No normal validation errors available for threshold selection.")

    return float(_percentile(normal_errors, percentile))