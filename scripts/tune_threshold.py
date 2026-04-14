"""CLI for sweeping decision thresholds on a saved binary-classification checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MIMIIDataset
from src.data.transforms import BaselineLogMelTransform
from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.cnn_baseline import BaselineCNN
from src.training.callbacks import load_checkpoint, save_json
from src.training.losses import build_baseline_loss
from src.training.train import select_device
from src.utils.io import to_portable_path


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for threshold sweeping."""
    parser = argparse.ArgumentParser(
        description="Sweep classification thresholds for a saved checkpoint and summarize the tradeoffs."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to a saved split manifest CSV.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to the saved checkpoint to analyze.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split used to tune the threshold. Validation is recommended.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit thresholds to sweep. If omitted, a default grid is used.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="recall",
        choices=["recall", "f1", "precision", "accuracy"],
        help="Metric used to select the best threshold.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override for threshold-tuning outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "threshold_tuning",
        help="Parent directory for threshold-tuning outputs.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also evaluate the selected threshold on the test split and save the result.",
    )
    return parser


def main() -> None:
    """Sweep thresholds, save a leaderboard, and optionally evaluate the chosen threshold on test."""
    args = build_parser().parse_args()
    thresholds = validate_thresholds(args.thresholds or build_default_thresholds())

    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    run_name = infer_run_name(checkpoint_path, args.run_name)
    output_run_dir = args.output_dir.expanduser().resolve() / run_name
    output_run_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    labels, probabilities, loss = collect_split_outputs(
        manifest_path=args.manifest_path,
        checkpoint_path=checkpoint_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    records = build_threshold_records(
        labels=labels,
        probabilities=probabilities,
        thresholds=thresholds,
        tuned_split=args.split,
        split_loss=loss,
    )
    summary_df = rank_threshold_records(pd.DataFrame(records), metric=args.metric)

    summary_csv_path = output_run_dir / f"{args.split}_threshold_summary.csv"
    summary_json_path = output_run_dir / f"{args.split}_threshold_summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary_df.to_dict(orient="records"), file, indent=2)

    best_record = summary_df.iloc[0].to_dict()
    selection_payload: dict[str, Any] = {
        "run_name": run_name,
        "manifest_path": to_portable_path(args.manifest_path),
        "checkpoint_path": to_portable_path(checkpoint_path),
        "tuned_split": args.split,
        "selection_metric": args.metric,
        "selected_threshold": float(best_record["threshold"]),
        "selected_metrics": {
            "accuracy": float(best_record["accuracy"]),
            "precision": float(best_record["precision"]),
            "recall": float(best_record["recall"]),
            "f1": float(best_record["f1"]),
            "roc_auc": float(best_record["roc_auc"]),
            "loss": float(best_record["loss"]),
            "confusion_matrix": [
                [int(best_record["tn"]), int(best_record["fp"])],
                [int(best_record["fn"]), int(best_record["tp"])],
            ],
        },
        "threshold_summary_csv": to_portable_path(summary_csv_path),
        "threshold_summary_json": to_portable_path(summary_json_path),
    }

    if args.evaluate_test:
        test_labels, test_probabilities, test_loss = collect_split_outputs(
            manifest_path=args.manifest_path,
            checkpoint_path=checkpoint_path,
            split="test",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        test_record = build_threshold_records(
            labels=test_labels,
            probabilities=test_probabilities,
            thresholds=[float(best_record["threshold"])],
            tuned_split="test",
            split_loss=test_loss,
        )[0]
        test_metrics_payload = {
            "run_name": run_name,
            "checkpoint_path": to_portable_path(checkpoint_path),
            "threshold": float(best_record["threshold"]),
            "split": "test",
            "loss": float(test_record["loss"]),
            "accuracy": float(test_record["accuracy"]),
            "precision": float(test_record["precision"]),
            "recall": float(test_record["recall"]),
            "f1": float(test_record["f1"]),
            "roc_auc": float(test_record["roc_auc"]),
            "confusion_matrix": [
                [int(test_record["tn"]), int(test_record["fp"])],
                [int(test_record["fn"]), int(test_record["tp"])],
            ],
        }
        test_metrics_path = output_run_dir / "selected_threshold_test_metrics.json"
        save_json(test_metrics_payload, test_metrics_path)
        selection_payload["test_metrics_path"] = to_portable_path(test_metrics_path)
        selection_payload["test_metrics"] = test_metrics_payload

    selection_path = output_run_dir / f"{args.split}_selected_threshold.json"
    save_json(selection_payload, selection_path)

    print(f"Threshold tuning complete for run '{run_name}'.")
    print(f"Summary CSV:      {summary_csv_path}")
    print(f"Summary JSON:     {summary_json_path}")
    print(f"Selection record: {selection_path}")
    print(
        f"Best threshold on {args.split}: {float(best_record['threshold']):.3f} | "
        f"{args.metric}={float(best_record[args.metric]):.4f} | "
        f"precision={float(best_record['precision']):.4f} | "
        f"recall={float(best_record['recall']):.4f} | "
        f"f1={float(best_record['f1']):.4f}"
    )
    if args.evaluate_test:
        print("Selected threshold was also evaluated on the test split.")


def collect_split_outputs(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[int], list[float], float]:
    """Run inference on one split and return labels, probabilities, and average loss."""
    transform = BaselineLogMelTransform()
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

    model = BaselineCNN().to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = build_baseline_loss()

    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels: list[int] = []
    probabilities: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device=device, dtype=torch.float32)
            batch_labels = batch["label"].to(device=device, dtype=torch.float32)

            logits = model(inputs)
            loss = criterion(logits, batch_labels)
            batch_probabilities = torch.sigmoid(logits)

            batch_size_actual = inputs.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

            labels.extend(batch_labels.detach().cpu().to(dtype=torch.int64).tolist())
            probabilities.extend(batch_probabilities.detach().cpu().tolist())

    if total_samples == 0:
        raise ValueError(f"Split '{split}' produced zero samples during threshold tuning.")

    return labels, probabilities, total_loss / total_samples


def build_threshold_records(
    labels: list[int],
    probabilities: list[float],
    thresholds: list[float],
    tuned_split: str,
    split_loss: float,
) -> list[dict[str, Any]]:
    """Compute metrics for every candidate threshold."""
    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        predictions = [1 if probability >= threshold else 0 for probability in probabilities]
        metrics = compute_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
        )
        confusion_matrix = metrics["confusion_matrix"]
        records.append(
            {
                "split": tuned_split,
                "threshold": float(threshold),
                "loss": float(split_loss),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "roc_auc": float(metrics.get("roc_auc", float("nan"))),
                "tn": int(confusion_matrix[0][0]),
                "fp": int(confusion_matrix[0][1]),
                "fn": int(confusion_matrix[1][0]),
                "tp": int(confusion_matrix[1][1]),
            }
        )
    return records


def rank_threshold_records(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sort thresholds by the selected metric and stable tie-breakers."""
    return summary_df.sort_values(
        by=[metric, "f1", "precision", "accuracy", "threshold"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def build_default_thresholds() -> list[float]:
    """Return a compact default threshold grid."""
    return [round(index * 0.05, 2) for index in range(2, 19)]


def validate_thresholds(thresholds: list[float]) -> list[float]:
    """Ensure all thresholds lie strictly between 0 and 1."""
    unique_thresholds = sorted(set(float(value) for value in thresholds))
    if not unique_thresholds:
        raise ValueError("Threshold list must not be empty.")
    for threshold in unique_thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Thresholds must lie strictly between 0 and 1, got {threshold}.")
    return unique_thresholds


def infer_run_name(checkpoint_path: Path, run_name: str | None) -> str:
    """Infer run folder name from an explicit value or checkpoint parent folder."""
    if run_name is not None:
        return run_name
    if checkpoint_path.name in {"best.pt", "last.pt"} and checkpoint_path.parent.name:
        return checkpoint_path.parent.name
    return checkpoint_path.stem


if __name__ == "__main__":
    main()
