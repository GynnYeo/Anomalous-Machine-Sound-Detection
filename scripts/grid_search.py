"""CLI entry point for small hyperparameter grid searches."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import run_evaluation
from src.training.train import run_training


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for a small grid search over training settings."""
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter grid search using the existing training and evaluation pipeline."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to a saved split manifest CSV.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="grid",
        help="Prefix used when naming each run in the search.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3],
        help="One or more learning rates to include in the grid.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="One or more batch sizes to include in the grid.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train each run.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for each training run.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints",
        help="Parent directory for run-specific checkpoint folders.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "metrics",
        help="Parent directory for run-specific metric folders.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "grid_search",
        help="Directory where the overall grid-search summary files will be written.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="f1",
        choices=["f1", "recall", "accuracy", "roc_auc", "best_val_loss"],
        help="Metric used to rank runs in the final summary.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if its test_metrics.json already exists.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Optional early stopping patience applied to each training run.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--pos-weights",
        type=str,
        nargs="+",
        default=["none"],
        help="Positive-class weights to include in the grid. Use numeric values or 'auto' or 'none'.",
    )
    return parser


def main() -> None:
    """Run all hyperparameter combinations, then persist a summary leaderboard."""
    args = build_parser().parse_args()

    pos_weight_specs = [parse_pos_weight_spec(value) for value in args.pos_weights]
    combinations = list(itertools.product(args.learning_rates, args.batch_sizes, pos_weight_specs))
    if not combinations:
        raise ValueError("Grid search produced zero hyperparameter combinations.")

    print(
        f"Starting grid search with {len(combinations)} combinations "
        f"for manifest '{args.manifest_path.expanduser().resolve()}'."
    )

    records: list[dict[str, Any]] = []
    for learning_rate, batch_size, pos_weight_spec in combinations:
        run_name = build_run_name(
            run_prefix=args.run_prefix,
            learning_rate=learning_rate,
            batch_size=batch_size,
            pos_weight_spec=pos_weight_spec,
        )
        metrics_path = args.metrics_dir.expanduser().resolve() / run_name / "test_metrics.json"

        print(
            f"[Grid Search] run_name={run_name} | "
            f"learning_rate={learning_rate} | batch_size={batch_size} | "
            f"pos_weight={format_pos_weight_spec(pos_weight_spec)}"
        )

        if args.skip_existing and metrics_path.exists():
            print(f"Skipping existing run because metrics already exist: {metrics_path}")
            evaluation_results = load_json(metrics_path)
            history = load_json(args.metrics_dir.expanduser().resolve() / run_name / "history.json")
        else:
            history = run_training(
                manifest_path=args.manifest_path,
                epochs=args.epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_workers=args.num_workers,
                seed=args.seed,
                run_name=run_name,
                checkpoint_dir=args.checkpoint_dir,
                history_dir=args.metrics_dir,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                pos_weight=pos_weight_spec["value"],
                auto_pos_weight=pos_weight_spec["mode"] == "auto",
            )
            evaluation_results = run_evaluation(
                manifest_path=args.manifest_path,
                checkpoint_path=args.checkpoint_dir.expanduser().resolve() / run_name / "best.pt",
                batch_size=batch_size,
                num_workers=args.num_workers,
                run_name=run_name,
                metrics_dir=args.metrics_dir,
            )

        records.append(
            {
                "run_name": run_name,
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "pos_weight_mode": pos_weight_spec["mode"],
                "pos_weight_requested": pos_weight_spec["value"],
                "pos_weight_effective": history.get("pos_weight"),
                "epochs_requested": int(args.epochs),
                "seed": int(args.seed),
                "best_epoch": history.get("best_epoch"),
                "best_val_loss": history.get("best_val_loss"),
                "test_loss": evaluation_results.get("loss"),
                "accuracy": evaluation_results.get("accuracy"),
                "precision": evaluation_results.get("precision"),
                "recall": evaluation_results.get("recall"),
                "f1": evaluation_results.get("f1"),
                "roc_auc": evaluation_results.get("roc_auc"),
                "checkpoint_path": evaluation_results.get("checkpoint_path"),
                "metrics_path": evaluation_results.get("metrics_path"),
            }
        )

    summary_df = pd.DataFrame(records)
    ascending = args.sort_by == "best_val_loss"
    summary_df = summary_df.sort_values(by=args.sort_by, ascending=ascending).reset_index(drop=True)

    summary_dir = args.summary_dir.expanduser().resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = summary_dir / f"{args.run_prefix}_summary.csv"
    summary_json_path = summary_dir / f"{args.run_prefix}_summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary_df.to_dict(orient="records"), file, indent=2)

    print("\nGrid search complete.")
    print(f"Summary CSV:  {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print("Top runs:")
    print(summary_df.head(5).to_string(index=False))


def build_run_name(
    run_prefix: str,
    learning_rate: float,
    batch_size: int,
    pos_weight_spec: dict[str, Any],
) -> str:
    """Build a compact run name from the grid-search hyperparameters."""
    learning_rate_text = format(learning_rate, ".0e").replace("-", "m")
    pos_weight_text = format_pos_weight_spec(pos_weight_spec).replace(".", "p")
    return f"{run_prefix}_lr{learning_rate_text}_bs{batch_size}_pw{pos_weight_text}"


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk for summary reuse."""
    resolved_path = Path(path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{resolved_path}'.")
    return payload


def parse_pos_weight_spec(value: str) -> dict[str, Any]:
    """Parse a grid-search positive-class weight spec."""
    normalized = value.strip().lower()
    if normalized == "auto":
        return {"mode": "auto", "value": None}
    if normalized == "none":
        return {"mode": "none", "value": None}

    parsed_value = float(value)
    if parsed_value <= 0:
        raise ValueError(f"Positive-class weights must be positive, got {value}.")
    return {"mode": "manual", "value": parsed_value}


def format_pos_weight_spec(spec: dict[str, Any]) -> str:
    """Return a short stable text form for a positive-class weight spec."""
    if spec["mode"] in {"auto", "none"}:
        return spec["mode"]
    return str(spec["value"])


if __name__ == "__main__":
    main()
