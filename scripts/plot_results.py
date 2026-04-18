"""CLI entry point for training, evaluation, and experiment-summary plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import (
    load_json,
    plot_confusion_matrix,
    plot_grid_search_metric,
    plot_loss_curves,
    plot_threshold_tradeoff,
    plot_validation_accuracy,
)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for plot generation from saved artifacts."""
    parser = argparse.ArgumentParser(
        description="Generate plots from saved training, evaluation, grid-search, or threshold-tuning artifacts."
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help="Optional path to a saved training history JSON file.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Optional path to a saved test metrics JSON file.",
    )
    parser.add_argument(
        "--curves-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "curves",
        help="Parent directory for run-specific plot folders.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override for the plot output folder.",
    )
    parser.add_argument(
        "--grid-summary-path",
        type=Path,
        default=None,
        help="Optional path to a saved grid-search summary JSON file.",
    )
    parser.add_argument(
        "--grid-metric",
        type=str,
        default="recall",
        help="Metric to plot from the grid-search summary.",
    )
    parser.add_argument(
        "--grid-top-k",
        type=int,
        default=8,
        help="Number of top grid-search runs to include in the summary plot.",
    )
    parser.add_argument(
        "--threshold-summary-path",
        type=Path,
        default=None,
        help="Optional path to a saved threshold summary JSON file.",
    )
    parser.add_argument(
        "--selected-threshold-path",
        type=Path,
        default=None,
        help="Optional path to a selected-threshold JSON file used to mark the chosen threshold.",
    )
    return parser


def main() -> None:
    """Parse CLI args, generate requested plots, and print saved files."""
    args = build_parser().parse_args()
    if (
        args.history_path is None
        and args.metrics_path is None
        and args.grid_summary_path is None
        and args.threshold_summary_path is None
    ):
        raise ValueError(
            "Provide at least one plotting input: "
            "--history-path, --metrics-path, --grid-summary-path, or --threshold-summary-path."
        )

    resolved_run_name = _infer_run_name(
        history_path=args.history_path,
        metrics_path=args.metrics_path,
        run_name=args.run_name,
        grid_summary_path=args.grid_summary_path,
        threshold_summary_path=args.threshold_summary_path,
    )
    curves_run_dir = _resolve_output_dir(args.curves_dir, resolved_run_name)

    generated_paths: list[Path] = []
    if args.history_path is not None:
        history = load_json(args.history_path)
        generated_paths.append(
            plot_loss_curves(history, curves_run_dir / "loss_curve.png")
        )
        generated_paths.append(
            plot_validation_accuracy(history, curves_run_dir / "val_accuracy_curve.png")
        )

    if args.metrics_path is not None:
        metrics = load_json(args.metrics_path)
        if not isinstance(metrics, dict):
            raise ValueError(f"Expected a JSON object in '{args.metrics_path}'.")
        confusion_matrix = metrics.get("confusion_matrix")
        if confusion_matrix is None:
            raise KeyError(
                f"Metrics file '{args.metrics_path}' is missing required key 'confusion_matrix'."
            )
        generated_paths.append(
            plot_confusion_matrix(confusion_matrix, curves_run_dir / "confusion_matrix.png")
        )

    if args.grid_summary_path is not None:
        grid_rows = load_json(args.grid_summary_path)
        if not isinstance(grid_rows, list):
            raise ValueError(f"Expected a JSON array in '{args.grid_summary_path}'.")
        generated_paths.append(
            plot_grid_search_metric(
                grid_rows,
                curves_run_dir / f"grid_search_{args.grid_metric}.png",
                metric=args.grid_metric,
                top_k=args.grid_top_k,
            )
        )

    if args.threshold_summary_path is not None:
        threshold_rows = load_json(args.threshold_summary_path)
        if not isinstance(threshold_rows, list):
            raise ValueError(f"Expected a JSON array in '{args.threshold_summary_path}'.")
        selected_threshold = _load_selected_threshold(args.selected_threshold_path)
        generated_paths.append(
            plot_threshold_tradeoff(
                threshold_rows,
                curves_run_dir / "threshold_tradeoff.png",
                selected_threshold=selected_threshold,
            )
        )

    print("Generated plots:")
    for path in generated_paths:
        print(f"  {path}")


def _infer_run_name(
    history_path: Path | None,
    metrics_path: Path | None,
    run_name: str | None,
    grid_summary_path: Path | None,
    threshold_summary_path: Path | None,
) -> str:
    """Infer the run name from an explicit value or artifact parent folders."""
    if run_name is not None:
        return run_name
    if history_path is not None:
        return history_path.expanduser().resolve().parent.name
    if metrics_path is not None:
        return metrics_path.expanduser().resolve().parent.name
    if grid_summary_path is not None:
        stem = grid_summary_path.expanduser().resolve().stem
        return stem.removesuffix("_summary")
    if threshold_summary_path is not None:
        return threshold_summary_path.expanduser().resolve().parent.name
    return "baseline_run"


def _load_selected_threshold(path: Path | None) -> float | None:
    """Return the selected threshold value when a sidecar JSON file is provided."""
    if path is None:
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    threshold_value = payload.get("threshold", payload.get("selected_threshold"))
    if threshold_value is None:
        raise KeyError(
            f"Selected-threshold file '{path}' is missing key "
            "'threshold' or 'selected_threshold'."
        )
    return float(threshold_value)


def _resolve_output_dir(curves_dir: Path, run_name: str) -> Path:
    """Create the requested output directory, falling back when Windows ACLs block artifacts."""
    requested_dir = curves_dir.expanduser().resolve() / run_name
    try:
        requested_dir.mkdir(parents=True, exist_ok=True)
        return requested_dir
    except PermissionError:
        fallback_dir = (PROJECT_ROOT / "report_figures" / run_name).resolve()
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Warning: could not write under the requested curves directory. "
            f"Falling back to '{fallback_dir}'."
        )
        return fallback_dir


if __name__ == "__main__":
    main()
