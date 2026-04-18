"""CLI entry point for test-set evaluation of the available CNN classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for classifier evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CNN checkpoint on the test split."
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
        help="Path to the saved best checkpoint to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for test evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override for the metrics output folder.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "metrics",
        help="Parent directory for run-specific evaluation metrics folders.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline_cnn",
        help="Model architecture to evaluate. Examples: baseline_cnn, deeper_cnn, wider_cnn.",
    )
    return parser


def main() -> None:
    """Parse CLI args and launch reusable evaluation."""
    args = build_parser().parse_args()

    run_evaluation(
        manifest_path=args.manifest_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        run_name=args.run_name,
        metrics_dir=args.metrics_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
