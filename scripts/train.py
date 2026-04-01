"""CLI entry point for baseline CNN training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import run_training


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN on a saved MIMII split manifest."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to a saved split manifest CSV.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total number of epochs to train for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
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
        help="Random seed for model initialization and dataloader shuffling.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline_cnn",
        help="Run folder name used under checkpoint and history parent directories.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints",
        help="Parent directory for run-specific checkpoint folders.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "metrics",
        help="Parent directory for run-specific history folders.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    return parser


def main() -> None:
    """Parse CLI args and launch the reusable baseline training pipeline."""
    args = build_parser().parse_args()

    run_training(
        manifest_path=args.manifest_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        history_dir=args.history_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
