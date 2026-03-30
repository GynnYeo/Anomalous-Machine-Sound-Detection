"""CLI for building a master index and deterministic split manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.index_dataset import build_master_index, save_master_index
from src.data.split import (
    build_split_for_scope,
    infer_scope_name,
    save_split_manifest,
    summarize_splits,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic train/val/test split manifests for MIMII audio data."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Root directory containing raw WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Directory where CSV outputs will be written.",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default=None,
        help="Optional machine type filter, for example 'fan'.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle group IDs before splitting.",
    )
    parser.add_argument(
        "--write-master-index",
        action="store_true",
        help="Also persist data/splits/master_index.csv.",
    )
    return parser


def print_summary(summary: dict[str, object]) -> None:
    """Print split summary statistics for quick inspection."""
    print(f"Total files: {summary['total_files']}")
    print("Files per split:")
    for split_name, count in summary["files_per_split"].items():
        print(f"  {split_name}: {count}")

    print("Class counts per split:")
    for split_name, class_counts in summary["class_counts_per_split"].items():
        counts_text = ", ".join(
            f"{label}={count}" for label, count in class_counts.items()
        )
        print(f"  {split_name}: {counts_text}")

    print("Class proportions per split:")
    for split_name, class_proportions in summary["class_proportions_per_split"].items():
        proportions_text = ", ".join(
            f"{label}={proportion:.3f}" for label, proportion in class_proportions.items()
        )
        print(f"  {split_name}: {proportions_text}")

    print("Number of groups per split:")
    for split_name, count in summary["groups_per_split"].items():
        print(f"  {split_name}: {count}")


def main() -> None:
    """Build the dataset index, create scoped splits, and write CSV outputs."""
    parser = build_parser()
    args = parser.parse_args()

    master_index_df = build_master_index(args.data_root)
    if args.write_master_index:
        master_index_path = save_master_index(
            index_df=master_index_df,
            output_path=args.output_dir / "master_index.csv",
        )
        print(f"Master index written to: {master_index_path}")

    split_df = build_split_for_scope(
        index_df=master_index_df,
        machine_type=args.machine_type,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    scope_name = infer_scope_name(split_df)
    manifest_path = save_split_manifest(
        split_df=split_df,
        output_path=args.output_dir / f"{scope_name}_split_seed{args.seed}.csv",
    )

    print(f"Split manifest written to: {manifest_path}")
    print_summary(summarize_splits(split_df))


if __name__ == "__main__":
    main()
