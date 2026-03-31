"""Small sanity check for the baseline dataset preprocessing pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MIMIIDataset
from src.data.transforms import BaselineLogMelTransform


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for the preprocessing sanity check."""
    parser = argparse.ArgumentParser(
        description="Load one preprocessed sample from a saved split manifest."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to a saved split manifest CSV.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to inspect: train, val, or test.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within the selected split.",
    )
    return parser


def main() -> None:
    """Load one transformed sample and print its basic structure."""
    args = build_parser().parse_args()

    dataset = MIMIIDataset(
        manifest_path=args.manifest_path,
        split=args.split,
        transform=BaselineLogMelTransform(),
        return_metadata=True,
    )
    sample = dataset[args.index]

    print(f"Keys: {sorted(sample.keys())}")
    print(f"Input shape: {tuple(sample['input'].shape)}")
    print(f"Label: {sample['label']}")
    if "metadata" in sample:
        print(f"Metadata keys: {sorted(sample['metadata'].keys())}")


if __name__ == "__main__":
    main()
