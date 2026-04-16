from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.evaluation.evaluate_autoencoder import run_autoencoder_evaluation

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained autoencoder.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", default="all_machines_autoencoder_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_autoencoder_evaluation(
        manifest_path=args.manifest_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()