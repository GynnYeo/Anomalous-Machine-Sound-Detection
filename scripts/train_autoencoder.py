from __future__ import annotations

import argparse

from src.training.train_autoencoder import run_autoencoder_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train convolutional autoencoder.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="all_machines_autoencoder")
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_autoencoder_training(
        manifest_path=args.manifest_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        run_name=args.run_name,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()