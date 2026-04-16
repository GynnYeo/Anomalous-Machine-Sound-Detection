from __future__ import annotations

import argparse

from src.evaluation.plot_autoencoder import plot_reconstruction_error_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot autoencoder reconstruction errors.")
    parser.add_argument("--errors-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_reconstruction_error_distribution(
        errors_json_path=args.errors_path,
        output_path=args.output_path,
        split=args.split,
    )
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()