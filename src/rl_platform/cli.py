import argparse
import sys
from pathlib import Path

from rl_platform.train import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="rlp", description="RL Platform CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run a training session")
    train_parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    if args.command == "train":
        results = train(args.config)
        print("\nTraining complete")
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
