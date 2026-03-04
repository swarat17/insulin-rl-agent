"""
CLI entry point for training an RL dosing agent.

Usage:
    python scripts/train.py --algorithm ppo --config configs/ppo.yaml
    python scripts/train.py --algorithm sac --config configs/sac.yaml --seed 123
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an insulin dosing RL agent.")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["dqn", "ppo", "sac"],
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML hyperparameter config file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config=config, algorithm=args.algorithm, seed=args.seed)
    checkpoint_path = trainer.run()
    print(f"\nCheckpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
