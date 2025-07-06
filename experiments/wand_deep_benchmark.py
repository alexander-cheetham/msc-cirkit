"""Run deep circuit Nyström benchmark with wandb tracking."""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from src.config import DeepBenchmarkConfig
from src.benchmarks import WandbDeepCircuitBenchmark


def main():
    parser = argparse.ArgumentParser(description="Run deep circuit wandb benchmark")
    parser.add_argument("--layers", type=int, default=2, help="Number of alternating layers")
    args = parser.parse_args()

    config = DeepBenchmarkConfig(num_layers=args.layers)
    print(f"Starting deep wandb experiment on {config.device}")
    benchmark = WandbDeepCircuitBenchmark(config)
    benchmark.run_full_benchmark()
    wandb.finish()


if __name__ == "__main__":
    main()
