"""Main script to run benchmarks with wandb integration."""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from src.config import BenchmarkConfig
from src.benchmarks import WandbCircuitBenchmark

def main():
    """Run benchmark with wandb tracking."""

    parser = argparse.ArgumentParser(description="Run wandb benchmark")
    parser.add_argument(
        "--powers-of-two",
        action="store_true",
        help="Use powers of two for number of units and keep input_units=sum_units",
    )
    parser.add_argument(
        "--min-exp",
        type=int,
        default=5,
        help="Minimum exponent for powers of two (2**min_exp)",
    )
    parser.add_argument(
        "--max-exp",
        type=int,
        default=9,
        help="Maximum exponent for powers of two (2**max_exp)",
    )

    args = parser.parse_args()
    
    # config = BenchmarkConfig(
    #     input_units=[10, 20, 30, 40],
    #     sum_units=[10, 20, 30, 40],
    #     ranks=[5, 10, 20, 30],
    #     batch_sizes=[1024, 2048, 4096],
    #     project_name="kronecker-vs-nystrom",
    #     experiment_name="full_benchmark"
    # )
    # config = BenchmarkConfig(
    #     input_units=[30, 40, 50, 60, 70],      # Creates matrices: 900, 1600, 2500, 3600, 4900
    #     sum_units=[30, 40, 50, 60, 70],        # Same for output dimensions
    #     ranks=[20, 50, 100, 200, 300],         # Scaled up ranks
    #     batch_sizes=[128, 256, 512],           # Larger batches for GPU efficiency
    #     project_name="kronecker-vs-nystrom",
    #     experiment_name="full_benchmark_large_scale"
    # )

    # config = BenchmarkConfig(
    #     input_units=[50, 70, 100, 120],        # Creates: 2500, 4900, 10000, 14400
    #     sum_units=[50, 70, 100, 120],
    #     ranks=[50, 100, 200, 400, 600],        # Higher ranks for larger matrices
    #     batch_sizes=[256, 512, 1024],          # Larger batches
    #     project_name="kronecker-vs-nystrom",
    #     experiment_name="full_benchmark_very_large"
    # )
    if args.powers_of_two:
        units = [2 ** i for i in range(args.min_exp, args.max_exp + 1)]
        config = BenchmarkConfig(
            input_units=units,
            sum_units=units,
            ranks=[50, 100, 200, 400, 600, 2000, 5000, 10000, 20000],
            batch_sizes=[256, 512],
            project_name="kronecker-vs-nystrom",
            experiment_name="full_benchmark_pow2",
            powers_of_two=True,
            min_exp=args.min_exp,
            max_exp=args.max_exp,
        )
    else:
        config = BenchmarkConfig(
            input_units=[50, 70, 100, 120, 200],
            sum_units=[50, 70, 100, 120, 200],
            ranks=[50, 100, 200, 400, 600, 2000, 5000, 10000, 20000],
            batch_sizes=[256, 512],
            project_name="kronecker-vs-nystrom",
            experiment_name="full_benchmark_very_large_more_ranks",
            powers_of_two=False,
            min_exp=None,
            max_exp=None,
        )
    
    print(f"Starting wandb experiment on {config.device}")
    benchmark = WandbCircuitBenchmark(config)
    results = benchmark.run_full_benchmark()
    
    # Save artifacts
    artifact = wandb.Artifact(
        name="benchmark_results",
        type="dataset",
        description="Complete benchmark results"
    )
    
    # results_df = results.get_dataframe()
    # results_df.to_csv("results/data/wandb_benchmark_results.csv", index=False)
    # artifact.add_file("results/data/wandb_benchmark_results.csv")
    
    # wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    main()
