"""Main script to run benchmarks with wandb integration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from src.config import BenchmarkConfig
from src.benchmarks import WandbCircuitBenchmark  # You'll need to add this to benchmarks.py

def main():
    """Run benchmark with wandb tracking."""
    
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
    config = BenchmarkConfig(
        input_units=[50, 70, 100, 120,200],        # Creates: 2500, 4900, 10000, 14400
        sum_units=[50, 70, 100, 120,200],
        ranks=[50, 100, 200, 400, 600, 2000,5000, 10000,20000],        # Higher ranks for larger matrices
        batch_sizes=[256, 512],          # Larger batches
        project_name="kronecker-vs-nystrom",
        experiment_name="full_benchmark_very_large_more_ranks"
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