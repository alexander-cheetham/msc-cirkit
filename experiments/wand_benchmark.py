"""Main script to run benchmarks with wandb integration."""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
from src.config import BenchmarkConfig
from src.benchmarks import WandbCircuitBenchmark
from src.circuit_types import CIRCUIT_BUILDERS

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

    parser.add_argument(
        "--circuit-structure",
        choices=["one_sum", "deep_cp_circuit", "MNIST"],
        default="one_sum",
        help="Type of circuit to benchmark",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Depth for deep_cp_circuit",
    )
    parser.add_argument(
        "--region-graph",
        type=str,
        default="quad-tree-4",
        help="Region graph to use for MNIST circuits",
    )
    parser.add_argument(
        "--distributed",
        choices=["none", "dp", "ddp", "fsdp"],
        default="none",
        help="Parallel backend: none, dp, ddp or fsdp",
    )

    args = parser.parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[launch] local_rank={local_rank}")
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
            circuit_structure=args.circuit_structure,
            depth=args.depth,
            region_graph=args.region_graph,
            distributed=args.distributed,
            local_rank=local_rank,
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
            circuit_structure=args.circuit_structure,
            depth=args.depth,
            region_graph=args.region_graph,
            distributed=args.distributed,
            local_rank=local_rank,
        )
    
    if config.distributed in {"ddp", "fsdp"}:
        import torch.distributed as dist
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            rank = dist.get_rank()
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        print(f"[setup] backend={config.distributed} rank={rank} device={device}")
    else:
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        print(f"[setup] running without distributed on device {device}")

    print(f"Starting wandb experiment on {config.device}")
    # Build the symbolic circuit once using the selected builder.
    builder = CIRCUIT_BUILDERS[config.circuit_structure]
    builder_kwargs = {
        "num_input_units": config.input_units[0],
        "num_sum_units": config.sum_units[0],
    }
    if config.circuit_structure == "deep_cp_circuit":
        builder_kwargs["depth"] = config.depth
    if config.circuit_structure == "MNIST":
        builder_kwargs["region_graph"] = config.region_graph

    symbolic_circuit = builder(**builder_kwargs)
    benchmark = WandbCircuitBenchmark(config, symbolic_circuit)
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
    if config.distributed in {"ddp", "fsdp"}:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
