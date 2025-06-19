#!/usr/bin/env python
# benchmark_kron.py
"""
Benchmark TorchKroneckerLayer vs torch.kron with Weights & Biases.
Logs execution time and peak memory to W&B, handling OOMs gracefully.

Usage examples
--------------
# Single run
python benchmark_kron.py --matrix_size 256 --implementation TorchKroneckerLayer

# Sweep from the CLI (matrix sizes 128,256) and both implementations
wandb sweep sweep_config.yaml
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
"""
import argparse
import gc
import time
from statistics import mean

import numpy as np
import psutil
import torch
import os
os.environ.setdefault("WANDB__REQUIRE_LEGACY_SERVICE", "TRUE")  # env override

import wandb
wandb.require("legacy-service")    

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def create_input(matrix_size: int, device: torch.device) -> torch.Tensor:
    """Return a (1, 2, N²) tensor where the two rows are random matrices flattened."""
    a = torch.randn(matrix_size, matrix_size, device=device).flatten()
    b = torch.randn(matrix_size, matrix_size, device=device).flatten()
    return torch.stack([a, b]).unsqueeze(0)


def measure(fn, *args, device: torch.device):
    """
    Time `fn(*args)` and return (elapsed_s, peak_memory_mb).
    If CUDA is available, GPU memory is reported; otherwise, RSS delta is used.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        rss_before = psutil.Process().memory_info().rss

    start = time.perf_counter()
    out = fn(*args)
    # Make sure all CUDA kernels have finished
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        rss_after = psutil.Process().memory_info().rss
        mem_mb = (rss_after - rss_before) / 1024 ** 2

    return out, elapsed, mem_mb


def run_one_rep(matrix_size: int, implementation: str, device: torch.device, num_folds: int):
    """
    Execute a single repetition of the selected implementation.
    Returns (elapsed_s, mem_mb) or raises RuntimeError (handled by caller).
    """
    x = create_input(matrix_size, device)

    if implementation == "TorchKroneckerLayer":
        from cirkit.backend.torch.layers.inner import TorchKroneckerLayer

        layer = TorchKroneckerLayer(
            num_input_units=matrix_size ** 2,
            arity=2,
            num_folds=num_folds,
        ).to(device)
        fn, args = layer, (x,)
    elif implementation == "torch.kron":
        # x has shape (1, 2, N²) → take a, b as matrices
        a = x[0, 0].view(matrix_size, matrix_size)
        b = x[0, 1].view(matrix_size, matrix_size)
        fn, args = torch.kron, (a, b)
    else:
        raise ValueError(f"Unknown implementation '{implementation}'")

    # Timed call
    _, elapsed, mem_mb = measure(fn, *args, device=device)

    # Clean-up
    del fn, args, x
    if implementation == "TorchKroneckerLayer":
        del layer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return elapsed, mem_mb


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MATRIX_SIZE", type=int, default=256, help="N for NxN matrices")
    parser.add_argument("--implementation", choices=["TorchKroneckerLayer", "torch.kron"], required=True)
    parser.add_argument("--k", type=int, default=20, help="Repetitions per configuration")
    parser.add_argument("--num_folds", type=int, default=1, help="TorchKroneckerLayer num_folds")
    parser.add_argument("--project", default="kronecker-benchmarks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise W&B
    run = wandb.init(
        project=args.project,
        config={
            "MATRIX_SIZE": args.MATRIX_SIZE,
            "implementation": args.implementation,
            "k": args.k,
            "num_folds": args.num_folds,
            "device": str(device),
        },
    )

    times, mems = [], []
    for rep in range(args.k):
        try:
            elapsed, mem_mb = run_one_rep(
                args.MATRIX_SIZE,
                args.implementation,
                device=device,
                num_folds=args.num_folds,
            )
            times.append(elapsed)
            mems.append(mem_mb)
            wandb.log({"rep": rep, "time_s": elapsed, "memory_mb": mem_mb})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Log OOM and abandon the remaining reps
                wandb.log({"rep": rep, "oom": True})
                print(f"[OOM] rep {rep} — aborting further reps for this run.")
                break
            else:
                raise

    # Aggregate statistics (skip if nothing ran)
    if times:
        for series, name in [(times, "time_s"), (mems, "memory_mb")]:
            run.summary[f"{name}_min"] = min(series)
            run.summary[f"{name}_max"] = max(series)
            run.summary[f"{name}_mean"] = mean(series)
            run.summary[f"{name}_std"] = float(np.std(series))
            run.summary["MATRIX_SIZE"] = args.MATRIX_SIZE
            run.summary["implementation"] = args.implementation

    run.finish()


if __name__ == "__main__":
    main()
