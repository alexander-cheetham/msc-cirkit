"""Core benchmarking logic."""

import traceback
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from .config import BenchmarkConfig
from .profilers import WandbMemoryProfiler, FLOPCounter
from .visualisation import create_wandb_visualisations
from dataclasses import asdict
import matplotlib.pyplot as plt
import wandb
from cirkit.pipeline import PipelineContext, compile as compile_circuit
from cirkit.symbolic.circuit import Circuit
import cirkit.symbolic.functional as SF
from .circuit_types import CIRCUIT_BUILDERS
from cirkit.symbolic.io import plot_circuit
import os

try:
    wandb.require("legacy-service")
except wandb.errors.UnsupportedError:
    # ignore if the legacy-service requirement isn’t supported
    pass


def compile_symbolic(circuit: Circuit, *, device: str, rank: int | None = None, opt: bool = False):
    """Compile a symbolic circuit with optional Nyström optimization."""
    ctx = PipelineContext(
        backend="torch",
        semiring="complex-lse-sum",
        fold=False,
        optimize=opt,
        nystrom_rank=rank,
    )
    compiled = compile_circuit(circuit, ctx, nystrom_rank=rank).to(device).eval()
    return compiled


class WandbCircuitBenchmark:
    """Benchmark suite with wandb integration.
    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.
    """

    def __init__(self, config: BenchmarkConfig, base_symbolic_circuit: str = None):
        self.config = config
        self.base_symbolic_circuit = base_symbolic_circuit

        # Determine distributed rank
        self.rank = getattr(config, "local_rank", 0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                self.rank = torch.distributed.get_rank()
            except RuntimeError:
                print("Warning: could not determine distributed rank")
        device_idx = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        print(f"[init] Device {device_idx}, rank {self.rank}")
        
        def _wandb_log(data, **kwargs):
            if self.rank == 0:
                wandb.log(data, **kwargs)

        def _print_rank0(*args, **kwargs):
            if self.rank == 0:
                print(*args, **kwargs)

        self.wandb_log = _wandb_log
        self.print_rank0 = _print_rank0

        # Initialize wandb run only once; disable on other ranks
        init_kwargs = dict(
            project=config.project_name,
            name=config.experiment_name,
            tags=config.tags,
            notes=config.notes,
            config=asdict(config),
        )
        if self.rank != 0:
            init_kwargs["mode"] = "disabled"
        self.run = wandb.init(**init_kwargs)

        # Log power-of-two configuration if enabled
        if self.config.powers_of_two:
            self.wandb_log(
                {
                    "config/min_exp": self.config.min_exp,
                    "config/max_exp": self.config.max_exp,
                },
                step=0,
            )

        # Create wandb table for detailed results. Include depth to allow
        # analysis of deep circuits.
        self.results_table = wandb.Table(
            columns=[
                "depth",
                "n_input",
                "n_sum",
                "rank",
                "batch_size",
                "matrix_size",
                "orig_time_ms",
                "nystrom_time_ms",
                "speedup",
                "theoretical_speedup",
                "orig_memory_mb",
                "nystrom_memory_mb",
                "memory_reduction",
                "orig_gflops",
                "nystrom_gflops",
                "flop_reduction",
                "nll_diff",
                "efficiency",
            ]
        )

        # Summary metrics
        self.summary_metrics = {
            "speedups": [],
            "memory_reductions": [],
            "nll_diffs": [],
            "efficiencies": [],
        }

    def apply_parallel_wrapper(self, model: nn.Module) -> nn.Module:
        """Wrap model for multi-GPU inference based on config.distributed."""
        mode = getattr(self.config, "distributed", "none")
        if mode == "dp" and torch.cuda.device_count() > 1:
            return nn.DataParallel(model)
        if mode == "ddp" and torch.distributed.is_available() and torch.distributed.is_initialized():
            device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
            return nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        if mode == "fsdp" and torch.distributed.is_available() and torch.distributed.is_initialized():
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return FSDP(model)
        return model

    def create_test_input(self, batch_size: int, input_dim: int, device: str):
        """Create test input tensor with correct shape for circuit."""
        # For squared circuit: num_variables = input_dim^2
        num_variables = input_dim**2
        if self.config.circuit_structure == "MNIST":
            # MNIST circuits use categorical input layers expecting discrete
            # pixel values in the range [0, 255]. ``torch.randn`` would
            # generate floating point values that after casting to ``long`` in
            # the Categorical layer might be negative or exceed the number of
            # categories, triggering CUDA index errors.  ``torch.randint``
            # ensures values fall within the valid range.
            num_variables = 784
            return torch.randint(0, 256, (batch_size, num_variables), device=device)
        return torch.randn(batch_size, num_variables, device=device)

    def time_forward_pass(self, circuit, test_input, num_warmup=10, num_trials=100):
        """Time forward pass with wandb logging"""
        times = []

        # Warmup
        for _ in range(num_warmup):
            _ = circuit(test_input)

        if test_input.is_cuda:
            torch.cuda.synchronize()

        # Time individual trials for variance analysis
        for i in range(num_trials):
            start = time.perf_counter()
            _ = circuit(test_input)

            if test_input.is_cuda:
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)
        return {
            "mean": times.mean(),
            "std": times.std(),
            "min": times.min(),
            "max": times.max(),
            "median": np.median(times),
        }

    def benchmark_single_configuration(
        self,
        n_input: int,
        n_sum: int,
        rank: int,
        batch_size: int,
        step: int,
        depth: int | None = None,
    ) -> Dict:
        """Run benchmark for single configuration with wandb logging.

        The function now gracefully handles out-of-memory errors. When such an
        error occurs, the configuration is logged to wandb and ``None`` is
        returned so that downstream aggregation and plots are not corrupted.
        """

        # Log current configuration
        if self.config.powers_of_two and n_input == n_sum:
            matrix_label = f"2^{int(np.log2(n_input**2))}"
        else:
            matrix_label = f"{n_input**2}x{n_sum**2}"

        log_dict = {
            "config/n_input": n_input,
            "config/n_sum": n_sum,
            "config/rank": rank,
            "config/batch_size": batch_size,
            "config/matrix_dims": matrix_label,
            "step": step,
        }
        if depth is not None:
            log_dict["config/depth"] = depth
        self.wandb_log(log_dict)

        try:
            # Build symbolic circuit and log its structure
            symbolic = self.base_symbolic_circuit
            # pre_path = "circuit_pre_square.png"
            # plot_circuit(symbolic, out_path=pre_path)
            # wandb.log({"charts/circuit_pre_square": wandb.Image(pre_path)})
            # if os.path.exists(pre_path):
            #     os.remove(pre_path)

            # Square the circuit and log its structure
            symbolic = SF.multiply(symbolic, symbolic)
            # post_path = "circuit_post_square.png"
            # plot_circuit(symbolic, out_path=post_path)
            # wandb.log({"charts/circuit_post_square": wandb.Image(post_path)})
            # if os.path.exists(post_path):
            #     os.remove(post_path)

            # Compile baseline and Nyström versions
            original_circuit = compile_symbolic(
                symbolic, device=self.config.device, rank=None, opt=False
            )
            nystrom_circuit = compile_symbolic(
                symbolic, device=self.config.device, opt=True, rank=rank
            )

            original_circuit = self.apply_parallel_wrapper(original_circuit)
            nystrom_circuit = self.apply_parallel_wrapper(nystrom_circuit)

            # Create test input
            test_input = self.create_test_input(batch_size, n_input, "cpu")

            # Time forward passes
            orig_times = self.time_forward_pass(
                original_circuit,
                test_input,
                self.config.num_warmup,
                self.config.num_trials,
            )

            nystrom_times = self.time_forward_pass(
                nystrom_circuit,
                test_input,
                self.config.num_warmup,
                self.config.num_trials,
            )

            # Log timing distributions
            self.wandb_log(
                {
                    "timing/original_mean_ms": orig_times["mean"] * 1000,
                    "timing/original_std_ms": orig_times["std"] * 1000,
                    "timing/nystrom_mean_ms": nystrom_times["mean"] * 1000,
                    "timing/nystrom_std_ms": nystrom_times["std"] * 1000,
                    "timing/speedup": orig_times["mean"] / nystrom_times["mean"],
                    "step": step,
                }
            )

            # Memory profiling with wandb logging
            orig_memory = WandbMemoryProfiler.profile_and_log(
                original_circuit,
                test_input,
                device=self.config.device,
                prefix="memory/original",
                rank=self.rank,
            )

            nystrom_memory = WandbMemoryProfiler.profile_and_log(
                nystrom_circuit,
                test_input,
                device=self.config.device,
                prefix="memory/nystrom",
                rank=self.rank,
            )

            # FLOP counting
            F = 1  # Number of folds
            orig_flops = FLOPCounter.kronecker_forward(batch_size, F, n_sum, n_input)
            nystrom_flops = FLOPCounter.nystrom_forward(
                batch_size, F, n_sum, n_input, rank
            )

            self.wandb_log(
                {
                    "flops/original_gflops": orig_flops / 1e9,
                    "flops/nystrom_gflops": nystrom_flops / 1e9,
                    "flops/reduction": 1 - (nystrom_flops / orig_flops),
                    "step": step,
                }
            )

            # Approximation metrics
            with torch.no_grad():
                orig_output = original_circuit(test_input).real
                nystrom_output = nystrom_circuit(test_input).real

                # TODO: verify that these formulas for NLL and KL divergence are
                # consistent with how the circuits represent probabilities. The
                # current implementation assumes the circuit outputs log
                # likelihoods for each sample.

                nll_orig = -orig_output
                nll_nystrom = -nystrom_output
                nll_diff_per_sample = (nll_nystrom - nll_orig).abs()
                nll_diff = nll_diff_per_sample.mean()

                self.wandb_log(
                    {
                        "accuracy/nll_diff": nll_diff.item(),
                        "accuracy/nll_diff_std": nll_diff_per_sample.std().item(),
                        "accuracy/nll_max": nll_diff_per_sample.max().item(),
                        "step": step,
                    }
                )

            # Calculate all metrics
            speedup = orig_times["mean"] / nystrom_times["mean"]
            theoretical_speedup = FLOPCounter.theoretical_speedup(n_sum, n_input, rank)
            memory_reduction = 1 - (nystrom_memory / max(orig_memory, 1e-6))
            efficiency = speedup / theoretical_speedup

            # Update summary metrics
            self.summary_metrics["speedups"].append(speedup)
            self.summary_metrics["memory_reductions"].append(memory_reduction)
            self.summary_metrics["nll_diffs"].append(nll_diff.item())
            self.summary_metrics["efficiencies"].append(efficiency)

            # Add row to results table
            self.results_table.add_data(
                depth if depth is not None else self.config.depth,
                n_input,
                n_sum,
                rank,
                batch_size,
                matrix_label,
                orig_times["mean"] * 1000,
                nystrom_times["mean"] * 1000,
                speedup,
                theoretical_speedup,
                orig_memory,
                nystrom_memory,
                memory_reduction,
                orig_flops / 1e9,
                nystrom_flops / 1e9,
                1 - (nystrom_flops / orig_flops),
                nll_diff.item(),
                efficiency,
            )

            # Log efficiency metrics
            self.wandb_log(
                {
                    "efficiency/actual_vs_theoretical": efficiency,
                    "efficiency/memory_reduction": memory_reduction,
                    "efficiency/speedup": speedup,
                    "step": step,
                }
            )
            return {
                "depth": depth if depth is not None else self.config.depth,
                "n_input": n_input,
                "n_sum": n_sum,
                "rank": rank,
                "batch_size": batch_size,
                "speedup": speedup,
                "memory_reduction": memory_reduction,
                "nll_diff": nll_diff.item(),
                "efficiency": efficiency,
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.print_rank0(
                    f"  OOM for input={n_input}, sum={n_sum}, rank={rank}, batch={batch_size}"
                )
                self.wandb_log(
                    {
                        "errors/type": "out_of_memory",
                        "errors/message": str(e),
                        "config/n_input": n_input,
                        "config/n_sum": n_sum,
                        "config/rank": rank,
                        "config/batch_size": batch_size,
                        "config/depth": (
                            depth if depth is not None else self.config.depth
                        ),
                        "step": step,
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
            else:
                raise

    def run_full_benchmark(self):
        """Run complete benchmark suite with wandb tracking"""

        step = 0
        # Pre-compute the number of configurations that will actually be
        # benchmarked. When ``powers_of_two`` is enabled we only test square
        # matrices, so skip any non-square combinations in this count.
        total_configs = 0
        depth_range = (
            range(2, self.config.depth + 1)
            if self.config.circuit_structure == "deep_cp_circuit"
            else [self.config.depth]
        )
        for depth in depth_range:
            for n_input in self.config.input_units:
                for n_sum in self.config.sum_units:
                    if self.config.powers_of_two and n_input != n_sum:
                        continue
                    for rank in self.config.ranks:
                        if rank >= min(n_input**2, n_sum**2):
                            continue
                        total_configs += len(self.config.batch_sizes)

        # Create progress bar in wandb
        progress = 0

        for depth in depth_range:
            for n_input in self.config.input_units:
                for n_sum in self.config.sum_units:
                    # When powers-of-two mode is active, only benchmark square matrices
                    if self.config.powers_of_two and n_input != n_sum:
                        continue
                    for rank in self.config.ranks:
                        # Skip if rank too large
                        if rank >= min(n_input**2, n_sum**2):
                            continue

                        for batch_size in self.config.batch_sizes:
                            progress += 1
                            self.wandb_log({"progress": progress / total_configs})

                            self.print_rank0(
                                f"[{progress}/{total_configs}] Benchmarking: depth={depth}, "
                                f"input={n_input}, sum={n_sum}, "
                                f"nystrom_rank={rank}, batch={batch_size}, gpu_rank={self.rank}"
                            )

                            try:
                                builder = CIRCUIT_BUILDERS[
                                    self.config.circuit_structure
                                ]
                                builder_kwargs = {}
                                if self.config.circuit_structure == "deep_cp_circuit":
                                    builder_kwargs["depth"] = depth
                                if self.config.circuit_structure == "MNIST":
                                    builder_kwargs["region_graph"] = (
                                        self.config.region_graph
                                    )
                                if n_input is not None:
                                    builder_kwargs["num_input_units"] = n_input
                                if n_sum is not None:
                                    builder_kwargs["num_sum_units"] = n_sum

                                self.base_symbolic_circuit = builder(**builder_kwargs)
                                result = self.benchmark_single_configuration(
                                    n_input, n_sum, rank, batch_size, step, depth=depth
                                )
                                if result is not None:
                                    step += 1

                            except Exception as e:
                                self.print_rank0(f"  Failed: {e}")
                                tb_str = traceback.format_exc()
                                self.print_rank0(f"Error details:\n{tb_str}")
                                self.wandb_log({"errors/count": 1, "errors/message": str(e)})
                                continue

        # Log final summary statistics
        self.log_summary_statistics()

        # Log results table
        self.wandb_log({"results_table": self.results_table})

        # Create and log visualizations
        if self.rank == 0:
            create_wandb_visualisations(self.results_table, self.config)

        return self.results_table

    def log_summary_statistics(self):
        """Log summary statistics to wandb"""
        summary = {
            "summary/avg_speedup": np.mean(self.summary_metrics["speedups"]),
            "summary/max_speedup": np.max(self.summary_metrics["speedups"]),
            "summary/min_speedup": np.min(self.summary_metrics["speedups"]),
            "summary/avg_memory_reduction": np.mean(
                self.summary_metrics["memory_reductions"]
            ),
            "summary/avg_nll_diff": np.mean(self.summary_metrics["nll_diffs"]),
            "summary/avg_efficiency": np.mean(self.summary_metrics["efficiencies"]),
        }

        # Find best configurations
        speedups = np.array(self.summary_metrics["speedups"])
        errors = np.array(self.summary_metrics["nll_diffs"])

        # Best speedup with NLL difference < 1e-2
        good_accuracy_mask = errors < 0.01
        if good_accuracy_mask.any():
            best_speedup_good_accuracy = speedups[good_accuracy_mask].max()
            summary["summary/best_speedup_low_nll"] = best_speedup_good_accuracy

        self.wandb_log(summary)
