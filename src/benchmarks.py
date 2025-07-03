"""Core benchmarking logic."""

import torch
import time
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from .config import BenchmarkConfig
from nystromlayer import NystromSumLayer
from .circuit_manip import build_and_compile_circuit,replace_sum_layers, fix_address_book_modules
from .profilers import WandbMemoryProfiler, FLOPCounter
from .visualisation import create_wandb_visualisations
import copy
from dataclasses import asdict
import matplotlib.pyplot as plt
import wandb
wandb.require("legacy-service")    



class WandbCircuitBenchmark:
    """Benchmark suite with wandb integration"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
        # Initialize wandb run
        self.run = wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            tags=config.tags,
            notes=config.notes,
            config=asdict(config)
        )

        # Log power-of-two configuration if enabled
        if self.config.powers_of_two:
            wandb.log({
                "config/min_exp": self.config.min_exp,
                "config/max_exp": self.config.max_exp,
            }, step=0)
        
        # Create wandb table for detailed results
        self.results_table = wandb.Table(columns=[
            "n_input", "n_sum", "rank", "batch_size", "matrix_size",
            "orig_time_ms", "nystrom_time_ms", "speedup", "theoretical_speedup",
            "orig_memory_mb", "nystrom_memory_mb", "memory_reduction",
            "orig_gflops", "nystrom_gflops", "flop_reduction",
            "rel_error", "efficiency"
        ])
        
        # Summary metrics
        self.summary_metrics = {
            "speedups": [],
            "memory_reductions": [],
            "errors": [],
            "efficiencies": []
        }
    
    def create_test_input(self, batch_size: int, input_dim: int, device: str):
        """Create test input tensor with correct shape for circuit."""
        # For squared circuit: num_variables = input_dim^2
        num_variables = input_dim ** 2
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
            "median": np.median(times)
        }
    
    
    def benchmark_single_configuration(
        self, 
        n_input: int, 
        n_sum: int, 
        rank: int, 
        batch_size: int,
        step: int
    ) -> Dict:
        """Run benchmark for single configuration with wandb logging"""
        
        # Log current configuration
        if self.config.powers_of_two and n_input == n_sum:
            matrix_label = f"2^{int(np.log2(n_input**2))}"
        else:
            matrix_label = f"{n_input**2}x{n_sum**2}"

        wandb.log({
            "config/n_input": n_input,
            "config/n_sum": n_sum,
            "config/rank": rank,
            "config/batch_size": batch_size,
            "config/matrix_dims": matrix_label,
            "step": step
        })
        
        # Build circuits
        original_circuit = build_and_compile_circuit(n_input, n_sum)
        original_circuit = original_circuit.to(self.config.device).eval()

        

        nystrom_circuit = copy.deepcopy(original_circuit)
        replace_sum_layers(nystrom_circuit,rank=rank)
        fix_address_book_modules(nystrom_circuit)

        # Create test input
        test_input = self.create_test_input(batch_size, n_input, self.config.device)
      
        # Time forward passes
        orig_times = self.time_forward_pass(
            original_circuit, test_input, 
            self.config.num_warmup, self.config.num_trials
        )
        
        nystrom_times = self.time_forward_pass(
            nystrom_circuit, test_input,
            self.config.num_warmup, self.config.num_trials
        )
        
        # Log timing distributions
        wandb.log({
            "timing/original_mean_ms": orig_times["mean"] * 1000,
            "timing/original_std_ms": orig_times["std"] * 1000,
            "timing/nystrom_mean_ms": nystrom_times["mean"] * 1000,
            "timing/nystrom_std_ms": nystrom_times["std"] * 1000,
            "timing/speedup": orig_times["mean"] / nystrom_times["mean"],
            "step": step
        })
        
        # Memory profiling with wandb logging
        orig_memory = WandbMemoryProfiler.profile_and_log(
            original_circuit, test_input, 
            device=self.config.device, 
            prefix="memory/original"
        )
        
        nystrom_memory = WandbMemoryProfiler.profile_and_log(
            nystrom_circuit, test_input,
            device=self.config.device,
            prefix="memory/nystrom"
        )
        
        # FLOP counting
        F = 1  # Number of folds
        orig_flops = FLOPCounter.kronecker_forward(batch_size, F, n_sum, n_input)
        nystrom_flops = FLOPCounter.nystrom_forward(batch_size, F, n_sum, n_input, rank)
        
        wandb.log({
            "flops/original_gflops": orig_flops / 1e9,
            "flops/nystrom_gflops": nystrom_flops / 1e9,
            "flops/reduction": 1 - (nystrom_flops / orig_flops),
            "step": step
        })
        
        # Approximation error
        with torch.no_grad():
            orig_output = original_circuit(test_input)
            nystrom_output = nystrom_circuit(test_input)
            
            abs_error = (orig_output - nystrom_output).norm()
            rel_error = abs_error / orig_output.norm()
            
            # Log error distribution
            error_per_sample = (orig_output - nystrom_output).norm(dim=-1) / orig_output.norm(dim=-1)
            
            wandb.log({
                "accuracy/rel_error": rel_error.item(),
                "accuracy/abs_error": abs_error.item(),
                "accuracy/error_mean": error_per_sample.mean().item(),
                "accuracy/error_std": error_per_sample.std().item(),
                "accuracy/error_max": error_per_sample.max().item(),
                "step": step
            })
        
        # Calculate all metrics
        speedup = orig_times["mean"] / nystrom_times["mean"]
        theoretical_speedup = FLOPCounter.theoretical_speedup(n_sum, n_input, rank)
        memory_reduction = 1 - (nystrom_memory / max(orig_memory, 1e-6))
        efficiency = speedup / theoretical_speedup
        
        # Update summary metrics
        self.summary_metrics["speedups"].append(speedup)
        self.summary_metrics["memory_reductions"].append(memory_reduction)
        self.summary_metrics["errors"].append(rel_error.item())
        self.summary_metrics["efficiencies"].append(efficiency)
        
        # Add row to results table
        self.results_table.add_data(
            n_input, n_sum, rank, batch_size, matrix_label,
            orig_times["mean"] * 1000, nystrom_times["mean"] * 1000,
            speedup, theoretical_speedup,
            orig_memory, nystrom_memory, memory_reduction,
            orig_flops / 1e9, nystrom_flops / 1e9, 1 - (nystrom_flops / orig_flops),
            rel_error.item(), efficiency
        )
        
        # Log efficiency metrics
        wandb.log({
            "efficiency/actual_vs_theoretical": efficiency,
            "efficiency/memory_reduction": memory_reduction,
            "efficiency/speedup": speedup,
            "step": step
        })
        
        return {
            'n_input': n_input,
            'n_sum': n_sum,
            'rank': rank,
            'batch_size': batch_size,
            'speedup': speedup,
            'memory_reduction': memory_reduction,
            'rel_error': rel_error.item(),
            'efficiency': efficiency
        }
    
    def run_full_benchmark(self):
        """Run complete benchmark suite with wandb tracking"""
        
        step = 0
        total_configs = (
            len(self.config.input_units) * 
            len(self.config.sum_units) * 
            len(self.config.ranks) * 
            len(self.config.batch_sizes)
        )
        
        # Create progress bar in wandb
        progress = 0
        
        for n_input in self.config.input_units:
            for n_sum in self.config.sum_units:
                for rank in self.config.ranks:
                    # Skip if rank too large
                    if rank >= min(n_input**2, n_sum**2):
                        continue
                    
                    for batch_size in self.config.batch_sizes:
                        progress += 1
                        wandb.log({"progress": progress / total_configs})
                        
                        print(f"[{progress}/{total_configs}] Benchmarking: "
                              f"input={n_input}, sum={n_sum}, "
                              f"rank={rank}, batch={batch_size}")
                        
                        try:
                            result = self.benchmark_single_configuration(
                                n_input, n_sum, rank, batch_size, step
                            )
                            step += 1
                            
                        except Exception as e:
                            print(f"  Failed: {e}")
                            wandb.log({"errors/count": 1, "errors/message": str(e)})
                            continue
        
        # Log final summary statistics
        self.log_summary_statistics()
        
        # Log results table
        wandb.log({"results_table": self.results_table})
        
        # Create and log visualizations
        create_wandb_visualisations(self.results_table, self.config)
        
        return self.results_table
    
    def log_summary_statistics(self):
        """Log summary statistics to wandb"""
        summary = {
            "summary/avg_speedup": np.mean(self.summary_metrics["speedups"]),
            "summary/max_speedup": np.max(self.summary_metrics["speedups"]),
            "summary/min_speedup": np.min(self.summary_metrics["speedups"]),
            "summary/avg_memory_reduction": np.mean(self.summary_metrics["memory_reductions"]),
            "summary/avg_error": np.mean(self.summary_metrics["errors"]),
            "summary/avg_efficiency": np.mean(self.summary_metrics["efficiencies"]),
        }
        
        # Find best configurations
        speedups = np.array(self.summary_metrics["speedups"])
        errors = np.array(self.summary_metrics["errors"])
        
        # Best speedup with <1% error
        good_accuracy_mask = errors < 0.01
        if good_accuracy_mask.any():
            best_speedup_good_accuracy = speedups[good_accuracy_mask].max()
            summary["summary/best_speedup_1pct_error"] = best_speedup_good_accuracy
        
        wandb.log(summary)
    
