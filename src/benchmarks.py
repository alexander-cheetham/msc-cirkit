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
        wandb.log({
            "config/n_input": n_input,
            "config/n_sum": n_sum,
            "config/rank": rank,
            "config/batch_size": batch_size,
            "config/matrix_dims": f"{n_input**2}x{n_sum**2}",
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
            n_input, n_sum, rank, batch_size, f"{n_input**2}x{n_sum**2}",
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
        self.create_wandb_visualizations()
        
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
    
    def create_wandb_visualizations(self):
        """Create custom visualizations for wandb using raw data."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract raw data from wandb table
        if not self.results_table.data:
            print("No data to visualize")
            return
        
        # Get column indices
        columns = self.results_table.columns
        col_indices = {col: idx for idx, col in enumerate(columns)}
        
        # Extract data as numpy arrays for easier manipulation
        data_array = np.array(self.results_table.data)
        
        # Extract specific columns
        n_inputs = data_array[:, col_indices['n_input']].astype(int)
        ranks = data_array[:, col_indices['rank']].astype(int)
        speedups = data_array[:, col_indices['speedup']].astype(float)
        rel_errors = data_array[:, col_indices['rel_error']].astype(float)
        efficiencies = data_array[:, col_indices['efficiency']].astype(float)
        matrix_sizes = data_array[:, col_indices['matrix_size']]
        
        # Get unique values
        unique_n_inputs = sorted(set(n_inputs))
        unique_ranks = sorted(set(ranks))
        unique_matrix_sizes = sorted(set(matrix_sizes))
        
        # 1. Speedup vs Rank scatter plot
        fig_speedup = plt.figure(figsize=(10, 6))
        for n in unique_n_inputs:
            # Filter data for this n_input
            mask = n_inputs == n
            n_ranks = ranks[mask]
            n_speedups = speedups[mask]
            
            plt.scatter(n_ranks, n_speedups, label=f'n={n}', s=50, alpha=0.7)
        
        plt.xlabel('Rank')
        plt.ylabel('Speedup Factor')
        plt.title('Speedup vs Rank')
        plt.legend()
        plt.grid(True, alpha=0.3)
        wandb.log({"charts/speedup_vs_rank": wandb.Image(fig_speedup)})
        plt.close()
        
        # 2. Error vs Rank (log scale)
        fig_error = plt.figure(figsize=(10, 6))
        for n in unique_n_inputs:
            # Filter data for this n_input
            mask = n_inputs == n
            n_ranks = ranks[mask]
            n_errors = rel_errors[mask]
            
            # Sort by rank for line plot
            sort_idx = np.argsort(n_ranks)
            n_ranks_sorted = n_ranks[sort_idx]
            n_errors_sorted = n_errors[sort_idx]
            
            plt.semilogy(n_ranks_sorted, n_errors_sorted, 'o-', 
                        label=f'n={n}', markersize=8)
        
        plt.xlabel('Rank')
        plt.ylabel('Relative Error')
        plt.title('Approximation Error vs Rank')
        plt.legend()
        plt.grid(True, alpha=0.3)
        wandb.log({"charts/error_vs_rank": wandb.Image(fig_error)})
        plt.close()
        
        # 3. Trade-off: Error vs Speedup with rank as color
        fig_tradeoff = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            speedups, rel_errors, 
            c=ranks, cmap='viridis', s=50, alpha=0.7
        )
        plt.xlabel('Speedup Factor')
        plt.ylabel('Relative Error')
        plt.yscale('log')
        plt.title('Accuracy vs Performance Trade-off')
        plt.colorbar(scatter, label='Rank')
        plt.grid(True, alpha=0.3)
        wandb.log({"charts/tradeoff": wandb.Image(fig_tradeoff)})
        plt.close()
        
        # 4. Efficiency heatmap (manual implementation without seaborn)
        fig_efficiency = plt.figure(figsize=(12, 8))
        
        # Create efficiency matrix manually
        efficiency_matrix = np.full((len(unique_ranks), len(unique_matrix_sizes)), np.nan)
        
        for i, rank in enumerate(unique_ranks):
            for j, mat_size in enumerate(unique_matrix_sizes):
                # Find matching entry
                mask = (ranks == rank) & (matrix_sizes == mat_size)
                if mask.any():
                    efficiency_matrix[i, j] = efficiencies[mask][0]
        
        # Create heatmap manually
        im = plt.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto')
        plt.colorbar(im, label='Efficiency')
        
        # Set ticks and labels
        plt.xticks(range(len(unique_matrix_sizes)), unique_matrix_sizes, rotation=45)
        plt.yticks(range(len(unique_ranks)), unique_ranks)
        plt.xlabel('Matrix Size')
        plt.ylabel('Rank')
        
        # Add text annotations
        for i in range(len(unique_ranks)):
            for j in range(len(unique_matrix_sizes)):
                if not np.isnan(efficiency_matrix[i, j]):
                    plt.text(j, i, f'{efficiency_matrix[i, j]:.2f}',
                            ha='center', va='center')
        
        plt.title('Efficiency: Actual/Theoretical Speedup')
        plt.tight_layout()
        wandb.log({"charts/efficiency_heatmap": wandb.Image(fig_efficiency)})
        plt.close()
        
        # 5. Additional useful plots
        
        # Memory reduction vs Matrix size
        fig_memory = plt.figure(figsize=(10, 6))
        memory_reductions = data_array[:, col_indices['memory_reduction']].astype(float)
        
        for rank in unique_ranks:
            mask = ranks == rank
            sizes = matrix_sizes[mask]
            mem_red = memory_reductions[mask]
            
            # Sort for plotting
            unique_sizes_for_rank = sorted(set(sizes))
            avg_mem_red = []
            for size in unique_sizes_for_rank:
                size_mask = (matrix_sizes == size) & (ranks == rank)
                if size_mask.any():
                    avg_mem_red.append(memory_reductions[size_mask].mean())
            
            if avg_mem_red:
                plt.plot(range(len(unique_sizes_for_rank)), avg_mem_red, 
                        'o-', label=f'rank={rank}', markersize=8)
        
        plt.xticks(range(len(unique_matrix_sizes)), unique_matrix_sizes, rotation=45)
        plt.xlabel('Matrix Size')
        plt.ylabel('Memory Reduction')
        plt.title('Memory Reduction by Matrix Size and Rank')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        wandb.log({"charts/memory_reduction": wandb.Image(fig_memory)})
        plt.close()