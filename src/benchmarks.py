"""Core benchmarking logic."""

import traceback
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import BenchmarkConfig
from .profilers import WandbMemoryProfiler, FLOPCounter
from .visualisation import create_wandb_visualisations
from dataclasses import asdict
import matplotlib.pyplot as plt
import wandb
from cirkit.backend.torch.layers.inner import TorchSumLayer
from src.nystromlayer import NystromSumLayer
from cirkit.pipeline import PipelineContext, compile as compile_circuit
from cirkit.symbolic.circuit import Circuit
import cirkit.symbolic.functional as SF
from .circuit_types import CIRCUIT_BUILDERS
from cirkit.symbolic.io import plot_circuit
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

LN2 = np.log(2.0)
import os

try:
    wandb.require("legacy-service")
except wandb.errors.UnsupportedError:
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


def sync_sumlayer_weights(
    original: nn.Module, nystrom: nn.Module, *, pivot: str = "uniform", rank: int | None = None
) -> None:
    """Copy weights from ``original`` to ``nystrom`` for matching layers."""
    orig_layers = [m for m in original.modules() if isinstance(m, TorchSumLayer)]
    nys_layers = [m for m in nystrom.modules() if isinstance(m, NystromSumLayer)]
    if len(orig_layers) != len(nys_layers):
        raise ValueError("Layer count mismatch when syncing weights")

    import faulthandler
    faulthandler.enable(all_threads=True)
    total = len(orig_layers)
    interval = max(1, total // 4)

    for i, (o, n) in enumerate(zip(orig_layers, nys_layers), start=1):
        start = time.perf_counter()
        faulthandler.dump_traceback_later(30, repeat=False)
        try:
            if rank is not None:
                n.rank = int(rank)
                n.rank_param.data.fill_(n.rank)
            n.pivot = pivot
            n._build_factors_from(o)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[{datetime.now()}] Exception on layer {i}/{total}: {e}", flush=True)
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()

        if i % interval == 0 or i == total:
            pct = int(100 * i / total)
            print(f"[{datetime.now()}] Weight Sync Progress: {pct}% ({i}/{total})", flush=True)


class WandbCircuitBenchmark:
    """Benchmark suite with wandb integration."""

    def __init__(self, config: BenchmarkConfig, base_symbolic_circuit: str = None):
        self.config = config
        self.base_symbolic_circuit = base_symbolic_circuit
        self.run = wandb.init(
            project=config.project_name, name=config.experiment_name,
            tags=config.tags, notes=config.notes, config=asdict(config),
        )
        if self.config.powers_of_two:
            wandb.log({"config/min_exp": self.config.min_exp, "config/max_exp": self.config.max_exp}, step=0)
        self.results_table = wandb.Table(
            columns=[
                "depth", "n_input", "n_sum", "rank", "sampling_method", "batch_size",
                "matrix_size", "orig_time_ms", "nystrom_time_ms", "speedup",
                "theoretical_speedup", "orig_memory_mb", "nystrom_memory_mb",
                "memory_reduction", "orig_gflops", "nystrom_gflops", "flop_reduction",
                "orig_bpd", "nystrom_bpd", "bpd_diff", "nll_diff", "efficiency",
            ]
        )
        self.summary_metrics = {
            "speedups": [], "memory_reductions": [], "nll_diffs": [],
            "efficiencies": [], "bpd_diffs": [],
        }

    def compute_dynamic_ranks(self, n_input: int, n_sum: int) -> List[int]:
        base_units = min(n_input, n_sum)
        base = base_units ** 2
        percentages = self.config.rank_percentages
        ranks = [max(1, int(base * p)) for p in percentages]
        return sorted(set(ranks))

    def create_test_input(self, batch_size: int, input_dim: int, device: str):
        num_variables = input_dim**2
        print(f"Creating test input of size {batch_size} on {device}", flush=True)
        if self.config.circuit_structure == "MNIST":
            print("MNIST circuit structure detected", flush=True)
            if not hasattr(self, "_mnist_dataset"):
                try:
                    self._mnist_dataset = datasets.MNIST(root="./.data", train=False, download=True)
                    print("MNIST dataset loaded successfully.", flush=True)
                except Exception as e:
                    print(f"Failed to load MNIST dataset: {e}", flush=True)
                    return torch.randint(0, 256, (batch_size, num_variables), device=device)
            dataset = self._mnist_dataset
            idx = torch.randint(len(dataset), (batch_size,))
            images = dataset.data[idx].to(device)
            return images.view(batch_size, 784).long()
        return torch.randn(batch_size, num_variables, device=device)

    def time_forward_pass(self, circuit, test_input, num_warmup, num_trials):
        times = []
        for _ in range(num_warmup):
            _ = circuit(test_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(num_trials):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            _ = circuit(test_input)
            end_time.record()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time) / 1000)
        return {"mean": np.mean(times), "std": np.std(times), "min": np.min(times), "max": np.max(times)}

    def benchmark_single_configuration(
        self, n_input: int, n_sum: int, rank: int, initial_batch_size: int, step: int, *,
        pivot: str, depth: int | None = None,
    ) -> dict:
        """
        Run benchmark on a single GPU, testing models sequentially to save memory.
        Includes automatic batch size reduction on OOM.
        """
        stage = "initialization"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"--- Starting configuration: input={n_input}, sum={n_sum}, rank={rank}, initial_batch={initial_batch_size} on {device} ---", flush=True)

        try:
            # --- Variables to store results from each sequential run ---
            orig_times, nystrom_times = None, None
            orig_output, nystrom_output = None, None
            
            # Build the base symbolic circuit once
            stage = "symbolic_compilation"
            print(f"[{datetime.now()}] Stage: {stage}", flush=True)
            symbolic = SF.multiply(self.base_symbolic_circuit, self.base_symbolic_circuit)

            # =====================================================================
            #  Step 1: Benchmark the Original Circuit
            # =====================================================================
            stage = "original_model_benchmark"
            print(f"[{datetime.now()}] --- Benchmarking ORIGINAL model ---", flush=True)
            original_circuit = compile_symbolic(symbolic, device=device, rank=None, opt=False)
            if self.config.circuit_structure == "MNIST":
                cache_path = f"./model_cache/checkpoints/mnist_{n_input}_{n_sum}_epoch10.pt"
                if os.path.exists(cache_path):
                    original_circuit.load_state_dict(torch.load(cache_path, map_location=device)["model_state_dict"])
                else:
                    raise FileNotFoundError(f"Checkpoint not found at {cache_path}")

            physical_batch_size = initial_batch_size
            while physical_batch_size >= 1:
                try:
                    print(f"[{datetime.now()}] Attempting ORIGINAL with batch size: {physical_batch_size}", flush=True)
                    torch.cuda.empty_cache()
                    test_input = self.create_test_input(physical_batch_size, n_input, device)
                    
                    orig_times = self.time_forward_pass(original_circuit, test_input, self.config.num_warmup, self.config.num_trials)
                    
                    with torch.no_grad():
                        if self.config.circuit_structure == "MNIST":
                            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (255 * x.view(-1)).long())])
                            data_test = datasets.MNIST(root="./.data", train=False, download=True, transform=transform)
                            test_dataloader = DataLoader(data_test, shuffle=False, batch_size=physical_batch_size)
                            orig_batches = [original_circuit(batch.to(device)).real for batch, _ in tqdm(test_dataloader, desc="Collating original")]
                            orig_output = torch.cat(orig_batches, dim=0).to("cpu")
                        else:
                            orig_output = original_circuit(test_input).real.to("cpu")
                    print(f"[{datetime.now()}] Success with ORIGINAL model, batch size {physical_batch_size}", flush=True)
                    break # Success
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[{datetime.now()}] OOM on ORIGINAL model with batch size {physical_batch_size}", flush=True)
                        physical_batch_size //= 2
                        if physical_batch_size < 1:
                            print(f"[{datetime.now()}] Cannot run ORIGINAL model even with batch size 1.", flush=True)
                            wandb.log({"errors/type": "oom_failure_original", "errors/message": str(e), "config/n_input": n_input, "config/n_sum": n_sum, "config/rank": rank, "step": step})
                            return {"status": "oom_failure"}
                        continue
                    else: raise e
            
            # --- Free up memory ---
            #del test_input
            if 'orig_batches' in locals(): del orig_batches
            torch.cuda.empty_cache()
            print(f"[{datetime.now()}] Original model cleared from GPU memory.", flush=True)

            # =====================================================================
            #  Step 2: Benchmark the Nyström Circuit
            # =====================================================================
            stage = "nystrom_model_benchmark"
            print(f"[{datetime.now()}] --- Benchmarking NYSTRÖM model ---", flush=True)
            
            
            nystrom_circuit = compile_symbolic(symbolic, device=device, opt=True, rank=rank)
            print(f"[{datetime.now()}] Syncing weights for Nyström model...", flush=True)
            sync_sumlayer_weights(original_circuit, nystrom_circuit, pivot=pivot, rank=rank)
            del original_circuit # Free GPU memory
            
            # Batch size is already determined by the original model run. We must use the same.
            while physical_batch_size >= 1:
                try:
                    print(f"[{datetime.now()}] Attempting NYSTROM with batch size: {physical_batch_size}", flush=True)
                    torch.cuda.empty_cache()
                    #test_input = self.create_test_input(physical_batch_size, n_input, device)

                    nystrom_times = self.time_forward_pass(nystrom_circuit, test_input, self.config.num_warmup, self.config.num_trials)

                    del test_input  # Free memory after timing

                    with torch.no_grad():
                        if self.config.circuit_structure == "MNIST":
                            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (255 * x.view(-1)).long())])
                            data_test = datasets.MNIST(root="./.data", train=False, download=True, transform=transform)
                            test_dataloader = DataLoader(data_test, shuffle=False, batch_size=physical_batch_size)
                            nyst_batches = [nystrom_circuit(batch.to(device)).real for batch, _ in tqdm(test_dataloader, desc="Collating nystrom")]
                            nystrom_output = torch.cat(nyst_batches, dim=0).to("cpu")
                        else:
                            nystrom_output = nystrom_circuit(test_input).real.to("cpu")
                    print(f"[{datetime.now()}] Success with NYSTROM model, batch size {physical_batch_size}", flush=True)
                    break # Success
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[{datetime.now()}] OOM on NYSTROM model with batch size {physical_batch_size}", flush=True)
                        physical_batch_size //= 2 # Reduce batch size for both models and restart
                        print(f"[{datetime.now()}] Batch size is too large even for Nystrom model. Restarting entire benchmark for this config with new batch size {physical_batch_size}", flush=True)
                        # We need to restart the whole function with a smaller batch size
                        return benchmark_single_configuration(self, n_input, n_sum, rank, physical_batch_size, step, pivot=pivot, depth=depth)
                    else: raise e
            
            # --- Free up memory ---
            del nystrom_circuit
            if 'nyst_batches' in locals(): del nyst_batches
            torch.cuda.empty_cache()
            print(f"[{datetime.now()}] Nystrom model cleared from GPU memory.", flush=True)

            # =====================================================================
            #  Step 3: Calculate Final Metrics on CPU
            # =====================================================================
            stage = "final_metric_calculation"
            print(f"[{datetime.now()}] Stage: {stage}", flush=True)
            nll_orig, nll_nystrom = -orig_output, -nystrom_output
            nll_diff_per_sample = (nll_nystrom - nll_orig).abs()
            nll_diff = nll_diff_per_sample.mean()
            data_dim = n_input ** 2
            if self.config.circuit_structure == "MNIST":
                data_dim = 784
            orig_bpd = (-orig_output.mean() / (data_dim * LN2)).item()
            nystrom_bpd = (-nystrom_output.mean() / (data_dim * LN2)).item()
            bpd_diff = abs(orig_bpd - nystrom_bpd)

            print(f"[{datetime.now()}] Successfully completed run with batch size {physical_batch_size}", flush=True)

                    

            # --- Final Logging and Metrics ---
            matrix_label = f"2^{int(np.log2(n_input**2))}" if self.config.powers_of_two and n_input == n_sum else f"{n_input**2}x{n_sum**2}"
            wandb.log({
                "config/n_input": n_input, "config/n_sum": n_sum, "config/rank": rank,
                "config/physical_batch_size": physical_batch_size, # Log the actual batch size that worked
                "config/matrix_dims": matrix_label, "step": step, "config/pivot": pivot,
                "accuracy/nll_diff": nll_diff.item(), "bpd/original": orig_bpd,
                "bpd/nystrom": nystrom_bpd, "bpd/diff": bpd_diff
            })
            
            # Placeholder values for memory/flops, replace with your actual profilers if available
            orig_memory, nystrom_memory, orig_flops, nystrom_flops = 0, 0, 1, 1

            speedup = orig_times["mean"] / nystrom_times["mean"]
            theoretical_speedup = FLOPCounter.theoretical_speedup(n_sum, n_input, rank)
            memory_reduction = 1 - (nystrom_memory / max(orig_memory, 1e-6))
            efficiency = speedup / theoretical_speedup

            self.summary_metrics["speedups"].append(speedup)
            self.summary_metrics["memory_reductions"].append(memory_reduction)
            self.summary_metrics["nll_diffs"].append(nll_diff.item())
            self.summary_metrics["efficiencies"].append(efficiency)
            self.summary_metrics["bpd_diffs"].append(bpd_diff)

            if self.results_table:
                self.results_table.add_data(
                    depth or self.config.depth, n_input, n_sum, rank, pivot, physical_batch_size,
                    matrix_label, orig_times["mean"] * 1000, nystrom_times["mean"] * 1000, speedup,
                    theoretical_speedup, orig_memory, nystrom_memory, memory_reduction,
                    orig_flops / 1e9, nystrom_flops / 1e9, 1 - (nystrom_flops / orig_flops),
                    orig_bpd, nystrom_bpd, bpd_diff, nll_diff.item(), efficiency
                )
            
            return {"status": "success"}

        except Exception as e:
            print(f"[{datetime.now()}] FATAL UNHANDLED ERROR during stage: '{stage}'", flush=True)
            print(f"  Configuration: input={n_input}, sum={n_sum}, rank={rank}, batch={initial_batch_size}", flush=True)
            print(f"  Error: {traceback.format_exc()}", flush=True)
            wandb.log({"errors/type": "fatal_exception", "errors/message": str(e), "step": step})
            return None

    def run_full_benchmark(self):
        step = 0
        methods = self.config.approximation_methods or [self.config.pivot]
        # ... (The rest of this function remains largely the same, as the complexity
        #      has been moved into benchmark_single_configuration)
        
        # NOTE: This calculation is just for the progress bar and can remain as is.
        total_configs = 0
        depth_range = range(2, self.config.depth + 1) if self.config.circuit_structure == "deep_cp_circuit" else [self.config.depth]
        for depth in depth_range:
            for n_input in self.config.input_units:
                for n_sum in self.config.sum_units:
                    if n_input != n_sum: continue
                    ranks_to_use = self.compute_dynamic_ranks(n_input, n_sum) if self.config.use_dynamic_ranks else self.config.ranks
                    for rank in ranks_to_use:
                        if rank >= min(n_input**2, n_sum**2): continue
                        total_configs += len(self.config.batch_sizes) * len(methods)

        progress = 0
        for depth in depth_range:
            for n_input in self.config.input_units:
                for n_sum in self.config.sum_units:
                    if n_input != n_sum: continue
                    ranks_to_use = self.compute_dynamic_ranks(n_input, n_sum) if self.config.use_dynamic_ranks else self.config.ranks
                    for rank in ranks_to_use:
                        if rank >= min(n_input**2, n_sum**2): continue
                        for batch_size in self.config.batch_sizes:
                            for pivot in methods:
                                progress += 1
                                wandb.log({"progress": progress / total_configs})
                                print(f"--- [{progress}/{total_configs}] Starting Benchmark Config ---", flush=True)
                                
                                try:
                                    builder = CIRCUIT_BUILDERS[self.config.circuit_structure]
                                    builder_kwargs = {"num_input_units": n_input, "num_sum_units": n_sum}
                                    if self.config.circuit_structure == "deep_cp_circuit":
                                        builder_kwargs["depth"] = depth
                                    if self.config.circuit_structure == "MNIST":
                                        builder_kwargs["region_graph"] = self.config.region_graph
                                    
                                    self.base_symbolic_circuit = builder(**builder_kwargs)
                                    result = self.benchmark_single_configuration(
                                        n_input, n_sum, rank, batch_size, step,
                                        pivot=pivot, depth=depth
                                    )
                                    if result is not None:
                                        step += 1
                                except Exception as e:
                                    print(f"  Outer loop caught a fatal error: {e}", flush=True)
                                    tb_str = traceback.format_exc()
                                    print(f"Error details:\n{tb_str}", flush=True)
                                    wandb.log({"errors/count": 1, "errors/message": str(e)})
                                    continue
        
        self.log_summary_statistics()
        wandb.log({"results_table": self.results_table})
        create_wandb_visualisations(self.results_table, self.config)
        return self.results_table

    def log_summary_statistics(self):
        """Log summary statistics to wandb"""
        if not self.summary_metrics["speedups"]:
             print("No successful runs to summarize.", flush=True)
             return
        summary = {
            "summary/avg_speedup": np.mean(self.summary_metrics["speedups"]),
            "summary/max_speedup": np.max(self.summary_metrics["speedups"]),
            "summary/min_speedup": np.min(self.summary_metrics["speedups"]),
            "summary/avg_memory_reduction": np.mean(self.summary_metrics["memory_reductions"]),
            "summary/avg_nll_diff": np.mean(self.summary_metrics["nll_diffs"]),
            "summary/avg_efficiency": np.mean(self.summary_metrics["efficiencies"]),
            "summary/avg_bpd_diff": np.mean(self.summary_metrics["bpd_diffs"]),
        }
        speedups = np.array(self.summary_metrics["speedups"])
        errors = np.array(self.summary_metrics["nll_diffs"])
        good_accuracy_mask = errors < 0.01
        if good_accuracy_mask.any():
            summary["summary/best_speedup_low_nll"] = speedups[good_accuracy_mask].max()
        wandb.log(summary)