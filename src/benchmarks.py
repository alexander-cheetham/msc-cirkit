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
from cirkit.backend.torch.queries import IntegrateQuery
from cirkit.backend.torch.layers import TorchCategoricalLayer
from cirkit.utils.scope import Scope
import faulthandler
from torch.utils.data import DataLoader, TensorDataset
import re
import random

LN2 = np.log(2.0)
import os

try:
    wandb.require("legacy-service")
except wandb.errors.UnsupportedError:
    pass


def load_mnist_weights_for_one_sum(one_sum_circuit: torch.nn.Module, mnist_checkpoint_path: str, device: str):
    """
    Loads weights from a pre-trained MNIS circuit checkpoint into a one_sum circuit.
    """
    print(f"--- Loading MNIST weights for one_sum circuit from {mnist_checkpoint_path} ---")

    # 1. Parse n_input and n_sum from the checkpoint path
    match = re.search(r'mnist_(\d+)_(\d+)_epoch10.pt', os.path.basename(mnist_checkpoint_path))
    if not match:
        raise ValueError(f"Could not parse n_input and n_sum from checkpoint path: {mnist_checkpoint_path}")
    mnist_n_input = int(match.group(1))
    mnist_n_sum = int(match.group(2))
    print(f"Inferred MNIST_COMPLEX model dimensions from checkpoint name: n_input={mnist_n_input}, n_sum={mnist_n_sum}")

    # 2. Load the state dictionary from the MNIST checkpoint
    checkpoint = torch.load(mnist_checkpoint_path, map_location=device)
    
    # 3. Create a temporary instance of the MNIST_COMPLEX circuit to hold the weights
    mnist_builder = CIRCUIT_BUILDERS["MNIST"] 
    mnist_symbolic = mnist_builder(num_input_units=mnist_n_input, num_sum_units=mnist_n_sum,region_graph="quad-tree-4")
    mnist_symbolic = SF.multiply(mnist_symbolic, mnist_symbolic)  # Scale the weights by 0.5
    mnist_circuit = compile_symbolic(mnist_symbolic, device=device)
    mnist_circuit.load_state_dict(checkpoint["model_state_dict"])

    # 4. Find all TorchSumLayers in the loaded MNIST_COMPLEX circuit
    all_source_sum_layers = [m for m in mnist_circuit.modules() if isinstance(m, TorchSumLayer)]
    if not all_source_sum_layers:
        raise ValueError("No TorchSumLayer found in the loaded MNIST circuit.")
    
    # Select a random TorchSumLayer to be the source
    source_sum_layer = random.choice(all_source_sum_layers)
    print(f"Randomly selected a TorchSumLayer to copy weights from.")

    # 5. Find the target TorchSumLayer in the one_sum circuit
    target_sum_layer = next((m for m in one_sum_circuit.modules() if isinstance(m, TorchSumLayer)), None)
    if target_sum_layer is None:
        raise ValueError("No TorchSumLayer found in the target one_sum circuit.")
    
    # 6. Access the underlying TorchParameterNode and its torch.nn.Parameter.
    if not source_sum_layer.weight.nodes:
        raise ValueError("Source TorchParameter has no nodes.")
    if not target_sum_layer.weight.nodes:
        raise ValueError("Target TorchParameter has no nodes.")

    source_node = source_sum_layer.weight.nodes[0]
    target_node = target_sum_layer.weight.nodes[0]

    source_param = next(source_node.parameters(), None)
    target_param = next(target_node.parameters(), None)

    if source_param is None:
        raise ValueError("Source TorchParameterNode has no torch.nn.Parameter.")
    if target_param is None:
        raise ValueError("Target TorchParameterNode has no torch.nn.Parameter.")

    # 7. Perform a direct data copy.
    print(f"Copying weights of shape {source_param.data.shape}")
    with torch.no_grad():
        target_param.data.copy_(source_param.data)

    print("--- Finished loading MNIST weights for one_sum circuit ---")
    return one_sum_circuit


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
    print(f"Compiled circuit with rank {rank} on device {device}", flush=True)
    return compiled


def sync_sumlayer_weights(
    original: nn.Module, nystrom: nn.Module, *, pivot: str = "uniform", rank: int | None = None
) -> None:
    """Copy weights from ``original`` to ``nystrom`` for matching layers."""
    # Sync for TorchSumLayer and NystromSumLayer
    orig_sum_layers = [m for m in original.modules() if isinstance(m, TorchSumLayer)]
    nys_sum_layers = [m for m in nystrom.modules() if isinstance(m, NystromSumLayer)]
    if len(orig_sum_layers) != len(nys_sum_layers):
        print(f"{len(orig_sum_layers)},{len(nys_sum_layers)}")
        raise ValueError("Sum layer count mismatch when syncing weights")

    faulthandler.enable(all_threads=True)
    total_sum = len(orig_sum_layers)
    interval_sum = max(1, total_sum // 4)

    for i, (o, n) in enumerate(zip(orig_sum_layers, nys_sum_layers), start=1):
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
            print(f"[{datetime.now()}] Exception on sum layer {i}/{total_sum}: {e}", flush=True)
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()

        if i % interval_sum == 0 or i == total_sum:
            pct = int(100 * i / total_sum)
            print(f"[{datetime.now()}] Sum Layer Weight Sync Progress: {pct}% ({i}/{total_sum})", flush=True)

    # --- Step 2: Sync Categorical Layers (NEW & IMPROVED LOGIC) ---
    print("\n--- Syncing Categorical Layer (logits/probs) by Parameter Name ---", flush=True)

    # Create a dictionary of the original model's parameters for fast lookup.
    # This is the 'a' from your working snippet.
    original_params = dict(original.named_parameters())

    copied_params_count = 0
    with torch.no_grad():
        # Iterate through the Nystrom model's named parameters.
        for name, nys_param in nystrom.named_parameters():
            # Check if the parameter is a logit or probability tensor by its name.
            if 'logits' in name or 'probs' in name:
                if name in original_params:
                    # Use the proven method: copy the data directly.
                    # .copy_() is a safe, in-place operation.
                    orig_param = original_params[name]
                    nys_param.data.copy_(orig_param.data)
                    copied_params_count += 1
                else:
                    # This warning is helpful for debugging architecture mismatches.
                    print(f"Warning: Parameter '{name}' found in Nystrom model but not in original.")
    
    print(f"--- Finished syncing. Copied {copied_params_count} categorical parameters. ---\n",)

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
        if self.config.circuit_structure == "MNIST" or self.config.circuit_structure == "MNIST_COMPLEX":
            print(f"{self.config.circuit_structure} circuit structure detected", flush=True)
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

    def benchmark_nystrom_repeated(
        self,
        symbolic_circuit,
        original_circuit,
        test_input,
        orig_output,
        Z_bok_orig,
        orig_times,
        orig_memory,
        n_input,
        n_sum,
        rank,
        physical_batch_size,
        pivot,
        device,
        reps,
        data_test_mnist=None,
    ):
        """Benchmark Nystrom circuit multiple times and aggregate results."""
        nystrom_metrics = {
            "times": [],
            "memories": [],
            "nll_diffs": [],
            "bpd_diffs": [],
            "speedups": [],
            "efficiencies": [],
        }

        for i in range(reps):
            print(f"--- Nyström repetition {i+1}/{reps} ---")
            torch.cuda.empty_cache()

            nystrom_circuit = compile_symbolic(symbolic_circuit, device=device, opt=True, rank=rank)
            sync_sumlayer_weights(original_circuit, nystrom_circuit, pivot=pivot, rank=rank)

            nystrom_times = self.time_forward_pass(nystrom_circuit, test_input, self.config.num_warmup, self.config.num_trials)

            def nystrom_forward():
                return nystrom_circuit(test_input)
            device_type = 'cuda' if device.startswith('cuda') else 'cpu'
            nystrom_memory, _ = WandbMemoryProfiler.profile_gpu(nystrom_forward) if device_type == 'cuda' else WandbMemoryProfiler.profile_cpu(nystrom_forward)

            with torch.no_grad():
                if self.config.circuit_structure in ("MNIST", "MNIST_COMPLEX"):
                    data_test = data_test_mnist
                    test_dataloader = DataLoader(data_test, shuffle=False, batch_size=physical_batch_size)
                    nyst_batches = [nystrom_circuit(batch.to(device)).real for batch, _ in tqdm(test_dataloader, desc=f"Collating nystrom (rep {i+1})")]
                    nystrom_output = torch.cat(nyst_batches, dim=0).to("cpu")
                else:
                    nyst_batches = []
                    for j in range(0, test_input.size(0), physical_batch_size):
                        batch = test_input[j : j + physical_batch_size]
                        output_batch = nystrom_circuit(batch).real.to("cpu")
                        nyst_batches.append(output_batch)
                    nystrom_output = torch.cat(nyst_batches, dim=0)
                    sample_for_dataset = test_input[0:1]
                    fake_label = torch.ones(1, dtype=torch.long)
                    data_test = TensorDataset(sample_for_dataset, fake_label)

            iq_nystrom = IntegrateQuery(nystrom_circuit)
            sample_image, _ = next(iter(DataLoader(data_test, batch_size=1)))
            Z_bok_nys = iq_nystrom(sample_image.to(device), integrate_vars=Scope(nystrom_circuit.scope)).to("cpu")

            del nystrom_circuit
            if 'nyst_batches' in locals(): del nyst_batches
            torch.cuda.empty_cache()

            nll_orig = -(orig_output - Z_bok_orig[0][0].real)
            nll_nystrom = -(nystrom_output - Z_bok_nys[0][0].real)
            nll_diff_per_sample = (nll_nystrom - nll_orig).abs()
            nll_diff = nll_diff_per_sample.mean().item()

            data_dim = n_input ** 2
            if self.config.circuit_structure in ("MNIST", "MNIST_COMPLEX"):
                data_dim = 784
            orig_bpd = (-orig_output.mean() / (data_dim * LN2)).item()
            nystrom_bpd = (-nystrom_output.mean() / (data_dim * LN2)).item()
            bpd_diff = abs(orig_bpd - nystrom_bpd)

            speedup = orig_times["mean"] / nystrom_times["mean"]
            theoretical_speedup = FLOPCounter.theoretical_speedup(n_sum, n_input, rank)
            efficiency = speedup / theoretical_speedup

            nystrom_metrics["times"].append(nystrom_times["mean"])
            nystrom_metrics["memories"].append(nystrom_memory)
            nystrom_metrics["nll_diffs"].append(nll_diff)
            nystrom_metrics["bpd_diffs"].append(bpd_diff)
            nystrom_metrics["speedups"].append(speedup)
            nystrom_metrics["efficiencies"].append(efficiency)

        aggregated_metrics = {
            "time_mean": np.mean(nystrom_metrics["times"]),
            "time_std": np.std(nystrom_metrics["times"]),
            "time_min": np.min(nystrom_metrics["times"]),
            "time_max": np.max(nystrom_metrics["times"]),
            "memory_mean": np.mean(nystrom_metrics["memories"]),
            "memory_std": np.std(nystrom_metrics["memories"]),
            "nll_diff_mean": np.mean(nystrom_metrics["nll_diffs"]),
            "nll_diff_std": np.std(nystrom_metrics["nll_diffs"]),
            "bpd_diff_mean": np.mean(nystrom_metrics["bpd_diffs"]),
            "bpd_diff_std": np.std(nystrom_metrics["bpd_diffs"]),
            "speedup_mean": np.mean(nystrom_metrics["speedups"]),
            "speedup_std": np.std(nystrom_metrics["speedups"]),
            "efficiency_mean": np.mean(nystrom_metrics["efficiencies"]),
            "efficiency_std": np.std(nystrom_metrics["efficiencies"]),
        }
        return aggregated_metrics

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
        print(f"--- Starting configuration: input={n_input}, sum={n_sum}, rank={rank}, initial_batch={initial_batch_size} on {device}, sampling: {pivot} ---", flush=True)

        try:
            # --- Variables to store results from each sequential run ---
            orig_times, nystrom_times = None, None
            orig_output, nystrom_output = None, None
            orig_memory, nystrom_memory = 0, 0
            
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
            
            # if self.config.circuit_structure == "one_sum":
            #     mnist_checkpoint_path = f"./model_cache/checkpoints/mnist_{n_input}_{n_sum}_epoch10.pt"
            #     if os.path.exists(mnist_checkpoint_path):
            #         original_circuit = load_mnist_weights_for_one_sum(original_circuit, mnist_checkpoint_path, device)
            #     else:
            #         print(f"Info: MNIST_COMPLEX checkpoint not found at {mnist_checkpoint_path}. Using random weights for one_sum circuit.")
            
            if self.config.circuit_structure in ("MNIST", "MNIST_COMPLEX"):
                suffix = "mnist" if self.config.circuit_structure == "MNIST" else "mnist_complex"
                cache_path = f"./model_cache/checkpoints/{suffix}_{n_input}_{n_sum}_epoch10.pt"

                if not os.path.exists(cache_path):
                    raise FileNotFoundError(f"Checkpoint not found at {cache_path}")

                checkpoint = torch.load(cache_path, map_location=device)
                original_circuit.load_state_dict(checkpoint["model_state_dict"])

            physical_batch_size = initial_batch_size
            while physical_batch_size >= 1:
                try:
                    print(f"[{datetime.now()}] Attempting ORIGINAL with batch size: {physical_batch_size}", flush=True)
                    torch.cuda.empty_cache()
                    test_input = self.create_test_input(physical_batch_size, n_input, device)
                    
                    orig_times = self.time_forward_pass(original_circuit, test_input, self.config.num_warmup, self.config.num_trials)
                    
                    def orig_forward():
                        return original_circuit(test_input)
                    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
                    orig_memory, _ = WandbMemoryProfiler.profile_gpu(orig_forward) if device_type == 'cuda' else WandbMemoryProfiler.profile_cpu(orig_forward)

                    with torch.no_grad():
                        if self.config.circuit_structure == "MNIST" or self.config.circuit_structure == "MNIST_COMPLEX":
                            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (255 * x.view(-1)).long())])
                            data_test = datasets.MNIST(root="./.data", train=False, download=True, transform=transform)
                            test_dataloader = DataLoader(data_test, shuffle=False, batch_size=physical_batch_size)
                            orig_batches = [original_circuit(batch.to(device)).real for batch, _ in tqdm(test_dataloader, desc="Collating original")]
                            orig_output = torch.cat(orig_batches, dim=0).to("cpu")
                        else:
                            orig_output = original_circuit(test_input).real.to("cpu")
                            sample_for_dataset = test_input[0:1]
                            fake_label = torch.ones(1, dtype=torch.long)
                            data_test = TensorDataset(sample_for_dataset, fake_label)
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
            iq_orig = IntegrateQuery(original_circuit)
            sample_image, _ = next(iter(DataLoader(data_test, batch_size=1)))
            Z_bok_orig = iq_orig(sample_image.to(device), integrate_vars=Scope(original_circuit.scope)).to("cpu")
            
            if 'orig_batches' in locals(): del orig_batches
            torch.cuda.empty_cache()
            print(f"[{datetime.now()}] Original model's results computed.", flush=True)

            # =====================================================================
            #  Step 2: Benchmark the Nyström Circuit Repeatedly
            # =====================================================================
            stage = "nystrom_model_benchmark_repeated"
            print(f"[{datetime.now()}] --- Benchmarking NYSTRÖM model ({self.config.reps} repetitions) ---", flush=True)
            
            data_test_mnist = data_test if self.config.circuit_structure in ("MNIST", "MNIST_COMPLEX") else None
            aggregated_metrics = self.benchmark_nystrom_repeated(
                symbolic, original_circuit, test_input, orig_output, Z_bok_orig,
                orig_times, orig_memory, n_input, n_sum, rank, physical_batch_size,
                pivot, device, self.config.reps, data_test_mnist=data_test_mnist
            )
            
            del original_circuit
            del test_input
            torch.cuda.empty_cache()
            print(f"[{datetime.now()}] Nystrom benchmark repetitions complete.", flush=True)

            # =====================================================================
            #  Step 3: Calculate Final Metrics on CPU
            # =====================================================================
            stage = "final_metric_calculation"
            print(f"[{datetime.now()}] Stage: {stage}", flush=True)
            
            data_dim = n_input ** 2
            if self.config.circuit_structure in ("MNIST", "MNIST_COMPLEX"):
                data_dim = 784
            orig_bpd = (-orig_output.mean() / (data_dim * LN2)).item()

            # --- Final Logging and Metrics ---
            matrix_label = f"2^{int(np.log2(n_input**2))}" if self.config.powers_of_two and n_input == n_sum else f"{n_input**2}x{n_sum**2}"
            wandb.log({
                "config/n_input": n_input, "config/n_sum": n_sum, "config/rank": rank,
                "config/physical_batch_size": physical_batch_size,
                "config/matrix_dims": matrix_label, "step": step, "config/pivot": pivot,
                "accuracy/nll_diff_mean": aggregated_metrics["nll_diff_mean"],
                "accuracy/nll_diff_std": aggregated_metrics["nll_diff_std"],
                "bpd/original": orig_bpd,
                "bpd/diff_mean": aggregated_metrics["bpd_diff_mean"],
                "bpd/diff_std": aggregated_metrics["bpd_diff_std"],
                "speedup/mean": aggregated_metrics["speedup_mean"],
                "speedup/std": aggregated_metrics["speedup_std"],
                "efficiency/mean": aggregated_metrics["efficiency_mean"],
                "efficiency/std": aggregated_metrics["efficiency_std"],
                "time/nystrom_mean_ms": aggregated_metrics["time_mean"] * 1000,
                "time/nystrom_std_ms": aggregated_metrics["time_std"] * 1000,
                "memory/nystrom_mean_mb": aggregated_metrics["memory_mean"],
                "memory/nystrom_std_mb": aggregated_metrics["memory_std"],
            })
            
            theoretical_speedup = FLOPCounter.theoretical_speedup(n_sum, n_input, rank)
            memory_reduction = 1 - (aggregated_metrics["memory_mean"] / max(orig_memory, 1e-6))

            self.summary_metrics["speedups"].append(aggregated_metrics["speedup_mean"])
            self.summary_metrics["memory_reductions"].append(memory_reduction)
            self.summary_metrics["nll_diffs"].append(aggregated_metrics["nll_diff_mean"])
            self.summary_metrics["efficiencies"].append(aggregated_metrics["efficiency_mean"])
            self.summary_metrics["bpd_diffs"].append(aggregated_metrics["bpd_diff_mean"])

            if self.results_table:
                self.results_table.add_data(
                    depth or self.config.depth, n_input, n_sum, rank, pivot, physical_batch_size,
                    matrix_label, orig_times["mean"] * 1000, aggregated_metrics["time_mean"] * 1000,
                    aggregated_metrics["speedup_mean"],
                    theoretical_speedup, orig_memory, aggregated_metrics["memory_mean"], memory_reduction,
                    1, 1, 0, # flops are placeholders
                    orig_bpd, orig_bpd + aggregated_metrics["bpd_diff_mean"],
                    aggregated_metrics["bpd_diff_mean"],
                    aggregated_metrics["nll_diff_mean"],
                    aggregated_metrics["efficiency_mean"]
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
                                    if self.config.circuit_structure == "MNIST" or self.config.circuit_structure == "MNIST_COMPLEX":
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