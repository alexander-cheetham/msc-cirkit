import time
import copy
from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from tqdm import trange,tqdm
import traceback
import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming these imports are in your project structure
from src.benchmarks import compile_symbolic, load_mnist_weights_for_one_sum
from src.circuit_types import CIRCUIT_BUILDERS
from src.nystromlayer import NystromSumLayer
from cirkit.backend.torch.layers.inner import TorchSumLayer
import cirkit.symbolic.functional as SF
from cirkit.backend.torch.queries import IntegrateQuery
from cirkit.utils.scope import Scope
from src.benchmarks import sync_sumlayer_weights

import cProfile
import pstats

LN2 = np.log(2.0)

def create_test_input( batch_size: int, input_dim: int, device: str):
        num_variables = input_dim**2
        print(f"Creating test input of size {batch_size} on {device}", flush=True)
        return torch.randn(batch_size, num_variables, device=device)



def _flatten_pairs(pairs: torch.Tensor, base: int) -> torch.Tensor:
    """Convert (k,2) pairs to flat indices using ``base``."""
    return pairs[:, 0] * base + pairs[:, 1]

def _compute_bpd(model: torch.nn.Module, dataloader: DataLoader, device: str) -> float:
    """Evaluate global bits-per-dimension of ``model`` on ``dataloader``."""
    model.eval()
    total = 0.0
    count = 0
    data_dim = None
    iq_nystrom = IntegrateQuery(model)
    sample_image, _ = next(iter(dataloader))
    Z_bok_nys = iq_nystrom(sample_image.to(device), integrate_vars=Scope(model.scope))
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            if data_dim is None:
                data_dim = batch.shape[1]
            out = model(batch).real - Z_bok_nys[0][0].real
            total += out.mean().item()
            count += 1
    
    return (total / (count * data_dim * LN2))

def _evaluate_with_pivots(model: torch.nn.Module, layer_path: str,
                           I_flat: torch.Tensor, J_flat: torch.Tensor,
                           dataloader: DataLoader, device: str,return_model=False):
    """Return BPD for ``model`` where ``layer_path`` is approximated using pivots.

    ``layer_path`` is the dotted module path (as returned by ``named_modules``).
    """
    model_copy = copy.deepcopy(model)
    target = dict(model_copy.named_modules())[layer_path]
    nystrom = NystromSumLayer(target, rank=len(I_flat),)
    pivots = [(torch.tensor(I_flat, dtype=torch.long, device=device), torch.tensor(J_flat, dtype=torch.long, device=device))] * target.num_folds
    nystrom._build_factors_from(target, pivots=pivots)
    # replace layer
    parent_path, _, attr = layer_path.rpartition(".")
    parent = dict(model_copy.named_modules())[parent_path] if parent_path else model_copy
    setattr(parent, attr, nystrom)
    # --- DEBUG: Verify layer replacement ---
    new_layer = dict(model_copy.named_modules())[layer_path]
    # print(f"DEBUG: Layer '{layer_path}' has been replaced with: {type(new_layer).__name__}")
    # --- END DEBUG ---
    if return_model:
        return _compute_bpd(model_copy, dataloader, device), model_copy
    else:
        return _compute_bpd(model_copy, dataloader, device)

def _evaluate_against_local(model: torch.nn.Module, layer_path: str,
                           dataloader: DataLoader, device: str,rank) -> float:
    """Return BPD for ``model`` where ``layer_path`` is approximated using pivots.

    ``layer_path`` is the dotted module path (as returned by ``named_modules``).
    """
    model_copy = copy.deepcopy(model)
    target = dict(model_copy.named_modules())[layer_path]
    nystrom = NystromSumLayer(target, rank=rank,pivot="cur")
    nystrom._build_factors_from(target,)
    # replace layer
    parent_path, _, attr = layer_path.rpartition(".")
    parent = dict(model_copy.named_modules())[parent_path] if parent_path else model_copy
    setattr(parent, attr, nystrom)
    # --- DEBUG: Verify layer replacement ---
    new_layer = dict(model_copy.named_modules())[layer_path]
    # print(f"DEBUG: Layer '{layer_path}' has been replaced with: {type(new_layer).__name__}")
    # --- END DEBUG ---
    return _compute_bpd(model_copy, dataloader, device)

def _topk_columns_via_l2(M: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return flat column indices of ``k`` columns of Kron(M, M) with the largest L2 scores,
    using torch.linalg.norm for clarity and correctness.
    """
    col_l2_norms = torch.linalg.norm(M, ord=2, dim=0)
    col_l2_norms_sq = col_l2_norms.pow(2)
    scores = torch.outer(col_l2_norms_sq, col_l2_norms_sq).flatten()
    topk_indices = torch.topk(scores, k).indices
    return topk_indices



def select_rows_gold_standard(model: torch.nn.Module, layer_path: str, k: int,
                              dataloader: DataLoader, device: str) -> torch.Tensor:
    """Greedy row selection using *all* columns as reference set."""
    original_layer = dict(model.named_modules())[layer_path]
    
    base_weight = original_layer.weight._nodes[0]()
    _, K_o_base, K_i_base = base_weight.shape
    K_i = K_i_base**2
    J_ref_flat = list(range(0, K_i, 1))
    
    available = J_ref_flat.copy()
    selected = []

    for t in trange(k, desc="Gold Row Selection"):
        def eval_cand(cand):
            I_flat = selected + [cand]
            with torch.inference_mode():
                model_copy = copy.deepcopy(model)
                layer_copy = dict(model_copy.named_modules())[layer_path]
                parent_path, _, attr = layer_path.rpartition(".")
                parent_copy = dict(model_copy.named_modules())[parent_path] if parent_path else model_copy
                
                nystrom = NystromSumLayer(layer_copy, rank=len(I_flat))
                pivots = [(torch.tensor(I_flat, dtype=torch.long, device=device),
                           torch.tensor(J_ref_flat, dtype=torch.long, device=device))] * layer_copy.num_folds
                nystrom._build_factors_from(layer_copy, pivots=pivots)
                
                setattr(parent_copy, attr, nystrom)
                bpd = _compute_bpd(model_copy, dataloader, device)
            return cand, bpd

        best_bpd = float("inf")
        best_idx = None
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(eval_cand, cand) for cand in available]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating Cands (k={t+1})", leave=False):
                cand, bpd = fut.result()
                if bpd < best_bpd:
                    best_bpd, best_idx = bpd, cand
        selected.append(best_idx)
        available.remove(best_idx)
    return torch.tensor(selected, dtype=torch.long, device=device)

def select_columns_given_rows(model: torch.nn.Module, layer_path: str, k: int,
                               I_pairs: torch.Tensor, dataloader: DataLoader, device: str) -> torch.Tensor:
    """Greedy column selection given a fixed set of rows."""
    original_layer = dict(model.named_modules())[layer_path]
    
    base_weight = original_layer.weight._nodes[0]()
    _, K_o_base, K_i_base = base_weight.shape
    K_i = K_i_base**2
    available = list(range(0, K_i, 1))
    selected = []
    
    for t in trange(k, desc="Gold Col Selection"):
        def eval_cand(cand):
            J_flat = selected + [cand]
            with torch.inference_mode():
                model_copy = copy.deepcopy(model)
                layer_copy = dict(model_copy.named_modules())[layer_path]
                parent_path, _, attr = layer_path.rpartition(".")
                parent_copy = dict(model_copy.named_modules())[parent_path] if parent_path else model_copy
                
                nystrom = NystromSumLayer(layer_copy, rank=len(I_pairs))
                pivots = [(I_pairs, torch.tensor(J_flat, dtype=torch.long, device=device))] * layer_copy.num_folds
                nystrom._build_factors_from(layer_copy, pivots=pivots)
                
                setattr(parent_copy, attr, nystrom)
                bpd = _compute_bpd(model_copy, dataloader, device)
            return cand, bpd

        best_bpd = float("inf")
        best_idx = None
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(eval_cand, cand) for cand in available]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating Cands (k={t+1})", leave=False):
                cand, bpd = fut.result()
                if bpd < best_bpd:
                    best_bpd, best_idx = bpd, cand
        selected.append(best_idx)
        available.remove(best_idx)
    return torch.tensor(selected, dtype=torch.long, device=device)

def run_experiment(units: int, k_rank: int, device: str):
    """
    Runs the full BPD evaluation for a given number of units and a specific rank.
    
    Args:
        units (int): The number of units for the model.
        k_rank (int): The rank of the Nystrom approximation.
        device (str): The device to run on ('cuda' or 'cpu').
        
    Returns:
        dict: A dictionary containing the results for this run.
    """
    print("\n" + "="*50)
    print(f"--- Starting Experiment for units={units}, rank={k_rank} ---")
    print("="*50)

    # --- Step 1: Define Symbolic Circuit and Load Data ---
    builder = CIRCUIT_BUILDERS["MNIST"]
    circuit = builder(num_input_units=units, num_sum_units=units,region_graph='quad-tree-4')
    symbolic_circuit = SF.multiply(circuit, circuit)
    
    X, y = torch.load("mnist_flat_int64_cache.pt")
    cached = TensorDataset(X, y)
    dataloader = DataLoader(cached, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

    # --- Step 2: Create and Load the ORIGINAL Model ---
    print("--- Compiling and loading Original Model ---")
    original_model = compile_symbolic(symbolic_circuit, device=device, rank=None)
    mnist_checkpoint_path = f"./model_cache/checkpoints/mnist_complex_{units}_{units}_epoch10.pt"
    #original_model = load_mnist_weights_for_one_sum(original_model, mnist_checkpoint_path, device=device)
    cache_path = f"./model_cache/checkpoints/mnist_{units}_{units}_epoch10.pt"
    checkpoint = torch.load(cache_path, map_location=device)
    original_model.load_state_dict(checkpoint["model_state_dict"])
    layer_path = next(name for name, m in original_model.named_modules() if isinstance(m, TorchSumLayer))

    # --- Step 3: Find the "Gold Standard" Pivots ---
    print(f"\n--- Finding 'Gold Standard' pivots (rank={k_rank}) using greedy BPD search ---")
    start_time = time.time()
    I_gold = select_rows_gold_standard(copy.deepcopy(original_model), layer_path, k_rank, dataloader, device)
    J_gold = select_columns_given_rows(copy.deepcopy(original_model), layer_path, k_rank, I_gold, dataloader, device)
    gold_time = time.time() - start_time
    print(f"Finished finding 'Gold' pivots in {gold_time:.2f}s")

    # --- Step 4: Create and Sync the Approximated Models ---
    print(f"\n--- Compiling 'Gold' Approximated Model (rank={k_rank}) ---")
    gold_model = compile_symbolic(symbolic_circuit, device=device, opt=True, rank=k_rank)
    sync_sumlayer_weights(original_model, gold_model, pivots=[[I_gold, J_gold]])
    
    print(f"\n--- Compiling 'Local CUR' Approximated Model (rank={k_rank}) ---")
    local_cur_model = compile_symbolic(symbolic_circuit, device=device, opt=True, rank=k_rank)
    sync_sumlayer_weights(original_model, local_cur_model, pivot="cur")
    
    print(f"\n--- Compiling 'Local Uniform' Approximated Model (rank={k_rank}) ---")
    local_uniform_model = compile_symbolic(symbolic_circuit, device=device, opt=True, rank=k_rank)
    sync_sumlayer_weights(original_model, local_uniform_model,)

    # --- Step 5: Evaluate All Models ---
    print("\n--- Final BPD Evaluation ---")
    
    print("Evaluating Original model...")
    orig_bpd = _compute_bpd(original_model, dataloader, device)
    
    print("Evaluating Gold model...")
    gold_bpd = _compute_bpd(gold_model, dataloader, device)

    print("Evaluating Local CUR model...")
    local_cur_bpd = _compute_bpd(local_cur_model, dataloader, device)
    
    print("Evaluating Local Uniform model...")
    local_uniform_bpd = _compute_bpd(local_uniform_model, dataloader, device)

    # --- Step 6: Return results as a dictionary ---
    results = {
        'units': units,
        'k_rank': k_rank,
        'pivot_search_time_s': gold_time,
        'bpd_original': orig_bpd,
        'bpd_gold': gold_bpd,
        'bpd_local_cur': local_cur_bpd,
        'bpd_local_uniform': local_uniform_bpd,
        'bpd_diff_gold': abs(orig_bpd - gold_bpd),
        'bpd_diff_local_cur': abs(orig_bpd - local_cur_bpd),
        'bpd_diff_local_uniform': abs(orig_bpd-local_uniform_bpd)
    }
    
    print("\n" + "-"*40)
    print(f"--- Results for units={units}, rank={k_rank} ---")
    print("-"*40)
    print(f"Pivot search time: {results['pivot_search_time_s']:.2f}s")
    print(f"\nBPD Scores (lower is better):")
    print(f"  - Original Model:         {results['bpd_original']:.6f}")
    print(f"  - Gold Approx (Greedy):   {results['bpd_gold']:.6f}")
    print(f"  - Local Approx (CUR):     {results['bpd_local_cur']:.6f}")
    print(f"  - Local Approx (Uniform): {results['bpd_local_uniform']:.6f}")
    print(f"\nBPD Difference from Original:")
    print(f"  - Gold Approx Diff:       {results['bpd_diff_gold']:.6f}")
    print(f"  - Local Approx Diff (CUR):  {results['bpd_diff_local_cur']:.6f}")
    print(f"  - Local Approx Diff (Uniform):{results['bpd_diff_local_uniform']:.6f}")
    print("-"*40)
    
    return results

def main():
    """
    Main function to run experiments, saving each result incrementally.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unit_list = [4] # Using a smaller list for your testing
    all_results_for_summary = [] # Keep a list in memory for the final summary print

    # Define the output filename at the start
    output_filename = "bpd_results_by_rank.csv"

    # Optional: Clean up old results file at the start of a new session
    # if os.path.exists(output_filename):
    #     print(f"Removing old results file: {output_filename}")
    #     os.remove(output_filename)

    for units in unit_list:
        total_columns = units ** 2
        ranks_to_test = sorted(list({
            1,
            max(1, int(0.10 * total_columns)),
            max(1, int(0.30 * total_columns)),
            max(1, int(0.60 * total_columns)),
        }))
        
        print(f"\n{'*'*60}")
        print(f"Processing for units = {units}. Ranks to be tested: {ranks_to_test}")
        print(f"{'*'*60}")
        
        for rank in ranks_to_test:
            try:
                # --- Run the experiment for one configuration ---
                result = run_experiment(units, rank, device)
                
                # --- This is the new incremental saving logic ---
                
                # 1. Add the result to our in-memory list for the final summary
                all_results_for_summary.append(result)
                
                # 2. Create a mini-DataFrame containing only the new result
                new_result_df = pd.DataFrame([result])
                
                # 3. Check if the file already exists to determine if we need a header
                write_header = not os.path.exists(output_filename)
                
                # 4. Append the new result to the CSV file
                new_result_df.to_csv(
                    output_filename, 
                    mode='a',          # 'a' stands for append
                    header=write_header, # Only write header if the file doesn't exist
                    index=False
                )
                
                print(f"--- Successfully saved result for (units={units}, rank={rank}) to {output_filename} ---\n")

            except Exception as e:
                print(f"\nERROR: An exception occurred during run for units={units}, rank={rank}: {e}")
                print("-" * 20, "TRACEBACK", "-" * 20)
                traceback.print_exc()
                print("-" * 51)
                print("Skipping to the next configuration.")
                continue
            
    # --- Final summary after all loops are complete ---
    
    # Create the final DataFrame from the in-memory list
    final_summary_df = pd.DataFrame(all_results_for_summary)
    
    print("\n" + "="*50)
    print("---           Experiment Complete            ---")
    print("="*50)
    print(f"All runs finished. All results have been saved incrementally to '{output_filename}'")
    
    if not final_summary_df.empty:
        print("\nFinal Results Summary:")
        print(final_summary_df.to_string())
    else:
        print("\nNo results were generated in this session.")
        
    print("="*50)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        main()
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        print("\n--- Profiler Summary ---")
        stats.print_stats(30)