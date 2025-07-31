import os
import torch
import numpy as np
import traceback
import csv  # <-- Import the csv module
from typing import List, Dict, Tuple

# --- Mock Imports for Standalone Running ---
# In your project, replace these with your actual imports
from src.circuit_types import CIRCUIT_BUILDERS
import cirkit.symbolic.functional as SF
from src.benchmarks import compile_symbolic
from cirkit.backend.torch.layers import TorchSumLayer
from torch.nn import Module

# --- Configuration ---
CACHE_DIR = "model_cache/checkpoints"
UNITS_TO_VISUALIZE = [4, 8, 16, 32]
EPOCHS_TO_VISUALIZE = [1, 5, 10] # Epochs to load from checkpoints
REGION_GRAPH_TYPE = 'quad-tree-4'
OUTPUT_CSV_FILE = "spectral_rank_analysis.csv" # <-- Define the output filename

# --- Helper Functions (Reused and Adapted) ---

def build_model(n_units: int, region_graph_type: str, device: str) -> torch.nn.Module:
    """Builds the model architecture."""
    builder = CIRCUIT_BUILDERS['MNIST']
    symbolic = builder(region_graph=region_graph_type, num_input_units=n_units, num_sum_units=n_units)
    symbolic = SF.multiply(symbolic, symbolic)
    model = compile_symbolic(symbolic, device=device).to(device)
    model.eval()
    return model

def load_weights_for_epoch(model: torch.nn.Module, epoch: int, n_units: int, cache_dir: str):
    """Loads a state_dict into an existing model."""
    model_filename = f"mnist_{n_units}_{n_units}_epoch{epoch}.pt"
    model_path = os.path.join(cache_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    device = next(model.parameters()).device
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)

def get_singular_values(layer: TorchSumLayer) -> np.ndarray:
    """
    Extracts all singular values from a single TorchSumLayer, handling various
    tensor shapes and sorting in descending order.
    """
    if not hasattr(layer, 'weight') or not callable(layer.weight):
        return np.array([], dtype=np.float32)

    W_tensor = layer.weight()
    if W_tensor is None:
        return np.array([], dtype=np.float32)

    W = W_tensor.detach().cpu().numpy()

    if W.ndim > 2:
        W = np.squeeze(W)

    singular_values = np.array([], dtype=np.float32)

    if W.ndim >= 2:
        try:
            singular_values = np.linalg.svd(W, compute_uv=False)
        except np.linalg.LinAlgError as e:
            print(f"  - SVD failed for a layer with shape {W.shape}: {e}")
            return np.array([], dtype=np.float32)
    elif W.ndim == 1:
        singular_values = np.array([np.linalg.norm(W)])
    
    if singular_values.size > 0:
        return np.sort(singular_values)[::-1].astype(np.float32)
    
    return np.array([], dtype=np.float32)

def get_all_sum_layers(model: torch.nn.Module) -> List[Tuple[str, TorchSumLayer]]:
    """Identifies and returns ALL TorchSumLayers with weights in the model."""
    sum_layers = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, TorchSumLayer) and hasattr(module, 'weight') and callable(module.weight)
    ]
    return sum_layers

# --- Core Rank Calculation and Analysis ---

def calculate_rank_for_95_variance(singular_values: np.ndarray) -> Tuple[int, int]:
    """
    Calculates the rank required to capture 95% of the variance.
    Returns a tuple of (max_rank, rank_for_95_variance).
    """
    max_rank = singular_values.size
    if max_rank == 0:
        return 0, 0

    squared_svs = singular_values**2
    total_variance = np.sum(squared_svs)

    if total_variance < 1e-12:
        return max_rank, 0

    cumulative_variance_ratio = np.cumsum(squared_svs) / total_variance
    index_95 = np.searchsorted(cumulative_variance_ratio, 0.95, side='left')
    rank_95 = index_95 + 1
    
    return max_rank, rank_95

def print_results_table(results: List[Dict]):
    """Prints a formatted table of the analysis results."""
    if not results:
        print("No results to display.")
        return
    headers = ["Units", "Epoch", "Num Layers", "Max Rank", "Avg 95% Rank", "Std 95% Rank", "Min 95% Rank", "Max 95% Rank"]
    col_formats = {
        "Units":        "{:<6}", "Epoch":        "{:<6}", "Num Layers":   "{:<11}",
        "Max Rank":     "{:<9}", "Avg 95% Rank": "{:<15.2f}", "Std 95% Rank": "{:<15.2f}",
        "Min 95% Rank": "{:<13}", "Max 95% Rank": "{:<13}"
    }
    header_line = " | ".join([f"{h:<{len(col_formats[h].format(0))}}" for h in headers])
    print(header_line)
    print("-" * len(header_line))
    for res in results:
        row_str = " | ".join([col_formats[h].format(res[h]) for h in headers])
        print(row_str)

# --- NEW: Function to save results to a CSV file ---
def save_results_to_csv(results: List[Dict], output_filename: str):
    """Saves the analysis results to a CSV file."""
    if not results:
        print("No results to save.")
        return

    # Use the same headers as the printed table for consistency
    fieldnames = ["Units", "Epoch", "Num Layers", "Max Rank", "Avg 95% Rank", "Std 95% Rank", "Min 95% Rank", "Max 95% Rank"]
    
    try:
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults successfully saved to {output_filename}")
    except IOError as e:
        print(f"\nError: Could not save results to file. {e}")

# --- Main Execution Logic ---

def main():
    """Main function to run spectral rank analysis."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs_to_process = [0] + EPOCHS_TO_VISUALIZE
    analysis_results = []

    for n_units in UNITS_TO_VISUALIZE:
        print(f"\n--- Processing Model with {n_units} units ---")
        
        for epoch in epochs_to_process:
            print(f"* Analyzing Epoch {epoch}...")
            try:
                model = build_model(n_units, REGION_GRAPH_TYPE, device)
                if epoch == 0:
                    print("  - Analyzing randomly initialized weights (Epoch 0).")
                else:
                    load_weights_for_epoch(model, epoch, n_units, CACHE_DIR)
                
                all_layers = get_all_sum_layers(model)
                if not all_layers:
                    print("  -> No weighted TorchSumLayers found. Skipping.")
                    continue
                
                ranks_for_95_variance = []
                max_ranks_in_model = []

                for name, layer in all_layers:
                    sv_array = get_singular_values(layer)
                    max_rank, rank_95 = calculate_rank_for_95_variance(sv_array)
                    if max_rank > 0:
                        ranks_for_95_variance.append(rank_95)
                        max_ranks_in_model.append(max_rank)
                
                if ranks_for_95_variance:
                    stats = {
                        "Units": n_units,
                        "Epoch": epoch,
                        "Num Layers": len(ranks_for_95_variance),
                        "Max Rank": max(max_ranks_in_model) if max_ranks_in_model else 0,
                        "Avg 95% Rank": np.mean(ranks_for_95_variance),
                        "Std 95% Rank": np.std(ranks_for_95_variance),
                        "Min 95% Rank": np.min(ranks_for_95_variance),
                        "Max 95% Rank": np.max(ranks_for_95_variance),
                    }
                    analysis_results.append(stats)
                    print(f"  - Done. Found {stats['Num Layers']} layers. Avg rank: {stats['Avg 95% Rank']:.2f}")

            except FileNotFoundError as e:
                print(f"  -> Skipping epoch {epoch}: {e}")
            except Exception as e:
                print(f"  -> An unexpected error occurred for epoch {epoch}: {e}")
                traceback.print_exc()

    # --- Print and Save Final Results ---
    print("\n\n--- Spectral Rank Analysis Summary ---")
    print_results_table(analysis_results)
    save_results_to_csv(analysis_results, OUTPUT_CSV_FILE) # <-- Save results to file

if __name__ == '__main__':
    # --- Create dummy checkpoints if they don't exist ---
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    for n_units in UNITS_TO_VISUALIZE:
        for epoch in EPOCHS_TO_VISUALIZE:
            model_filename = f"mnist_{n_units}_{n_units}_epoch{epoch}.pt"
            model_path = os.path.join(CACHE_DIR, model_filename)
            if not os.path.exists(model_path):
                print(f"Creating dummy checkpoint: {model_path}")
                dummy_model = build_model(n_units, REGION_GRAPH_TYPE, 'cpu')
                with torch.no_grad():
                    for param in dummy_model.parameters():
                        param.mul_(torch.randn_like(param) * epoch * 0.5) 
                torch.save({'model_state_dict': dummy_model.state_dict()}, model_path)

    main()