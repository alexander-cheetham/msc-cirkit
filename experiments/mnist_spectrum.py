import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import traceback
from typing import List, Dict, Tuple
from scipy.stats import gaussian_kde

# --- Mock Imports for Standalone Running ---
# In your project, replace these with your actual imports
from src.circuit_types import CIRCUIT_BUILDERS
import cirkit.symbolic.functional as SF
from src.benchmarks import compile_symbolic
from cirkit.backend.torch.layers import TorchSumLayer
from torch.nn import Module, Sequential, Linear, ModuleList

# --- Configuration ---
CACHE_DIR = "model_cache/checkpoints"
UNITS_TO_VISUALIZE = [4,8,16,32]
EPOCHS_TO_VISUALIZE = [1, 5, 10] # Epochs to load from checkpoints
REGION_GRAPH_TYPE = 'quad-tree-4'
OUTPUT_DIR = "spectral_visualizations_io"

# --- Helper Functions ---

def build_model(n_units: int, region_graph_type: str, device: str) -> torch.nn.Module:
    """Builds the model architecture."""
    # This function is kept as is, assuming it correctly builds your model.
    # Replace with your actual model building logic if different.
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
    print(f"  - Successfully loaded weights for epoch {epoch}.")

def get_singular_values(layer: TorchSumLayer) -> np.ndarray:
    """
    Extracts all singular values from a single TorchSumLayer, sorted in
    descending order.
    """
    if not hasattr(layer, 'weight') or not callable(layer.weight):
        return np.array([], dtype=np.float32)

    W_tensor = layer.weight()
    if W_tensor is None:
        return np.array([], dtype=np.float32)

    W = W_tensor.detach().cpu().numpy()

    if W.ndim == 3 and W.shape[0] == 1:
        W = np.squeeze(W, axis=0)
    
    singular_values = np.array([], dtype=np.float32)

    print(f"W NDIM: {W.ndim}, Shape: {W.shape}")

    if W.ndim >= 2:
        try:
            # SVD on a (M, N) matrix produces min(M, N) singular values.
            # NOTE: If your weights are shape (1, N), you will only get 1 SV.
            singular_values = np.linalg.svd(W, compute_uv=False)
        except np.linalg.LinAlgError as e:
            print(f"  - SVD failed for a layer with shape {W.shape}: {e}")
            return np.array([], dtype=np.float32)
    elif W.ndim == 1:
        singular_values = np.array([np.linalg.norm(W)])
    elif W.ndim == 0:
        singular_values = np.array([np.abs(W.item())])
    
    ## CHANGE: Remove 95% cutoff and sort all values ##
    if singular_values.size > 0:
        # Sort all values in descending order for the scree plot
        sorted_svs = np.sort(singular_values)[::-1]
        return sorted_svs.astype(np.float32)
    
    return np.array([], dtype=np.float32)

def get_input_output_layers(model: torch.nn.Module) -> Dict[str, List[Tuple[str, TorchSumLayer]]]:
    """Identifies and returns the first and second-to-last TorchSumLayers."""
    sum_layers = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, TorchSumLayer) and hasattr(module, 'weight') and callable(module.weight)
    ]
    if not sum_layers:
        return {}
    
    print(f"  - Found {len(sum_layers)} weighted TorchSumLayers. Selecting I/O layers.")
    grouped_layers = {}
    if len(sum_layers) > 0:
        grouped_layers["Input Layer"] = [sum_layers[0]]
    if len(sum_layers) > 1:
        grouped_layers["Output Layer"] = [sum_layers[-2]]
    return grouped_layers

# --- Corrected and Enhanced Scree Plot Function ---

def plot_scree_grid_for_io(
    all_layers_spectral_data: Dict,
    target_layers: Dict[str, List[Tuple[str, torch.nn.Module]]],
    epochs: List[int],
    n_units: int,
    output_dir: str
):
    """
    Generates a visually appealing grid of scree plots for Input and Output layers.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    row_labels = [label for label in ["Input Layer", "Output Layer"] if label in target_layers]
    if not row_labels:
        print("  -> No target layers to plot.")
        return

    fig, axes = plt.subplots(
        nrows=len(row_labels),
        ncols=len(epochs),
        figsize=(len(epochs) * 5, len(row_labels) * 4.5),
        sharex=True,
        sharey='row'
    )
    if len(row_labels) == 1: axes = np.array([axes])
    if len(epochs) == 1: axes = axes.reshape(-1, 1)

    print("\n--- Generating Scree Plot Grid for Input/Output Layers ---")

    for row_idx, group_name in enumerate(row_labels):
        layer_name, _ = target_layers[group_name][0]

        for col_idx, epoch in enumerate(epochs):
            ax = axes[row_idx, col_idx]
            singular_values = all_layers_spectral_data.get(layer_name, {}).get(epoch)

            if singular_values is None or singular_values.size == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            else:
                try:
                    # X-axis: Rank of the singular value (1, 2, 3, ...)
                    ranks = np.arange(1, len(singular_values) + 1)
                    
                    # Plot the connecting line
                    ax.plot(ranks, singular_values, marker='', linestyle='-', color='#888888', alpha=0.7, zorder=1)
                    # Plot the individual points for clarity
                    ax.scatter(ranks, singular_values, color='#004488', s=15, zorder=2, label='Singular Values')

                    # --- NEW: Calculate and plot the 95% variance line ---
                    if singular_values.size > 1:
                        # Variance is proportional to the square of the singular values
                        squared_svs = singular_values**2
                        total_variance = np.sum(squared_svs)
                        
                        if total_variance > 1e-9: # Avoid division by zero
                            cumulative_variance_ratio = np.cumsum(squared_svs) / total_variance
                            
                            # Find the rank (index + 1) where the ratio exceeds 0.95
                            # np.searchsorted is efficient for this on a sorted array
                            try:
                                index_95 = np.searchsorted(cumulative_variance_ratio, 0.95)
                                rank_95 = index_95 + 1
                                
                                # Add a vertical line at that rank
                                ax.axvline(x=rank_95, color='#D55E00', linestyle='--', linewidth=1.5,
                                           label=f'95% Variance (Rank {rank_95})', zorder=3)
                                
                            except IndexError:
                                # This would happen if for some reason the threshold is not met,
                                # which is unlikely with cumsum.
                                pass 
                    # --- END of new section ---
                    
                    # Log scale on Y-axis is crucial for scree plots
                    # ax.set_yscale('log')
                    
                    # Ensure x-axis ticks are integers, especially for few SVs
                    if len(ranks) < 20:
                        ax.set_xticks(ranks)
                        ax.tick_params(axis='x', rotation=45)

                except Exception as e:
                    print(f"Could not create scree plot for {group_name} at epoch {epoch}: {e}")
                    ax.text(0.5, 0.5, 'Plotting Error', ha='center', va='center', transform=ax.transAxes)

            # --- Formatting ---
            if row_idx == 0:
                title = "Initial State (Epoch 0)" if epoch == 0 else f"Epoch {epoch}"
                ax.set_title(title, fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(f"{group_name}\nSingular Value ", fontsize=12)
            ax.grid(True, which='major', linestyle='-', linewidth=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5)

    fig.suptitle(f'Scree Plots of Layer Weights (Units: {n_units})', fontsize=18, y=1.0)
    fig.supxlabel('Singular Value Rank (Largest to Smallest)', fontsize=14, y=0.01)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"scree_plot_grid_units-{n_units}.png"
    output_path = os.path.join(output_dir, plot_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved scree plot grid to {output_path}")

# --- Main Execution Logic ---

def main():
    """Main function to run targeted spectral visualization."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    epochs_to_process = [0] + EPOCHS_TO_VISUALIZE

    for n_units in UNITS_TO_VISUALIZE:
        print(f"\n--- Processing Model with {n_units} units ---")
        
        temp_model = build_model(n_units, REGION_GRAPH_TYPE, device)
        target_layers = get_input_output_layers(temp_model)
        if not target_layers:
            print("  -> Could not identify target layers. Skipping.")
            continue
        
        all_target_layer_names = [layer[0][0] for layer in target_layers.values()]
        all_layers_spectral_data = {name: {} for name in all_target_layer_names}

        for epoch in epochs_to_process:
            print(f"\n* Analyzing Epoch {epoch}...")
            try:
                model = build_model(n_units, REGION_GRAPH_TYPE, device)
                if epoch == 0:
                    print("  - Analyzing randomly initialized weights (Epoch 0).")
                else:
                    load_weights_for_epoch(model, epoch, n_units, CACHE_DIR)
                
                current_layers = dict(model.named_modules())
                for layer_name in all_target_layer_names:
                    if layer_name in current_layers:
                        sv_array = get_singular_values(current_layers[layer_name])
                        if sv_array.size > 0:
                            all_layers_spectral_data[layer_name][epoch] = sv_array

            except FileNotFoundError as e:
                print(f"  -> Skipping epoch {epoch}: {e}")
            except Exception as e:
                print(f"  -> An unexpected error occurred for epoch {epoch}: {e}")
                traceback.print_exc()

        plot_scree_grid_for_io(
            all_layers_spectral_data,
            target_layers,
            epochs_to_process,
            n_units,
            OUTPUT_DIR
        )

    print("\nGrid visualization process complete.")

if __name__ == '__main__':
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
                        param.mul_(epoch * 0.5) 
                torch.save({'model_state_dict': dummy_model.state_dict()}, model_path)

    main()