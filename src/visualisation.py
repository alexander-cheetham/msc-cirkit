"""Visualization functions for benchmark results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict

def plot_benchmark_results(df: pd.DataFrame, save_path: str = None):
    """Create comprehensive visualization of benchmark results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Kronecker vs Nyström: Benchmark Results', fontsize=16)
    
    # Implementation from previous code
    # ... (all the plotting code)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_summary_report(df: pd.DataFrame) -> Dict:
    """Generate summary statistics from benchmark results."""
    summary = {
        "avg_speedup": df['actual_speedup'].mean(),
        "max_speedup": df['actual_speedup'].max(),
        "avg_memory_reduction": df['memory_reduction'].mean(),
        "avg_error": df['rel_error'].mean(),
    }
    
    # Find optimal configurations
    good_accuracy = df[df['rel_error'] < 0.01]
    if len(good_accuracy) > 0:
        best = good_accuracy.loc[good_accuracy['actual_speedup'].idxmax()]
        summary['best_config'] = {
            'n_input': best['n_input'],
            'n_sum': best['n_sum'],
            'rank': best['rank'],
            'speedup': best['actual_speedup']
        }
    
    return summary


def create_wandb_visualisations(results_table, config) -> None:
    """Create custom visualisations for wandb using raw data."""
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb

    if not results_table.data:
        print("No data to visualize")
        return

    columns = results_table.columns
    col_indices = {col: idx for idx, col in enumerate(columns)}

    data_array = np.array(results_table.data)

    n_inputs = data_array[:, col_indices['n_input']].astype(int)
    ranks = data_array[:, col_indices['rank']].astype(int)
    speedups = data_array[:, col_indices['speedup']].astype(float)
    rel_errors = data_array[:, col_indices['rel_error']].astype(float)
    efficiencies = data_array[:, col_indices['efficiency']].astype(float)
    matrix_sizes = data_array[:, col_indices['matrix_size']]

    unique_n_inputs = sorted(set(n_inputs))
    unique_ranks = sorted(set(ranks))
    if config.powers_of_two:
        matrix_exps = [int(str(m).split('^')[1]) for m in matrix_sizes]
        unique_matrix_exps = sorted(set(matrix_exps))
        unique_matrix_sizes = [f"2^{e}" for e in unique_matrix_exps]
    else:
        unique_matrix_sizes = sorted(set(matrix_sizes))

    fig_speedup = plt.figure(figsize=(10, 6))
    for n in unique_n_inputs:
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

    fig_error = plt.figure(figsize=(10, 6))
    for n in unique_n_inputs:
        mask = n_inputs == n
        n_ranks = ranks[mask]
        n_errors = rel_errors[mask]
        sort_idx = np.argsort(n_ranks)
        n_ranks_sorted = n_ranks[sort_idx]
        n_errors_sorted = n_errors[sort_idx]
        plt.semilogy(n_ranks_sorted, n_errors_sorted, 'o-', label=f'n={n}', markersize=8)
    plt.xlabel('Rank')
    plt.ylabel('Relative Error')
    plt.title('Approximation Error vs Rank')
    plt.legend()
    plt.grid(True, alpha=0.3)
    wandb.log({"charts/error_vs_rank": wandb.Image(fig_error)})
    plt.close()

    fig_tradeoff = plt.figure(figsize=(10, 6))
    scatter = plt.scatter(speedups, rel_errors, c=ranks, cmap='viridis', s=50, alpha=0.7)
    plt.xlabel('Speedup Factor')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.title('Accuracy vs Performance Trade-off')
    plt.colorbar(scatter, label='Rank')
    plt.grid(True, alpha=0.3)
    wandb.log({"charts/tradeoff": wandb.Image(fig_tradeoff)})
    plt.close()

    fig_efficiency = plt.figure(figsize=(12, 8))
    efficiency_matrix = np.full((len(unique_ranks), len(unique_matrix_sizes)), np.nan)
    for i, rank in enumerate(unique_ranks):
        for j, mat_size in enumerate(unique_matrix_sizes):
            mask = (ranks == rank) & (matrix_sizes == mat_size)
            if mask.any():
                efficiency_matrix[i, j] = efficiencies[mask][0]
    im = plt.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, label='Efficiency')
    plt.xticks(range(len(unique_matrix_sizes)), unique_matrix_sizes, rotation=45)
    plt.yticks(range(len(unique_ranks)), unique_ranks)
    plt.xlabel('Matrix Size')
    plt.ylabel('Rank')
    for i in range(len(unique_ranks)):
        for j in range(len(unique_matrix_sizes)):
            if not np.isnan(efficiency_matrix[i, j]):
                plt.text(j, i, f'{efficiency_matrix[i, j]:.2f}', ha='center', va='center')
    plt.title('Efficiency: Actual/Theoretical Speedup')
    plt.tight_layout()
    wandb.log({"charts/efficiency_heatmap": wandb.Image(fig_efficiency)})
    plt.close()

    fig_memory = plt.figure(figsize=(10, 6))
    memory_reductions = data_array[:, col_indices['memory_reduction']].astype(float)
    for rank in unique_ranks:
        mask = ranks == rank
        sizes = matrix_sizes[mask]
        mem_red = memory_reductions[mask]
        if config.powers_of_two:
            unique_sizes_for_rank = sorted(set(sizes), key=lambda s: int(str(s).split('^')[1]))
        else:
            unique_sizes_for_rank = sorted(set(sizes))
        avg_mem_red = []
        for size in unique_sizes_for_rank:
            size_mask = (matrix_sizes == size) & (ranks == rank)
            if size_mask.any():
                avg_mem_red.append(memory_reductions[size_mask].mean())
        if avg_mem_red:
            plt.plot(range(len(unique_sizes_for_rank)), avg_mem_red, 'o-', label=f'rank={rank}', markersize=8)
    plt.xticks(range(len(unique_matrix_sizes)), unique_matrix_sizes, rotation=45)
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory Reduction')
    plt.title('Memory Reduction by Matrix Size and Rank')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"charts/memory_reduction": wandb.Image(fig_memory)})
    plt.close()
