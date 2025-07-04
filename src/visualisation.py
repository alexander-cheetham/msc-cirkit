"""Visualization functions for benchmark results."""

import matplotlib.pyplot as plt
import numpy as np
import wandb


def plot_speedup_vs_rank(n_inputs, ranks, speedups, unique_n_inputs):
    """Plot speedup against rank for each input size."""
    fig = plt.figure(figsize=(10, 6))
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
    wandb.log({"charts/speedup_vs_rank": wandb.Image(fig)})
    plt.close()




def plot_error_vs_rank(n_inputs, ranks, rel_errors, unique_n_inputs, error_label):
    """Plot approximation error against rank."""
    fig = plt.figure(figsize=(10, 6))
    for n in unique_n_inputs:
        mask = n_inputs == n
        n_ranks = ranks[mask]

        n_errors = rel_errors[mask]
        sort_idx = np.argsort(n_ranks)
        n_ranks_sorted = n_ranks[sort_idx]
        n_errors_sorted = n_errors[sort_idx]
        plt.semilogy(n_ranks_sorted, n_errors_sorted, 'o-', label=f'n={n}', markersize=8)
    plt.xlabel('Rank')

    plt.ylabel(error_label)
    plt.title('Approximation Error vs Rank')
    plt.legend()
    plt.grid(True, alpha=0.3)
    wandb.log({"charts/error_vs_rank": wandb.Image(fig)})
    plt.close()


def plot_kl_nll_vs_rank(n_inputs, ranks, kl_divs, nll_diffs, unique_n_inputs):
    """Plot KL divergence and NLL difference against rank in separate subplots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    metrics = [(kl_divs, "KL Divergence"), (nll_diffs, "NLL Difference")]
    for ax, (errors, label) in zip(axes, metrics):
        for n in unique_n_inputs:
            mask = n_inputs == n
            n_ranks = ranks[mask]
            n_errors = errors[mask]
            sort_idx = np.argsort(n_ranks)
            n_ranks_sorted = n_ranks[sort_idx]
            n_errors_sorted = n_errors[sort_idx]
            ax.plot(n_ranks_sorted, n_errors_sorted, 'o-', label=f'n={n}', markersize=8)
        ax.set_xlabel('Rank')
        ax.set_ylabel(label)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Approximation Error vs Rank')
    fig.tight_layout()
    wandb.log({"charts/error_vs_rank": wandb.Image(fig)})
    plt.close()


def plot_tradeoff(speedups, errors, ranks, error_label, log_y=True):
    """Plot accuracy vs performance trade-off for a single error metric."""
    fig = plt.figure(figsize=(10, 6))
    scatter = plt.scatter(speedups, errors, c=ranks, cmap='viridis', s=50, alpha=0.7)
    plt.xlabel('Speedup Factor')
    plt.ylabel(error_label)
    if log_y:
        plt.yscale('log')
    if error_label in ("KL Divergence", "Negative Log-Likelihood"):
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Accuracy vs Performance Trade-off')
    plt.colorbar(scatter, label='Rank')
    plt.grid(True, alpha=0.3)
    wandb.log({"charts/tradeoff": wandb.Image(fig)})
    plt.close()


def plot_tradeoff_kl_nll(speedups, kl_divs, nll_diffs, ranks):
    """Plot trade-off between speedup and KL divergence and NLL difference in separate subplots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    metrics = [(kl_divs, "KL Divergence"), (nll_diffs, "NLL Difference")]
    scatters = []
    for ax, (errors, label) in zip(axes, metrics):
        sc = ax.scatter(speedups, errors, c=ranks, cmap='viridis', s=50, alpha=0.7)
        scatters.append(sc)
        ax.set_xlabel('Speedup Factor')
        ax.set_ylabel(label)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_title(f'Accuracy vs Performance: {label}')
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(right=0.86, wspace=0.3)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(scatters[0], cax=cbar_ax, label='Rank')
    wandb.log({"charts/tradeoff": wandb.Image(fig)})
    plt.close()



def plot_memory_reduction(ranks, matrix_sizes, memory_reductions, unique_ranks, unique_matrix_sizes, powers_of_two=False):
    """Plot memory reduction for each matrix size and rank."""
    fig = plt.figure(figsize=(10, 6))
    for rank in unique_ranks:
        mask = ranks == rank
        sizes = matrix_sizes[mask]
        if powers_of_two:
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
    wandb.log({"charts/memory_reduction": wandb.Image(fig)})
    plt.close()


def create_wandb_visualisations(results_table, config) -> None:
    """Create and log wandb visualisations from a results table."""

    if not results_table.data:
        print("No data to visualize")
        return

    columns = results_table.columns
    col_indices = {col: idx for idx, col in enumerate(columns)}

    data_array = np.array(results_table.data)

    n_inputs = data_array[:, col_indices['n_input']].astype(int)
    ranks = data_array[:, col_indices['rank']].astype(int)
    speedups = data_array[:, col_indices['speedup']].astype(float)

    kl_divs = data_array[:, col_indices['kl_div']].astype(float) if 'kl_div' in col_indices else None
    nll_diffs = data_array[:, col_indices['nll_diff']].astype(float) if 'nll_diff' in col_indices else None
    rel_errors = data_array[:, col_indices['rel_error']].astype(float) if 'rel_error' in col_indices else None

    matrix_sizes = data_array[:, col_indices['matrix_size']]
    memory_reductions = data_array[:, col_indices['memory_reduction']].astype(float)

    unique_n_inputs = sorted(set(n_inputs))
    unique_ranks = sorted(set(ranks))

    if config.powers_of_two:
        matrix_exps = [int(str(m).split('^')[1]) for m in matrix_sizes]
        unique_matrix_exps = sorted(set(matrix_exps))
        unique_matrix_sizes = [f"2^{e}" for e in unique_matrix_exps]
    else:
        unique_matrix_sizes = sorted(set(matrix_sizes))

    plot_speedup_vs_rank(n_inputs, ranks, speedups, unique_n_inputs)

    if kl_divs is not None and nll_diffs is not None:
        plot_kl_nll_vs_rank(n_inputs, ranks, kl_divs, nll_diffs, unique_n_inputs)
        plot_tradeoff_kl_nll(speedups, kl_divs, nll_diffs, ranks)
    elif rel_errors is not None:
        plot_error_vs_rank(n_inputs, ranks, rel_errors, unique_n_inputs, 'Relative Error')
        plot_tradeoff(speedups, rel_errors, ranks, 'Relative Error', log_y=True)
    elif kl_divs is not None:
        plot_error_vs_rank(n_inputs, ranks, kl_divs, unique_n_inputs, 'KL Divergence')
        plot_tradeoff(speedups, kl_divs, ranks, 'KL Divergence', log_y=False)
    elif nll_diffs is not None:
        plot_error_vs_rank(n_inputs, ranks, nll_diffs, unique_n_inputs, 'NLL Difference')
        plot_tradeoff(speedups, nll_diffs, ranks, 'NLL Difference', log_y=False)

    plot_memory_reduction(ranks, matrix_sizes, memory_reductions, unique_ranks, unique_matrix_sizes, powers_of_two=config.powers_of_two)

