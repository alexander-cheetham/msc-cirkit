"""Visualization functions for benchmark results."""

import matplotlib.pyplot as plt
import numpy as np
import wandb


def plot_speedup_vs_rank(
    n_inputs, ranks, speedups, sampling_methods, unique_n_inputs, unique_methods
):
    """Plot speedup against rank for each input size and sampling method."""
    fig = plt.figure(figsize=(10, 6))
    markers = {m: ms for m, ms in zip(unique_methods, ["o", "s", "^", "D", "v"])}

    for n in unique_n_inputs:
        for method in unique_methods:
            mask = (n_inputs == n) & (sampling_methods == method)
            n_ranks = ranks[mask]
            n_speedups = speedups[mask]
            if n_ranks.size == 0:
                continue
            plt.scatter(
                n_ranks,
                n_speedups,
                label=f"n={n}, {method}",
                marker=markers.get(method, "o"),
                s=50,
                alpha=0.7,
            )
    plt.xlabel("Rank")
    plt.ylabel("Speedup Factor")
    plt.title("Speedup vs Rank")
    plt.legend()
    plt.grid(True, alpha=0.3)
    wandb.log({"charts/speedup_vs_rank": wandb.Image(fig)})
    plt.close()


def plot_error_vs_rank(
    n_inputs,
    ranks,
    rel_errors,
    sampling_methods,
    unique_n_inputs,
    unique_methods,
    error_label,
):
    """Plot mean approximation error vs rank with 95% CI shaded."""
    fig = plt.figure(figsize=(10, 6))
    markers = {m: ms for m, ms in zip(unique_methods, ["o", "s", "^", "D", "v"])}
    linestyles = {m: ls for m, ls in zip(unique_methods, ["-", "--", ":", "-.", ":"])}

    for n in unique_n_inputs:
        for method in unique_methods:
            mask = (n_inputs == n) & (sampling_methods == method)
            r = ranks[mask]
            e = rel_errors[mask]
            if r.size == 0:
                continue

            uniq_r = np.unique(r)
            means, lows, highs = [], [], []

            for ur in uniq_r:
                errs = e[r == ur]
                m = errs.mean()
                if errs.size > 1:
                    se = errs.std(ddof=1) / np.sqrt(errs.size)
                else:
                    se = 0.0
                delta = 1.96 * se
                means.append(m)
                lows.append(max(m - delta, np.finfo(float).tiny))
                highs.append(m + delta)

            means = np.array(means)
            lows = np.array(lows)
            highs = np.array(highs)

            sort_idx = np.argsort(uniq_r)
            x = uniq_r[sort_idx]
            y = means[sort_idx]
            y_low = lows[sort_idx]
            y_high = highs[sort_idx]

            plt.semilogy(
                x,
                y,
                marker=markers.get(method, "o"),
                linestyle=linestyles.get(method, "-"),
                markersize=6,
                label=f"{n}, {method}",
            )
            plt.fill_between(x, y_low, y_high, alpha=0.2)

    plt.xlabel("Rank")
    plt.ylabel(error_label)
    plt.title("Approximation Error vs Rank")
    leg = plt.legend(title="n_inputs/n_sums, method")
    leg._legend_box.align = "left"
    plt.grid(True, which="both", alpha=0.3)

    wandb.log({"charts/error_vs_rank": wandb.Image(fig)})
    plt.close()



def plot_tradeoff(speedups, rel_errors, ranks, sampling_methods, unique_methods, error_label):
    """Plot accuracy vs performance trade-off, distinguished by sampling method."""
    fig = plt.figure(figsize=(10, 6))
    markers = {m: ms for m, ms in zip(unique_methods, ["o", "s", "^", "D", "v"])}
    for method in unique_methods:
        mask = sampling_methods == method
        scatter = plt.scatter(
            speedups[mask],
            rel_errors[mask],
            c=ranks[mask],
            cmap="viridis",
            marker=markers.get(method, "o"),
            s=50,
            alpha=0.7,
            label=method,
        )
    plt.xlabel("Speedup Factor")
    plt.ylabel(error_label)
    plt.yscale("log")
    plt.title("Accuracy vs Performance Trade-off")
    plt.colorbar(scatter, label="Rank")
    plt.legend(title="sampling")
    plt.grid(True, alpha=0.3)
    wandb.log({"charts/tradeoff": wandb.Image(fig)})
    plt.close()


def plot_efficiency_heatmap(
    ranks,
    matrix_sizes,
    efficiencies,
    sampling_methods,
    unique_ranks,
    unique_matrix_sizes,
    unique_methods,
):
    """Plot heatmap of efficiency values for each sampling method."""
    for method in unique_methods:
        fig = plt.figure(figsize=(12, 8))
        efficiency_matrix = np.full((len(unique_ranks), len(unique_matrix_sizes)), np.nan)
        mask_method = sampling_methods == method
        for i, rank in enumerate(unique_ranks):
            for j, mat_size in enumerate(unique_matrix_sizes):
                mask = mask_method & (ranks == rank) & (matrix_sizes == mat_size)
                if mask.any():
                    efficiency_matrix[i, j] = efficiencies[mask][0]
        im = plt.imshow(efficiency_matrix, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, label="Efficiency")
        plt.xticks(range(len(unique_matrix_sizes)), unique_matrix_sizes, rotation=45)
        plt.yticks(range(len(unique_ranks)), unique_ranks)
        plt.xlabel("Matrix Size")
        plt.ylabel("Rank")
        for i in range(len(unique_ranks)):
            for j in range(len(unique_matrix_sizes)):
                if not np.isnan(efficiency_matrix[i, j]):
                    plt.text(
                        j, i, f"{efficiency_matrix[i, j]:.2f}", ha="center", va="center"
                    )
        plt.title(f"Efficiency: Actual/Theoretical Speedup ({method})")
        plt.tight_layout()
        wandb.log({f"charts/efficiency_heatmap_{method}": wandb.Image(fig)})
        plt.close()


def plot_memory_reduction(
    ranks,
    matrix_sizes,
    memory_reductions,
    sampling_methods,
    rank_percentages,
    unique_methods,
    powers_of_two=False,
):
    """Plot memory reduction for each matrix size at fixed rank‑percentages."""

    def to_numeric(s):
        if isinstance(s, str) and "x" in s:
            return int(s.split("x", 1)[0])
        if powers_of_two and isinstance(s, str) and "^" in s:
            return 2 ** int(s.split("^", 1)[1])
        return int(s)

    numeric_sizes = np.array([to_numeric(s) for s in matrix_sizes])
    unique_sizes = sorted(set(numeric_sizes))

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {m: ms for m, ms in zip(unique_methods, ["o", "s", "^", "D", "v"])}
    linestyles = {m: ls for m, ls in zip(unique_methods, ["-", "--", ":", "-.", ":"])}

    for pct in rank_percentages:
        for method in unique_methods:
            avg_mem = []
            for N in unique_sizes:
                expected_rank = int(pct * N)
                mask = (
                    (numeric_sizes == N)
                    & (ranks == expected_rank)
                    & (sampling_methods == method)
                )
                avg_mem.append(
                    memory_reductions[mask].mean() if mask.any() else np.nan
                )

            ax.plot(
                unique_sizes,
                avg_mem,
                marker=markers.get(method, "o"),
                linestyle=linestyles.get(method, "-"),
                label=f"{int(pct*100)}% of N, {method}",
                markersize=8,
            )

    ax.set_xticks(unique_sizes)
    ax.set_xticklabels([f"{N}x{N}" for N in unique_sizes], rotation=45)
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Memory Reduction")
    ax.set_title("Memory Reduction by Matrix Size and Rank Percentage")
    ax.legend(title="Rank = % of N, method")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    wandb.log({"charts/memory_reduction": wandb.Image(fig)})
    plt.close(fig)


def plot_error_vs_depth(
    depths, errors, sampling_methods, unique_depths, unique_methods, error_label
):
    """Plot approximation error as a function of circuit depth."""
    fig = plt.figure(figsize=(10, 6))
    markers = {m: ms for m, ms in zip(unique_methods, ["o", "s", "^", "D", "v"])}
    linestyles = {m: ls for m, ls in zip(unique_methods, ["-", "--", ":", "-.", ":"])}
    for method in unique_methods:
        avg_err = [errors[(depths == d) & (sampling_methods == method)].mean() for d in unique_depths]
        plt.semilogy(
            unique_depths,
            avg_err,
            marker=markers.get(method, "o"),
            linestyle=linestyles.get(method, "-"),
            markersize=8,
            label=method,
        )
    plt.xlabel("Depth")
    plt.ylabel(error_label)
    plt.title("Error vs Depth")
    plt.grid(True, alpha=0.3)
    plt.legend(title="sampling")
    wandb.log({"charts/error_vs_depth": wandb.Image(fig)})
    plt.close()


def plot_speedup_vs_depth(
    depths, speedups, sampling_methods, unique_depths, unique_methods
):
    """Plot speedup as a function of circuit depth."""
    fig = plt.figure(figsize=(10, 6))
    markers = {m: ms for m, ms in zip(unique_methods, ["o", "s", "^", "D", "v"])}
    linestyles = {m: ls for m, ls in zip(unique_methods, ["-", "--", ":", "-.", ":"])}
    for method in unique_methods:
        avg_speedup = [
            speedups[(depths == d) & (sampling_methods == method)].mean()
            for d in unique_depths
        ]
        plt.plot(
            unique_depths,
            avg_speedup,
            marker=markers.get(method, "o"),
            linestyle=linestyles.get(method, "-"),
            markersize=8,
            label=method,
        )
    plt.xlabel("Depth")
    plt.ylabel("Speedup Factor")
    plt.title("Speedup vs Depth")
    plt.grid(True, alpha=0.3)
    plt.legend(title="sampling")
    wandb.log({"charts/speedup_vs_depth": wandb.Image(fig)})
    plt.close()


def create_wandb_visualisations(results_table, config) -> None:
    """Create and log wandb visualisations from a results table."""

    if not results_table.data:
        print("No data to visualize")
        return

    columns = results_table.columns
    col_indices = {col: idx for idx, col in enumerate(columns)}

    data_array = np.array(results_table.data)

    n_inputs = data_array[:, col_indices["n_input"]].astype(int)
    ranks = data_array[:, col_indices["rank"]].astype(int)
    speedups = data_array[:, col_indices["speedup"]].astype(float)
    if "sampling_method" in col_indices:
        sampling_methods = data_array[:, col_indices["sampling_method"]]
        sampling_methods = np.where(
            sampling_methods == "", "uniform", sampling_methods
        )
    else:
        sampling_methods = np.array(["uniform"] * len(data_array))
    if "rel_error" in col_indices:
        rel_errors = data_array[:, col_indices["rel_error"]].astype(float)
        error_label = "Relative Error"
    elif "bpd_diff" in col_indices:
        rel_errors = data_array[:, col_indices["bpd_diff"]].astype(float)
        error_label = "|ΔBPD|"
    elif "nll_diff" in col_indices:
        rel_errors = data_array[:, col_indices["nll_diff"]].astype(float)
        error_label = "ΔNLL"
    else:
        rel_errors = None
        error_label = "Error"
    efficiencies = data_array[:, col_indices["efficiency"]].astype(float)
    matrix_sizes = data_array[:, col_indices["matrix_size"]]
    memory_reductions = data_array[:, col_indices["memory_reduction"]].astype(float)
    depths = (
        data_array[:, col_indices["depth"]].astype(int)
        if "depth" in col_indices
        else None
    )

    unique_n_inputs = sorted(set(n_inputs))
    unique_ranks = sorted(set(ranks))
    unique_methods = sorted(set(sampling_methods))

    if config.powers_of_two:
        matrix_exps = [int(str(m).split("^")[1]) for m in matrix_sizes]
        unique_matrix_exps = sorted(set(matrix_exps))
        unique_matrix_sizes = [f"2^{e}" for e in unique_matrix_exps]
    else:
        unique_matrix_sizes = sorted(set(matrix_sizes))
    print(f"matrix sizes {matrix_sizes}")

    plot_speedup_vs_rank(
        n_inputs, ranks, speedups, sampling_methods, unique_n_inputs, unique_methods
    )
    if rel_errors is not None:
        plot_error_vs_rank(
            n_inputs,
            ranks,
            rel_errors,
            sampling_methods,
            unique_n_inputs,
            unique_methods,
            error_label,
        )
        plot_tradeoff(
            speedups,
            rel_errors,
            ranks,
            sampling_methods,
            unique_methods,
            error_label,
        )
    if depths is not None:
        unique_depths = sorted(set(depths))
        if rel_errors is not None:
            plot_error_vs_depth(
                depths,
                rel_errors,
                sampling_methods,
                unique_depths,
                unique_methods,
                error_label,
            )
        plot_speedup_vs_depth(
            depths,
            speedups,
            sampling_methods,
            unique_depths,
            unique_methods,
        )
    plot_efficiency_heatmap(
        ranks,
        matrix_sizes,
        efficiencies,
        sampling_methods,
        unique_ranks,
        unique_matrix_sizes,
        unique_methods,
    )
    plot_memory_reduction(
        ranks,
        matrix_sizes,
        memory_reductions,
        sampling_methods,
        config.rank_percentages,
        unique_methods,
        powers_of_two=config.powers_of_two,
    )
