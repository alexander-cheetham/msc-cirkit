# Low-Rank Approximation for Kronecker-Product Layers in the `cirkit` Library

This project implements and benchmarks a CUR-based approximation for linear layers with Kronecker-product weights, specifically targeting layers from the `cirkit` library. The goal is to accelerate the forward pass and reduce the memory footprint of these layers while maintaining acceptable accuracy.

## Key Features

*   **CUR-based Low-Rank Approximation**: Replaces `TorchSumLayer` from `cirkit` with a `NystromSumLayer` that uses a low-rank approximation of the weight matrix.
*   **Kronecker Product Support**: The Nyström approximation is designed to work with weights that are Kronecker products of smaller matrices, without materializing the full weight matrix.
*   **Multiple Pivot Selection Strategies**: Implements several pivot selection strategies for the Nyström method, including:
    *   `uniform`: Uniform random sampling of columns.
    *   `l2`: Importance sampling based on the L2 norm of the columns.
    *   `cur`: Leverage score-based sampling.
*   **Comprehensive Benchmarking Suite**: Includes a suite of benchmarks for comparing the performance of the Nyström layer against the original `TorchSumLayer`. The benchmarks measure:
    *   Speedup of the forward pass.
    *   Memory reduction.
    *   Approximation error (NLL difference and BPD difference).
*   **Extensible Circuit Architecture**: Supports different circuit structures for benchmarking, including simple one-sum circuits, deep CP-circuits, and MNIST classifiers.
*   **Integration with `wandb`**: Logs all benchmark results, including metrics, plots, and system information, to Weights & Biases for easy analysis and visualization.

## How it Works

The core of this project is the `NystromSumLayer`, which approximates a `TorchSumLayer`. A `TorchSumLayer` has a weight matrix `W` of size `(F, K_o, K_i)`, where `F` is the number of folds, `K_o` is the number of output units, and `K_i` is the number of input units. In the context of squared circuits, the weight matrix of a layer is often a Kronecker product of a smaller matrix with itself, i.e., `W = A ⊗ A`. This results in a very large weight matrix that can be computationally expensive to work with.

The `NystromSumLayer` avoids materializing the full `W` matrix. Instead, it approximates `W` with a low-rank matrix `W_lr = U V^T`, where `U` and `V` are smaller matrices of size `(F, K_o, s)` and `(F, K_i, s)` respectively, and `s` is the rank of the approximation. The factors `U` and `V` are constructed using the Nyström method, which involves selecting a small number of "pivot" columns from `W` and using them to reconstruct the entire matrix.

The project implements different strategies for selecting these pivot columns, which can have a significant impact on the accuracy of the approximation.

## Benchmarking

The `benchmarks.py` script allows you to run a variety of experiments to compare the Nyström approximation with the exact Kronecker product. The configuration for the benchmarks is defined in the `BenchmarkConfig` dataclass in `src/config.py`.

To run the benchmarks, you can use the `experiments/wand_benchmark.py` script. This script will run the benchmarks with the specified configuration and log the results to `wandb`.

The benchmark results include:

*   **Speedup vs. Rank**: How the speedup of the forward pass changes with the rank of the Nyström approximation.
*   **Error vs. Rank**: How the approximation error changes with the rank.
*   **Accuracy vs. Performance Trade-off**: A plot showing the trade-off between speedup and approximation error.
*   **Efficiency Heatmap**: A heatmap showing the efficiency of the Nyström approximation (actual speedup / theoretical speedup) for different matrix sizes and ranks.
*   **Memory Reduction**: How much memory is saved by using the Nyström approximation.

## Development

Install dependencies with uv:

```bash
uv venv
uv pip install -e .[test]
```

Install optional graph visualization dependencies:

```bash
uv pip install -e .[graphviz]
```

Run tests:

```bash
.venv/bin/pytest -q
```
