#!/usr/bin/env python
# coding: utf-8
"""
Train a CirKit circuit on the 2-ring toy dataset and log everything to wandb.
A single run’s hyper-parameters come from wandb.config, so the same file
doubles as the sweep entry-point.
"""
import kron_logger 
import functools, random, argparse, math, os, sys, time
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import os
os.environ.setdefault("WANDB__REQUIRE_LEGACY_SERVICE", "TRUE")  # env override

import wandb
wandb.require("legacy-service")       # switch backend → skips wandb-core
            #  ← NEW
from artificial import crossing_rings_sample
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# 0-bis · Plot helper  (copy/paste your exact function here)
#       • Only extra code: collect figures in a list and return them
# ──────────────────────────────────────────────────────────────────────
def plot_density_comparison_return(
    X: np.ndarray,
    trained_circuits: list,
    grid_res: int = 400,          #     ← 400 is fast enough for sweeps
    batch_size: int = 4096,
    device: torch.device = None
):
    """
    Wrapper around your original plot_density_comparison.
    It returns a list[Figure] so we can hand them to wandb.Image.
    """
    figures = []                       # NEW

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    """
    For a set of 2D points X (shape = (N, 2)) and a list of trained density models (circuits),
    compute the “true” Gaussian KDE over X on a grid, then for each circuit evaluate its learned
    density on that same grid and plot a side-by-side comparison.

    Args:
        X (np.ndarray): Array of shape (N, 2) containing the original samples (e.g. from crossing_rings_sample).
        trained_circuits (list): A list of PyTorch modules, each of which, when called on a Tensor of shape (B, 2),
                                 returns a Tensor of log-densities of shape (B,).
        grid_res (int, optional): Resolution of the square grid along each axis (default=1000). 
                                  The grid will be grid_res × grid_res points.
        batch_size (int, optional): How many grid points to process at once to avoid OOM (default=4096).
        device (torch.device, optional): If None, automatically picks CUDA if available, otherwise CPU.

    Returns:
        None. Displays one figure per circuit, with two subplots (true KDE vs. learned density).
    """
    # ────────────────────────────────────────────────────────────────────────────
    # 0. Compute “true” KDE once
    # ────────────────────────────────────────────────────────────────────────────
    # X is assumed to be shape (N, 2)
    x = X[:, 0]
    y = X[:, 1]
    xy = np.vstack([x, y])                      # shape = (2, N)
    kde = gaussian_kde(xy)                      # “true” density estimator

    # Build a square grid (grid_res × grid_res) over the data range
    xmin, xmax = x.min() - 1, x.max() + 1
    ymin, ymax = y.min() - 1, y.max() + 1

    # Create meshgrid of shape (grid_res, grid_res)
    xx, yy = np.mgrid[
        xmin : xmax : grid_res*1j,
        ymin : ymax : grid_res*1j
    ]  # both xx and yy have shape (grid_res, grid_res)

    positions = np.vstack([xx.ravel(), yy.ravel()])  # shape = (2, grid_res²)

    # Evaluate “true” KDE on that grid and reshape for plotting
    Z_true = np.reshape(kde(positions).T, xx.shape)  # (grid_res, grid_res)

    # ────────────────────────────────────────────────────────────────────────────
    # 1. Determine device
    # ────────────────────────────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # ────────────────────────────────────────────────────────────────────────────
    # 2. Loop over each circuit, evaluate model density on the grid, and plot
    # ────────────────────────────────────────────────────────────────────────────
    for idx, circuit in enumerate(trained_circuits):
        # Move model to device and set to eval mode
        circuit = circuit.to(device)
        circuit.eval()

        # Convert grid points to a torch.Tensor of shape (grid_res², 2) on `device`
        grid_tensor = torch.from_numpy(positions.T).float().to(device)

        n_points = grid_tensor.shape[0]           # grid_res²
        log_probs_chunks = []

        # Evaluate in batches to avoid OOM
        with torch.no_grad():
            for i in range(0, n_points, batch_size):
                chunk = grid_tensor[i : i + batch_size]     # shape = (≤batch_size, 2)
                logp_chunk = circuit(chunk)                 # shape = (≤batch_size,)
                log_probs_chunks.append(logp_chunk.cpu())

        # Concatenate all log-probs and exponentiate to get p_model
        log_probs_all = torch.cat(log_probs_chunks, dim=0)      # shape = (grid_res²,)
        p_model = torch.exp(log_probs_all).numpy()              # shape = (grid_res²,)
        p_vals = torch.exp(log_probs_all).numpy()           # shape = (N,)

        # (6)  (Optional) If you need a normalized PMF over the finite grid:
        p_vals /= p_vals.sum() #TODO: CHECK THIS IS A LEGAL MANOEUVRE

        # Reshape into (grid_res, grid_res)
        Z_model = p_model.reshape(xx.shape)                     # (grid_res, grid_res)

        # ────────────────────────────────────────────────────────────────────────────
        # 3. Plot “true” KDE vs. model’s learned density side by side
        # ────────────────────────────────────────────────────────────────────────────
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

        # ─── Left: True KDE ───────────────────────────────────────────────────────
        im0 = ax0.imshow(
            np.rot90(Z_true),
            cmap='viridis',
            extent=[xmin, xmax, ymin, ymax],
            aspect='auto'
        )
        ax0.scatter(x, y, c='white', s=5, edgecolor='k', linewidth=0.2)
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.set_title(f'Circuit #{idx}: True KDE')
        fig.colorbar(im0, ax=ax0, label='Density')

        # ─── Right: Model’s Learned Density ───────────────────────────────────────
        im1 = ax1.imshow(
            np.rot90(Z_model),
            cmap='viridis',
            extent=[xmin, xmax, ymin, ymax],
            aspect='auto'
        )
        ax1.scatter(x, y, c='white', s=5, edgecolor='k', linewidth=0.2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Circuit #{idx}: Model Density')
        fig.colorbar(im1, ax=ax1, label='Density')

        plt.suptitle(f'Density Comparison for Circuit #{idx}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figures.append(fig)
    return figures


# ──────────────────────────────────────────────────────────────────────
# 1.  Dataset helpers
# ──────────────────────────────────────────────────────────────────────
def make_dataloaders(n_points: int,
                     test_split: float,
                     batch_size: int,
                     seed: int = 42) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """Generate the 2-ring dataset, then build Torch DataLoaders."""
    rng = np.random.default_rng(seed)
    X = crossing_rings_sample(n_points)          # (N, 2)

    X_tensor = torch.from_numpy(X).float()
    dataset   = TensorDataset(X_tensor)

    n_total   = len(dataset)
    n_train   = int((1.0 - test_split) * n_total)
    n_test    = n_total - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl, X


# ──────────────────────────────────────────────────────────────────────
# 2.  Circuit factory
# ──────────────────────────────────────────────────────────────────────
from cirkit.symbolic.circuit import Circuit
from cirkit.templates.region_graph import RandomBinaryTree, RegionGraph
from cirkit.symbolic.layers import GaussianLayer
from cirkit.symbolic.parameters import mixing_weight_factory
from cirkit.templates.utils import Parameterization, parameterization_to_factory

def define_circuit(rg: RegionGraph,
                   num_input_units: int,
                   num_sum_units: int,
                   sum_prod_layer: str = "cp") -> Circuit:
    """Return an over-parameterized Circuit given a region-graph."""
    # Input layer (Gaussian) is over-parameterized by num_input_units →
    input_factory = lambda scope, _: GaussianLayer(scope=scope,
                                                   num_output_units=num_input_units)

    # Sum-layer parameterization (softmax-normal initialisation)
    sum_param   = Parameterization(activation="softmax", initialization="normal")
    sum_factory = parameterization_to_factory(sum_param)

    # Special case: mixing (n-ary sum) layers
    nary_factory = functools.partial(mixing_weight_factory,
                                     param_factory=sum_factory)

    circuit = rg.build_circuit(
        input_factory        = input_factory,
        sum_weight_factory   = sum_factory,
        nary_sum_weight_factory = nary_factory,
        num_input_units      = num_input_units,
        num_sum_units        = num_sum_units,
        sum_product          = sum_prod_layer,
    )
    return circuit


# ──────────────────────────────────────────────────────────────────────
# 3.  Training & evaluation
# ──────────────────────────────────────────────────────────────────────
from cirkit.pipeline import compile as cirkit_compile
from torch.cuda.amp import autocast, GradScaler        # mixed precision (optional)
from cirkit.pipeline import compile
from cirkit.pipeline import PipelineContext




def train_one_run(cfg):
    """Single experiment driven by values inside wandb.config (cfg)."""

    ctx = PipelineContext(
    backend='torch',      # Use the PyTorch backend
    # Specify the backend compilation flags next
    semiring='lse-sum',   # Use the 'lse-sum' semiring
    fold=True,            # Enable circuit folding
    # -------- Enable layer optimizations -------- #
    optimize=False, # NOTE: THIS IS AN IMPORTANT FLAG (false for kronecker true for einsum)
    # -------------------------------------------- #
)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    train_dl, test_dl, X = make_dataloaders(
        n_points   = cfg.dataset_size,
        test_split = cfg.test_split,
        batch_size = cfg.batch_size,
        seed       = cfg.seed)

    # Build region graph & circuit
    rg = RandomBinaryTree(2, depth=None, num_repetitions=1)
    net = define_circuit(rg,
                         num_input_units = cfg.num_input_units,
                         num_sum_units   = cfg.num_sum_units,
                         sum_prod_layer  = cfg.sum_product)
    net = ctx.compile(net).to(device)

    # Optimiser & scaler
    opt     = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    scaler  = GradScaler(enabled=cfg.mixed_precision)

    steps   = 0
    for epoch in range(cfg.epochs):
        net.train()
        running_loss = 0.0
        n_seen       = 0

        for (x_batch,) in train_dl:
            x_batch = x_batch.view(x_batch.size(0), -1).to(device)
            opt.zero_grad()

            with autocast(enabled=cfg.mixed_precision):
                logp = net(x_batch)                       # (B,)
                loss = -logp.mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * x_batch.size(0)
            n_seen       += x_batch.size(0)
            steps        += 1

            # Log every N steps
            if steps % cfg.log_every == 0:
                wandb.log({"train/nll": running_loss / n_seen,
                           "epoch": epoch,
                           "steps": steps},
                          step=steps)
                running_loss = 0.0
                n_seen       = 0

        # ─ Evaluate each epoch ───────────────────────────────
        net.eval()
        with torch.no_grad():
            total_ll = 0.0
            total_N  = 0
            for (x_batch,) in test_dl:
                x_batch = x_batch.view(x_batch.size(0), -1).to(device)
                logp    = net(x_batch)
                total_ll += logp.sum().item()
                total_N  += x_batch.size(0)

        avg_test_nll = -total_ll / total_N
        bpd          = avg_test_nll / (2 * math.log(2.0))
        wandb.log({"test/nll": avg_test_nll,
                   "test/bpd": bpd,
                   "epoch": epoch},
                  step=steps)
    figs = plot_density_comparison_return(X, [net], grid_res=400,
                                          batch_size=4096, device=device)
    for i, fig in enumerate(figs):
        wandb.log({f"plots/density_{i}": wandb.Image(fig)}, step=steps)
        plt.close(fig)                       # free memory/GPU
    # Save the final circuit (state_dict) as an artifact
    ckpt_path = os.path.join(wandb.run.dir,
                             f"circuit_input{cfg.num_input_units}_sum{cfg.num_sum_units}.pt")
    torch.save(net.state_dict(), ckpt_path)
    wandb.save(ckpt_path)

    return avg_test_nll


# ──────────────────────────────────────────────────────────────────────
# 4.  Script entry-point for *manual* runs  (sweeps call this too)
# ──────────────────────────────────────────────────────────────────────
default_config = dict(
    # Data
    dataset_size = 1000,
    test_split   = 0.2,
    batch_size   = 256,
    seed         = 42,

    # Model
    num_input_units = 5,      # will be overridden by the sweep
    num_sum_units   = 5,      # ditto
    sum_product     = "cp",   # {cp, tucker}

    # Training
    epochs        = 2000,
    lr            = 1e-2,
    mixed_precision = True,
    log_every     = 250
)

def main():
    wandb.init(project="cirkit_density", config=default_config)
    cfg = wandb.config
    final_nll = train_one_run(cfg)
    wandb.summary["final_test_nll"] = final_nll

if __name__ == "__main__":
    main()

