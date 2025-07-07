from types import MethodType
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import GaussianLayer, SumLayer
from nystromlayer import NystromSumLayer as SymbolicNystromSumLayer

def build_circuit_one_sum(
    self,
    *,
    input_factory,        # like lambda scope, n: GaussianLayer(scope, n)
    sum_weight_factory,   # your ParameterFactory for the SumLayer
    num_input_units: int, # # of outputs per Gaussian leaf
    num_sum_units: int,# # of mixtures in the one SumLayer
    debug=False,
) -> Circuit:
    # 1) Find all the leaves in the region graph:
    leaves = [node for node in self.topological_ordering()
              if not self.region_inputs(node)]
    
    layers = []
    in_layers = {}
    
    # 2) Build one GaussianLayer per leaf
    gaussians = []
    for leaf in leaves:
        gauss = input_factory(leaf.scope, num_input_units)
        layers.append(gauss)
        gaussians.append(gauss)
    
    # 3) Build *one* SumLayer mixing them all
    sum_layer = SumLayer(
        num_input_units=num_input_units,
        num_output_units=num_sum_units,
        arity=len(gaussians),
        weight_factory=sum_weight_factory,
    )
    layers.append(sum_layer)
    in_layers[sum_layer] = gaussians
    
    # 4) Return a circuit whose only output is that top‐sum
    if debug:
        print(layers,"---------------\n\n\n",in_layers,"---------------\n\n\n",[sum_layer])
    return Circuit(layers, in_layers, outputs=[sum_layer])



import functools
from types import MethodType
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from cirkit.symbolic.parameters import mixing_weight_factory

def define_circuit_one_sum(
    num_input_units: int = 3,
    num_sum_units:   int = 2,
) -> Circuit:
    # ── 1) Build the region‐graph (just to get leaf scopes) ────────────
    rg = RandomBinaryTree(1, depth=None, num_repetitions=1, seed=42)
    
    # ── 2) Attach our star‐builder ────────────────────────────────────
    rg.build_circuit = MethodType(build_circuit_one_sum, rg)
    
    # ── 3) Make the factories ─────────────────────────────────────────
    input_factory = lambda scope, n: GaussianLayer(scope=scope, num_output_units=n)
    p = Parameterization(activation="softmax", initialization="normal")
    sum_param_factory = parameterization_to_factory(p)
    # (we don’t need an n‐ary mixing factory here, just the base factory)
    
    # ── 4) Build & return ─────────────────────────────────────────────
    return rg.build_circuit(
        input_factory=input_factory,
        sum_weight_factory=sum_param_factory,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
    )


def build_circuit_one_nystrom(
    self,
    *,
    input_factory,
    weight_factory,
    rank: int,
    num_input_units: int,
    num_sum_units: int,
    debug: bool = False,
) -> Circuit:
    """Build a circuit using a symbolic Nyström sum layer."""

    leaves = [node for node in self.topological_ordering() if not self.region_inputs(node)]

    layers: list = []
    in_layers: dict = {}

    gaussians = []
    for leaf in leaves:
        gauss = input_factory(leaf.scope, num_input_units)
        layers.append(gauss)
        gaussians.append(gauss)

    sum_layer = SymbolicNystromSumLayer(
        num_input_units=num_input_units,
        num_output_units=num_sum_units,
        rank=rank,
        arity=len(gaussians),
        U_factory=weight_factory,
        V_factory=weight_factory,
    )
    layers.append(sum_layer)
    in_layers[sum_layer] = gaussians

    if debug:
        print(layers, "---------------\n\n\n", in_layers, "---------------\n\n\n", [sum_layer])

    return Circuit(layers, in_layers, outputs=[sum_layer])


def define_circuit_one_nystrom(
    num_input_units: int = 3,
    num_sum_units: int = 2,
    rank: int = 2,
) -> Circuit:
    rg = RandomBinaryTree(1, depth=None, num_repetitions=1, seed=42)
    rg.build_circuit = MethodType(build_circuit_one_nystrom, rg)

    input_factory = lambda scope, n: GaussianLayer(scope=scope, num_output_units=n)
    p = Parameterization(activation="softmax", initialization="normal")
    param_factory = parameterization_to_factory(p)

    return rg.build_circuit(
        input_factory=input_factory,
        weight_factory=param_factory,
        rank=rank,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
    )

import torch
import matplotlib.pyplot as plt

def spectrum_diagnostics(W: torch.Tensor, *, energy_thresh: float = 0.95):
    """
    Plot the singular-value spectrum of a 2-D (m × n) matrix and annotate the
    effective rank k_eff — the smallest k whose leading singular values capture
    `energy_thresh` of the Frobenius energy.

    Parameters
    ----------
    W : torch.Tensor
        The matrix (or batched matrix) to analyse.  If batched, all leading
        batch dimensions are collapsed before the SVD.
    energy_thresh : float, default 0.95
        Energy fraction (0 < energy_thresh ≤ 1) that defines k_eff.

    Returns
    -------
    k_eff : int
        Effective rank satisfying the chosen energy threshold.
    """
    # ---- 1. ensure 2-D, compute singular values --------------------------------
    W2d = W.reshape(-1, *W.shape[-2:]) if W.ndim > 2 else W            # (m, n)
    with torch.no_grad():
        s = torch.linalg.svdvals(W).flatten()                        # 1-D σ₁ ≥ …

    # ---- 2. energy and effective rank ------------------------------------------
    energy   = torch.cumsum(s.square(), 0)
    total_e  = energy[-1]
    k_eff    = int(torch.searchsorted(energy,
                                      energy_thresh * total_e).item()) + 1

    # ---- 3. plotting ------------------------------------------------------------
    fig, ax_scree = plt.subplots()

    # (a) Scree plot – singular values (log scale)
    ax_scree.semilogy(s.cpu(), marker='o', lw=0.8, label='σᵢ')
    ax_scree.set_xlabel('index $i$')
    ax_scree.set_ylabel('singular value $σ_i$ (log)')
    ax_scree.set_title('Spectrum diagnostics')
    ax_scree.grid(True, which='both', ls=':')

    # (b) Cumulative-energy curve (secondary axis, linear scale)
    ax_cum = ax_scree.twinx()
    ax_cum.plot((energy / total_e).cpu(), marker='x', c='tab:red',
                label='cum. energy')
    ax_cum.set_ylabel('cumulative energy')
    ax_cum.axhline(energy_thresh, ls='--', lw=1, c='tab:red')
    ax_cum.axvline(k_eff - 1, ls='--', lw=1, c='tab:gray')   # index is 0-based
    ax_cum.set_ylim(0, 1.05)

    # merge legends
    lines, labels = ax_scree.get_legend_handles_labels()
    lines2, labels2 = ax_cum.get_legend_handles_labels()
    ax_scree.legend(lines + lines2, labels + labels2, loc='best')

    plt.tight_layout()
    plt.show()

    return k_eff

