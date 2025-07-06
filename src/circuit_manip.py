from types import MethodType
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import (
    GaussianLayer,
    SumLayer,
    ProductLayer,
    Scope,
)
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from symbolic_nystrom import NystromSumLayer as SymbolicNystromSumLayer

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



import cirkit.symbolic.functional as SF
from cirkit.symbolic.io import plot_circuit
from cirkit.pipeline import PipelineContext
from helpers import define_circuit_one_sum

def build_and_compile_circuit(input_units: int, sum_units: int):
    """
    Build a one‐sum circuit with the given number of input and sum units,
    compile it in a sum‐product semiring, and return only the compiled circuit.
    All intermediate variables are deleted before returning.
    """
    # Build symbolic network
    net = define_circuit_one_sum(input_units, sum_units)

    # Compile network to a torch backend
    ctx = PipelineContext(
        backend="torch",
        semiring="sum-product",
        fold=False,
        optimize=False
    )
    cc = ctx.compile(net).cpu().eval()

    # Build and plot the partition‐function circuit
    symbolic_circuit_partition_func = SF.multiply(net, net)
    plot_circuit(symbolic_circuit_partition_func)

    # Compile the partition function circuit
    csc = ctx.compile(symbolic_circuit_partition_func)

    # (Optional) inspect a particular parameter
    kronparameter = csc.layers[1].weight._nodes[4]

    # Delete everything except the result
    del net, cc, ctx, symbolic_circuit_partition_func, kronparameter
    return csc


def build_deep_symbolic_circuit(
    num_layers: int,
    num_input_units: int,
    num_sum_units: int,
    *,
    rank: int,
    use_nystrom: bool = False,
) -> Circuit:
    """Construct a deep symbolic circuit alternating sum and product layers."""

    num_leaves = 2 ** num_layers
    layers: list = []
    in_layers: dict = {}

    p = Parameterization(activation="softmax", initialization="normal")
    weight_factory = parameterization_to_factory(p)

    current = []
    for i in range(num_leaves):
        g = GaussianLayer(scope=Scope({i}), num_output_units=num_input_units)
        layers.append(g)
        current.append(g)

    for depth in range(num_layers):
        next_layers = []
        is_sum = depth % 2 == 0
        for j in range(0, len(current), 2):
            ins = current[j : j + 2]
            if is_sum:
                out_units = 1 if depth == num_layers - 1 else num_sum_units
                if use_nystrom:
                    sl = SymbolicNystromSumLayer(
                        num_input_units=ins[0].num_output_units,
                        num_output_units=out_units,
                        rank=rank,
                        arity=len(ins),
                        U_factory=weight_factory,
                        V_factory=weight_factory,
                    )
                else:
                    sl = SumLayer(
                        num_input_units=ins[0].num_output_units,
                        num_output_units=out_units,
                        arity=len(ins),
                        weight_factory=weight_factory,
                    )
                layers.append(sl)
                in_layers[sl] = ins
                next_layers.append(sl)
            else:
                pl = ProductLayer(
                    ins[0].num_output_units,
                    ins[0].num_output_units,
                    arity=len(ins),
                )
                layers.append(pl)
                in_layers[pl] = ins
                next_layers.append(pl)
        current = next_layers

    return Circuit(layers, in_layers, outputs=current)


def compile_deep_circuit(
    num_layers: int,
    num_input_units: int,
    num_sum_units: int,
    *,
    rank: int,
    use_nystrom: bool = False,
) -> Circuit:
    """Build and compile a deep circuit to a torch backend."""

    symbolic = build_deep_symbolic_circuit(
        num_layers,
        num_input_units,
        num_sum_units,
        rank=rank,
        use_nystrom=use_nystrom,
    )
    ctx = PipelineContext(
        backend="torch",
        semiring="sum-product",
        fold=False,
        optimize=False,
    )
    return ctx.compile(symbolic).cpu().eval()

# --- 1. Imports --------------------------------------------------------------
import torch.nn as nn
from cirkit.backend.torch.layers.inner import TorchSumLayer        # the baseline
from nystromlayer import NystromSumLayer                           # your wrapper

# --- 2. Recursive graph-walk -------------------------------------------------
def replace_sum_layers(module: nn.Module, *, rank: int) -> None:
    """
    Walk `module` and in-place replace every TorchSumLayer with NystromSumLayer
    of the same weight but compressed to the given Nyström rank.

    Parameters
    ----------
    module : nn.Module         # csc, or any sub-module
    rank   : int               # target Nyström rank `s`
    """
    for name, child in list(module.named_children()):             
        if isinstance(child, TorchSumLayer):
            # swap in-place 
            setattr(module, name, NystromSumLayer(child, rank=rank))
        else:
            # descend the tree 
            replace_sum_layers(module=child, rank=rank)

def fix_address_book_modules(circuit, verbose=False) -> bool:
    """Replace old TorchSumLayer with NystromSumLayer in address book"""
    if not hasattr(circuit, '_address_book'):
        return False

    addr_book = circuit._address_book

    # Find the NystromSumLayer in the circuit
    nystrom_layer = None
    for name, module in circuit.named_modules():
        if isinstance(module, NystromSumLayer):
            nystrom_layer = module
            if verbose:
                print(f"Found NystromSumLayer at: {name}")
            break

    if nystrom_layer is None:
        if verbose:
            print("ERROR: No NystromSumLayer found in circuit!")
        return False

    # Update _entry_modules
    if hasattr(addr_book, '_entry_modules'):
        for i, module in enumerate(addr_book._entry_modules):
            if verbose:
                print(f"Entry {i}: {type(module).__name__ if module else 'None'}")

            if isinstance(module, TorchSumLayer) and not isinstance(module, NystromSumLayer):
                addr_book._entry_modules[i] = nystrom_layer

                # Verify the update
                if verbose:
                    print(f"  After update: {type(addr_book._entry_modules[i]).__name__}")
                    print(f"  Is NystromSumLayer? {isinstance(addr_book._entry_modules[i], NystromSumLayer)}")
                return True

    return False
