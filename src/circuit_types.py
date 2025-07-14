"""Predefined circuit builders used in benchmarks."""

from types import MethodType
from typing import Callable, Dict, Optional

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import GaussianLayer, SumLayer
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.templates.utils import Parameterization, parameterization_to_factory


def build_circuit_one_sum(
    self,
    *,
    input_factory: Callable,
    sum_weight_factory: Callable,
    num_input_units: int,
    num_sum_units: int,
) -> Circuit:
    """Helper to build a single-sum circuit."""
    leaves = [n for n in self.topological_ordering() if not self.region_inputs(n)]
    layers = []
    in_layers = {}
    gaussians = []
    for leaf in leaves:
        g = input_factory(leaf.scope, num_input_units)
        layers.append(g)
        gaussians.append(g)
    sum_layer = SumLayer(
        num_input_units=num_input_units,
        num_output_units=num_sum_units,
        arity=len(gaussians),
        weight_factory=sum_weight_factory,
    )
    layers.append(sum_layer)
    in_layers[sum_layer] = gaussians
    return Circuit(layers, in_layers, outputs=[sum_layer])


def make_random_binary_tree_circuit(
    depth: int,
    *,
    num_input_units: Optional[int] = None,
    num_sum_units: Optional[int] = None,
    seed: int = 0,
) -> Circuit:
    """Create a random binary tree circuit of the given depth."""

    rg = RandomBinaryTree(depth, seed=seed)

    input_factory = lambda scope, n: GaussianLayer(scope=scope, num_output_units=n)
    p = Parameterization(activation="softmax", initialization="normal")
    sum_param_factory = parameterization_to_factory(p)

    build_kwargs = {
        "input_factory": input_factory,
        "sum_weight_factory": sum_param_factory,
        "num_input_units": num_input_units or 1,
        "num_sum_units": num_sum_units or 1,
        "num_classes": 1,
        "sum_product": "cp",
    }

    if depth == 1:
        rg.build_circuit = MethodType(build_circuit_one_sum, rg)
        build_kwargs.pop("sum_product")
        build_kwargs.pop("num_classes")

    return rg.build_circuit(**build_kwargs)


def make_mnist_circuit(
    *,
    num_input_units: Optional[int] = None,
    num_sum_units: Optional[int] = None,
    seed: int = 0,
) -> Circuit:
    """Placeholder for an MNIST-suitable circuit."""
    return make_random_binary_tree_circuit(
        3, num_input_units=num_input_units, num_sum_units=num_sum_units, seed=seed
    )


CIRCUIT_BUILDERS: Dict[str, Callable[..., Circuit]] = {
    "one_sum": lambda **kw: make_random_binary_tree_circuit(1, **kw),
    "deep_cp_circuit": make_random_binary_tree_circuit,
    "MNIST": make_mnist_circuit,
}
