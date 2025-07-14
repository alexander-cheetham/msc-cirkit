"""Predefined circuit builders used in benchmarks."""

from types import MethodType
from typing import Callable, Dict, Optional

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import GaussianLayer, SumLayer
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.templates.utils import Parameterization, parameterization_to_factory


def define_circuit_one_sum(num_input_units=2, num_sum_units=2):
    rg = RandomBinaryTree(1, depth=None, num_repetitions=1, seed=42)
    rg.build_circuit = MethodType(_build_circuit_one_sum, rg)
    input_factory = lambda scope, n: GaussianLayer(scope=scope, num_output_units=n)
    p = Parameterization(activation="softmax", initialization="normal")
    sum_param_factory = parameterization_to_factory(p)
    return rg.build_circuit(
        input_factory=input_factory,
        sum_weight_factory=sum_param_factory,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
    )


def make_random_binary_tree_circuit(
    depth: int,
    *,
    num_input_units: Optional[int] = 1,
    num_sum_units: Optional[int] = 1,
    seed: int = 0,
) -> Circuit:
    print(depth, num_input_units, num_sum_units, seed)
    rg = RandomBinaryTree(depth, seed=seed)
    input_factory = lambda scope, n: GaussianLayer(scope=scope, num_output_units=n)
    p = Parameterization(activation="softmax", initialization="normal")
    sum_param_factory = parameterization_to_factory(p)
    circuit = rg.build_circuit(
        input_factory=input_factory,
        sum_product="cp",
        sum_weight_factory=sum_param_factory,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        num_classes=1,
    )
    return circuit


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
    "one_sum": define_circuit_one_sum,
    "deep_cp_circuit": make_random_binary_tree_circuit,
    "MNIST": make_mnist_circuit,
}