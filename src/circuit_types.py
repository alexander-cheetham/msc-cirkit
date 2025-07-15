"""Predefined circuit builders used in benchmarks."""

from types import MethodType
from typing import Callable, Dict, Optional

from cirkit.templates import data_modalities

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import GaussianLayer, SumLayer
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.templates.utils import Parameterization, parameterization_to_factory
from helpers import build_circuit_one_sum
from cirkit.symbolic.circuit import are_compatible


def define_circuit_one_sum(num_input_units=2, num_sum_units=2):
    rg = RandomBinaryTree(1, depth=None, num_repetitions=1, seed=42)
    rg.build_circuit = MethodType(build_circuit_one_sum, rg)
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


def make_squarable_mnist_circuit(
    *,
    region_graph,
    num_input_units: Optional[int] = 64,
    num_sum_units: Optional[int] = 64,
) -> Circuit:
    """Construct a symbolic circuit tailored for MNIST data.

    region_graph: any region graph object compatible with the circuit.
    """
    circuit = data_modalities.image_data(
        (1, 28, 28),
        region_graph=region_graph,
        input_layer="categorical",
        num_input_units=num_input_units,
        sum_product_layer="cp",
        num_sum_units=num_sum_units,
        num_classes=1,
        sum_weight_param=Parameterization(
            activation="softmax",
            initialization="normal",
        ),
    )
    if not are_compatible(circuit, circuit):
        raise ValueError("The provided region_graph produces an incompatible circuit.")
    return circuit


CIRCUIT_BUILDERS: Dict[str, Callable[..., Circuit]] = {
    "one_sum": define_circuit_one_sum,
    "deep_cp_circuit": make_random_binary_tree_circuit,
    "MNIST": make_squarable_mnist_circuit,
}
