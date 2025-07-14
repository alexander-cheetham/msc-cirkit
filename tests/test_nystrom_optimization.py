import pytest
import cirkit.pipeline as _cp
from cirkit.pipeline import PipelineContext

try:
    import torch
except Exception:
    torch = None
    pytest.skip("torch not installed", allow_module_level=True)

try:
    from types import MethodType
    from cirkit.symbolic.circuit import Circuit
    from cirkit.symbolic.layers import GaussianLayer, SumLayer
    from cirkit.templates.region_graph import RandomBinaryTree
    from cirkit.templates.utils import Parameterization, parameterization_to_factory
    import cirkit.symbolic.functional as SF
    from nystromlayer import NystromSumLayer
except Exception:
    pytest.skip("cirkit library not installed", allow_module_level=True)


def _build_circuit_one_sum(self, *, input_factory, sum_weight_factory, num_input_units, num_sum_units):
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


def define_deep_cp_circuit(num_input_units=2, num_sum_units=2):
    rg = RandomBinaryTree(4, seed=0)
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
    return SF.multiply(circuit, circuit)


def test_nystrom_flag_replaces_layers():
    circuit = define_circuit_one_sum(2, 2)
    circuit = SF.multiply(circuit, circuit)
    ctx = PipelineContext(
        backend="torch", semiring="sum-product", fold=False, optimize=True, nystrom=True
    )
    from cirkit.pipeline import compile as compile_circuit
    compiled = compile_circuit(circuit, ctx).cpu().eval()
    x = torch.randn(1, 4)
    out = compiled(x)
    assert out.shape[0] == 1
    found = False
    for m in compiled.modules():
        if isinstance(m, NystromSumLayer):
            found = True
            assert hasattr(m, "U")
            assert hasattr(m, "V")
    assert found


def test_nystrom_no_match_raises():
    circuit = define_circuit_one_sum(2, 2)
    ctx = PipelineContext(
        backend="torch", semiring="sum-product", fold=False, optimize=True, nystrom=True
    )
    from cirkit.pipeline import compile as compile_circuit
    with pytest.raises(ValueError):
        compile_circuit(circuit, ctx)


def test_flag_off_leaves_layers():
    circuit = define_circuit_one_sum(2, 2)
    circuit = SF.multiply(circuit, circuit)
    ctx = PipelineContext(
        backend="torch", semiring="sum-product", fold=False, optimize=True
    )
    from cirkit.pipeline import compile as compile_circuit
    compiled = compile_circuit(circuit, ctx, nystrom=False).cpu().eval()
    assert not any(isinstance(m, NystromSumLayer) for m in compiled.modules())


@pytest.mark.xfail(reason="upstream incompatibility")
def test_nystrom_deep_network():
    circuit = define_deep_cp_circuit(2, 2)
    ctx = PipelineContext(
        backend="torch", semiring="sum-product", fold=False, optimize=True, nystrom=True
    )
    from cirkit.pipeline import compile as compile_circuit
    compiled = compile_circuit(circuit, ctx).cpu().eval()
    assert sum(isinstance(m, NystromSumLayer) for m in compiled.modules()) > 0
