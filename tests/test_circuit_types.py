import pytest

try:
    from src.circuit_types import make_random_binary_tree_circuit
    from cirkit.symbolic.circuit import Circuit
except Exception:
    pytest.skip("cirkit library not installed", allow_module_level=True)


def test_builder_depth_affects_size():
    c1 = make_random_binary_tree_circuit(1, num_input_units=2, num_sum_units=2)
    c2 = make_random_binary_tree_circuit(3, num_input_units=2, num_sum_units=2)
    assert isinstance(c1, Circuit) and isinstance(c2, Circuit)
    assert len(c2.layers) > len(c1.layers)


def test_builder_honors_units():
    c = make_random_binary_tree_circuit(1, num_input_units=5, num_sum_units=3)
    top = c.outputs[0]
    assert top.num_output_units == 3
    assert top.num_input_units == 5
