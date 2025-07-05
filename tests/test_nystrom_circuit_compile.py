import pytest

try:
    import torch
except Exception:
    torch = None
    pytest.skip("torch not installed", allow_module_level=True)

try:
    from helpers import define_circuit_one_sum
    from cirkit.pipeline import PipelineContext
    import cirkit.symbolic.functional as SF
    from src.circuit_manip import replace_sum_layers, fix_address_book_modules
    from nystromlayer import NystromSumLayer
except Exception:
    pytest.skip("cirkit library not installed", allow_module_level=True)


def test_compile_and_infer():
    circuit = define_circuit_one_sum(num_input_units=2, num_sum_units=2)
    circuit = SF.multiply(circuit, circuit)
    ctx = PipelineContext(backend="torch", semiring="sum-product", fold=False, optimize=False)
    compiled = ctx.compile(circuit).cpu().eval()
    replace_sum_layers(compiled, rank=1)
    fix_address_book_modules(compiled)
    x = torch.randn(1, 4)
    out = compiled(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 1

    found = False
    for m in compiled.modules():
        if isinstance(m, NystromSumLayer):
            found = True
            assert hasattr(m, "U")
            assert hasattr(m, "V")
            assert hasattr(m, "weight_orig")
            assert m.weight.shape == m.weight_orig.shape
    assert found

