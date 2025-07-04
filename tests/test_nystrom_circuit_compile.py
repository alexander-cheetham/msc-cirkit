import pytest

try:
    import torch
except Exception:
    torch = None
    pytest.skip("torch not installed", allow_module_level=True)

try:
    from helpers import define_circuit_one_nystrom
    from cirkit.pipeline import PipelineContext
except Exception:
    pytest.skip("cirkit library not installed", allow_module_level=True)


def test_compile_and_infer():
    circuit = define_circuit_one_nystrom(num_input_units=2, num_sum_units=2, rank=1)
    ctx = PipelineContext(backend="torch", semiring="sum-product", fold=False, optimize=False)
    compiled = ctx.compile(circuit).cpu().eval()
    x = torch.randn(1, 4)
    out = compiled(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 1

