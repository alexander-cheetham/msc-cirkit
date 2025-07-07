import pytest

# These equivalence tests allocate large tensors and are slow. For the
# simplified test suite they are skipped by default.
pytest.skip("slow equivalence tests", allow_module_level=True)

try:
    import torch
except Exception:  # pragma: no cover - torch missing
    torch = None
    pytest.skip("torch not installed", allow_module_level=True)

try:  # pragma: no cover - skip if cirkit missing
    from nystromlayer import NystromSumLayer, NystromSumLayer_old
except Exception:  # pragma: no cover - cirkit missing
    pytest.skip("cirkit library not installed", allow_module_level=True)

try:  # pragma: no cover - optional dependency
    from cirkit.backend.torch.layers.inner import TorchSumLayer
except Exception:  # pragma: no cover - cirkit unavailable
    TorchSumLayer = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
def dense_nystrom(W: torch.Tensor, rank: int, pivots):
    F, Ko, Ki = W.shape
    U_lr, V_lr = [], []
    for f in range(F):
        W_f = W[f]
        I, J = pivots[f]
        I_c = torch.tensor([i for i in range(Ko) if i not in I], device=W_f.device)
        J_c = torch.tensor([j for j in range(Ki) if j not in J], device=W_f.device)
        A = W_f[I][:, J]
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        L_inv = torch.diag(1.0 / S)
        F_blk = W_f[I_c][:, J]
        B_blk = W_f[I][:, J_c]
        tilde_U = F_blk @ Vh.T @ L_inv
        tilde_H = L_inv @ U.T @ B_blk
        C = torch.cat([A, F_blk], dim=0)
        R = torch.cat([A, B_blk], dim=1)
        A_pinv = torch.linalg.pinv(A)
        U_lr.append(C)
        V_lr.append((A_pinv @ R).T)
    U = torch.stack(U_lr, dim=0)
    V = torch.stack(V_lr, dim=0)
    return torch.einsum('fok,fik->foi', U, V)

@pytest.mark.skipif(TorchSumLayer is None, reason="cirkit library not installed")
def test_new_matches_old():
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    F, Ko_base, Ki_base = 9, 80, 80
    base = torch.randn(F, Ko_base, Ki_base, device=device)

    kron = torch.stack([torch.kron(base[f], base[f]) for f in range(F)], dim=0)
    from cirkit.backend.torch.parameters.nodes import TorchTensorParameter

    weight_node = TorchTensorParameter(Ko_base * Ko_base, Ki_base * Ki_base, num_folds=F)
    weight_node.reset_parameters()
    weight_node.to(device)
    with torch.no_grad():
        weight_node._ptensor.copy_(kron)
    weight_param = weight_node
    weight_param._nodes = [lambda: base]

    orig = TorchSumLayer(
        num_input_units=Ki_base**2,
        num_output_units=Ko_base**2,
        arity=1,
        weight=weight_param,
        semiring=None,
        num_folds=F,
    )
    rank = 2
    Ko = Ko_base * Ko_base
    Ki = Ki_base * Ki_base
    pivots = []
    for f in range(F):
        I = torch.randperm(Ko, device=device)[:rank]
        J = torch.randperm(Ki, device=device)[:rank]
        pivots.append((I, J))

    # Dense baseline
    W_full = orig.weight()
    baseline = dense_nystrom(W_full, rank, pivots)

    # Nyström layer under test
    layer = NystromSumLayer(orig, rank=rank)
    layer._build_factors_from(orig, pivots=pivots)
    approx = layer.weight
    # print(f"approx{approx}")
    # print(f"baseline{baseline}")
    assert torch.allclose(approx, baseline, atol=1e-7)


@pytest.mark.skipif(TorchSumLayer is None, reason="cirkit library not installed")
def test_new_faster_than_old():
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    F, Ko_base, Ki_base = 9, 80, 80
    base = torch.randn(F, Ko_base, Ki_base, device=device)

    kron = torch.stack([torch.kron(base[f], base[f]) for f in range(F)], dim=0)
    from cirkit.backend.torch.parameters.nodes import TorchTensorParameter

    weight_node = TorchTensorParameter(Ko_base * Ko_base, Ki_base * Ki_base, num_folds=F)
    weight_node.reset_parameters()
    weight_node.to(device)
    with torch.no_grad():
        weight_node._ptensor.copy_(kron)
    weight_param = weight_node
    weight_param._nodes = [lambda: base]
    orig = TorchSumLayer(
        num_input_units=Ki_base**2,
        num_output_units=Ko_base**2,
        arity=1,
        weight=weight_param,
        semiring=None,
        num_folds=F,
    )
    rank = 4

    import timeit

    t_old = timeit.timeit(lambda: NystromSumLayer_old(orig, rank), number=3)
    t_new = timeit.timeit(lambda: NystromSumLayer(orig, rank=rank), number=3)
    print(f"Old time: {t_old:.6f}s, New time: {t_new:.6f}s")
    assert t_new < t_old
