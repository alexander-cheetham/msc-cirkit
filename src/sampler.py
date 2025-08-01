"""Sampling columns/rows of Kron(A,A) with probabilities proportional to squared L2 norms using the product structure; no explicit Kronecker materialization."""

from __future__ import annotations

from typing import Protocol, Tuple, Optional

import torch
from torch import Tensor
from torch.distributions import Categorical


class KronL2Sampler(Protocol):
    """Callable protocol for Kronecker-L2 sampling."""

    def __call__(
        self,
        A: Tensor,
        count: int,
        axis: int,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, Tensor, Tuple[torch.LongTensor, torch.LongTensor]]:
        """Sample ``count`` indices from ``kron(A, A)`` along ``axis``."""
        ...


def kron_l2_sampler(
    A: Tensor,
    count: int,
    axis: int,
    *,
    seed: Optional[int] = None,
) -> Tuple[torch.LongTensor, Tensor, Tuple[torch.LongTensor, torch.LongTensor]]:
    """Sample indices from ``kron(A, A)`` with probability proportional to L2 norm squared.

    Parameters
    ----------
    A : Tensor, shape (m, n)
        Base matrix for the Kronecker product.
    count : int
        Number of samples to draw.
    axis : int
        ``0`` to sample rows, ``1`` to sample columns of ``kron(A, A)``.
    seed : int | None, optional
        Optional random seed for reproducibility.

    Returns
    -------
    flat_indices : LongTensor
        Flattened indices in the Kronecker matrix.
    weights : Tensor
        Importance weights ``1/(count * p_joint)``.
    pair_indices : tuple(LongTensor, LongTensor)
        Tuple ``(i1, i2)`` if sampling rows or ``(j1, j2)`` if sampling columns.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    m, n = A.shape
    device = A.device

    if axis == 1:
        norms = A.pow(2).sum(dim=0)  # column norms
        dim = n
    else:
        norms = A.pow(2).sum(dim=1)  # row norms
        dim = m

    total = norms.sum()
    if total == 0:
        probs = torch.full_like(norms, 1.0 / norms.numel())
    else:
        probs = norms / total

    dist = Categorical(probs=probs)
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    idx1 = dist.sample((count,), generator=g)
    idx2 = dist.sample((count,), generator=g)

    joint_prob = probs[idx1] * probs[idx2]
    flat = idx1 * dim + idx2
    weights = 1.0 / (count * joint_prob)

    return flat.long(), weights, (idx1.long(), idx2.long())


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    flat, w, (j1, j2) = kron_l2_sampler(A, count=5, axis=1, seed=42)
    print("Column sample indices:", flat)
    print("Weights:", w[:3])

    flat_r, w_r, (i1, i2) = kron_l2_sampler(A, count=5, axis=0, seed=123)
    print("Row sample indices:", flat_r)
    print("Weights:", w_r[:3])
