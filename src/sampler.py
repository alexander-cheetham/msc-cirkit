"""Sampling columns/rows of ``kron(A, A)`` by L2-norm importance without
materialising the Kronecker product.

The sampler oversamples and returns a set of unique flattened indices for the
Kronecker matrix.  These indices can be fed into a ``kron_block`` function to
reconstruct the corresponding rows or columns on the fly.
"""
from __future__ import annotations

from typing import Optional, Protocol

import torch
from torch import Tensor
from torch.distributions import Categorical


class KronL2Sampler(Protocol):
    """Callable protocol for Kronecker L2-norm sampling.

    Parameters
    ----------
    A: Tensor
        Base matrix ``A`` with shape ``(m, n)``.
    target_rank: int
        Number of *unique* indices required.
    axis: int
        ``0`` to sample rows of ``kron(A, A)``, ``1`` for columns.
    oversampling_p: int, optional
        Additional samples to draw to reduce the chance of duplicates.
    seed: int | None, optional
        Random seed for reproducibility.
    Returns
    -------
    LongTensor
        ``target_rank`` unique flattened indices along the requested axis.
    """

    def __call__(
        self,
        A: Tensor,
        target_rank: int,
        axis: int,
        *,
        oversampling_p: int = 10,
        seed: Optional[int] = None,
    ) -> torch.LongTensor:
        ...


def kron_l2_sampler(
    A: Tensor,
    target_rank: int,
    axis: int,
    *,
    oversampling_p: int = 10,
    seed: Optional[int] = None,
) -> torch.LongTensor:
    """Return ``target_rank`` unique indices from ``kron(A, A)``.

    The function samples according to squared L2 norms of rows or columns of
    ``A``.  Oversampling is performed to obtain enough unique indices when
    sampling with replacement.
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

    def sample_once(num: int) -> Tensor:
        idx1 = dist.sample((num,), generator=g)
        idx2 = dist.sample((num,), generator=g)
        return (idx1 * dim + idx2).long()

    need = target_rank + oversampling_p
    idx = sample_once(need)
    uniq = torch.unique(idx)

    while uniq.numel() < target_rank:
        need = target_rank - uniq.numel() + oversampling_p
        extra = sample_once(need)
        uniq = torch.unique(torch.cat([uniq, extra]))

    return uniq[:target_rank]


def create_submatrix(A: Tensor, rank: int, *, seed: Optional[int] = None) -> Tensor:
    """Construct a ``rank``×``rank`` submatrix of ``kron(A, A)``.

    Sampling is performed using :func:`kron_l2_sampler` for both rows and
    columns, then ``kron_block`` reconstructs the submatrix without explicitly
    forming the Kronecker product.
    """
    m, n = A.shape
    rows = kron_l2_sampler(A, rank, axis=0, seed=seed)
    cols = kron_l2_sampler(A, rank, axis=1, seed=None if seed is None else seed + 1)

    def kron_block(r: Tensor, c: Tensor) -> Tensor:
        r0 = torch.div(r, m, rounding_mode="floor")
        r1 = r % m
        c0 = torch.div(c, n, rounding_mode="floor")
        c1 = c % n
        return A[r0][:, c0] * A[r1][:, c1]

    return kron_block(rows, cols)


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(4, 3)

    rows = kron_l2_sampler(A, target_rank=5, axis=0, seed=0)
    cols = kron_l2_sampler(A, target_rank=5, axis=1, seed=1)
    print("Row indices:", rows)
    print("Column indices:", cols)

    sub = create_submatrix(A, rank=5, seed=2)
    print("Submatrix shape:", sub.shape)
