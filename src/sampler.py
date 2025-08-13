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


import torch
from torch import Tensor

def kron_l2_sampler(
        A: Tensor,
        target_rank: int,
        axis: int,
        *,
        oversampling_p: int = 10,
        seed: int | None = None,
        max_extra_iters: int = 5,  # cap to avoid pathological loops
    ) -> torch.LongTensor:
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (rows) or 1 (columns)")

        m, n = A.shape
        if axis == 1:
            max_val = A.max() # Find the maximum value in the tensor
            A_exp_stable = torch.exp(A - max_val)
            norms = A_exp_stable.sum(dim=0)
            dim = n
        else:
            max_val = A.max() # Find the maximum value in the tensor
            A_exp_stable = torch.exp(A - max_val)
            norms = A_exp_stable.sum(dim=1)
            dim = m

        total = norms.sum()
        if total == 0 or not torch.isfinite(total):
            probs = torch.full_like(norms, 1.0 / norms.numel())
        else:
            probs = norms / total

        # sanitize probs: ensure no NaNs / infs and normalize
        if not torch.isfinite(probs).all():
            print("Warning: non-finite values in probabilities, replacing with zeros.")

            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        s = probs.sum()
        if s == 0:
            probs = torch.full_like(probs, 1.0 / probs.numel())
        else:
            probs = probs / s

        # explicit generator to avoid global state mutation
        gen = None
        if seed is not None:
            gen = torch.Generator(probs.device).manual_seed(seed)

        def sample_once(num: int) -> Tensor:
            # vectorized draws instead of Categorical.sample
            if gen is not None:
                idx1 = torch.multinomial(probs, num, replacement=True, generator=gen)
                idx2 = torch.multinomial(probs, num, replacement=True, generator=gen)
            else:
                idx1 = torch.multinomial(probs, num, replacement=True)
                idx2 = torch.multinomial(probs, num, replacement=True)
            return (idx1 * dim + idx2).long()

        need = target_rank 
        idx = sample_once(need)

        return idx


# kron_cur_min.py
import torch
from typing import Optional, Tuple, Dict

# ---- SVD + leverage on A (rank-k or full/statistical) -----------------------
def _topk_svd_torch(A: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    if k is not None:
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
    return U, S, Vh

def leverage_scores_A_torch(A: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
    U, _, Vh = _topk_svd_torch(A, k)
    row_lev = U.abs().pow(2).sum(dim=1)            # ||U_{i,*}||^2
    col_lev = Vh.conj().T.abs().pow(2).sum(dim=1)  # ||V_{j,*}||^2
    sum_val = int(round(float(row_lev.sum().item())))
    return row_lev, col_lev, sum_val

# ---- Minimal CUR planner for F = kron(A, A) ---------------------------------
def kron_cur_plan_torch_min(
    A: torch.Tensor,
    *,
    c: int,
    r: int,
    k: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    return_flat_indices: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Plan leverage-score CUR sampling for F = kron(A, A) *without* forming F.

    Returns a dict with ONLY small tensors:
      - I_pairs: (r,2) row-pair indices for F   (values in [0..m-1])
      - J_pairs: (c,2) col-pair indices for F   (values in [0..n-1])
      - row_scale: (r,) Exactly(r) scaling:  1/sqrt(r * p_row_pair)
      - col_scale: (c,) Exactly(c) scaling:  1/sqrt(c * p_col_pair)
      - k_used: int rank used for leverages
      - (optional) I_flat, J_flat: flattened indices in [0..m*m-1] / [0..n*n-1]
    """
    device, dtype = A.device, A.dtype
    m, n = A.shape

    # 1) leverage on A (rank-k or statistical)
    row_lev, col_lev, sum_val = leverage_scores_A_torch(A, k=k)
    p_row_1D = (row_lev / row_lev.sum()).to(device=device, dtype=dtype)
    p_col_1D = (col_lev / col_lev.sum()).to(device=device, dtype=dtype)

    # 2) sample pair indices for F using product distributions (no m^2/n^2 vectors)
    I1 = torch.multinomial(p_row_1D, num_samples=r, replacement=True, generator=generator)
    I2 = torch.multinomial(p_row_1D, num_samples=r, replacement=True, generator=generator)
    J1 = torch.multinomial(p_col_1D, num_samples=c, replacement=True, generator=generator)
    J2 = torch.multinomial(p_col_1D, num_samples=c, replacement=True, generator=generator)
    I_pairs = torch.stack([I1, I2], dim=1)  # (r,2)
    J_pairs = torch.stack([J1, J2], dim=1)  # (c,2)

    # 3) Exactly(r)/Exactly(c) rescaling vectors (pair probabilities)
    p_row_pairs = p_row_1D[I1] * p_row_1D[I2]  # (r,)
    p_col_pairs = p_col_1D[J1] * p_col_1D[J2]  # (c,)
    row_scale = (r * p_row_pairs).reciprocal().sqrt().to(dtype)
    col_scale = (c * p_col_pairs).reciprocal().sqrt().to(dtype)

    plan: Dict[str, torch.Tensor] = dict(
        I_pairs=I_pairs,
        J_pairs=J_pairs,
        row_scale=row_scale,
        col_scale=col_scale,
        k_used=torch.tensor(sum_val, device=device),
    )

    if return_flat_indices:
        I_flat = I_pairs[:, 0] * m + I_pairs[:, 1]  # (r,)
        J_flat = J_pairs[:, 0] * n + J_pairs[:, 1]  # (c,)
        plan["I_flat"] = I_flat
        plan["J_flat"] = J_flat

    return plan

