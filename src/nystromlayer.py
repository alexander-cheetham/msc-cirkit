import torch
import torch.nn as nn
from torch import Tensor
from .sampler import kron_l2_sampler

class NystromSumLayer_old(nn.Module):
    """
    Block-Nyström low-rank replacement for a fold-summed linear layer.
    
    Parameters
    ----------
    original_layer : nn.Module
        A layer whose `weight()` returns a 3-D tensor (F, K_o, K_i).
    rank : int
        Target Nyström rank *s*.
    learnable_rank : bool, optional
        If True, keeps `rank` as a non-trainable nn.Parameter so it moves
        with the model state dict; otherwise stores it as a buffer.
    """
    def __init__(self, original_layer: nn.Module, rank: int):
        super().__init__()
        self.original_layer = original_layer
        self.semiring = getattr(original_layer, "semiring", None)
        self.rank = int(rank)                                  

        # --- Keep rank inside the Module state -------------------------------
        rank_tensor = torch.tensor(self.rank, dtype=torch.int32,
                                   device=original_layer.weight().device)
        self.rank_param = nn.Parameter(rank_tensor.float(),
                                           requires_grad=False) 

        # ---------------------------------------------------------------------
        # Prepare Nyström factors
        # ---------------------------------------------------------------------
        with torch.no_grad():                                   # saves memory 
            #TODO: IMPROVE THIS BECAUSE IT STILL MATERIALISES THE MATRIX
            W = original_layer.weight()                         # (F, K_o, K_i)
            F_, K_o, K_i = W.shape
            s = self.rank                                       # local alias

            U_lr, V_lr = [], []

            for f in range(F_):
                W_f = W[f]                                      # (K_o, K_i)

                # 0. random pivot indices  -------------------------------
                I = torch.randperm(K_o, device=W_f.device)[:s]
                J = torch.randperm(K_i, device=W_f.device)[:s]
                I_c = torch.tensor([i for i in range(K_o) if i not in I],
                                   device=W_f.device)
                J_c = torch.tensor([j for j in range(K_i) if j not in J],
                                   device=W_f.device)

                # 1. SVD of pivot block  -------------------------------
                A = W_f[I][:, J]                                # (s, s)
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)  

                # 2. Extend singular vectors ----------------------------
                L_inv = torch.diag(1.0 / S)
                F_blk = W_f[I_c][:, J]                          # (K_o-s, s)
                B_blk = W_f[I][:, J_c]                          # (s, K_i-s)
                tilde_U = F_blk @ Vh.T @ L_inv
                tilde_H = L_inv @ U.T @ B_blk

                # 3. Assemble Nyström factors --------------------------
                C   = torch.cat([A, F_blk], dim=0)              # (K_o, s)
                R   = torch.cat([A, B_blk], dim=1)              # (s, K_i)
                A_pinv = torch.linalg.pinv(A)                   # (s, s)

                U_lr.append(C)                                  # (K_o, s)
                V_lr.append((A_pinv @ R).T)                     # (K_i, s)


            # Stack over folds and register as parameters (trainable)
            self.U = nn.Parameter(torch.stack(U_lr, dim=0))     # (F, K_o, s)
            self.V = nn.Parameter(torch.stack(V_lr, dim=0))     # (F, K_i, s)

    # -------------------------------------------------------------------------
    # Forward pass:  x  ->  U_lr V_lrᵀ  ---------------------------------------
    # -------------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        x : Tensor, shape (F, B, K_i)
        returns : Tensor, shape (F, B, K_o)
        """
        # First multiply by Vᵀ  (einsum handles broadcasting)  
        temp = torch.einsum('fbi,fir->fbr', x, self.V)
        # Then multiply by U    ---------------------------------------------
        return torch.einsum('fbr,for->fbo', temp, self.U)


from cirkit.backend.torch.layers.inner import TorchSumLayer
class NystromSumLayer(TorchSumLayer):
    def __init__(
        self,
        original_layer: TorchSumLayer,
        *,
        rank: int,
        learnable_rank: bool = False,
        pivot: str = "random",
        semiring=None,
    ):
        # ------------------------------------------------------------------
        # 0 · Call the parent ctor with the same signature it expects
        # ------------------------------------------------------------------

        if semiring is None:
            semiring = original_layer.semiring
        super().__init__(
            num_input_units=original_layer.num_input_units,
            num_output_units=original_layer.num_output_units,
            arity=original_layer.arity,
            weight=original_layer.weight,
            semiring=semiring,
            num_folds=original_layer.num_folds,
        )
        # ------------------------------------------------------------------
        # 1 · Rank bookkeeping
        # ------------------------------------------------------------------
        self.rank = int(rank)
        self.weight_orig = original_layer.weight()
        self.pivot = pivot
        # buffer → moves with .to()/ .cuda() but is not trainable
        self.register_buffer(
            "rank_param", torch.tensor(self.rank, dtype=torch.int32), persistent=False
        )

        # ------------------------------------------------------------------
        # 2 · Build Nyström factors from the *dense* weight we just copied
        # ------------------------------------------------------------------
        with torch.no_grad():
            self._build_factors_from(original_layer)
        del self.weight 

    
    # ------------------------------------------------------------------
    # 3 · We no longer need the dense weight inside the compressed layer.
    #     Replace it with a lightweight property to keep TorchSumLayer’s
    #     API intact, but avoid storing duplicate data.
    # ------------------------------------------------------------------
    @property
    def weight(self):
        w = torch.einsum("fok,fik->foi", self.U, self.V)
        return w

    @property
    def params(self):
        """Expose no trainable parameters to CirKit's reset mechanism."""
        return {}

    def reset_parameters(self):
        # Parameters are initialized during construction; nothing to reset
        pass

    # ------------------------------------------------------------------
    # Optional: expose a virtual dense weight for code that still calls
    # layer.weight() even after compression.
    # # ------------------------------------------------------------------

    def _get_name(self) -> str:          # nn.Module.__repr__ calls this
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # forward pass uses the evaluation semiring with Nyström factors
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        # x: (F, H, B, Ki) -> (F, B, H * Ki)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        # compute x · Vᵀ · U using a single einsum call under the semiring
        return self.semiring.einsum(
            "fbi,fir,for->fbo",
            inputs=(x,),
            operands=(self.V, self.U),
            dim=-1,
            keepdim=True,
        )

    # ------------------------------------------------------------------
    # helper ------------------------------------------------------------
    def _build_factors_from(self, original_layer: TorchSumLayer, pivots=None):
        """Construct Nyström factors without materialising the Kronecker weight.

        Parameters
        ----------
        original_layer : TorchSumLayer
            Layer providing the Kronecker-product weights.
        pivots : list[tuple[Tensor, Tensor]] | None
            Optional list of pivot index pairs ``(I, J)`` for each fold.  If not
            provided, random pivots are chosen as before.
        """
        with torch.no_grad():                                   # saves memory
            # ``original_layer.weight`` encodes the Kronecker product of a
            # smaller matrix with itself.  Extract that base matrix so we can
            # compute required sub-blocks without building the full product.
            if hasattr(original_layer.weight, "_nodes"):
                base_weight = original_layer.weight._nodes[0]()
                base_weight = torch.nn.functional.softmax(
                    base_weight, dim=-1
                )  # ensure probabilities sum to 1
                # assert torch.equal(
                #     torch.kron(base_weight, base_weight),
                #     original_layer.weight()
                # ), "Weight tensors differ!" 

                # print(f"kronecker of base weight matches materialised ORIGINAL CIRCUIT")
            else:
                base_weight = original_layer.weight()
            # Prepare weights depending on the semiring. For log-space
            # computation ("lse-sum"), convert the logits to log-probabilities;
            # otherwise operate on probabilities in linear space.
            F_, K_o_base, K_i_base = base_weight.shape
            K_o = K_o_base * K_o_base
            K_i = K_i_base * K_i_base
            s = self.rank                                       # local alias

            U_lr, V_lr = [], []

            for f in range(F_):
                M_f = base_weight[f]

                def kron_block(rows, cols):
                    r0 = torch.div(rows, K_o_base, rounding_mode="floor")
                    r1 = rows % K_o_base
                    c0 = torch.div(cols, K_i_base, rounding_mode="floor")
                    c1 = cols % K_i_base
                    return self.semiring.mul(M_f[r0][:, c0], M_f[r1][:, c1])

                if pivots is not None:
                    I, J = pivots[f]
                else:
                    if self.pivot == "l2":
                        # Importance sample rows/columns based on L2 norms.
                        # ``kron_l2_sampler`` returns flat indices as well as
                        # the base index pairs (j1, j2) or (i1, i2).  Those can
                        # be fed back into ``kron_block`` to reconstruct
                        # individual columns/rows if needed.
                        I, _, _ = kron_l2_sampler(M_f, s, axis=0)
                        J, _, _ = kron_l2_sampler(M_f, s, axis=1)
                    else:
                        I = torch.randperm(K_o, device=M_f.device)[:s]
                        J = torch.randperm(K_i, device=M_f.device)[:s]
                mask_I = torch.ones(K_o, dtype=torch.bool, device=M_f.device)
                mask_I[I] = False
                I_c = mask_I.nonzero(as_tuple=False).flatten()
                mask_J = torch.ones(K_i, dtype=torch.bool, device=M_f.device)
                mask_J[J] = False
                J_c = mask_J.nonzero(as_tuple=False).flatten()

                A = kron_block(I, J)
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)

                L_inv = torch.diag(1.0 / S)
                F_blk = kron_block(I_c, J)
                B_blk = kron_block(I, J_c)
                tilde_U = F_blk @ Vh.T @ L_inv
                tilde_H = L_inv @ U.T @ B_blk

                C   = torch.cat([A, F_blk], dim=0)
                R   = torch.cat([A, B_blk], dim=1)
                A_pinv = torch.linalg.pinv(A)

                U_lr.append(C)
                V_lr.append((A_pinv @ R).T)

            # Stack over folds and register as parameters (trainable)
            self.U = nn.Parameter(torch.stack(U_lr, dim=0))     # (F, K_o, s)
            self.V = nn.Parameter(torch.stack(V_lr, dim=0))     # (F, K_i, s)

