import torch
import torch.nn as nn
from torch import Tensor

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
    ):
        # ------------------------------------------------------------------
        # 0 · Call the parent ctor with the same signature it expects
        # ------------------------------------------------------------------

        super().__init__(
            num_input_units=original_layer.num_input_units,
            num_output_units=original_layer.num_output_units,
            arity=original_layer.arity,
            weight=original_layer.weight,     
            semiring=original_layer.semiring,
            num_folds=original_layer.num_folds,
        )
        # ------------------------------------------------------------------
        # 1 · Rank bookkeeping
        # ------------------------------------------------------------------
        self.rank = int(rank)
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
        # debug prints
        print("U shape (F,Ko,Ki):", self.U.shape)
        print("V shape (F,H·Ki,Ki):", self.V.shape)
    
        # einsum: sum over k, keep o from U and i from V
        w = torch.einsum("fok,fik->foi", self.U, self.V)
    
        print("🔧 Computed weight with shape", tuple(w.shape))
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
    # forward pass is unchanged from earlier answer
    # ------------------------------------------------------------------
    def forward(self, x):
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        temp = torch.einsum("fbi,fir->fbr", x, self.V)
        return torch.einsum("fbr,for->fbo", temp, self.U)

    # ------------------------------------------------------------------
    # helper ------------------------------------------------------------
    def _build_factors_from(self, original_layer: TorchSumLayer) -> None:
        """Compute Nyström factors from a dense weight matrix using SVD."""
        with torch.no_grad():
            W = original_layer.weight()
            F_, Ko, Ki = W.shape
            r = self.rank
            U_lr, V_lr = [], []
            for f in range(F_):
                M_f = W[f]
                U, S, Vh = torch.linalg.svd(M_f, full_matrices=False)
                U_lr.append(U[:, :r] * S[:r].sqrt())
                V_lr.append((S[:r].sqrt()[None, :] * Vh[:r, :]).T)
            self.U = nn.Parameter(torch.stack(U_lr, dim=0))
            self.V = nn.Parameter(torch.stack(V_lr, dim=0))

