import torch
import torch.nn as nn
from torch import Tensor
from .sampler import kron_l2_sampler,  kron_cur_plan_torch_min

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
        pivot: str = "uniform",
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
            provided, indices are chosen according to ``self.pivot``.
        """
        # --- Constants for the retry mechanism ---
        MAX_RETRIES = 3
        CONDITION_THRESHOLD = 1e7 # Threshold for what's considered ill-conditioned

        with torch.no_grad():                                   # saves memory

            base_weight = type(original_layer.weight)(original_layer.weight._nodes, original_layer.weight._in_nodes, [original_layer.weight._nodes[-2]])()
            
            F_, K_o_base, K_i_base = base_weight.shape
            K_o = K_o_base * K_o_base
            K_i = K_i_base * K_i_base
            s = self.rank                                       # local alias

            U_lr, V_lr = [], []

            def kron_block(rows, cols):
                    r0 = torch.div(rows, K_o_base, rounding_mode="floor")
                    r1 = rows % K_o_base
                    c0 = torch.div(cols, K_i_base, rounding_mode="floor")
                    c1 = cols % K_i_base
                    return self.semiring.mul(M_f[r0][:, c0], M_f[r1][:, c1])
                    #return M_f[r0][:, c0]* M_f[r1][:, c1]

            # Helper: map pair indices -> flat Kronecker indices
            def _flat_pairs_to_indices(pairs, base):
                # pairs: (s,2) with entries in [0..base-1]
                return pairs[:, 0] * base + pairs[:, 1]
            

            for f in range(F_):
                M_f = base_weight[f]
                

                nystrom_success = False
                for attempt in range(MAX_RETRIES):
                    # --- 1. Sample Pivots ---
                    if pivots is not None:
                        I, J = pivots[f]
                    else:
                        # Using L1 norm sampler for better stability
                        if self.pivot == "l2":
                           I = kron_l2_sampler(M_f, target_rank=s, axis=0)
                           J = kron_l2_sampler(M_f, target_rank=s, axis=1)
                           row_scale = torch.ones(s, dtype=M_f.dtype, device=M_f.device)
                           col_scale = torch.ones(s, dtype=M_f.dtype, device=M_f.device)
                        elif self.pivot == "cur":
                            M_for_sampling = M_f.to(torch.float32)
                            plan = kron_cur_plan_torch_min(
                                M_for_sampling, r=s, c=s, k=min(s, K_o_base, K_i_base),
                                generator=None, return_flat_indices=False
                            )
                            I = _flat_pairs_to_indices(plan["I_pairs"].to(M_f.device), K_o_base)
                            J = _flat_pairs_to_indices(plan["J_pairs"].to(M_f.device), K_i_base)
                            row_scale = plan["row_scale"].to(device=M_f.device, dtype=M_f.dtype)
                            col_scale = plan["col_scale"].to(device=M_f.device, dtype=M_f.dtype)

                        else: # Fallback to uniform or original L2
                           I = torch.randperm(K_o, device=M_f.device)[:s]
                           J = torch.randperm(K_i, device=M_f.device)[:s]
                           row_scale = torch.ones(s, dtype=M_f.dtype, device=M_f.device)
                           col_scale = torch.ones(s, dtype=M_f.dtype, device=M_f.device)

                    # --- 2. Form Pivot Matrix and Check Condition ---
                    A = kron_block(I, J)
                    A_complex32 = A.to(torch.complex64)
                    cond_num = torch.linalg.cond(A_complex32)

                    #print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Condition Number = {cond_num}")

                    if not torch.isinf(cond_num) and cond_num < CONDITION_THRESHOLD:
                        # --- 3. SUCCESS PATH: Build Nyström Factors ---
                        #print("Condition number acceptable. Proceeding with Nyström.")
                        mask_I = torch.ones(K_o, dtype=torch.bool, device=M_f.device)
                        mask_I[I] = False
                        I_c = mask_I.nonzero(as_tuple=False).flatten()
                        mask_J = torch.ones(K_i, dtype=torch.bool, device=M_f.device)
                        mask_J[J] = False
                        J_c = mask_J.nonzero(as_tuple=False).flatten()

                        F_blk = kron_block(I_c, J)
                        B_blk = kron_block(I, J_c)

                        C = torch.cat([A, F_blk], dim=0)
                        C = C * col_scale[None, :]      
                        R = torch.cat([A, B_blk], dim=1)
                        R = row_scale[:, None] * R   
                        
                        A = (row_scale[:, None] * A) * col_scale[None, :].to(torch.complex64)
                        A_pinv = torch.linalg.pinv(A, rcond=1e-6)

                        U_f = C
                        V_f = (A_pinv @ R).T
                        
                        U_lr.append(U_f)
                        V_lr.append(V_f)
                        nystrom_success = True
                        break # Exit the retry loop
                
                if not nystrom_success:
                    # --- 4. FALLBACK PATH: Use Full SVD ---
                    # print(f"WARNING: All {MAX_RETRIES} attempts failed for fold {f}. "
                    #       "Falling back to full SVD. This will be slower but is stable.")
                    
                    # Materialize the full Kronecker product matrix
                    W_f = torch.kron(M_f, M_f)
                    
                    # Compute its SVD and truncate to the target rank 's'
                    U_svd, S_svd, Vh_svd = torch.linalg.svd(W_f.to(torch.complex64))
                    actual_rank = min(s, S_svd.shape[0])
                    # If the actual rank is 0, it means the matrix is a zero matrix. Handle this case.
                    if actual_rank == 0:
                        print(f"Warning: Layer produced a zero matrix. Skipping factorization.")

                    # Use the actual_rank for slicing
                    U_svd_s = U_svd[:, :actual_rank]
                    S_svd_s = S_svd[:actual_rank]
                    Vh_svd_s = Vh_svd[:actual_rank, :]

                    # --- END MODIFICATION ---

                    # Create U and V factors from the SVD components.
                    # This part of your code is correct and does not need to change.
                    S_sqrt = torch.sqrt(S_svd_s)
                    U_f = U_svd_s @ torch.diag(S_sqrt).to(U_svd_s.dtype)
                    V_f = (Vh_svd_s.T @ torch.diag(S_sqrt).to(Vh_svd_s.dtype))
                    
                    U_lr.append(U_f)
                    V_lr.append(V_f)
                
                

            # --- 5. Finalize Parameters ---
            # Stack over folds and register as parameters (trainable)
            self.U = nn.Parameter(torch.stack(U_lr, dim=0))
            self.V = nn.Parameter(torch.stack(V_lr, dim=0))
