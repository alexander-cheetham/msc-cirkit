import torch
import torch.nn as nn
from torch import Tensor
from .sampler import kron_l2_sampler,  kron_cur_plan_torch_min
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.parameters.nodes import TorchTensorParameter
import gc # Added for garbage collection
from datetime import datetime # Added for logging timestamps


from cirkit.backend.torch.layers.inner import TorchSumLayer
class NystromSumLayer(TorchSumLayer):
    def __init__(
        self,
        original_layer: TorchSumLayer,
        *,
        rank,
        learnable_rank: bool = False,
        pivot: str = "uniform",
        semiring=None,
    ):

        # Create a dummy weight to satisfy the parent constructor
        dummy_node = TorchTensorParameter(
            original_layer.num_output_units,
            original_layer.num_input_units,
            num_folds=original_layer.num_folds,
            requires_grad=False,
        )
        dummy_weight = TorchParameter(
            modules=[dummy_node],
            in_modules={},
            outputs=[dummy_node],
        )
        dummy_weight.reset_parameters()

        if semiring is None:
            semiring = original_layer.semiring
        super().__init__(
            num_input_units=original_layer.num_input_units,
            num_output_units=original_layer.num_output_units,
            arity=original_layer.arity,
            weight=dummy_weight,
            semiring=semiring,
            num_folds=original_layer.num_folds,
        )
        # ------------------------------------------------------------------
        # 1 · Rank bookkeeping
        # ------------------------------------------------------------------
        self.rank = rank
        self.pivot = pivot
        # buffer → moves with .to()/ .cuda() but is not trainable
        self.register_buffer(
            "rank_param", torch.tensor(self.rank, dtype=torch.int32), persistent=False
        )

    
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
                        row_scale = torch.ones(I.shape[0], dtype=M_f.dtype, device=M_f.device)
                        col_scale = torch.ones(J.shape[0], dtype=M_f.dtype, device=M_f.device)
                    else:
                        # Using L1 norm sampler for better stability
                        if self.pivot == "l2":
                           I = kron_l2_sampler(M_f, target_rank=s, axis=0)
                           J = kron_l2_sampler(M_f, target_rank=s, axis=1)
                           row_scale = torch.ones(s, dtype=M_f.dtype, device=M_f.device)
                           col_scale = torch.ones(s, dtype=M_f.dtype, device=M_f.device)
                        elif self.pivot == "cur":
                            M_for_sampling = M_f
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
                    cond_num = torch.linalg.cond(A)

                    #print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Condition Number = {cond_num}")

                    if True:
                        # --- 3. SUCCESS PATH: Build Nyström Factors ---
                        #print("Condition number acceptable. Proceeding with Nyström.")
                        A = (row_scale[:, None] * A) * col_scale[None, :]
                        A_pinv = torch.linalg.pinv(A, rcond=1e-6)

                        
                        # U  K_o x s (all rows, pivot columns)
                        # V  K_i x s (all columns, pivot rows)
                        
                        # For U, we need the full column space defined by the sampled columns J
                        all_rows = torch.arange(K_o, device=M_f.device)
                        U_f = kron_block(all_rows, J) # K_o x s
                        U_f = U_f * col_scale[None, :] # Apply column scaling for CUR
                        
                        # For R, we need the full row space defined by the sampled rows I
                        all_cols = torch.arange(K_i, device=M_f.device)
                        R_f = kron_block(I, all_cols) # s x K_i
                        R_f = row_scale[:, None] * R_f # Apply row scaling for CUR

                        # Now, construct the final V factor
                        V_f = (A_pinv @ R_f).T # (K_i x s)
                        
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
                    U_svd, S_svd, Vh_svd = torch.linalg.svd(W_f)
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
            self.register_buffer("U", torch.stack(U_lr, dim=0))
            self.register_buffer("V", torch.stack(V_lr, dim=0))
