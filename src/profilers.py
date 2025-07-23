"""Memory and computation profilers."""

import torch
import tracemalloc
import numpy as np
import torch.distributed as dist
import wandb
import os
try:
    wandb.require("legacy-service")
except wandb.errors.UnsupportedError:
    # ignore if the legacy-service requirement isn’t supported
    pass

class WandbMemoryProfiler:
    """Profile memory usage and log to wandb"""
    

    def profile_gpu(func, *args, **kwargs):
        if not torch.cuda.is_available():
            return 0, {}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_mb = torch.cuda.memory_allocated() / (1024**2)

        # Run and sync
        result = func(*args, **kwargs)
        torch.cuda.synchronize()

        peak_mb    = torch.cuda.max_memory_allocated() / (1024**2)
        current_mb = torch.cuda.memory_allocated() / (1024**2)

        # Build tensor on ALL ranks with identical dtype/shape
        dist.barrier()
        t = torch.tensor([initial_mb, peak_mb, current_mb],
                        device='cuda', dtype=torch.float32)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # everyone participates, but only rank0 cares about the value
            torch.distributed.reduce(t, dst=0, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.barrier()

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            stats = {
                "initial_memory_mb":  t[0].item(),
                "peak_memory_mb":     t[1].item(),
                "current_memory_mb":  t[2].item(),
                "memory_increase_mb": t[1].item() - t[0].item(),
            }
            return t[1].item(), stats

        return 0, {}
    
    @staticmethod
    def profile_cpu(func, *args, **kwargs):
        import tracemalloc
        
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / (1024**2)
        stats = {
            "peak_memory_mb": peak_mb,
            "current_memory_mb": current / (1024**2)
        }
        
        return peak_mb, stats
    
    @classmethod
    def profile_and_log(cls, func, *args, device='cuda', prefix="", rank=0, **kwargs):
        """Profile and log to wandb with prefix"""
        if device == 'cuda' and torch.cuda.is_available():
            peak, stats = cls.profile_gpu(func, *args, **kwargs)
        else:
            peak, stats = cls.profile_cpu(func, *args, **kwargs)
        
        # Log to wandb with prefix
        wandb_stats = {f"{prefix}/{k}": v for k, v in stats.items()}
        if rank == 0:
            wandb.log(wandb_stats)
        
        return peak
class FLOPCounter:
    """Count theoretical FLOPs for different operations."""
    
    @staticmethod
    def kronecker_forward(batch_size: int, F: int, K_o: int, K_i: int) -> int:
        """FLOPs for full Kronecker product forward pass."""
        return 2 * F * batch_size * (K_o**2) * (K_i**2)
    
    @staticmethod
    def nystrom_forward(batch_size: int, F: int, K_o: int, K_i: int, rank: int) -> int:
        """FLOPs for Nyström approximation forward pass."""
        flops = 2 * F * batch_size * (K_i**2) * rank
        flops += 2 * F * batch_size * rank * (K_o**2)
        return flops
    
    @staticmethod
    def theoretical_speedup(K_o: int, K_i: int, rank: int) -> float:
        """Theoretical speedup factor."""
        original = (K_o**2) * (K_i**2)
        nystrom = rank * ((K_i**2) + (K_o**2))
        return original / nystrom
