"""Memory and computation profilers."""

import torch
import tracemalloc
import numpy as np
import wandb
try:
    wandb.require("legacy-service")
except wandb.errors.UnsupportedError:
    # ignore if the legacy-service requirement isn’t supported
    pass

class WandbMemoryProfiler:
    """Profile memory usage and log to wandb"""
    
    @staticmethod
    def profile_gpu(func, *args, **kwargs):
        if not torch.cuda.is_available():
            return 0, {}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / (1024**2)
        
        # Run function
        result = func(*args, **kwargs)
        
        # Force synchronization
        if isinstance(result, torch.Tensor) and result.is_cuda:
            torch.cuda.synchronize()
        
        # Get memory stats
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
        current_memory = torch.cuda.memory_allocated() / (1024**2)
        
        stats = {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "current_memory_mb": current_memory,
            "memory_increase_mb": peak_memory - initial_memory
        }
        
        return peak_memory, stats
    
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
    def profile_and_log(cls, func, *args, device='cuda', prefix="", **kwargs):
        """Profile and log to wandb with prefix"""
        if device == 'cuda' and torch.cuda.is_available():
            peak, stats = cls.profile_gpu(func, *args, **kwargs)
        else:
            peak, stats = cls.profile_cpu(func, *args, **kwargs)
        
        # Log to wandb with prefix
        wandb_stats = {f"{prefix}/{k}": v for k, v in stats.items()}
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
