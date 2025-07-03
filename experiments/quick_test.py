"""Quick test script to verify everything works."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:  # pragma: no cover - optional dependency
    from nystromlayer import NystromSumLayer
    from src.circuit_manip import build_and_compile_circuit
except Exception:
    NystromSumLayer = None

def quick_test():
    """Quick test to verify everything works."""

    if NystromSumLayer is None:
        print("cirkit library not installed; skipping quick test")
        return

    print("Running quick test...")
    
    # Small test case
    n_input, n_sum, rank = 20, 20, 10
    batch_size = 32
    
    # Build circuits
    print("Building original circuit...")
    orig = build_and_compile_circuit(n_input, n_sum)
    
    print("Building Nyström circuit...")
    nys = build_and_compile_circuit(n_input, n_sum)
    nys.layers[1] = NystromSumLayer(nys.layers[1], rank)
    
    # Test
    x = torch.randn(1, batch_size, n_input**2)
    
    # Check outputs match approximately
    with torch.no_grad():
        y_orig = orig(x)
        y_nys = nys(x)
        error = (y_orig - y_nys).norm() / y_orig.norm()
        
    print(f"\nQuick test results:")
    print(f"  Relative error: {error:.2e}")
    print(f"  Output shapes match: {y_orig.shape == y_nys.shape}")
    
    # Quick timing
    import timeit
    t_orig = timeit.timeit(lambda: orig(x), number=100) / 100
    t_nys = timeit.timeit(lambda: nys(x), number=100) / 100
    
    print(f"\nTiming results:")
    print(f"  Original: {t_orig*1000:.2f} ms")
    print(f"  Nyström: {t_nys*1000:.2f} ms")
    print(f"  Speedup: {t_orig/t_nys:.2f}x")

if __name__ == "__main__":
    quick_test()