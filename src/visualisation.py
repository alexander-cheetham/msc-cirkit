"""Visualization functions for benchmark results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_benchmark_results(df: pd.DataFrame, save_path: str = None):
    """Create comprehensive visualization of benchmark results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Kronecker vs Nyström: Benchmark Results', fontsize=16)
    
    # Implementation from previous code
    # ... (all the plotting code)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_summary_report(df: pd.DataFrame) -> Dict:
    """Generate summary statistics from benchmark results."""
    summary = {
        "avg_speedup": df['actual_speedup'].mean(),
        "max_speedup": df['actual_speedup'].max(),
        "avg_memory_reduction": df['memory_reduction'].mean(),
        "avg_error": df['rel_error'].mean(),
    }
    
    # Find optimal configurations
    good_accuracy = df[df['rel_error'] < 0.01]
    if len(good_accuracy) > 0:
        best = good_accuracy.loc[good_accuracy['actual_speedup'].idxmax()]
        summary['best_config'] = {
            'n_input': best['n_input'],
            'n_sum': best['n_sum'],
            'rank': best['rank'],
            'speedup': best['actual_speedup']
        }
    
    return summary