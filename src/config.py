"""Configuration classes for benchmarking experiments."""

import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Model architecture
    input_units: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    sum_units: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    ranks: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 50])
    
    # Training settings
    batch_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    num_warmup: int = 10
    num_trials: int = 100
    
    # Hardware
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experiment metadata for wandb
    project_name: str = "kronecker-vs-nystrom"
    experiment_name: str = "benchmark"
    tags: List[str] = field(default_factory=lambda: ["benchmark", "nystrom", "kronecker"])
    notes: str = "Comparing Kronecker product vs Nyström approximation in squared circuits"