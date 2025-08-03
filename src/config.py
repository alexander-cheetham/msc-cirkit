"""Configuration classes for benchmarking experiments."""

import socket
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
    use_dynamic_ranks: bool = True
    rank_percentages: List[float] = field(
        default_factory=lambda: [0.1,0.2, 0.3, 0.6,]
    )

    # Pivot selection strategy for Nyström approximation
    pivot: str = field(
        default="uniform",
        metadata={"help": "Pivot strategy for Nyström layers ('uniform' or 'l2')"},
    )

    # Optional list of Nyström sampling methods to benchmark.  If ``None``
    # only ``pivot`` is used.  This allows running multiple Nyström
    # approximations (e.g. both ``'uniform'`` and ``'l2'``) in a single
    # benchmark while retaining backward compatibility with older
    # configurations that specified a single ``pivot``.
    approximation_methods: Optional[List[str]] = None
    
    # Training settings
    batch_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    num_warmup: int = 2
    num_trials: int = 20
    
    # Hardware
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experiment metadata for wandb
    project_name: str = "kronecker-vs-nystrom"
    experiment_name: str = "benchmark"
    tags: List[str] = field(default_factory=lambda: ["benchmark", "nystrom", "kronecker"])
    notes: str = "Comparing Kronecker product vs Nyström approximation in squared circuits"
    hostname: str = field(
        default_factory=socket.gethostname,
        metadata={"help": "Hostname of the machine running the benchmark"},
    )

    # Optional power-of-two configuration
    powers_of_two: bool = False
    min_exp: Optional[int] = None
    max_exp: Optional[int] = None
    circuit_structure: str = field(
        default="one_sum",
        metadata={"help": "Type of circuit to benchmark"},
    )
    depth: int = field(
        default=3,
        metadata={"help": "Depth of the circuit"},
    )
    region_graph: Optional[str] = field(
        default=None,
        metadata={"help": "Region graph for MNIST circuits"},
    )
