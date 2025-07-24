import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
import torch
import importlib
import wandb
from types import SimpleNamespace
from torchvision import datasets
from src.config import BenchmarkConfig
from src.circuit_types import make_squarable_mnist_circuit
import cirkit.symbolic.functional as SF


def test_create_test_input_mnist_inference(monkeypatch):
    # Disable wandb functionality to avoid network calls
    monkeypatch.setattr(wandb, "require", lambda *a, **k: None)
    monkeypatch.setattr(wandb, "init", lambda *a, **k: SimpleNamespace(log=lambda *a, **k: None))

    benchmarks = importlib.import_module("src.benchmarks")
    compile_symbolic = benchmarks.compile_symbolic
    WandbCircuitBenchmark = benchmarks.WandbCircuitBenchmark

    class DummyMNIST:
        def __init__(self, *a, **k):
            self.data = torch.arange(28 * 28, dtype=torch.uint8).view(1, 28, 28)

        def __len__(self):
            return 1

    monkeypatch.setattr(datasets, "MNIST", DummyMNIST)

    config = BenchmarkConfig(circuit_structure="MNIST", region_graph="quad-tree-4", input_units=[1], sum_units=[1])
    bench = WandbCircuitBenchmark.__new__(WandbCircuitBenchmark)
    bench.config = config

    circuit = make_squarable_mnist_circuit(region_graph="quad-tree-4", num_input_units=1, num_sum_units=1)
    circuit = SF.multiply(circuit, circuit)
    compiled = compile_symbolic(circuit, device=config.device)

    test_input = bench.create_test_input(1, 28, config.device)
    assert test_input.shape == (1, 28 * 28)
    assert test_input.dtype == torch.long
    # Ensure the tensor comes from the dummy dataset rather than being random
    assert torch.equal(test_input[0], DummyMNIST().data.view(-1).long())

    output = compiled(test_input)
    assert output.shape[0] == 1
