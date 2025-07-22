import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from src.benchmarks import WandbCircuitBenchmark
from src.config import BenchmarkConfig
import wandb
from types import SimpleNamespace


def test_apply_parallel_wrapper_dp(monkeypatch):
    bench = WandbCircuitBenchmark.__new__(WandbCircuitBenchmark)
    bench.config = BenchmarkConfig(distributed="dp")
    model = nn.Linear(10, 10)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    wrapped = bench.apply_parallel_wrapper(model)
    assert isinstance(wrapped, nn.DataParallel)

def test_apply_parallel_wrapper_none():
    bench = WandbCircuitBenchmark.__new__(WandbCircuitBenchmark)
    bench.config = BenchmarkConfig(distributed="none")
    model = nn.Linear(10, 10)
    wrapped = bench.apply_parallel_wrapper(model)
    assert wrapped is model


def test_wandb_init_disabled_on_nonzero_rank(monkeypatch):
    """wandb.init should be called with mode='disabled' when rank != 0"""
    calls = {}

    def fake_init(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(log=lambda *a, **k: None)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr("wandb.init", fake_init)
    monkeypatch.setattr("wandb.Table", lambda *a, **k: SimpleNamespace())

    WandbCircuitBenchmark(BenchmarkConfig())
    assert calls.get("mode") == "disabled"


def test_wandb_log_only_rank_zero(monkeypatch):
    calls = []

    monkeypatch.setattr(wandb, "init", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(wandb, "Table", lambda *a, **k: SimpleNamespace())

    def fake_log(data, **kwargs):
        calls.append(data)

    monkeypatch.setattr(wandb, "log", fake_log)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)

    bench = WandbCircuitBenchmark(BenchmarkConfig())
    bench.rank = 1
    bench.wandb_log({"a": 1})
    assert calls == []
    bench.rank = 0
    bench.wandb_log({"b": 2})
    assert calls == [{"b": 2}]

