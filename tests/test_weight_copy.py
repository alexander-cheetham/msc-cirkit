import torch
import os
import sys
import pytest
import re
import cirkit.symbolic.functional as SF

# Add project root to path to allow imports from src and cirkit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.benchmarks import compile_symbolic
from src.circuit_types import CIRCUIT_BUILDERS
from cirkit.backend.torch.layers.inner import TorchSumLayer

# The function to be tested, defined here temporarily
def load_mnist_weights_for_one_sum(one_sum_circuit: torch.nn.Module, mnist_checkpoint_path: str, device: str):
    """
    Loads weights from a pre-trained MNIST_COMPLEX circuit checkpoint into a one_sum circuit.
    """
    print(f"--- Loading MNIST_COMPLEX weights for one_sum circuit from {mnist_checkpoint_path} ---")

    # 1. Parse n_input and n_sum from the checkpoint path
    match = re.search(r'mnist_complex_(\d+)_(\d+)_epoch10.pt', os.path.basename(mnist_checkpoint_path))
    if not match:
        raise ValueError(f"Could not parse n_input and n_sum from checkpoint path: {mnist_checkpoint_path}")
    mnist_n_input = int(match.group(1))
    mnist_n_sum = int(match.group(2))
    print(f"Inferred MNIST_COMPLEX model dimensions from checkpoint name: n_input={mnist_n_input}, n_sum={mnist_n_sum}")

    # 2. Load the state dictionary from the MNIST checkpoint
    checkpoint = torch.load(mnist_checkpoint_path, map_location=device)
    
    # 3. Create a temporary instance of the MNIST_COMPLEX circuit to hold the weights
    mnist_builder = CIRCUIT_BUILDERS["MNIST_COMPLEX"] 
    mnist_symbolic = mnist_builder(num_input_units=mnist_n_input, num_sum_units=mnist_n_sum,region_graph="quad-tree-4")
    mnist_symbolic = SF.multiply(mnist_symbolic, mnist_symbolic)  # Scale the weights by 0.5
    mnist_circuit = compile_symbolic(mnist_symbolic, device=device)
    mnist_circuit.load_state_dict(checkpoint["model_state_dict"])

    # 4. Find the source TorchSumLayer in the loaded MNIST_COMPLEX circuit
    source_sum_layer = next((m for m in mnist_circuit.modules() if isinstance(m, TorchSumLayer)), None)
    if source_sum_layer is None:
        raise ValueError("No TorchSumLayer found in the loaded MNIST_COMPLEX circuit.")

    # 5. Find the target TorchSumLayer in the one_sum circuit
    target_sum_layer = next((m for m in one_sum_circuit.modules() if isinstance(m, TorchSumLayer)), None)
    if target_sum_layer is None:
        raise ValueError("No TorchSumLayer found in the target one_sum circuit.")
    
    # 6. Access the underlying TorchParameterNode and its torch.nn.Parameter.
    if not source_sum_layer.weight.nodes:
        raise ValueError("Source TorchParameter has no nodes.")
    if not target_sum_layer.weight.nodes:
        raise ValueError("Target TorchParameter has no nodes.")

    source_node = source_sum_layer.weight.nodes[0]
    target_node = target_sum_layer.weight.nodes[0]

    source_param = next(source_node.parameters(), None)
    target_param = next(target_node.parameters(), None)

    if source_param is None:
        raise ValueError("Source TorchParameterNode has no torch.nn.Parameter.")
    if target_param is None:
        raise ValueError("Target TorchParameterNode has no torch.nn.Parameter.")

    # 7. Perform a direct data copy.
    print(f"Copying weights of shape {source_param.data.shape}")
    with torch.no_grad():
        target_param.data.copy_(source_param.data)

    print("--- Finished loading MNIST_COMPLEX weights for one_sum circuit ---")
    return one_sum_circuit

# Helper function to create a dummy checkpoint if one doesn't exist
def create_dummy_checkpoint_if_needed(path, n_input, n_sum):
    if not os.path.exists(path):
        print(f"Creating dummy checkpoint at {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        builder = CIRCUIT_BUILDERS["MNIST_COMPLEX"]
        symbolic = builder(num_input_units=n_input, num_sum_units=n_sum)
        circuit = compile_symbolic(symbolic, device='cpu')
        for param in circuit.parameters():
            param.data.uniform_()
        torch.save({"model_state_dict": circuit.state_dict()}, path)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires a GPU for full compatibility, but will run on CPU if not available.")
def test_one_sum_weight_copying():
    """
    Tests that weights are correctly copied from an MNIST_COMPLEX checkpoint
    to a one_sum circuit.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_input = 16
    n_sum = 16

    checkpoint_dir = "./model_cache/checkpoints/"
    checkpoint_path = os.path.join(checkpoint_dir, f"mnist_complex_{n_input}_{n_sum}_epoch10.pt")
    create_dummy_checkpoint_if_needed(checkpoint_path, n_input, n_sum)

    one_sum_builder = CIRCUIT_BUILDERS["one_sum"]
    one_sum_symbolic = one_sum_builder(num_input_units=n_input, num_sum_units=n_sum)
    one_sum_circuit = compile_symbolic(one_sum_symbolic, device=device)

    modified_one_sum_circuit = load_mnist_weights_for_one_sum(one_sum_circuit, checkpoint_path, device)

    source_checkpoint = torch.load(checkpoint_path, map_location=device)
    source_builder = CIRCUIT_BUILDERS["MNIST_COMPLEX"]
    source_symbolic = source_builder(num_input_units=n_input, num_sum_units=n_sum,region_graph="quad-tree-4")
    source_symbolic = SF.multiply(source_symbolic, source_symbolic)
    source_circuit = compile_symbolic(source_symbolic, device=device)
    
    source_circuit.load_state_dict(source_checkpoint["model_state_dict"])

    source_layer = next(m for m in source_circuit.modules() if isinstance(m, TorchSumLayer))
    target_layer = next(m for m in modified_one_sum_circuit.modules() if isinstance(m, TorchSumLayer))

    source_param = next(source_layer.weight.nodes[0].parameters())
    target_param = next(target_layer.weight.nodes[0].parameters())

    assert torch.allclose(source_param.data, target_param.data), "The weights were not copied correctly."
    print("Test passed: Weights are all close.")


if __name__ == "__main__":
    test_one_sum_weight_copying()