import os
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.config import BenchmarkConfig
from src.circuit_types import CIRCUIT_BUILDERS
from src.benchmarks import compile_symbolic
import cirkit.symbolic.functional as SF

CACHE_DIR = "model_cache"


def train_mnist_circuit(symbolic, device, checkpoint_dir: str, checkpoint_prefix: str):
    """Compile and train ``symbolic`` circuit on MNIST.

    Parameters
    ----------
    symbolic : Circuit
        Symbolic circuit to compile and train.
    device : str
        Device to train on.
    checkpoint_dir : str
        Directory to store epoch checkpoints.
    checkpoint_prefix : str
        Base name for checkpoint files.
    """

    circuit = compile_symbolic(symbolic, device=device)
    circuit = circuit.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (255 * x.view(-1)).long()),
    ])
    dataset = datasets.MNIST('datasets', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=256)

    num_steps = len(dataloader)

    optimizer = optim.Adam(circuit.parameters(), lr=0.01)

    num_epochs = 10
    step_idx = 0
    running_loss = 0.0
    running_samples = 0

<<<<<<< HEAD
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch_idx in range(num_epochs):
        for batch_idx, (batch, _) in enumerate(dataloader):
            batch = batch.to(device)
            print(
                f"Epoch {epoch_idx + 1}/{num_epochs}, Step {batch_idx + 1}/{num_steps}"
            )
=======
    for epoch_idx in range(num_epochs):
        for batch, _ in dataloader:
            print(f"Epoch {epoch_idx + 1}/{num_epochs}, Step {step_idx + 1}")
            batch = batch.to(device)
>>>>>>> 73dd102 (code for spectrum analysis and Experiment 1: Is it actually likely nystrom approximations are actually a good fit)
            loss = -torch.mean(circuit(batch).real)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.detach() * len(batch)
            running_samples += len(batch)
            step_idx += 1
            if step_idx % 500 == 0:
                avg = running_loss / running_samples
                print(f"Step {step_idx}: Average NLL: {avg:.3f}")
                running_loss = 0.0
                running_samples = 0

        checkpoint_path = os.path.join(
            checkpoint_dir, f"{checkpoint_prefix}_epoch{epoch_idx + 1}.pt"
        )
        torch.save(circuit.state_dict(), checkpoint_path)
        print(f"Saved checkpoint {checkpoint_path}")

    return circuit


def main():
    parser = argparse.ArgumentParser(description="Train MNIST circuits and cache them")
    parser.add_argument('--cache-dir', default=CACHE_DIR, help='Directory to store trained models')
    parser.add_argument(
        '--powers-of-two',
        action='store_true',
        help='Use powers of two for number of units and keep input_units=sum_units',
    )
    parser.add_argument(
        '--min-exp',
        type=int,
        default=5,
        help='Minimum exponent for powers of two (2**min_exp)',
    )
    parser.add_argument(
        '--max-exp',
        type=int,
        default=9,
        help='Maximum exponent for powers of two (2**max_exp)',
    )
    parser.add_argument(
        '--region-graph',
        type=str,
        default='quad-tree-4',
        help='Region graph to use for MNIST circuits',
    )
    args = parser.parse_args()

    if args.powers_of_two:
        units = [2 ** i for i in range(args.min_exp, args.max_exp + 1)]
    else:
        default_cfg = BenchmarkConfig()
        units = default_cfg.input_units

    config = BenchmarkConfig(
        circuit_structure="MNIST",
        input_units=units,
        sum_units=units,
        powers_of_two=args.powers_of_two,
        min_exp=args.min_exp if args.powers_of_two else None,
        max_exp=args.max_exp if args.powers_of_two else None,
        region_graph=args.region_graph,
    )
    builder = CIRCUIT_BUILDERS['MNIST']
    os.makedirs(args.cache_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.cache_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for n_in in config.input_units:
        for n_sum in config.sum_units:
            cache_file = os.path.join(args.cache_dir, f"mnist_{n_in}_{n_sum}.pt")
            if os.path.exists(cache_file):
                print(f"Skipping existing model {cache_file}")
                continue
            print(
                f"Training circuit with {n_in} input units and {n_sum} sum units"
            )
            symbolic = builder(
                region_graph=config.region_graph,
                num_input_units=n_in,
                num_sum_units=n_sum,
            )
            symbolic = SF.multiply(symbolic, symbolic)
<<<<<<< HEAD
            checkpoint_prefix = f"mnist_{n_in}_{n_sum}"
            circuit = train_mnist_circuit(
                symbolic,
                config.device,
                checkpoint_dir,
                checkpoint_prefix,
            )
=======
            print(f"Training circuit with {n_in} input units and {n_sum} sum units")
            circuit = train_mnist_circuit(symbolic, config.device)
>>>>>>> 73dd102 (code for spectrum analysis and Experiment 1: Is it actually likely nystrom approximations are actually a good fit)
            torch.save(circuit.state_dict(), cache_file)
            print(f"Saved {cache_file}")


if __name__ == '__main__':
    main()
