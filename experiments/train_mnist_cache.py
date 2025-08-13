import os
import argparse
import torch
import multiprocessing
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from itertools import product

from src.config import BenchmarkConfig
from src.circuit_types import CIRCUIT_BUILDERS
from src.benchmarks import compile_symbolic
import cirkit.symbolic.functional as SF
import shutil

CACHE_DIR = "model_cache"

def find_latest_checkpoint(checkpoint_dir, prefix):
    """Return (latest_path, start_epoch) or (None, 0) if none."""
    ckpts = []
    for fn in os.listdir(checkpoint_dir):
        if fn.startswith(prefix) and fn.endswith('.pt') and 'epoch' in fn:
            # filename like prefix_epoch{n}.pt
            try:
                ep = int(fn.split('epoch')[-1].split('.pt')[0])
                ckpts.append((ep, os.path.join(checkpoint_dir, fn)))
            except ValueError:
                continue
    if not ckpts:
        return None, 0
    # pick highest epoch
    start_epoch, latest_path = max(ckpts, key=lambda x: x[0])
    return latest_path, start_epoch  # resume from next epoch

def train_and_save(args):
    n_in, config, cache_dir, checkpoint_dir = args
    prefix = f"mnist_complex_{n_in}_{n_in}"
    cache_file = os.path.join(cache_dir, f"{prefix}.pt")

    # If final model exists, skip entirely
    if os.path.exists(cache_file):
        print(f"Skipping existing model {cache_file}")
        return

    # Build symbolic circuit
    builder = CIRCUIT_BUILDERS[config.circuit_structure]
    symbolic = builder(
        region_graph=config.region_graph,
        num_input_units=n_in,
        num_sum_units=n_in,
    )
    symbolic = SF.multiply(symbolic, symbolic)

    # Train (with resume logic)
    circuit = train_mnist_circuit(
        symbolic,
        config.device,
        checkpoint_dir,
        prefix
    )

    # Save the final state_dict
    torch.save(circuit.state_dict(), cache_file)
    print(f"Saved final model {cache_file}")

    # Optional Drive sync (Colab)
    from google.colab import drive
    drive.mount('/content/drive')
    shutil.copytree("msc-cirkit/checkpoints/",
                    "/content/drive/MyDrive/checkpoints/",
                    dirs_exist_ok=True)
    shutil.copytree("msc-cirkit/model_cache/",
                    "/content/drive/MyDrive/model_cache/",
                    dirs_exist_ok=True)

def train_mnist_circuit(symbolic, device, checkpoint_dir: str, prefix: str):
    # Compile and move to device
    circuit = compile_symbolic(symbolic, device=device).to(device)

    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (255 * x.view(-1)).long()),
    ])
    dataset = datasets.MNIST('datasets', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=256)

    optimizer = optim.Adam(circuit.parameters(), lr=0.01)
    num_epochs = 10
    step_idx = 0
    running_loss = 0.0
    running_samples = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    # **Resume logic**: find latest checkpoint and load if present
    latest_ckpt, start_epoch = find_latest_checkpoint(checkpoint_dir, prefix)
    if latest_ckpt:
        checkpoint = torch.load(latest_ckpt, map_location=device)
        circuit.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move any optimizer tensors to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Resumed from {latest_ckpt}, starting at epoch {start_epoch+1}")

    # Training loop, resuming from start_epoch
    for epoch_idx in range(start_epoch, num_epochs):
        for batch_idx, (batch, _) in enumerate(dataloader, start=1):
            batch = batch.to(device)
            print(f"[{multiprocessing.current_process().name}] "
                  f"Epoch {epoch_idx+1}/{num_epochs} — Step {batch_idx}/{len(dataloader)}")
            loss = -torch.mean(circuit(batch).real)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.detach().item() * batch.size(0)
            running_samples += batch.size(0)
            step_idx += 1

            if step_idx % 500 == 0:
                avg = running_loss / running_samples
                print(f"Step {step_idx}: Avg NLL = {avg:.3f}")
                running_loss = running_samples = 0

        # checkpoint end of epoch
        ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_epoch{epoch_idx+1}.pt")
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': circuit.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint {ckpt_path}")

    return circuit

def main():
    parser = argparse.ArgumentParser(description="Train MNIST circuits in parallel")
    parser.add_argument('--cache-dir', default=CACHE_DIR, help='Directory to store trained models')
    parser.add_argument('--powers-of-two', action='store_true',
                        help='Use powers of two for number of units')
    parser.add_argument('--min-exp', type=int, default=5, help='Min exponent (2**min-exp)')
    parser.add_argument('--max-exp', type=int, default=9, help='Max exponent (2**max-exp)')
    parser.add_argument('--region-graph', type=str, default='quad-tree-4',
                        help='Region graph to use for MNIST circuits')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    # Determine unit sizes
    if args.powers_of_two:
        units = [2 ** i for i in range(args.min_exp, args.max_exp + 1)]
    else:
        units = BenchmarkConfig().input_units

    config = BenchmarkConfig(
        circuit_structure="MNIST_COMPLEX",
        input_units=units,
        sum_units=units,
        powers_of_two=args.powers_of_two,
        min_exp=args.min_exp if args.powers_of_two else None,
        max_exp=args.max_exp if args.powers_of_two else None,
        region_graph=args.region_graph,
    )
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.cache_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.cache_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Only equal (n, n) combos
    tasks = [(n, config, args.cache_dir, checkpoint_dir)
             for n in config.input_units if n in config.sum_units]

    # Spawn-mode multiprocessing for CUDA safety
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=args.workers) as pool:
        pool.map(train_and_save, tasks)


if __name__ == '__main__':
    main()
