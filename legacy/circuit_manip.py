"""Legacy circuit manipulation utilities."""

import torch.nn as nn
from cirkit.backend.torch.layers.inner import TorchSumLayer
from nystromlayer import NystromSumLayer

__all__ = ["replace_sum_layers", "fix_address_book_modules"]


def replace_sum_layers(module: nn.Module, *, rank: int) -> None:
    """LEGACY: recursively replace :class:`TorchSumLayer` with
    :class:`NystromSumLayer`.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, TorchSumLayer):
            setattr(module, name, NystromSumLayer(child, rank=rank))
        else:
            replace_sum_layers(module=child, rank=rank)


def fix_address_book_modules(circuit, verbose: bool = False) -> bool:
    """LEGACY: update circuit address book after layer replacement."""
    if not hasattr(circuit, "_address_book"):
        return False

    addr_book = circuit._address_book
    nystrom_layer = None
    for name, module in circuit.named_modules():
        if isinstance(module, NystromSumLayer):
            nystrom_layer = module
            if verbose:
                print(f"Found NystromSumLayer at: {name}")
            break

    if nystrom_layer is None:
        if verbose:
            print("ERROR: No NystromSumLayer found in circuit!")
        return False

    if hasattr(addr_book, "_entry_modules"):
        for i, module in enumerate(addr_book._entry_modules):
            if verbose:
                print(
                    f"Entry {i}: {type(module).__name__ if module else 'None'}"
                )
            if isinstance(module, TorchSumLayer) and not isinstance(
                module, NystromSumLayer
            ):
                addr_book._entry_modules[i] = nystrom_layer
                if verbose:
                    print(
                        f"  After update: {type(addr_book._entry_modules[i]).__name__}"
                    )
                    print(
                        f"  Is NystromSumLayer? {isinstance(addr_book._entry_modules[i], NystromSumLayer)}"
                    )
                return True
    return False
