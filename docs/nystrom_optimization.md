# Nyström optimization integration

This repository extends the default CirKit torch backend with a new layer shatter rule that replaces suitable dense Sum layers with `NystromSumLayer`. Below is a high level summary of the key modifications and how they interact.

## Changes in `optimization/layers.py`

- **`NystromPattern`** – new `LayerOptPattern` subclass matching a single dense `TorchSumLayer` with arity `1` and a Kronecker structured weight. The `match` method simply returns `True`, so every layer that fits the structural constraints is eligible for replacement.
- **`apply_nystrom_sum`** – new apply function that constructs a `NystromSumLayer` directly from the matched dense layer. The rank is chosen as `min(num_input_units, num_output_units)` for simplicity.
- **Default rule map** – `NystromPattern` is registered in `DEFAULT_LAYER_SHATTER_OPT_RULES` before the existing `DenseKroneckerPattern` rule. This ensures the Nyström rule is tried first and does not interfere with the remaining shatter or fuse rules.

## Compiler updates

- **`TorchCompiler.compile`** now accepts a `nystrom` boolean flag. The flag is stored in `self._flags` and restored after compilation so nested compilations are handled correctly.
- **`compile_pipeline`** propagates the flag only to the main circuit. Subcircuits compile with normal optimizations so the Nyström rule only fires once on the top level.
- **`is_nystrom_enabled`** property exposes the active flag.
- **`_post_process_circuit`** forwards the flag to `_optimize_circuit` so the optimizer knows whether to include the Nyström rule.
- **`_optimize_circuit`** disables the flag after the shatter step if the rule matched once, preventing repeated replacement passes.
- **`_optimize_layers`** accepts the flag and filters the pattern list accordingly. When `nystrom` is true and the shatter pass finds no matches, a `ValueError` is raised which allows tests to assert failure conditions.
- When the Nyström flag is enabled the optimizer skips other layer and parameter
  rules until after the replacement, avoiding interference from additional
  optimisations.

These changes are wired through `PipelineContext.compile` and the module level `compile` helper so callers simply pass `nystrom=True` when desired.

## Test overview

`tests/test_nystrom_optimization.py` exercises three scenarios:

1. **Happy path** – constructs a small two-layer circuit, squares it and compiles with `nystrom=True`. The resulting compiled model is checked to contain a `NystromSumLayer` with built attributes `U` and `V`.
2. **No match** – compiling a single dense layer with the flag set raises `ValueError`, proving that optimization only proceeds when the pattern actually matches.
3. **Flag off** – compiling the squared circuit with `nystrom=False` leaves the graph unchanged and no Nyström layers are produced.

Since the new rule is inserted alongside existing ones but only activated via the flag, standard compilation and other optimization rules continue to behave exactly as before.

## Circuit builders

Utilities for constructing example circuits live in :mod:`src.circuit_types`. The
function :func:`make_random_binary_tree_circuit` accepts a ``depth`` argument and
optionally ``num_input_units`` and ``num_sum_units``. ``depth=1`` recreates the
shallow one-sum circuit used throughout the tests while larger values build
deeper CP structures. A small stub ``make_mnist_circuit`` is also provided for
future dataset-specific experiments.
