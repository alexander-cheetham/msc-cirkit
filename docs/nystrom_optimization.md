# Nyström optimization integration

This repository extends the default CirKit torch backend with a new layer shatter rule that replaces suitable dense Sum layers with `NystromSumLayer`. Below is a high level summary of the key modifications and how they interact.

## Changes in `optimization/layers.py`

- **`NystromPattern`** – new `LayerOptPattern` subclass matching a single dense `TorchSumLayer` with arity `1` and a Kronecker structured weight. The `match` method simply returns `True`, so every layer that fits the structural constraints is eligible for replacement.
- **`apply_nystrom_sum`** – new apply function that constructs a `NystromSumLayer` directly from the matched dense layer. The rank is chosen as `min(num_input_units, num_output_units)` for simplicity.
- **Default rule map** – `NystromPattern` is registered in `DEFAULT_LAYER_SHATTER_OPT_RULES` before the existing `DenseKroneckerPattern` rule. This ensures the Nyström rule is tried first and does not interfere with the remaining shatter or fuse rules.

## Compiler updates

- **`TorchCompiler.compile`** now accepts an optional `nystrom_rank` argument. When this value is not ``None`` the Nyström rule is enabled for the top level circuit.
- **`compile_pipeline`** forwards the rank only to the main circuit so the rule fires at most once.
- **`is_nystrom_enabled`** checks whether a rank is set.
- **`_post_process_circuit`** passes the boolean result to `_optimize_circuit` so the optimizer knows whether to include the Nyström rule.
- **`_optimize_circuit`** disables the flag after the shatter step if the rule matched once, preventing repeated replacement passes.
- **`_optimize_layers`** filters the pattern list based on the flag. When active and the shatter pass finds no matches, a ``ValueError`` is raised which allows tests to assert failure conditions.
- When the optimization is active the optimizer skips other layer and parameter
  rules until after the replacement, avoiding interference from additional
  optimisations.

These changes are wired through `PipelineContext.compile` and the module level `compile` helper. Nyström optimization is enabled by providing a `nystrom_rank` value when compiling; otherwise the circuit is compiled normally.

## Test overview

`tests/test_nystrom_optimization.py` exercises three scenarios:

1. **Happy path** – constructs a small two-layer circuit, squares it and compiles with `nystrom_rank` set. The resulting compiled model is checked to contain a `NystromSumLayer` with built attributes `U` and `V`.
2. **No match** – compiling a single dense layer with a rank specified raises `ValueError`, proving that optimization only proceeds when the pattern actually matches.
3. **Disabled** – compiling the squared circuit without specifying a rank leaves the graph unchanged and no Nyström layers are produced.

Since the new rule is inserted alongside existing ones but only activated via the flag, standard compilation and other optimization rules continue to behave exactly as before.
