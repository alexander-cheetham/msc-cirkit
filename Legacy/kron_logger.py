# kron_logger.py  (unified patch for CP + Tucker)
import logging, sys, inspect
from datetime import datetime
import importlib
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Locate candidate product layers wherever CirKit puts them ───────────
layer_modules = [
    "cirkit.backend.torch.layers.inner",      # TorchKroneckerLayer
    "cirkit.backend.torch.layers.optimized",  # TorchTuckerLayer (and friends)
]

targets = []

for modname in layer_modules:
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError:
        continue            # CirKit not imported yet, or optional module absent
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if name in {"TorchKroneckerLayer", "TorchTuckerLayer"}:
            targets.append(cls)

if not targets:
    logging.warning("Kron-logger: no product layers found at import-time; "
                    "they’ll still be patched once the modules are imported.")
_call_counter = 0

def _make_wrapper(cls):
    orig = cls.forward
    def wrapper(self, x, *args, **kw):
        global _call_counter
        _call_counter += 1
        msg = (f"[{cls.__name__} #{_call_counter:04d}] "
               f"shape={tuple(x.shape)}  Ki={getattr(self,'num_input_units','?')} "
               f"arity={getattr(self,'arity','?')}")
        # 1) structured log
        logging.info(msg)
        # 2) plain stdout
        print(datetime.now().strftime("%H:%M:%S"), msg, flush=True)
        # 3) optional W&B scalar
        if "wandb" in sys.modules and sys.modules["wandb"].run is not None:
            import wandb
            wandb.log({"kronecker/layer_calls": _call_counter})
        return orig(self, x, *args, **kw)
    return wrapper

for cls in targets:
    cls.forward = _make_wrapper(cls)

logging.info(f"✓ Kron-logger patched {', '.join(c.__name__ for c in targets) or 'no layers (yet)'}")
