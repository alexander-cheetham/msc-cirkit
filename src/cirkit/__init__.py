from pathlib import Path
import sys
import importlib.util

# Append original cirkit package path to module search path for fallback
_curr = Path(__file__).resolve().parent
for p in sys.path:
    cand = Path(p) / 'cirkit'
    if cand.is_dir() and cand.resolve() != _curr:
        __path__.append(str(cand))
        break
