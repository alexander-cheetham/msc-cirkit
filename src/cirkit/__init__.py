from pathlib import Path
import sys
import importlib.util

# Append original cirkit package path to module search path for fallback
_curr = Path(__file__).resolve()
for p in sys.path:
    cand = Path(p) / 'cirkit' / '__init__.py'
    if cand.exists() and cand.resolve() != _curr:
        __path__.append(str(cand.parent))
        break
