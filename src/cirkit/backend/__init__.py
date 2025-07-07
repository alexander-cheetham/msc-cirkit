from pathlib import Path
import sys

_curr = Path(__file__).resolve()
for p in sys.path:
    cand = Path(p) / 'cirkit' / 'backend' / '__init__.py'
    if cand.exists() and cand.resolve() != _curr:
        __path__.append(str(cand.parent))
        break
