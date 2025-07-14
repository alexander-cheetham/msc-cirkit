from pathlib import Path
import sys

_curr = Path(__file__).resolve().parent
for p in sys.path:
    cand = Path(p) / 'cirkit' / 'backend' / 'torch'
    if cand.is_dir() and cand.resolve() != _curr:
        __path__.append(str(cand))
        break
