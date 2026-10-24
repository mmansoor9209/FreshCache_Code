"""Create the run-time output directories listed in configs/output_dirs.txt under OUTPUT_DIR (default outputs/) and DATA_DIR/derived.
Usage: python scripts/prepare_output_dirs.py"""
import os, pathlib, sys
if any(a in ("-h", "--help") for a in sys.argv[1:]): sys.exit(print(__doc__))
REPO = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", pathlib.Path(__file__).resolve().parents[1])).resolve()
OUT = pathlib.Path(os.environ.get("OUTPUT_DIR") or REPO / "outputs").resolve()
DATA = pathlib.Path(os.environ.get("DATA_DIR") or REPO / "data").resolve()
n = 0
for line in open(REPO / "configs/output_dirs.txt"):
    d = line.strip()
    if not d: continue
    p = OUT / d[len("outputs/"):] if d.startswith("outputs/") else DATA / d[len("data/"):] if d.startswith("data/") else None
    if p is None: continue
    p.mkdir(parents=True, exist_ok=True); n += 1
print(f"{n} directories ready under {OUT} and {DATA}")
