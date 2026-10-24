"""Dataset-size checks against configs/data/dataset_sizes.yaml.

check(name, actual) returns True when `actual` equals the configured size of the full benchmark. When DATA_DIR holds the
bundled sample (DATA_DIR/sample_summary.json exists, as in data/sample and in a re-fetched copy of it) the check is skipped:
it prints a message and returns True, so stages run on the sample instead of failing on its smaller size.
Usage from a script:  DS = <one-line loader>;  assert DS.check("primary_requests", len(stream))"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib
import yaml
REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve()
SIZES = yaml.safe_load(open(REPO_ROOT / "configs/data/dataset_sizes.yaml", encoding="utf-8"))["sizes"]
SAMPLE = (DATA_ROOT / "sample_summary.json").exists()


def expected(name):
    return SIZES[name]


def check(name, actual):
    """True if `actual` is the configured full-benchmark size; on the sample, skipped with a message (True)."""
    if SAMPLE:
        print(f"  size check skipped (DATA_DIR is the bundled sample): {name} = {actual}, full benchmark {SIZES[name]}", flush=True)
        return True
    if actual != SIZES[name]:
        print(f"  size check FAILED: {name} = {actual}, configs/data/dataset_sizes.yaml expects {SIZES[name]}", flush=True)
        return False
    return True
