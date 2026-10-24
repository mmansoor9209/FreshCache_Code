"""Shared labeller for the live-stream runs: the frozen two-judge rule (HELD abstention patterns first; Llama-3.1-8B and Qwen2.5-7B both CORRECT)
over every earlier judge pool (earlier runs, optional read-only pools, and the frozen project pools via Held) plus this folder's outputs."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "cross_workload")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams/recorded_stream_controls")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("cross_workload"); import composition_study_helpers as X   # noqa: E402
POOLS = [O.parent / d / "gpu/out" for d in ("composition_checks", "write_controls", "phi4_judge", "common_sample", "natural_workload")] + [O / "gpu/out"]
class Lab:
    def __init__(self):
        self.hl = X.Held(); self.J = {"llama8b": {}, "qwen7b": {}}
        for d in POOLS:
            for j in self.J:
                for f in sorted(d.glob(f"judge_{j}*.jsonl")):
                    for r in X.jl(f): self.J[j].setdefault(tuple(r["sig"]), r["label"])
    def label_answer(self, q, gold, a):
        if a is None: return None, None, None
        if any(x in a.strip().lower() for x in X.HELD): return "ABSTAIN", a, None
        s = (X.h16(q), X.h16(gold), X.h16(a)); l = []
        for j in ("llama8b", "qwen7b"):
            v = self.hl.j[j].get(s); v = v if v is not None else self.hl.GJ[j].get(s); l.append(v if v is not None else self.J[j].get(s))
        if None in l: return None, a, s
        return ("CORRECT" if l == ["CORRECT", "CORRECT"] else "WRONG"), a, s
