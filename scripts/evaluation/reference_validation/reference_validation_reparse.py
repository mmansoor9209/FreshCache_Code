"""Stage 2b: deterministic re-parse of the saved raw judge outputs (no model is re-run).
Deviates from the frozen parser, after inspection of UNPARSED/mis-parsed rows; the deviation is logged in config/deviations.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pathlib, re, collections; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
W = ((OUTPUT_ROOT / "reference_validation"))
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def ref(raw):
    u = raw.upper().replace(" ", "_")
    if re.match(r"^_*TEMPORARIL?Y_AMBIG", u): return "TEMPORALLY_AMBIGUOUS"          # misspelling seen in Mistral output
    hits = [(u.find(l), l) for l in ("VALID_AT_T", "INVALID_AT_T", "TEMPORALLY_AMBIGUOUS", "UNVERIFIABLE") if u.find(l) >= 0]
    return min(hits)[1] if hits else "UNPARSED"
def ans(raw):
    u = raw.upper()
    u = re.sub(r"\bINCORRECT\b", "WRONG", u)                                         # 'INCORRECT' must not match 'CORRECT'
    u = re.sub(r"^(WROONG|WONG)\b", "WRONG", u)                                     # misspellings seen in Mistral output
    hits = [(u.find(l), l) for l in ("CORRECT", "WRONG", "ABSTAIN", "UNRESOLVABLE") if u.find(l) >= 0]
    return min(hits)[1] if hits else "UNPARSED"
changes = {}
for f in sorted((W/"outputs/raw").glob("*.jsonl")):
    stage = f.stem.split("_")[0]; fn = ref if stage == "reference" else ans; rows = jl(f); c = collections.Counter()
    with open(W/"outputs/parsed"/f.name, "w", encoding="utf-8") as o:
        for r in rows:
            new = fn(r["raw"]); c[f"{r['label']}->{new}"] += int(new != r["label"])
            o.write(json.dumps({**r, "label_frozen_parser": r["label"], "label": new}, ensure_ascii=False) + "\n")
    changes[f.name] = {k: v for k, v in c.items() if v}
json.dump({"deviation": "parser reported after inspection of raw outputs; raw outputs unchanged; no model re-run",
           "rules": ["'INCORRECT' is read as WRONG (frozen parser matched the substring CORRECT)",
                     "Mistral misspellings TEMPORARILY_AMBIGUO* -> TEMPORALLY_AMBIGUOUS; WROONG/WONG -> WRONG"],
           "label_changes": changes}, open(W/"config/deviations.json", "w"), indent=1)
print(json.dumps(changes, indent=1))
