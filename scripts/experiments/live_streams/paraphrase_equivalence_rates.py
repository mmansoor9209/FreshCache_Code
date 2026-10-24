"""Paraphrase-fidelity rates of the two equivalence judges (App. D: share of the stream paraphrases each of Llama-3.1-8B and Qwen2.5-7B
rates equivalent). Counts the labels written by paraphrase_equivalence_judging.py (outputs/common_sample/gpu/out/c_paraeq_{judge}.jsonl).
Writes outputs/common_sample/paraphrase_equivalence_rates.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import collections, json, os, pathlib; OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = OUTPUT_ROOT / "common_sample"; OUT = {}
for j in ("llama8b", "qwen7b"):
    c = collections.Counter(json.loads(l)["label"] for l in open(O / f"gpu/out/c_paraeq_{j}.jsonl")); n = sum(c.values())
    OUT[j] = {"labels": dict(c), "N": n, "SAME_pct": round(100 * c["SAME"] / n, 2)}; print(j, c)
json.dump(OUT, open(O / "paraphrase_equivalence_rates.json", "w"), indent=1); print(json.dumps(OUT, indent=1))
