"""Assemble the FreshCache-Bench dataset provenance into the frozen release.

Ships exactly what exists. Page bodies are NOT redistributed; the longitudinal
panel carries content hashes, lengths, statuses, validators and observability
labels instead, which is what every reported temporal number is computed from.
"""
import json, csv, pathlib, hashlib, collections, sys

SRC  = pathlib.Path("<PROJECT_ROOT>")
DEST = pathlib.Path(__file__).resolve().parent.parent / "artifacts/release/bundle/data"
DEST.mkdir(parents=True, exist_ok=True)

def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---- base questions + retention ------------------------------------------
base = jl(SRC/"data/queries.jsonl")
clus = jl(SRC/"data/paraphrase_clusters.jsonl")
kept = {c["base_query_id"] for c in clus}
for q in base:
    q["retained"] = q["query_id"] in kept
n_kept = sum(q["retained"] for q in base)
print(f"base questions {len(base)}, retained {n_kept}, excluded {len(base)-n_kept}")

with open(DEST/"queries_base.jsonl","w",encoding="utf-8") as f:
    for q in base: f.write(json.dumps(q,ensure_ascii=False)+"\n")
with open(DEST/"paraphrase_clusters.jsonl","w",encoding="utf-8") as f:
    for c in clus: f.write(json.dumps(c,ensure_ascii=False)+"\n")
print(f"paraphrase rows {len(clus)}, clusters {len({c['cluster_id'] for c in clus})}")

# ---- split ----------------------------------------------------------------
split = json.load(open(SRC/"validation/heldout_baseline_tuning/split.json",encoding="utf-8"))
assert not (set(split["validation_clusters"]) & set(split["test_clusters"]))
json.dump(split, open(DEST/"split.json","w",encoding="utf-8"), indent=1)
print("split: seed",split["seed"],"val",split["validation"]["requests"],
      "test",split["test"]["requests"])

# ---- longitudinal panel ---------------------------------------------------
panel_src = SRC/"v13_corrected/corrected_benchmark_manifest.jsonl"
rows = jl(panel_src)
with open(DEST/"longitudinal_panel.jsonl","w",encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("panel rows",len(rows),"urls",len({r['url_hash'] for r in rows}))

# ---- validators (etag / last-modified / caching headers) ------------------
RUNS = ["run_00","rerun_1h","rerun_12h","rerun_24h","rerun_48h","rerun_7d"]
n=0
with open(DEST/"validators.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["url_hash","run","status_code","fetched_at",
        "content_hash","content_length","etag","last_modified","content_type","cache_control"])
    for run in RUNS:
        d=SRC/"data/snapshots"/run
        if not d.is_dir(): continue
        for p in sorted(d.glob("*.json")):
            try: s=json.load(open(p,encoding="utf-8"))
            except Exception: continue
            w.writerow([s.get("url_hash") or p.stem, run, s.get("status_code"),
                s.get("fetched_at"), s.get("content_hash"), s.get("content_length"),
                s.get("etag"), s.get("last_modified"), s.get("content_type"),
                s.get("cache_control")]); n+=1
print("validator rows",n)
have_etag=0
with open(DEST/"validators.csv",encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["etag"] not in ("","None") or r["last_modified"] not in ("","None"): have_etag+=1
print("rows carrying a validator:",have_etag)
