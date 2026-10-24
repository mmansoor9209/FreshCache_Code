"""
scripts/data/query_rich_features.py
Precomputes eight per-query MLP features for all queries (entity volatility
max/mean/high flag, similarity to the nearest other query, log number of URLs,
mean evidence-domain volatility, lexical Jaccard and entity overlap with the
base query) and saves them to data/query_rich_features.json.
Run: python scripts/data/query_rich_features.py
"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys; sys.path.insert(0, str(REPO_ROOT)); import json; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import math
import numpy as np
import spacy
from pathlib import Path
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

DATA = (DATA_ROOT)

# Entity type → volatility score
ENTITY_VOL = {
    "TIME":       0.90,
    "EVENT":      0.85,
    "MONEY":      0.70,
    "DATE":       0.65,
    "PERCENT":    0.55,
    "QUANTITY":   0.50,
    "CARDINAL":   0.45,
    "PRODUCT":    0.45,
    "ORG":        0.40,
    "PERSON":     0.35,
    "LAW":        0.30,
    "NORP":       0.22,
    "WORK_OF_ART":0.20,
    "FAC":        0.15,
    "GPE":        0.15,
    "ORDINAL":    0.15,
    "LOC":        0.10,
    "LANGUAGE":   0.05,
}
DEFAULT_VOL = 0.30

def entity_features(text):
    doc = nlp(text)
    scores = [ENTITY_VOL.get(e.label_, DEFAULT_VOL) for e in doc.ents]
    max_vol  = max(scores) if scores else 0.0
    mean_vol = float(np.mean(scores)) if scores else 0.0
    high_vol = 1.0 if max_vol >= 0.70 else 0.0
    entities = {e.text.lower() for e in doc.ents}
    return max_vol, mean_vol, high_vol, entities

def lexical_jaccard(a, b):
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

# ── Load queries ──────────────────────────────────────────────────────────────
print("Loading queries...")
queries = {q["query"]: q for q in
           (json.loads(l) for l in open(DATA / "queries.jsonl"))}

# ── Load paraphrase clusters to get base→paraphrase mapping ──────────────────
print("Loading paraphrase clusters...")
base_of = {}   # paraphrase_text → base_text
cluster_members = defaultdict(list)  # base_text → [paraphrase_texts]
for line in open(DATA / "paraphrase_clusters.jsonl"):
    rec = json.loads(line)
    base = rec.get("base_query") or rec.get("query")
    paraphrases = rec.get("paraphrases", [])
    for p in paraphrases:
        ptext = p if isinstance(p, str) else p.get("query", "")
        if ptext:
            base_of[ptext] = base
            cluster_members[base].append(ptext)
    cluster_members[base].append(base)
    base_of[base] = base

# ── Load sim matrix for sim_to_cached ────────────────────────────────────────
print("Loading similarity matrix...")
sim_matrix = np.load(DATA / "sim_matrix_bgem3.npy")

# Build query → row index mapping from paraphrase clusters
all_queries_ordered = []
for line in open(DATA / "paraphrase_clusters.jsonl"):
    rec = json.loads(line)
    base = rec.get("base_query") or rec.get("query")
    paraphrases = rec.get("paraphrases", [])
    all_queries_ordered.append(base)
    for p in paraphrases:
        ptext = p if isinstance(p, str) else p.get("query", "")
        if ptext:
            all_queries_ordered.append(ptext)

query_to_idx = {q: i for i, q in enumerate(all_queries_ordered)
                if i < sim_matrix.shape[0]}

# ── Load URL manifest for evidence metadata ───────────────────────────────────
print("Loading URL manifest...")
query_urls = defaultdict(list)   # query → list of domains
for line in open(DATA / "url_manifest.jsonl"):
    rec = json.loads(line)
    if rec.get("run_id") == "run_00" and rec.get("snapshot_available"):
        query_urls[rec["query"]].append(rec.get("domain", ""))

# Load domain volatility (fitted by risk model)
domain_vol_map = {}
try:
    import freshcache.risk_model as rm
    model = rm.RuleBasedRiskModel()
    model.calibrate(str(DATA / "change_log.jsonl"),
                    str(DATA / "url_manifest.jsonl"),
                    excluded_runs=["rerun_48h"])
    domain_vol_map = model._domain_vol   # dict: domain → float
except Exception as e:
    print(f"Warning: could not load domain_vol_map: {e}")

# ── Compute features for every query ─────────────────────────────────────────
print("Computing features...")
rich_features = {}

for qtext, qdata in queries.items():
    # Entity features
    max_ev, mean_ev, high_ev, ents = entity_features(qtext)

    # sim_to_cached: top-1 sim to any other query in the matrix
    idx = query_to_idx.get(qtext)
    if idx is not None:
        row = sim_matrix[idx].copy()
        row[idx] = 0.0   # exclude self
        sim_cached = float(np.max(row))
    else:
        sim_cached = 0.80   # fallback: typical paraphrase similarity

    # Evidence metadata
    domains = query_urls.get(qtext, [])
    n_urls = len(domains)
    log_n  = math.log(1 + n_urls) / math.log(6)   # normalise: 5 URLs → ~1.0
    ev_vols = [domain_vol_map.get(d, 0.1) for d in domains if d]
    mean_ev_dom = float(np.mean(ev_vols)) if ev_vols else 0.1

    # Query-query relation: use base query as the "cached" query
    base = base_of.get(qtext, qtext)
    lex_j = lexical_jaccard(qtext, base)
    _, _, _, base_ents = entity_features(base)
    ent_overlap = 1.0 if (ents & base_ents) else 0.0

    rich_features[qtext] = {
        "max_entity_vol":          round(max_ev,    4),
        "mean_entity_vol":         round(mean_ev,   4),
        "has_high_vol_entity":     high_ev,
        "sim_to_cached":           round(sim_cached, 4),
        "log_n_urls":              round(log_n,     4),
        "mean_evidence_domain_vol":round(mean_ev_dom, 4),
        "lexical_jaccard":         round(lex_j,     4),
        "entity_overlap":          ent_overlap,
    }

out = DATA / "query_rich_features.json"
json.dump(rich_features, open(out, "w"), indent=2)
print(f"Done. Saved {len(rich_features)} queries → {out}")