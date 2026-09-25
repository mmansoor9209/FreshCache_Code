#!/usr/bin/env python3
"""
s3b_label.py — build and judge the VALIDATION sufficiency labels used to select
gamma. Stages: build | judge --judge X | combine

Population: L2 hits on the validation cluster split, EXCLUDING every request in
the existing 300-case audit. Evidence: the actual served URL list at the page
versions L3 actually served (title + snippet + page text, served order).
Prompt and judges reuse the existing evidence-sufficiency protocol.
"""
from __future__ import annotations
import argparse, collections, json, os, pathlib, random, re, sys
import numpy as np

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)
OUT = HERE / "out"
TARGET, SEED = 600, 42
BODY_CAP, MINBODY = 2000, 200
JUDGES = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
PROMPT = """You are evaluating search-result evidence for retrieval-augmented QA.

Incoming query:
{q}

Search results served from the cache for this request:
{ev}

Determine whether these search results contain sufficient information or
credible evidence that could support answering the incoming query.

Return exactly one label:

SUFFICIENT
INSUFFICIENT
UNCLEAR

SUFFICIENT means at least one or more results provide evidence directly
relevant enough to support answering the query.
INSUFFICIENT means the results are largely unrelated or do not provide
useful evidence for answering the query.
UNCLEAR means relevance cannot be determined reliably from the available
evidence."""


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def build(_a):
    import experiment as exp, schedules as sc, engine_all as ea
    import mixed_age_v2 as ma, e2e_answer_grading as e2e
    import s2_engine as E
    assert (OUT / "prereg_gamma.json").exists(), "run s3a_prereg.py first"
    L = lambda p: exp.load_jsonl(pathlib.Path(p))                      # noqa
    recs = exp.build_query_records(L(exp.QUERIES_FILE), L(exp.MANIFEST_FILE),
                                   L(exp.PARAPHRASE_FILE))
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(recs)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(recs)
    full = sc.build_stream(recs, "zipf_uniform", 42)
    split = json.load(open(ROOT / "validation/heldout_baseline_tuning/split.json"))
    VAL = set(split["validation_clusters"])
    vs = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in VAL]
    _, rows, _ = E.replay(vs, gamma=None, support=None)
    byq = {r["query_id"]: r for _, r in vs}
    import csv as _csv
    audited = {c["incoming_query_id"] for c in _csv.DictReader(
        open(ROOT / "validation/l2_independent_audit_300/selected_cases.csv"))}
    hits = [q for q, d in rows.items() if d["tier"] == "L2"]
    elig = [q for q in hits if q not in audited]
    print(f"  validation L2 hits {len(hits):,} | excluded (in the 300 audit) "
          f"{len(hits) - len(elig)} | eligible {len(elig):,}")
    man = {}
    for m in (json.loads(l) for l in open(ROOT / "data/url_manifest.jsonl")):
        if m.get("run_id") == "run_00" and m["url_hash"] not in man:
            man[m["url_hash"]] = {"title": m.get("title", "") or "",
                                  "snippet": m.get("snippet", "") or ""}

    def band(s):
        for a, b in ((0.75, .80), (.80, .85), (.85, .90), (.90, .95), (.95, 1.01)):
            if a <= s < b:
                return f"{a:.2f}-{b:.2f}"
        return "other"
    strata = collections.defaultdict(list)
    for q in sorted(elig):
        strata[(byq[q]["freshness_class"], band(rows[q]["similarity"]))].append(q)
    rng = random.Random(SEED)
    keys = sorted(strata); per = max(1, TARGET // len(keys))
    samp = []
    for k in keys:
        v = sorted(strata[k]); rng.shuffle(v); samp += v[:per]
    pool = [q for k in keys for q in sorted(strata[k]) if q not in set(samp)]
    rng.shuffle(pool); samp += pool[:max(0, TARGET - len(samp))]
    samp = sorted(set(samp))[:TARGET]
    print(f"  sampled {len(samp)} over {len(keys)} strata (class x similarity band)")

    out = []
    for q in samp:
        d = rows[q]; r = byq[q]
        lines = []
        for uh, tv in d["ev"]:
            mt = man.get(uh, {})
            b = re.sub(r"\s+", " ", (e2e.load_snapshot_text(uh, ma.version_at(tv))
                                     or "")).strip()[:BODY_CAP]
            blk = f"{len(lines)+1}. {mt.get('title','').strip()}"
            if mt.get("snippet"):
                blk += f"\n   {mt['snippet'].strip()}"
            if len(b) >= MINBODY:
                blk += f"\n   PAGE TEXT: {b}"
            lines.append(blk)
        out.append({"query_id": q, "query": r["query"],
                    "freshness_class": r["freshness_class"],
                    "l2_similarity": d["similarity"],
                    "served_urls": "|".join(u for u, _ in d["ev"]),
                    "n_served": len(d["ev"]),
                    "evidence": "\n".join(lines) or "(no stored evidence)"})
    with open(OUT / "val_label_tasks.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"validation_l2_hits": len(hits), "excluded_audited": len(hits) - len(elig),
               "eligible": len(elig), "sampled": len(out), "seed": SEED,
               "strata": len(keys)},
              open(OUT / "s3b_build.json", "w"), indent=2)
    print(f"  wrote val_label_tasks.jsonl ({len(out)})")


def judge(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    o = OUT / f"val_support_{a.judge}.jsonl"
    done = {r["query_id"] for r in jl(o)}
    todo = [t for t in jl(OUT / "val_label_tasks.jsonl") if t["query_id"] not in done]
    print(f"  [{a.judge}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    tp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tp.read_text().strip() if tp.exists() else None
    tok = AutoTokenizer.from_pretrained(JUDGES[a.judge], token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGES[a.judge], dtype=torch.bfloat16, device_map="cuda:0", token=hf)
    model.eval()
    with open(o, "a", encoding="utf-8") as fh:
        for i, c in enumerate(todo, 1):
            tx = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(q=c["query"],
                                                           ev=c["evidence"])}],
                tokenize=False, add_generation_prompt=True)
            enc = tok([tx], return_tensors="pt", truncation=True,
                      max_length=8192).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            raw = tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0].strip().upper()
            lab = ("INSUFFICIENT" if "INSUFFICIENT" in raw else
                   "UNCLEAR" if "UNCLEAR" in raw else
                   "SUFFICIENT" if "SUFFICIENT" in raw else "UNPARSED")
            fh.write(json.dumps({"query_id": c["query_id"], "judge": a.judge,
                                 "label": lab}) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(f"    [{a.judge}] {i}/{len(todo)}", flush=True)
    print(f"  [{a.judge}] done", flush=True)


def combine(_a):
    import s2_engine as E
    T = {t["query_id"]: t for t in jl(OUT / "val_label_tasks.jsonl")}
    L = {r["query_id"]: r["label"] for r in jl(OUT / "val_support_llama8b.jsonl")}
    Q = {r["query_id"]: r["label"] for r in jl(OUT / "val_support_qwen7b.jsonl")}
    sup = E.Support(OUT)
    import experiment as exp, schedules as sc, engine_all as ea
    L_ = lambda p: exp.load_jsonl(pathlib.Path(p))                     # noqa
    recs = exp.build_query_records(L_(exp.QUERIES_FILE), L_(exp.MANIFEST_FILE),
                                   L_(exp.PARAPHRASE_FILE))
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(recs)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    ea.set_cluster_base(recs)
    full = sc.build_stream(recs, "zipf_uniform", 42)
    split = json.load(open(ROOT / "validation/heldout_baseline_tuning/split.json"))
    VAL = set(split["validation_clusters"])
    vs = [(t, r) for t, r in full if (r.get("cluster_id") or r["query_id"]) in VAL]
    _, rows, _ = E.replay(vs, gamma=None, support=None)
    byq = {r["query_id"]: r for _, r in vs}
    out = []
    for q, t in T.items():
        l, w = L.get(q), Q.get(q)
        lab = l if (l == w and l in ("SUFFICIENT", "INSUFFICIENT")) else \
            ("UNCLEAR_AGREED" if l == w else "JUDGE_DISAGREEMENT")
        out.append({"query_id": q, "label": lab, "llama8b": l, "qwen7b": w,
                    "support_score": sup.score(byq[q]["query"], rows[q]["ev"]),
                    "l2_similarity": t["l2_similarity"],
                    "freshness_class": t["freshness_class"],
                    "n_served": t["n_served"]})
    with open(OUT / "val_labels.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    c = collections.Counter(r["label"] for r in out)
    print(f"  labels: {dict(c)}")
    print(f"  usable (judge-agreed SUFFICIENT/INSUFFICIENT): "
          f"{c['SUFFICIENT'] + c['INSUFFICIENT']}/{len(out)}")
    ns = sum(1 for r in out if r["support_score"] is None)
    print(f"  unscoreable support (no page embedding): {ns}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "judge", "combine"])
    ap.add_argument("--judge", choices=list(JUDGES)); ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    {"build": build, "judge": judge, "combine": combine}[a.stage](a)
