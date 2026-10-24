#!/usr/bin/env python3
"""
quality_selection_gold_verification.py — gold verification for the validation-sample requests: paraphrase
fidelity (A) and timestamp-aware validity (C).

Prompts, evidence caps, decoding and the 2-of-3 tie-break are imported/copied
verbatim from outputs/archived_runs/gold_verification/expanded_evidence_3b.

Stages: build | fidelity --judge X | validity --judge X | tiebreak | combine
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import argparse, csv, json, os, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import Counter, defaultdict

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
HERE = ((OUTPUT_ROOT / "quality_selection"))
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
RV = (OUTPUT_ROOT / "gold_verification/timestamp_validity")
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "freshcache/simulation", "scripts/evaluation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/evaluation/answer_audits"))
os.chdir(ROOT)
import timestamp_validity_judging as RJ                                          # noqa: E402

SNAP = (DATA_ROOT / "snapshots")
MAXCHARS_TOTAL, MAXCHARS_PAGE, MINCHARS = 12000, 5000, 0
JUDGES = RJ.JUDGES
FID_SYS = (
    "You are evaluating a semantic cache for a question-answering system. "
    "You will be given two questions. Decide whether a single correct answer "
    "would satisfy BOTH questions.\n\nRules:\n"
    "- SAME       if both ask for the same fact about the same entity, however "
    "worded or in whatever language.\n"
    "- DIFFERENT  if they ask about different entities, different attributes, "
    "different time references, or different events; or if Question B is "
    "garbled or too vague to ask for that fact.\n"
    "- UNCLEAR    only if Question B cannot be interpreted well enough to decide."
    "\n\nReply with exactly one word: SAME, DIFFERENT, or UNCLEAR.")


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def build(_a):
    import tiered_cache_model as exp
    import page_version_replay as ma
    S = jl(HERE / "q_sample.jsonl")
    para = jl((DATA_ROOT / "paraphrase_clusters.jsonl"))
    base_of = {p["query_id"]: p.get("base_query_id") for p in para}
    Q = {q["query_id"]: q for q in jl((DATA_ROOT / "queries.jsonl"))}
    L = lambda p: exp.load_jsonl(pathlib.Path(p))                       # noqa
    recs = exp.build_query_records(L(exp.QUERIES_FILE), L(exp.MANIFEST_FILE),
                                   L(exp.PARAPHRASE_FILE))
    urls_of = {r["query_id"]: [u["url_hash"] for u in r["urls"]] for r in recs}
    pairs, units = [], {}
    for d in S:
        qid = d["query_id"]; b = base_of.get(qid) or qid
        bq = (Q.get(b) or {}).get("query", d["query"])
        if qid in base_of:
            pairs.append({"query_id": qid, "base_query_id": b,
                          "base_query": bq, "paraphrase": d["query"]})
        rd = ma.version_at(d["t"])
        units.setdefault((b, rd), {"base_query_id": b, "base_query": bq,
                                   "snapshot_round": rd, "gold": d["gold"],
                                   "aliases": d["aliases"],
                                   "gold_source": d["gold_source"],
                                   "gold_year": d["gold_year"],
                                   "requests": []})["requests"].append(qid)
    tasks = []
    for (b, rd), u in units.items():
        uh = []
        for m in u["requests"]:
            for x in urls_of.get(m, []):
                if x not in uh:
                    uh.append(x)
        parts = []
        for x in uh:
            p = SNAP / rd / f"{x}.json"
            if not p.exists():
                continue
            t = re.sub(r"\s+", " ", (json.load(open(p, encoding="utf-8"))
                                     .get("extracted_text") or "")).strip()[:MAXCHARS_PAGE]
            if len(t) > MINCHARS:
                parts.append(t)
        keep, tot = [], 0
        for t in parts:
            room = MAXCHARS_TOTAL - tot
            if room <= 0:
                break
            keep.append(t[:room]); tot += min(len(t), room)
        qt = RJ_qtype(u["base_query"])
        tasks.append({"unit_id": f"{b}@{rd}", **u,
                      "request_ids": "|".join(u["requests"]),
                      "gold_display": u["gold"],
                      "question_type": qt,
                      "mode": "AS_STATED" if qt in ("DATED", "HISTORICAL") else "CURRENT",
                      "evidence": "\n\n".join(keep), "n_pages_kept": len(keep),
                      "evidence_chars": tot, "has_evidence": int(bool(keep))})
    for f, data in ((HERE / "q_fid_pairs.jsonl", pairs),
                    (HERE / "q_val_tasks.jsonl", tasks)):
        with open(f, "w", encoding="utf-8") as fh:
            for x in data:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"  paraphrase pairs: {len(pairs)}  base-query requests: {len(S)-len(pairs)}")
    print(f"  validity units: {len(tasks)}  modes: "
          f"{dict(Counter(t['mode'] for t in tasks))}")
    print(f"  no-evidence units: {sum(1 for t in tasks if not t['has_evidence'])}")


RECENCY = re.compile(r"\b(current|currently|now|latest|most recent|recent|recently|"
                     r"today|this year|as of now|newest|present|ongoing|upcoming|"
                     r"next|still)\b", re.I)
YEAR = re.compile(r"\b(1[89]\d\d|20[0-2]\d)\b")
HIST = re.compile(r"\b(was|were|did|first|originally|founded|born|died|invented|"
                  r"discovered|premiered|established|used to)\b", re.I)


def RJ_qtype(q):
    r, y = bool(RECENCY.search(q)), bool(YEAR.search(q))
    if y and not r:
        return "DATED"
    if y and r:
        return "DATED_CURRENT"
    if r:
        return "CURRENT_SEEKING"
    return "HISTORICAL" if HIST.search(q) else "RECURRING_UNMARKED"


def load(mid):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tp.read_text().strip() if tp.exists() else None
    tok = AutoTokenizer.from_pretrained(mid, token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16,
                                             device_map="cuda:0", token=hf)
    m.eval()
    return m, tok


def gen1(model, tok, msgs, mx=8, ctx=8192):
    import torch
    tx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([tx], return_tensors="pt", truncation=True, max_length=ctx).to(model.device)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=mx, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    return tok.batch_decode(o[:, enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0].strip()


def fidelity(a):
    out = HERE / f"q_fid_{a.judge}.jsonl"
    done = {r["query_id"] for r in jl(out)}
    todo = [p for p in jl(HERE / "q_fid_pairs.jsonl") if p["query_id"] not in done]
    print(f"  [fid {a.judge}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    model, tok = load(JUDGES[a.judge])
    with open(out, "a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            raw = gen1(model, tok,
                       [{"role": "system", "content": FID_SYS},
                        {"role": "user", "content":
                         f"Question A: {p['base_query']}\nQuestion B: {p['paraphrase']}"}],
                       mx=6, ctx=2048)
            u = raw.upper()
            lab = ("DIFFERENT" if "DIFFERENT" in u else
                   "UNCLEAR" if "UNCLEAR" in u else
                   "SAME" if "SAME" in u else "UNPARSED")
            fh.write(json.dumps({"query_id": p["query_id"], "judge": a.judge,
                                 "label": lab, "raw": raw}) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(f"    [fid {a.judge}] {i}/{len(todo)}", flush=True)
    print(f"  [fid {a.judge}] done", flush=True)


def validity(a):
    out = HERE / f"q_val_{a.judge}.jsonl"
    done = {r["unit_id"] for r in jl(out)}
    tasks = jl(HERE / "q_val_tasks.jsonl")
    if a.only_disagreements:
        ids = {r["unit_id"] for r in jl(HERE / "q_val_tiebreak.jsonl")}
        tasks = [t for t in tasks if t["unit_id"] in ids]
    todo = [t for t in tasks if t["has_evidence"] and t["unit_id"] not in done]
    print(f"  [val {a.judge}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    model, tok = load(JUDGES[a.judge])
    with open(out, "a", encoding="utf-8") as fh:
        for i, t in enumerate(todo, 1):
            g = t["gold_display"]
            alt = [x for x in t["aliases"] if x.strip().lower() != g.strip().lower()]
            if alt:
                g += "  (also accepted: " + "; ".join(alt) + ")"
            sysmsg = RJ.SYS_AS_STATED if t["mode"] == "AS_STATED" else RJ.SYS_CURRENT
            user = RJ.USER.format(q=t["base_query"], gold=g,
                                  year=t["gold_year"] or "unknown",
                                  when=RJ.ROUND_LABEL.get(t["snapshot_round"],
                                                          t["snapshot_round"]),
                                  ev=t["evidence"])
            msgs = ([{"role": "user", "content": sysmsg + "\n\n" + user}]
                    if a.judge == "mistral7b" else
                    [{"role": "system", "content": sysmsg},
                     {"role": "user", "content": user}])
            raw = gen1(model, tok, msgs, mx=8, ctx=8192)
            fh.write(json.dumps({"unit_id": t["unit_id"], "judge": a.judge,
                                 "mode": t["mode"],
                                 "status": RJ.parse(raw, t["mode"]), "raw": raw},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(f"    [val {a.judge}] {i}/{len(todo)}", flush=True)
    print(f"  [val {a.judge}] done", flush=True)


def tiebreak(_a):
    L = {r["unit_id"]: r["status"] for r in jl(HERE / "q_val_llama8b.jsonl")}
    Q = {r["unit_id"]: r["status"] for r in jl(HERE / "q_val_qwen7b.jsonl")}
    dis = [u for u in L if u in Q and L[u] != Q[u]]
    with open(HERE / "q_val_tiebreak.jsonl", "w", encoding="utf-8") as f:
        for u in dis:
            f.write(json.dumps({"unit_id": u, "llama8b": L[u], "qwen7b": Q[u]}) + "\n")
    print(f"  validity disagreements needing a third judge: {len(dis)} of {len(L)}")


def combine(_a):
    S = jl(HERE / "q_sample.jsonl")
    FL = {r["query_id"]: r["label"] for r in jl(HERE / "q_fid_llama8b.jsonl")}
    FQ = {r["query_id"]: r["label"] for r in jl(HERE / "q_fid_qwen7b.jsonl")}
    VL = {r["unit_id"]: r["status"] for r in jl(HERE / "q_val_llama8b.jsonl")}
    VQ = {r["unit_id"]: r["status"] for r in jl(HERE / "q_val_qwen7b.jsonl")}
    VM = {r["unit_id"]: r["status"] for r in jl(HERE / "q_val_mistral7b.jsonl")}
    unit_of = {}
    for t in jl(HERE / "q_val_tasks.jsonl"):
        for q in t["request_ids"].split("|"):
            unit_of[q] = t["unit_id"]
    rows = []
    for d in S:
        q = d["query_id"]
        if q in FL:
            fl, fq = FL[q], FQ.get(q)
            fid_lab = fl if fl == fq else "JUDGE_DISAGREEMENT"
        else:
            fid_lab = "BASE_QUERY"
        u = unit_of.get(q)
        l, w = VL.get(u), VQ.get(u)
        if l is None or w is None:
            maj = "NO_LABEL"
        elif l == w:
            maj = l
        else:
            m = VM.get(u)
            maj = m if m in (l, w) else "NO_MAJORITY"
        rows.append({"query_id": q, "unit_id": u or "", "fidelity": fid_lab,
                     "validity_llama8b": l or "", "validity_qwen7b": w or "",
                     "validity_mistral7b": VM.get(u, ""),
                     "validity_majority": maj,
                     "evidence_valid": int(fid_lab in ("SAME", "BASE_QUERY")
                                           and maj == "VALID_AT_T")})
    with open(HERE / "q_verification.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader(); wtr.writerows(rows)
    print(f"  fidelity: {dict(Counter(r['fidelity'] for r in rows))}")
    print(f"  validity(majority): {dict(Counter(r['validity_majority'] for r in rows))}")
    print(f"  EVIDENCE-VALID validation requests: "
          f"{sum(r['evidence_valid'] for r in rows)} of {len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "fidelity", "validity",
                                      "tiebreak", "combine"])
    ap.add_argument("--judge", choices=list(JUDGES))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only-disagreements", action="store_true")
    a = ap.parse_args()
    {"build": build, "fidelity": fidelity, "validity": validity,
     "tiebreak": tiebreak, "combine": combine}[a.stage](a)
