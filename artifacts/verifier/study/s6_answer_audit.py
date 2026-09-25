#!/usr/bin/env python3
"""
s6_answer_audit.py — held-out 400-request answer audit: FreshCache vs
FreshCache-L2Verify(gamma frozen). Also judges actual-artifact evidence
sufficiency for both arms' served evidence.

Reuses the existing audit's fresh and FreshCache answers and judgments byte-for-
byte; generates only contexts that L2Verify actually changed. Protocol imported
from the existing audit (FORCED prompt, Llama-3.2-3B, greedy, 80 tokens, 1024
truncation; judges Llama-3.1-8B + Qwen2.5-7B, both must agree).

Stages: build | gen | judge --judge X | suff --judge X | analyze
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, pathlib, pickle, re, sys
from fractions import Fraction
import numpy as np

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in ("", "v13_corrected", "v14_baselines", "v9", "v16_exp12"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "mixed_age_full_policy_audit"))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)
OUT = HERE / "out"
A2 = ROOT / "validation" / "heldout_baseline_tuning" / "answer_audit_k1_16"
GEN = "meta-llama/Llama-3.2-3B-Instruct"
JUDGES = {"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
ABST = ("i don't know", "i do not know", "cannot be determined",
        "not enough information", "insufficient", "unable to determine",
        "no information")
FORCED = """You are a factual question answering assistant.
Answer the following question in one sentence using the provided context.
You must commit to a single best answer. Never reply that you do not know and
never state that the context is insufficient. If the context is incomplete,
give the most likely specific answer that the context points toward.

Question: {question}

Context:
{context}

Answer:"""
JUDGE = """You are grading a short answer against the reference answer.

QUESTION: {q}
REFERENCE ANSWER: {gold}
CANDIDATE ANSWER: {cand}

Does the candidate convey the same factual answer as the reference? Minor
wording, extra context, or added detail do not matter. Reply with exactly one
word: CORRECT or INCORRECT."""
SUFF = """You are evaluating search-result evidence for retrieval-augmented QA.

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
h = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]   # noqa


def jl(p):
    p = pathlib.Path(p)
    return ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            if p.exists() else [])


def build(_a):
    import prep2 as P2, mixed_age_v2 as ma, e2e_answer_grading as e2e
    G = json.load(open(OUT / "gamma_selection.json"))["selected_gamma"]
    rows = pickle.load(open(OUT / "test_rows.pkl", "rb"))
    FC, LV = rows["FreshCache"], rows[f"L2Verify_g{G}"]
    sample = json.load(open(A2 / "sample.json", encoding="utf-8"))["requests"]
    man = {}
    for m in (json.loads(l) for l in open(ROOT / "data/url_manifest.jsonl")):
        if m.get("run_id") == "run_00" and m["url_hash"] not in man:
            man[m["url_hash"]] = {"title": m.get("title", "") or "",
                                  "snippet": m.get("snippet", "") or ""}

    def evblock(ev):
        lines = []
        for uh, tv in ev:
            mt = man.get(uh, {})
            b = re.sub(r"\s+", " ", (e2e.load_snapshot_text(uh, ma.version_at(tv))
                                     or "")).strip()[:2000]
            blk = f"{len(lines)+1}. {mt.get('title','').strip()}"
            if mt.get("snippet"):
                blk += f"\n   {mt['snippet'].strip()}"
            if len(b) >= 200:
                blk += f"\n   PAGE TEXT: {b}"
            lines.append(blk)
        return "\n".join(lines) or "(no stored evidence)"

    out, tasks, suff = [], {}, {}
    for d in sample:
        q = d["query_id"]
        a, b = FC[q], LV[q]
        ev_a = [tuple(x) for x in (a.get("stored_answer_ev") if a["tier"] == "L1"
                                   else a.get("ev")) or []]
        ev_b = [tuple(x) for x in (b.get("stored_answer_ev") if b["tier"] == "L1"
                                   else b.get("ev")) or []]
        ctx_b = P2.ctx_for(tuple(ev_b))
        gq_b = b.get("gen_query", d["query"])
        key_b = ["gen", h(ctx_b), h(gq_b)]
        tasks[tuple(key_b)] = {"key": key_b, "question": gq_b, "context": ctx_b}
        same = (a["tier"] == b["tier"] and ev_a == ev_b)
        out.append({"query_id": q, "query": d["query"], "gold": d["gold"],
                    "fc_tier": a["tier"], "lv_tier": b["tier"],
                    "lv_verdict": b.get("verified"),
                    "lv_support_score": b.get("support_score"),
                    "serving_unchanged": int(same),
                    "fc_key": d["fc_key"], "fresh_key": d["fresh_key"],
                    "lv_key": key_b, "lv_ctx_sha": h(ctx_b),
                    "fc_ctx_sha": d["fc_ctx_sha"]})
        suff[(q, "fc")] = {"query_id": q, "arm": "fc", "query": d["query"],
                           "evidence": evblock(ev_a)}
        suff[(q, "lv")] = {"query_id": q, "arm": "lv", "query": d["query"],
                           "evidence": evblock(ev_b)}
    with open(OUT / "audit_sample.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    old = {tuple(t["key"]) for t in jl(A2 / "gen_tasks.jsonl")}
    need = [t for k, t in tasks.items() if k not in old]
    with open(OUT / "audit_gen_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in need:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(OUT / "audit_suff_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in suff.values():
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    ch = sum(1 for r in out if not r["serving_unchanged"])
    print(f"  400 held-out requests | L2Verify changed serving on {ch}")
    print(f"    verdicts: {dict(collections.Counter(r['lv_verdict'] for r in out))}")
    print(f"    FreshCache tiers: {dict(collections.Counter(r['fc_tier'] for r in out))}")
    print(f"    L2Verify   tiers: {dict(collections.Counter(r['lv_tier'] for r in out))}")
    print(f"  unique L2Verify contexts {len(tasks)} | new generations needed {len(need)}")
    print(f"  sufficiency tasks {len(suff)} (400 x 2 arms)")


def _load(mid):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tp = pathlib.Path("~/.cache/huggingface/token").expanduser()
    hf = tp.read_text().strip() if tp.exists() else None
    tok = AutoTokenizer.from_pretrained(mid, token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16,
                                             device_map="cuda:0", token=hf)
    m.eval(); return m, tok


def gen(a):
    import torch
    o = OUT / "audit_answers.jsonl"
    done = {tuple(r["key"]) for r in jl(o)}
    todo = [t for t in jl(OUT / "audit_gen_tasks.jsonl") if tuple(t["key"]) not in done]
    print(f"  {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return
    model, tok = _load(GEN)
    with open(o, "a", encoding="utf-8") as fh:
        for s in range(0, len(todo), a.batch):
            ch = todo[s:s + a.batch]
            pr = [FORCED.format(question=c["question"], context=c["context"]) for c in ch]
            enc = tok(pr, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)):
                t = t.strip()
                for sep in (".", "\n"):
                    if sep in t:
                        t = t[:t.index(sep) + 1].strip(); break
                fh.write(json.dumps({"key": c["key"], "answer": t or "I don't know."},
                                    ensure_ascii=False) + "\n")
            fh.flush()
    print("  generation done", flush=True)


def _ans():
    a = {tuple(r["key"]): r["answer"] for r in jl(A2 / "answers.jsonl")}
    for r in jl(OUT / "audit_answers.jsonl"):
        a[tuple(r["key"])] = r["answer"]
    return a


def judge(a):
    import torch
    ans = _ans()
    o = OUT / f"audit_judge_{a.judge}.jsonl"
    done = {tuple(r["sig"]) for r in jl(o)}
    old = {tuple(r["sig"]) for r in jl(A2 / f"judge_{a.judge}.jsonl")}
    items, seen = [], set()
    for d in jl(OUT / "audit_sample.jsonl"):
        for k in (d["lv_key"],):
            v = ans.get(tuple(k))
            if v is None:
                continue
            sig = (h(d["query"]), h(d["gold"]), h(v))
            if sig in old or sig in done or sig in seen:
                continue
            seen.add(sig)
            items.append({"sig": list(sig), "q": d["query"], "gold": d["gold"], "cand": v})
    print(f"  [{a.judge}] reusing {len(old)} existing judgments; {len(items)} new",
          flush=True)
    if not items:
        return
    model, tok = _load(JUDGES[a.judge])
    with open(o, "a", encoding="utf-8") as fh:
        for s in range(0, len(items), a.batch):
            ch = items[s:s + a.batch]
            tx = [tok.apply_chat_template(
                [{"role": "user", "content": JUDGE.format(q=c["q"], gold=c["gold"],
                                                          cand=c["cand"])}],
                tokenize=False, add_generation_prompt=True) for c in ch]
            enc = tok(tx, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)):
                u = t.upper()
                fh.write(json.dumps({"sig": c["sig"], "judge": a.judge,
                                     "label": ("INCORRECT" if "INCORRECT" in u
                                               else "CORRECT" if "CORRECT" in u
                                               else "UNPARSED")}) + "\n")
            fh.flush()
    print(f"  [{a.judge}] done", flush=True)


def suff(a):
    import torch
    o = OUT / f"audit_suff_{a.judge}.jsonl"
    done = {(r["query_id"], r["arm"]) for r in jl(o)}
    todo = [t for t in jl(OUT / "audit_suff_tasks.jsonl")
            if (t["query_id"], t["arm"]) not in done]
    print(f"  [suff {a.judge}] {len(done)} done, {len(todo)} to judge", flush=True)
    if not todo:
        return
    model, tok = _load(JUDGES[a.judge])
    with open(o, "a", encoding="utf-8") as fh:
        for i, c in enumerate(todo, 1):
            tx = tok.apply_chat_template(
                [{"role": "user", "content": SUFF.format(q=c["query"], ev=c["evidence"])}],
                tokenize=False, add_generation_prompt=True)
            enc = tok([tx], return_tensors="pt", truncation=True,
                      max_length=8192).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            u = tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip().upper()
            fh.write(json.dumps({"query_id": c["query_id"], "arm": c["arm"],
                                 "judge": a.judge,
                                 "label": ("INSUFFICIENT" if "INSUFFICIENT" in u
                                           else "UNCLEAR" if "UNCLEAR" in u
                                           else "SUFFICIENT" if "SUFFICIENT" in u
                                           else "UNPARSED")}) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(f"    [suff {a.judge}] {i}/{len(todo)}", flush=True)
    print(f"  [suff {a.judge}] done", flush=True)


def mcn(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(Fraction(1), Fraction(
        sum(math.comb(n, i) for i in range(k + 1)) * 2, 2 ** n)))


def wil(k, n, z=1.96):
    if not n:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * max(0, (c - m) / d), 3), round(100 * min(1, (c + m) / d), 3)]


def analyze(_a):
    ans = _ans()
    j = {}
    for name in JUDGES:
        j[name] = {tuple(r["sig"]): r["label"] for r in jl(A2 / f"judge_{name}.jsonl")}
        for r in jl(OUT / f"audit_judge_{name}.jsonl"):
            j[name][tuple(r["sig"])] = r["label"]
    S = jl(OUT / "audit_sample.jsonl")

    def outc(d, key):
        v = ans.get(tuple(key))
        if v is None:
            return None
        if any(p in v.strip().lower() for p in ABST):
            return "ABSTAIN"
        sig = (h(d["query"]), h(d["gold"]), h(v))
        a, b = j["llama8b"].get(sig), j["qwen7b"].get(sig)
        if a is None or b is None:
            return None
        return "CORRECT" if (a == "CORRECT" and b == "CORRECT") else "WRONG"

    rows = []
    for d in S:
        r = {"qid": d["query_id"], "fresh": outc(d, d["fresh_key"]),
             "fc": outc(d, d["fc_key"]), "lv": outc(d, d["lv_key"]),
             "changed": 1 - d["serving_unchanged"], "verdict": d["lv_verdict"],
             "fc_tier": d["fc_tier"], "lv_tier": d["lv_tier"]}
        if any(v is None for v in (r["fresh"], r["fc"], r["lv"])):
            continue
        rows.append(r)
    N = len(rows)
    FCc = sum(1 for r in rows if r["fresh"] == "CORRECT")
    res = {"N": N, "fresh_correct": FCc}
    for a in ("fc", "lv"):
        c = collections.Counter(r[a] for r in rows)
        w = sum(1 for r in rows if r[a] == "WRONG" and r["fresh"] == "CORRECT")
        res[a] = {"correct": c["CORRECT"], "abstain": c["ABSTAIN"],
                  "accuracy_pct": round(100 * c["CORRECT"] / N, 3),
                  "accuracy_ci95": wil(c["CORRECT"], N),
                  "wai": w, "wai_pct": round(100 * w / N, 3), "wai_ci95": wil(w, N),
                  "conditional_wai_pct": round(100 * w / FCc, 3) if FCc else None,
                  "conditional_wai_ci95": wil(w, FCc)}
    wa = lambda r, x: r[x] == "WRONG" and r["fresh"] == "CORRECT"          # noqa
    res["paired_wai"] = {"both": sum(1 for r in rows if wa(r, "fc") and wa(r, "lv")),
                         "fc_only": sum(1 for r in rows if wa(r, "fc") and not wa(r, "lv")),
                         "lv_only": sum(1 for r in rows if wa(r, "lv") and not wa(r, "fc"))}
    res["paired_wai"]["exact_mcnemar_p"] = mcn(res["paired_wai"]["fc_only"],
                                               res["paired_wai"]["lv_only"])
    cw = sum(1 for r in rows if r["fc"] == "CORRECT" and r["lv"] != "CORRECT")
    wc = sum(1 for r in rows if r["fc"] != "CORRECT" and r["lv"] == "CORRECT")
    res["paired_correctness"] = {"fc_only": cw, "lv_only": wc,
                                 "exact_mcnemar_p": mcn(cw, wc)}
    res["serving_changed"] = sum(r["changed"] for r in rows)
    # sufficiency
    sl = {(r["query_id"], r["arm"]): r["label"] for r in jl(OUT / "audit_suff_llama8b.jsonl")}
    sq = {(r["query_id"], r["arm"]): r["label"] for r in jl(OUT / "audit_suff_qwen7b.jsonl")}
    suffres = {}
    for arm in ("fc", "lv"):
        lab = []
        for r in rows:
            a, b = sl.get((r["qid"], arm)), sq.get((r["qid"], arm))
            lab.append(a if a == b else "JUDGE_DISAGREEMENT")
        c = collections.Counter(lab)
        agreed = c["SUFFICIENT"] + c["INSUFFICIENT"]
        suffres[arm] = {"n": len(lab), "SUFFICIENT": c["SUFFICIENT"],
                        "INSUFFICIENT": c["INSUFFICIENT"], "UNCLEAR": c["UNCLEAR"],
                        "JUDGE_DISAGREEMENT": c["JUDGE_DISAGREEMENT"],
                        "sufficient_pct_of_all": round(100 * c["SUFFICIENT"] / len(lab), 3),
                        "ci95_of_all": wil(c["SUFFICIENT"], len(lab)),
                        "sufficient_pct_of_agreed": round(100 * c["SUFFICIENT"] / agreed, 3) if agreed else None,
                        "ci95_of_agreed": wil(c["SUFFICIENT"], agreed)}
    # paired sufficiency
    pl = [(sl.get((r["qid"], "fc")) == sq.get((r["qid"], "fc")) and sl.get((r["qid"], "fc")),
           sl.get((r["qid"], "lv")) == sq.get((r["qid"], "lv")) and sl.get((r["qid"], "lv")))
          for r in rows]
    a_only = sum(1 for a, b in pl if a == "SUFFICIENT" and b != "SUFFICIENT")
    b_only = sum(1 for a, b in pl if b == "SUFFICIENT" and a != "SUFFICIENT")
    suffres["paired"] = {"fc_sufficient_only": a_only, "lv_sufficient_only": b_only,
                         "exact_mcnemar_p": mcn(a_only, b_only)}
    res["sufficiency"] = suffres
    json.dump(res, open(OUT / "audit_results.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "gen", "judge", "suff", "analyze"])
    ap.add_argument("--judge", choices=list(JUDGES)); ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    {"build": build, "gen": gen, "judge": judge, "suff": suff,
     "analyze": analyze}[a.stage](a)
