"""Automated-evaluation sensitivity on the common sample (work/common.pkl; B, B-minus(mask), B-minus(elig), D, D').
Existing protocols only: third judge family Mistral-7B (frozen JUDGE prompt via scripts/evaluation/composition_study_generation_judging.py judge2), Phi-4 (stronger-judge protocol: frozen JUDGE
prompt, chat template, greedy, 6 tokens), deterministic matching check (model_adjudication_analysis.det, read-only; sensitivity only, structured references).
Reference screening: the frozen reference-validity protocol (configs/protocols/reference_validity_protocol.json, sha-checked; three families) applied to the
POOLED evidence of all five policies for the request (union of pages, not one arm). Selection effect disclosed: the pool contains B's own list.
Modes: items | phi4 --gpu G | refval --judge J --gpu G | score."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import ast, collections, datetime, hashlib, json, math, os, re, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); MODE = sys.argv[1]
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); RV = (OUTPUT_ROOT / "unseen_queries/refval")
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if pathlib.Path(p).exists() else []
PROT = json.load(open((REPO_ROOT / "configs/protocols/reference_validity_protocol.json"))); assert hashlib.sha256(((REPO_ROOT / "configs/protocols/reference_validity_protocol.json")).read_bytes()).hexdigest() == open((REPO_ROOT / "configs/protocols/reference_validity_protocol.sha256")).read().split()[0]
def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
if MODE in ("phi4", "refval"):
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--gpu", required=True); ap.add_argument("--judge"); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu; si, sn = map(int, a.shard.split("/"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if MODE == "phi4":
        K = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(open(ROOT / "scripts/evaluation/generation_judging_stage.py").read()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "JUDGE"}
        tok = AutoTokenizer.from_pretrained("microsoft/phi-4"); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
        m = AutoModelForCausalLM.from_pretrained("microsoft/phi-4", dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
        out = O / f"gpu/out/sensitivity_phi4_shard{si}.jsonl"; done = {tuple(r["sig"]) for r in jl(out)}
        todo = [x for i, x in enumerate(jl(O / "gpu/in/sensitivity_answer_items.jsonl")) if i % sn == si and tuple(x["sig"]) not in done]; fh = open(out, "a", encoding="utf-8")
        for s in range(0, len(todo), 16):
            ch = todo[s:s + 16]
            enc = tok([tok.apply_chat_template([{"role": "user", "content": K["JUDGE"].format(q=c["q"], gold=c["gold"], cand=c["cand"])}], tokenize=False, add_generation_prompt=True) for c in ch],
                      return_tensors="pt", padding=True, truncation=True, max_length=2048).to(m.device)
            with torch.no_grad(): g = m.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
            for c, t in zip(ch, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                u = t.upper(); fh.write(json.dumps({"sig": c["sig"], "raw": t, "label": "INCORRECT" if "INCORRECT" in u else "CORRECT" if "CORRECT" in u else "UNPARSED"}) + "\n")
            fh.flush()
    else:   # frozen refval reference stage (reference_validation_judging.py logic, unchanged prompt/decoding/parse)
        D = PROT["decoding"]; mid = PROT["judges"][a.judge]; LAB = PROT["reference_stage"]["labels"]
        tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda:0"); m.eval()
        def parse(t):
            u = t.upper().replace(" ", "_"); hits = [(u.find(l), l) for l in LAB if u.find(l) >= 0]
            if not hits:
                for l, alt in (("VALID_AT_T", "VALID"), ("INVALID_AT_T", "INVALID"), ("TEMPORALLY_AMBIGUOUS", "AMBIGUOUS")):
                    if re.search(rf"\b{alt}\b", t.upper()): hits.append((0, l))
                hits = [x for x in hits if not (x[1] == "VALID_AT_T" and re.search(r"\bINVALID\b", t.upper()))] or hits
            return min(hits)[1] if hits else "UNPARSED"
        out = O / f"gpu/out/sensitivity_refval_{a.judge}_shard{si}.jsonl"; done = {r["task_id"] for r in jl(out)}
        todo = [t for i, t in enumerate(jl(O / "gpu/in/sensitivity_refval_tasks.jsonl")) if i % sn == si and t["task_id"] not in done]; fh = open(out, "a", encoding="utf-8")
        for t in todo:
            msgs = [{"role": "system", "content": t["system"]}, {"role": "user", "content": t["user"]}]
            if a.judge == "mistral7b": msgs = [{"role": "user", "content": t["system"] + "\n\n" + t["user"]}]
            enc = tok([tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)], return_tensors="pt", truncation=True, max_length=D["max_input_tokens"]).to(m.device)
            with torch.no_grad(): g = m.generate(**enc, max_new_tokens=D["max_new_tokens"], do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            raw = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            fh.write(json.dumps({"task_id": t["task_id"], "judge": a.judge, "raw": raw, "label": parse(raw), "input_tokens": int(enc["input_ids"].shape[1]),
                                 "truncated": bool(enc["input_ids"].shape[1] >= D["max_input_tokens"])}, ensure_ascii=False) + "\n"); fh.flush()
    sys.exit(0)
import pickle   # noqa: E402
sys.argv = [sys.argv[0], "score"]
import output_workspace; output_workspace.use("common_sample"); import composition_study_helpers as X   # noqa: E402
from common_sample_construction import goldof, REC, Lab, POL   # noqa: E402
C = pickle.load(open(O / "work/common.pkl", "rb")); rows, S = C["rows"], C["S"]; U, Nref, n = len(C["U"]), C["N_ref"], len(C["S"]); lab = Lab()
ANS = {(q, p): lab.answer(a["key"]) for (q, p), a in rows.items()}
if MODE == "items":
    it = {}
    for (q, p), a in ANS.items():
        if any(x in a.strip().lower() for x in X.HELD): continue
        s = [X.h16(REC[q]["query"]), X.h16(goldof(q)), X.h16(a)]; it.setdefault(tuple(s), {"sig": s, "q": REC[q]["query"], "gold": goldof(q), "cand": a})
    wjl(O / "gpu/in/sensitivity_answer_items.jsonl", list(it.values())); wjl(O / "gpu/in/judge2_items.jsonl", list(it.values()))   # judge2 file -> composition_study_generation_judging.py judge2 --judge mistral7b
    import answer_grading as e2e   # noqa: E402
    p2 = X.engine()["p2"]; S_ = PROT["reference_stage"]; tasks = []; T = pickle.load(open(O / "work/traces.pkl", "rb"))
    for q in S:
        pages = {}
        for p in POL:
            for uh, tv in T[p][q]["ev"]:
                rd = p2.snapshot_round_on_disk(uh, tv); pages.setdefault((uh, rd), None)
        ev, caps = [], []
        for (uh, rd) in sorted(pages):
            f = e2e.SNAP_DIR / rd / f"{uh}.json"
            if not f.exists(): continue
            d = json.load(open(f, encoding="utf-8")); t = re.sub(r"\s+", " ", d.get("extracted_text") or "").strip()
            if len(t) < 200: continue
            ev.append((d.get("fetched_at", "")[:10], t[:2000])); caps.append(d.get("fetched_at", "")[:10])
        if not ev: continue
        evid = "\n\n".join(f"[page {i+1}, captured {c}] {t}" for i, (c, t) in enumerate(ev))
        tasks.append({"task_id": q, "system": S_["system"], "user": S_["user"].format(question=REC[q]["query"], target=max(caps), reference=goldof(q), captured=", ".join(sorted(set(caps))), evidence=evid)})
    wjl(O / "gpu/in/sensitivity_refval_tasks.jsonl", tasks); print("answer items", len(it), "refval tasks", len(tasks), "of", n); sys.exit(0)
if MODE == "score":
    J = {"llama8b": {}, "qwen7b": {}, "mistral7b": {}, "phi4": {}}
    for f in sorted((O / "gpu/out").glob("judge_mistral7b*.jsonl")):
        for r in jl(f): J["mistral7b"][tuple(r["sig"])] = r["label"]
    for f in sorted((O / "gpu/out").glob("sensitivity_phi4_shard*.jsonl")):
        for r in jl(f): J["phi4"][tuple(r["sig"])] = r["label"]
    TABLE_ROWS = (OUTPUT_ROOT / "model_adjudication"); src = open((REPO_ROOT / "scripts/evaluation/reference_validation/model_adjudication_analysis.py")).read()
    g = {"__name__": "model_adjudication_analysis_copy", "__file__": str((REPO_ROOT / "scripts/evaluation/reference_validation/model_adjudication_analysis.py"))}; exec(compile(src[:src.index("FRESH_ARM =")], "model_adjudication_analysis_copy", "exec"), g); det = g["det"]
    def two(q, p):
        k = rows[(q, p)]["key"]; return lab.label(REC[q]["query"], goldof(q), k)
    def L(q, p, rule):
        a = ANS[(q, p)]
        if rule == "deterministic":
            k, l = det(goldof(q), a); return l if k else "NOT_STRUCTURED"
        if any(x in a.strip().lower() for x in X.HELD): return "ABSTAIN"
        l2, _, s = two(q, p)
        if rule == "primary_two_judge": return l2 or "UNRESOLVED"
        s = (X.h16(REC[q]["query"]), X.h16(goldof(q)), X.h16(a))
        lm, lp = J["mistral7b"].get(s), J["phi4"].get(s)
        if rule == "phi4": return "UNRESOLVED" if lp is None else ("CORRECT" if lp == "CORRECT" else "WRONG")
        if rule == "mistral7b": return "UNRESOLVED" if lm is None else ("CORRECT" if lm == "CORRECT" else "WRONG")
        ll = [lab.hl.j[j].get(s) or lab.hl.GJ[j].get(s) or lab.J[j].get(s) for j in ("llama8b", "qwen7b")] + [lm]
        if None in ll: return "UNRESOLVED"
        c = sum(x == "CORRECT" for x in ll); return ("CORRECT" if c >= 2 else "WRONG") if rule == "majority_2of3" else ("CORRECT" if c == 3 else "WRONG")
    RF = collections.defaultdict(dict)
    for f in sorted((O / "gpu/out").glob("sensitivity_refval_*_shard*.jsonl")):
        for r in jl(f): RF[r["task_id"]][r["judge"]] = r["label"]
    def refstat(q):
        v = list(RF.get(q, {}).values())
        if len(v) < 3: return "NOT_SCREENED"
        return "VALID" if sum(x == "VALID_AT_T" for x in v) >= 2 else ("INVALID" if sum(x == "INVALID_AT_T" for x in v) >= 2 else "UNSETTLED")
    REF = {q: refstat(q) for q in S}
    PAIRS = [("B", "D"), ("B", "Dp"), ("B", "Bm"), ("Bm", "Dp"), ("D", "Dp")]
    R = {"n": n, "reference_screen": dict(collections.Counter(REF.values())), "rules": {}}
    for subset in ("all", "reference_valid"):
        ids0 = S if subset == "all" else [q for q in S if REF[q] == "VALID"]
        for rule in ("primary_two_judge", "mistral7b", "phi4", "majority_2of3", "unanimous_3of3", "deterministic"):
            Lr = {(q, p): L(q, p, rule) for q in ids0 for p in POL}
            for a, b in PAIRS:
                ids = [q for q in ids0 if "UNRESOLVED" not in (Lr[(q, a)], Lr[(q, b)]) and "NOT_STRUCTURED" not in (Lr[(q, a)], Lr[(q, b)])]
                lo = sum(Lr[(q, a)] == "CORRECT" and Lr[(q, b)] != "CORRECT" for q in ids); ga = sum(Lr[(q, a)] != "CORRECT" and Lr[(q, b)] == "CORRECT" for q in ids)
                mu = (ga - lo) / len(ids) if ids else 0; d = [int(Lr[(q, b)] == "CORRECT") - int(Lr[(q, a)] == "CORRECT") for q in ids]
                se = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d) * (1 - len(d) / U)) if len(d) > 1 else 0; f = 100 * U / Nref
                R["rules"].setdefault(subset, {}).setdefault(rule, {})[f"{a}->{b}"] = {"n": len(ids), "unresolved_or_unstructured": len(ids0) - len(ids), "losses": lo, "gains": ga,
                    "p": round(X.mcnemar(lo, ga), 4), "stream_pp_if_frame_scaled": round(f * mu, 3), "ci": [round(f * (mu - 1.96 * se), 3), round(f * (mu + 1.96 * se), 3)]}
    agree = [(lab.label(REC[q]["query"], goldof(q), rows[(q, p)]["key"])[0], L(q, p, "phi4")) for q in S for p in POL]
    agree = [(x, y) for x, y in agree if x in ("CORRECT", "WRONG") and y in ("CORRECT", "WRONG")]
    po = sum(x == y for x, y in agree) / len(agree); pa = sum(x == "CORRECT" for x, _ in agree) / len(agree); pb = sum(y == "CORRECT" for _, y in agree) / len(agree); pe = pa * pb + (1 - pa) * (1 - pb)
    R["phi4_vs_two_judge"] = {"pairs": len(agree), "agreement_pct": round(100 * po, 1), "kappa": round((po - pe) / (1 - pe), 3)}
    json.dump(R, open(O / "outputs/judging_sensitivity.json", "w"), indent=1); print(json.dumps(R, indent=1))
