"""Read-only diagnostic of the recorded (frozen) L1 lexical admission rules.

Uses the exact frozen functions tiered_cache_model.semantic_equivalent / _eq_content_tokens / _eq_answer_type /
_entity_match / _entities / p_stale (outputs/frozen_inputs/files/tiered_cache_model.py) and, for the separate
L1-guard diagnostic only, the frozen l1_structural_guards.py. No rule, threshold, stopword, trigger or result is
changed; no replay, generation, judging, tuning or selection is run. Diagnostic re-computations (word-boundary
trigger matching, temporal words kept as content tokens) are clearly separate counterfactual flags and are never
used as a policy.
"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, csv, hashlib, json, os, pathlib, re, sys; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()

sys.dont_write_bytecode = True
FR = ((OUTPUT_ROOT / "frozen_inputs"))
FZ = ((OUTPUT_ROOT / "frozen_inputs")); F = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); OUT = ((OUTPUT_ROOT / "l1_rule_audit"))
GATE_DIR = (OUTPUT_ROOT / "l1_precision")
os.chdir(os.path.abspath(os.environ.get("FRESHCACHE_ROOT", ".")))          # tiered_cache_model.py resolves its data files relative to the project root
sys.path.insert(0, str(REPO_ROOT / "freshcache/method")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(REPO_ROOT / "")); sys.path.insert(0, str(REPO_ROOT / "scripts/data")); sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation")); sys.path.insert(0, str(REPO_ROOT / "freshcache/method"))
import numpy as np                         # noqa: E402
import tiered_cache_model as exp                   # noqa: E402
import l1_structural_guards as G                      # noqa: E402
assert pathlib.Path(exp.__file__).resolve() == F / "freshcache/method/tiered_cache_model.py"
assert pathlib.Path(G.__file__).resolve() == (REPO_ROOT / "freshcache/method/l1_structural_guards.py")

# ------------------------------------------------------------------ integrity of every input
FH = {r["frozen_path"]: r["sha256"] for r in csv.DictReader(open((REPO_ROOT / "configs/integrity/FILE_HASHES.csv"))) if r["mode"] == "copied"}
def sha(p, big=False):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
INTEG = {
    "freshcache/method/tiered_cache_model.py": sha(F / "freshcache/method/tiered_cache_model.py") == FH["frozen/freshcache/method/tiered_cache_model.py"],
    "l1_structural_guards.py": sha((REPO_ROOT / "freshcache/method/l1_structural_guards.py")) == FH["frozen/freshcache/method/l1_structural_guards.py"],
    "data/queries.jsonl (live == frozen)": sha(DATA_ROOT / "queries.jsonl") == FH.get("frozen/data/queries.jsonl"),
    "data/url_manifest.jsonl (live == frozen)": sha(DATA_ROOT / "url_manifest.jsonl") == FH.get("frozen/data/url_manifest.jsonl"),
    "data/paraphrase_clusters.jsonl (live == frozen)": sha(DATA_ROOT / "paraphrase_clusters.jsonl") == FH.get("frozen/data/paraphrase_clusters.jsonl"),
    "data/sim_matrix_bgem3.npy (hash-referenced)": sha(DATA_ROOT / "sim_matrix_bgem3.npy") == FH.get("hash_only/data/sim_matrix_bgem3.npy"),
}
assert all(v for k, v in INTEG.items() if not k.startswith("data/")), INTEG   # released code must match the manifest
if not all(INTEG.values()): print("WARNING: DATA_DIR inputs are not the paper's collection (expected for data/sample and re-fetched pages):", [k for k, v in INTEG.items() if not v])
assert pathlib.Path(exp.SIM_MATRIX_CACHE) == DATA_ROOT / "sim_matrix_bgem3.npy"

# ------------------------------------------------------------------ frozen workload (as audit_replay.main builds it)
records = exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE),
                                  exp.load_jsonl(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else [])
IDX = {r["query"]: i for i, r in enumerate(records)}           # identical to audit_replay.main / mixed-age replay
SIM = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
assert SIM.shape == (len(records), len(records))
QID_BY_TEXT = collections.defaultdict(list)
for r in records:
    QID_BY_TEXT[r["query"]].append(r["query_id"])
FC_BY_TEXT = {r["query"]: r["freshness_class"] for r in records}   # last record wins, as IDX
split_of = {r["query_id"]: r["split"] for r in csv.DictReader(open(FZ / "generated/request_lists/primary_replay.csv"))}
heldout_ids = {r["query_id"] for r in csv.DictReader(open(FZ / "generated/request_lists/heldout_replay.csv"))}
TEXT_BY_QID = {r["query_id"]: r["query"] for r in records}
HELD_TEXTS = {TEXT_BY_QID[q] for q in heldout_ids}
VAL_TEXTS = {TEXT_BY_QID[q] for q, s in split_of.items() if s != "test"}

# ------------------------------------------------------------------ recorded rule, decomposed (frozen functions)
STOP = exp._EQ_STOPWORDS
TEMP = frozenset("most recent latest current currently now today last next new".split())
assert TEMP <= STOP
TOKRE = re.compile(r"[a-z0-9]+|[가-힣]+")
def raw_tokens(q): return set(TOKRE.findall(q.lower()))
def is_ascii(s): return all(ord(c) < 128 for c in s)
def kw_boundary(kw, ql):
    return re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", ql) is not None

def atype_trace(q, table=None, boundary=False):
    """Mirror of exp._eq_answer_type returning (type, [triggers of that type that match]).
    boundary=True is the DIAGNOSTIC word-boundary variant for ASCII triggers (Hangul unchanged)."""
    ql = q.lower()
    for at, kws in (table or exp._EQ_ANSWER_TYPES).items():
        hits = [kw for kw in kws if (kw_boundary(kw, ql) if (boundary and is_ascii(kw)) else kw in ql)]
        if hits:
            return at, hits
    return None, []
NO_WON = {k: [w for w in v if w != "won"] for k, v in exp._EQ_ANSWER_TYPES.items()}

def jacc(a, b):
    u = a | b
    return (len(a & b) / len(u)) if u else None
def subset(a, b): return a != b and (a <= b or b <= a)
def lexical(a, b):
    j = jacc(a, b)
    return j is not None and j >= exp._EQ_JACCARD_MIN and not subset(a, b)
def at_ok(a1, a2): return a1 is None or a2 is None or a1 == a2

def analyse(q1, q2, sim):
    t1, t2 = exp._eq_content_tokens(q1), exp._eq_content_tokens(q2)
    j = jacc(t1, t2); sub = subset(t1, t2)
    a1, k1 = atype_trace(q1); a2, k2 = atype_trace(q2)
    assert a1 == exp._eq_answer_type(q1) and a2 == exp._eq_answer_type(q2)
    e1, e2 = exp._entities(q1), exp._entities(q2)
    ent = exp._entity_match(q1, q2)
    cos_ok = sim is not None and sim >= exp._EQ_SIM_FLOOR
    jac_ok = j is not None and j >= exp._EQ_JACCARD_MIN
    sem = exp.semantic_equivalent(q1, q2, sim if sim is not None else 1.0)
    assert sem == ((sim is None or cos_ok) and jac_ok and at_ok(a1, a2) and not sub)
    pub = bool((sim is None or cos_ok) and sem and ent)
    # ---- diagnostic flags (read-only; not policy)
    r1, r2 = raw_tokens(q1), raw_tokens(q2)
    w1 = atype_trace(q1, NO_WON)[0]; w2 = atype_trace(q2, NO_WON)[0]
    b1, bk1 = atype_trace(q1, boundary=True); b2, bk2 = atype_trace(q2, boundary=True)
    def within_only(q):   # ASCII triggers (any type) that match as substrings but never at a word boundary
        ql = q.lower()
        return sorted({kw for kws in exp._EQ_ANSWER_TYPES.values() for kw in kws
                       if is_ascii(kw) and kw in ql and not kw_boundary(kw, ql)})
    def hangul_inner(q):  # Hangul triggers found inside a longer Hangul run (often a particle; reported separately)
        runs = re.findall(r"[가-힣]+", q)
        return sorted({kw for kws in exp._EQ_ANSWER_TYPES.values() for kw in kws
                       if not is_ascii(kw) and any(kw in r and kw != r for r in runs)})
    n1, n2 = G.numeric_tokens(q1), G.numeric_tokens(q2)
    y1 = {x for x in n1 if x.startswith("Y:")}; y2 = {x for x in n2 if x.startswith("Y:")}
    tp1, tp2 = r1 & TEMP, r2 & TEMP
    k1_ = r1 - (STOP - TEMP); k2_ = r2 - (STOP - TEMP)     # temporal words kept as content tokens
    lex_pub, lex_keep = lexical(t1, t2), lexical(k1_, k2_)
    past = {"last", "latest", "recent", "current", "currently", "now", "today"}
    f = {
        "won_type_dependency": (w1 != a1) or (w2 != a2),
        "won_dependency_changes_type_check": at_ok(a1, a2) != at_ok(w1, w2),
        "won_inside_larger_token": any(("won" in tok and tok != "won") for tok in r1 | r2),
        "won_as_token": ("won" in r1) or ("won" in r2),
        "trigger_within_token_any": bool(within_only(q1) or within_only(q2)),
        "type_depends_on_within_token_match": (b1 != a1) or (b2 != a2),
        "within_token_changes_type_check": at_ok(a1, a2) != at_ok(b1, b2),
        "hangul_trigger_inside_longer_run": bool(hangul_inner(q1) or hangul_inner(q2)),
        "numeric_conflict": bool(n1) and bool(n2) and n1 != n2,
        "numeric_one_sided": bool(n1) != bool(n2),
        "year_conflict": bool(y1) and bool(y2) and y1 != y2,
        "negation_mismatch": G.has_negation(q1) != G.has_negation(q2),
        "temporal_terms_differ": tp1 != tp2,
        "temporal_terms_differ_excl_most_new": (tp1 - {"most", "new"}) != (tp2 - {"most", "new"}),
        "next_vs_past_present_conflict": (("next" in tp1) and bool(tp2 & past) and "next" not in tp2)
                                         or (("next" in tp2) and bool(tp1 & past) and "next" not in tp1),
        "temporal_removal_changes_lexical_decision": lex_pub != lex_keep,
        "temporal_removal_admits_(keep_would_reject)": lex_pub and not lex_keep,
        "numeric_guard_fail": not G.numeric_guard(q1, q2),
        "negation_guard_fail": not G.negation_guard(q1, q2),
        "comparative_guard_fail": not G.comparative_guard(q1, q2),
    }
    f["diag_guard_reject_numeric_negation_temporal"] = (f["numeric_guard_fail"] or f["negation_guard_fail"]
                                                        or f["temporal_terms_differ"])
    f["diag_guard_reject_numeric_negation_temporal_excl_most_new"] = (f["numeric_guard_fail"] or f["negation_guard_fail"]
                                                                      or f["temporal_terms_differ_excl_most_new"])
    f["structural_guards_fail_(numeric|negation|comparative)"] = not G.structural_guards(q1, q2)
    rec = {"cosine": None if sim is None else round(float(sim), 6),
           "raw_tokens_1": " ".join(sorted(r1)), "raw_tokens_2": " ".join(sorted(r2)),
           "content_tokens_1": " ".join(sorted(t1)), "content_tokens_2": " ".join(sorted(t2)),
           "jaccard": None if j is None else round(j, 4), "cosine_pass": None if sim is None else cos_ok,
           "jaccard_pass": jac_ok, "strict_subset": sub,
           "answer_type_1": a1, "trigger_1": "|".join(k1), "answer_type_2": a2, "trigger_2": "|".join(k2),
           "answer_type_pass": at_ok(a1, a2),
           "entities_1": "|".join(sorted(e1)), "entities_2": "|".join(sorted(e2)), "entity_pass": ent,
           "semantic_equivalent": sem, "recorded_lexical_admission": pub,
           "numeric_tokens_1": "|".join(sorted(n1)), "numeric_tokens_2": "|".join(sorted(n2)),
           "negation_1": G.has_negation(q1), "negation_2": G.has_negation(q2),
           "temporal_terms_1": "|".join(sorted(tp1)), "temporal_terms_2": "|".join(sorted(tp2)),
           "jaccard_temporal_kept": None if jacc(k1_, k2_) is None else round(jacc(k1_, k2_), 4),
           "strict_subset_temporal_kept": subset(k1_, k2_),
           "within_token_triggers": "|".join(within_only(q1) + ["//"] + within_only(q2)),
           "wordboundary_type_1": b1, "wordboundary_type_2": b2}
    rec.update({"flag_" + k: v for k, v in f.items()})
    return rec

# ------------------------------------------------------------------ 1. constructed unit tests
UNIT = [
    ("a1_won_currency_both", "a", "What is the exchange rate of the Korean won to the dollar?", "How many won does one US dollar buy?"),
    ("a2_won_verb_vs_price", "a", "Who won the Chevron Championship?", "What is the price of Chevron stock?"),
    ("a3_won_verb_vs_untyped", "a", "Which player won the award this season?", "Which player received the award this season?"),
    ("a4_won_verb_vs_score", "a", "Which team won the match and what was the final score?", "Which team won the match?"),
    ("a5_won_verb_both", "a", "Who won the league title?", "Which club won the league title?"),
    ("b1_point_in_appointed", "b", "Who was appointed chief executive of the firm?", "Who is the chief executive of the firm?"),
    ("b2_share_in_shared_vs_won", "b", "Who shared the physics prize?", "Who won the physics prize?"),
    ("b3_born_in_stubborn", "b", "Which character is stubborn in the novel?", "Which character is obstinate in the novel?"),
    ("b4_date_in_update_vs_news", "b", "What is the update on the candidate list?", "What is the news on the candidate list?"),
    ("b5_won_in_wonder", "b", "Who directed the wonder film?", "Who is the director of the wonder film?"),
    ("b6_hangul_gol_in_sigol", "b", "시골 날씨", "지방 날씨"),
    ("c1_year_conflict", "c", "Who won the league title in 2022?", "Who won the league title in 2018?"),
    ("c2_year_conflict_untyped", "c", "Which film topped the box office in 2019?", "Which film topped the box office in 2021?"),
    ("c3_count_conflict", "c", "What are the top 5 selling cars?", "What are the top 10 selling cars?"),
    ("c4_date_conflict", "c", "What happened at the summit on March 3?", "What happened at the summit on March 4?"),
    ("d1_negation_subset", "d", "Which countries do not use the euro?", "Which countries use the euro?"),
    ("d2_negation_reworded", "d", "Which foods are not safe for dogs to eat?", "Which foods are safe for dogs to eat daily?"),
    ("d3_never_vs_positive", "d", "Which teams have never reached the final round?", "Which teams have reached the final round often?"),
    ("e1_last_vs_next", "e", "What was the last film by this director?", "What is the next film by this director?"),
    ("e2_current_vs_plain_subset", "e", "Who is the current coach of the team?", "Who is the coach of the team?"),
    ("e3_today_vs_tomorrow", "e", "What is today's weather in the city?", "What is the weather in the city tomorrow?"),
    ("e4_latest_vs_first", "e", "What is the latest phone model from the company?", "What is the first phone model from the company?"),
    ("e5_new_vs_former", "e", "Who is the new president of the club?", "Who was the former president of the club?"),
    ("e6_recent_vs_next_with_reword", "e", "When was the most recent solar eclipse visible here?", "When is the next solar eclipse visible here?"),
]
TEMPORAL_UNIT = {"class": "TIMELESS", "age_s": 3600.0}
p_unit = exp.p_stale(TEMPORAL_UNIT["class"], TEMPORAL_UNIT["age_s"], "answer")
unit_rows = []
for cid, grp, q1, q2 in UNIT:
    sim = float(SIM[IDX[q1], IDX[q2]]) if (q1 in IDX and q2 in IDX) else None
    rec = analyse(q1, q2, sim)
    rec.update({"record_type": "unit_test", "case_id": cid, "unit_group": grp, "query_1": q1, "query_2": q2,
                "cosine_note": "real bge-m3 cosine" if sim is not None else "constructed text: cosine not available; lexical rules evaluated as if cosine >= 0.80",
                "temporal_result": f"PASS (p_stale={p_unit:.4f} <= {exp.EPS_ANSWER}) for a {TEMPORAL_UNIT['class']} entry aged 1h; temporal rule unchanged",
                "final_recorded_L1_admission": ("ADMIT if cosine >= 0.80" if sim is None and rec["recorded_lexical_admission"]
                                                 else "REJECT" if not rec["recorded_lexical_admission"] else "ADMIT")})
    unit_rows.append(rec)

# ------------------------------------------------------------------ 2. real pairs: every distinct text pair with cosine >= 0.80
U = np.array(sorted(set(IDX.values())), dtype=np.int64)
TEXT_AT = {i: t for t, i in IDX.items()}
pairs = {}   # (i, j) with i < j (record index) -> cosine SIM[i, j]
asym_max = 0.0
B = 256
for s in range(0, len(U), B):
    rows = U[s:s + B]
    blk = np.asarray(SIM[rows])[:, U]
    for r_pos, i in enumerate(rows):
        cols = np.nonzero(blk[r_pos, s + r_pos + 1:] >= exp._EQ_SIM_FLOOR)[0] + s + r_pos + 1
        for c in cols:
            j = int(U[c]); pairs[(int(i), j)] = float(blk[r_pos, c])
# symmetry check on the admitted-cosine set (the engine reads SIM[incoming, cached])
for (i, j), v in list(pairs.items()):
    back = float(SIM[j, i]); asym_max = max(asym_max, abs(back - v))
pair_rows = []
for (i, j), v in pairs.items():
    q1, q2 = TEXT_AT[i], TEXT_AT[j]
    rec = analyse(q1, q2, min(v, float(SIM[j, i])))   # min over orientations: conservative for the cosine test
    rec.update({"q1": q1, "q2": q2,
                "heldout_pair": q1 in HELD_TEXTS and q2 in HELD_TEXTS,
                "validation_pair": q1 in VAL_TEXTS and q2 in VAL_TEXTS,
                "same_cluster": bool({records[i].get("cluster_id")} & {records[j].get("cluster_id")})})
    pair_rows.append(rec)

FLAGS = [k for k in pair_rows[0] if k.startswith("flag_")] if pair_rows else []
def count(rows):
    n = len(rows)
    return {"n": n, **{k[5:]: sum(bool(r[k]) for r in rows) for k in FLAGS}}
POP = {
    "all_cos080": [r for r in pair_rows],
    "all_cos080_recorded_admitted": [r for r in pair_rows if r["recorded_lexical_admission"]],
    "heldout_cos080": [r for r in pair_rows if r["heldout_pair"]],
    "heldout_cos080_recorded_admitted": [r for r in pair_rows if r["heldout_pair"] and r["recorded_lexical_admission"]],
    "validation_cos080": [r for r in pair_rows if r["validation_pair"]],
    "validation_cos080_recorded_admitted": [r for r in pair_rows if r["validation_pair"] and r["recorded_lexical_admission"]],
}
COUNTS = {k: count(v) for k, v in POP.items()}
# decision-changing effects measured against the recorded decision (pairs with cosine >= 0.80)
def flips(rows):
    out = collections.Counter()
    for r in rows:
        pub = r["recorded_lexical_admission"]
        ent = r["entity_pass"]
        # won removed from triggers
        a1w = atype_trace(r["q1"], NO_WON)[0]; a2w = atype_trace(r["q2"], NO_WON)[0]
        won_dec = bool(r["jaccard_pass"] and not r["strict_subset"] and ent and at_ok(a1w, a2w))
        out["won_removed_admit_to_reject"] += pub and not won_dec
        out["won_removed_reject_to_admit"] += (not pub) and won_dec
        wb_dec = bool(r["jaccard_pass"] and not r["strict_subset"] and ent and at_ok(r["wordboundary_type_1"], r["wordboundary_type_2"]))
        out["wordboundary_admit_to_reject"] += pub and not wb_dec
        out["wordboundary_reject_to_admit"] += (not pub) and wb_dec
        j = r["jaccard_temporal_kept"]
        keep_dec = bool(j is not None and j >= exp._EQ_JACCARD_MIN and not r["strict_subset_temporal_kept"]
                        and r["answer_type_pass"] and ent)
        out["temporal_kept_admit_to_reject"] += pub and not keep_dec
        out["temporal_kept_reject_to_admit"] += (not pub) and keep_dec
    return dict(out)
FLIPS = {k: flips(v) for k, v in POP.items() if not k.endswith("admitted")}
REJECT_REASONS = {k: dict(collections.Counter(
    ("jaccard" if not r["jaccard_pass"] else "answer_type" if not r["answer_type_pass"] else
     "strict_subset" if r["strict_subset"] else "entity" if not r["entity_pass"] else "ADMIT") for r in v))
    for k, v in POP.items() if not k.endswith("admitted")}

# entity-check behaviour relative to the manuscript wording
ent_texts = sorted({t for r in pair_rows for t in (r["q1"], r["q2"])})
self_fail = [t for t in ent_texts if exp._entities(t) and not exp._entity_match(t, t)]
both_ent = [r for r in pair_rows if r["entities_1"] and r["entities_2"]]
ENTITY = {"distinct_texts_in_cos080_pairs": len(ent_texts),
          "texts_with_entities": sum(1 for t in ent_texts if exp._entities(t)),
          "texts_failing_entity_check_against_themselves": len(self_fail),
          "examples_self_fail": self_fail[:8],
          "pairs_both_with_entities": len(both_ent),
          "pairs_equal_entity_sets_but_entity_check_fails": sum(1 for r in both_ent if r["entities_1"] == r["entities_2"] and not r["entity_pass"]),
          "pairs_different_entity_sets_but_entity_check_passes": sum(1 for r in both_ent if r["entities_1"] != r["entities_2"] and r["entity_pass"])}

# ------------------------------------------------------------------ 3. realized held-out L1 hits and stricter arms
T1 = [json.loads(l) for l in open((OUTPUT_ROOT / "replay/shadow/reported/tier/inputs/l1_reuse_hits.jsonl"))]
ST = json.load(open((OUTPUT_ROOT / "replay/results/stricter_cpu_reported.json")))["heldout"]
arm_hits = {a: ST[a]["hits"] for a in ("recorded", "strict090", "precision_frozen")}
assert [h["query_id"] for h in arm_hits["recorded"]] == [h["query_id"] for h in T1]
assert all(a["pair_id"] == b["pair_id"] for a, b in zip(arm_hits["recorded"], T1))
# jury verdicts exactly as the frozen analysis computes them (answer_quality_evaluation.py:155-159 + answer_pools.jury), read from the frozen mirror
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine"))
import answer_pools as P   # noqa: E402
assert pathlib.Path(P.__file__).resolve() == F / "freshcache/replay_engine/answer_pools.py"
_missing = [f for f in P.JURY_FILES for j in ("llama3b",) if not (F / f.format(j=j)).exists()]
assert not _missing, _missing
P.R = F   # in-memory: read the byte-identical frozen mirror instead of the live tree
JY = P.jury(extra=[str((OUTPUT_ROOT / "replay/gpu/out/equiv_{j}.jsonl"))])
def jury(pid):
    v = {j: l for j, l in JY.get(pid, {}).items() if l in ("SAME", "DIFFERENT")}
    if len(v) < 2: return "UNJUDGED"
    c = collections.Counter(v.values()); return "JURY_TIE" if c["SAME"] == c["DIFFERENT"] else c.most_common(1)[0][0]
def mism(lab): return lab in ("DIFFERENT", "JURY_TIE")   # ties count DIFFERENT (manuscript Table 13)
MISMATCH = {a: (sum(mism(jury(h["pair_id"])) for h in hs), len(hs)) for a, hs in arm_hits.items()}
set090 = {(h["query_id"], h["pair_id"]) for h in arm_hits["strict090"]}
setG = {(h["query_id"], h["pair_id"]) for h in arm_hits["precision_frozen"]}
hit_rows = []
for h in T1:
    q1, q2 = h["incoming_query"], h["matched_query"]
    sim = float(SIM[IDX[q1], IDX[q2]])
    rec = analyse(q1, q2, sim)
    fc = FC_BY_TEXT[q2]; p = exp.p_stale(fc, h["age_s"], "answer")
    rec.update({"record_type": "realized_heldout_L1_hit", "case_id": h["pair_id"], "request_id": h["query_id"],
                "stored_query_ids": "|".join(QID_BY_TEXT[q2]), "query_1": q1, "query_2": q2,
                "stored_answer_query_equals_matched": h["stored_answer_query"] == q2,
                "temporal_result": f"{'PASS' if p <= exp.EPS_ANSWER else 'FAIL'} (entry class {fc}, age {h['age_s'] / 3600:.2f}h, p_stale={p:.4f})",
                "temporal_pass": p <= exp.EPS_ANSWER,
                "final_recorded_L1_admission": "ADMIT (realized hit)" if rec["recorded_lexical_admission"] and p <= exp.EPS_ANSWER else "INCONSISTENT",
                "jury_4_label": jury(h["pair_id"]), "l1_equivalence_4juror": h.get("equivalence_4juror"),
                "static_cos090_pass": sim >= 0.90,
                "static_cos090_guards_pass": sim >= 0.90 and G.structural_guards(q1, q2),
                "also_hit_in_strict090_arm": (h["query_id"], h["pair_id"]) in set090,
                "also_hit_in_090_guards_arm": (h["query_id"], h["pair_id"]) in setG})
    hit_rows.append(rec)
assert all(r["recorded_lexical_admission"] for r in hit_rows), "reconstruction does not reproduce a realized hit"
assert all(r["temporal_pass"] for r in hit_rows)
assert all(r["jury_4_label"] == r["l1_equivalence_4juror"] for r in hit_rows)
HITS = count(hit_rows)
HITS_BY_JURY = {lab: count([r for r in hit_rows if r["jury_4_label"] == lab]) for lab in ("SAME", "DIFFERENT", "JURY_TIE")}
AFFECT_KEYS = ["flag_won_type_dependency", "flag_won_inside_larger_token", "flag_type_depends_on_within_token_match",
               "flag_trigger_within_token_any", "flag_hangul_trigger_inside_longer_run", "flag_numeric_conflict",
               "flag_numeric_one_sided", "flag_negation_mismatch", "flag_temporal_terms_differ",
               "flag_temporal_removal_changes_lexical_decision", "flag_diag_guard_reject_numeric_negation_temporal"]
for r in hit_rows:
    r["affected_any"] = any(r[k] for k in AFFECT_KEYS)
HITS["affected_any"] = sum(r["affected_any"] for r in hit_rows)
HITS["distinct_pairs"] = len({r["case_id"] for r in hit_rows})
# the stricter arms' own hits, flagged the same way (separate diagnostic)
ARM = {}
for a in ("strict090", "precision_frozen"):
    rows = [analyse(h["incoming"], h["cached"], float(SIM[IDX[h["incoming"]], IDX[h["cached"]]])) for h in arm_hits[a]]
    for r, h in zip(rows, arm_hits[a]):
        r["jury"] = jury(h["pair_id"])
    ARM[a] = {"hits": len(rows), "mismatch_ties_diff": MISMATCH[a][0], **{k: v for k, v in count(rows).items() if k != "n"}}
RECON = {"recorded_hits": len(hit_rows), "mismatch_recorded": MISMATCH["recorded"],
         "mismatch_strict090": MISMATCH["strict090"], "mismatch_precision_frozen": MISMATCH["precision_frozen"],
         "recorded_hits_static_cos090": sum(r["static_cos090_pass"] for r in hit_rows),
         "recorded_hits_static_cos090_guards": sum(r["static_cos090_guards_pass"] for r in hit_rows),
         "recorded_hits_also_in_strict090_arm": sum(r["also_hit_in_strict090_arm"] for r in hit_rows),
         "recorded_hits_also_in_090_guards_arm": sum(r["also_hit_in_090_guards_arm"] for r in hit_rows),
         "hit_rows_all_arms": sum(len(v) for v in arm_hits.values()),
         "distinct_pairs_all_arms": len({h["pair_id"] for v in arm_hits.values() for h in v}),
         "arm_flag_counts": ARM}
# which recorded-hit pairs did the structural guards reject statically, by guard, and their jury labels
RG = collections.Counter()
for r in hit_rows:
    if r["static_cos090_pass"] and not r["static_cos090_guards_pass"]:
        for g in ("numeric_guard_fail", "negation_guard_fail", "comparative_guard_fail"):
            if r["flag_" + g]:
                RG[(g, r["jury_4_label"])] += 1
RECON["recorded_hits_cos090_but_guard_rejected_by_guard_and_jury"] = {f"{g}|{j}": v for (g, j), v in sorted(RG.items())}

# ------------------------------------------------------------------ outputs
COLS = ["record_type", "case_id", "unit_group", "request_id", "stored_query_ids", "query_1", "query_2", "cosine", "cosine_note",
        "raw_tokens_1", "raw_tokens_2", "content_tokens_1", "content_tokens_2", "jaccard", "jaccard_pass", "strict_subset",
        "answer_type_1", "trigger_1", "answer_type_2", "trigger_2", "answer_type_pass", "entities_1", "entities_2", "entity_pass",
        "temporal_result", "semantic_equivalent", "recorded_lexical_admission", "final_recorded_L1_admission",
        "numeric_tokens_1", "numeric_tokens_2", "negation_1", "negation_2", "temporal_terms_1", "temporal_terms_2",
        "jaccard_temporal_kept", "strict_subset_temporal_kept", "within_token_triggers", "wordboundary_type_1", "wordboundary_type_2",
        ] + FLAGS + ["affected_any", "jury_4_label", "static_cos090_pass", "static_cos090_guards_pass",
                     "also_hit_in_strict090_arm", "also_hit_in_090_guards_arm"]
with open(OUT / "C4_AUDIT.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
    for r in unit_rows + hit_rows:
        w.writerow(r)
with open(OUT / "workload_pairs_cos080.csv", "w", newline="", encoding="utf-8") as f:
    cols = ["q1", "q2", "heldout_pair", "validation_pair", "same_cluster"] + [c for c in COLS if c in pair_rows[0]]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(pair_rows)
RES = {"integrity": INTEG, "records": len(records), "distinct_texts": len(U), "heldout_texts": len(HELD_TEXTS),
       "validation_texts": len(VAL_TEXTS), "texts_in_both_splits": len(HELD_TEXTS & VAL_TEXTS),
       "pairs_cos080": len(pairs), "sim_asymmetry_max_on_pairs": asym_max,
       "population_counts": COUNTS, "decision_flips": FLIPS, "first_failing_rule": REJECT_REASONS, "entity_check": ENTITY,
       "realized_hits": HITS, "realized_hits_by_jury": HITS_BY_JURY, "reconciliation": RECON,
       "unit_temporal_p": p_unit}
json.dump(RES, open(OUT / "lexical_rule_results.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in RES.items() if k not in ("population_counts", "realized_hits_by_jury")}, indent=1, default=str))
print(json.dumps(RES["population_counts"], indent=1))
print(json.dumps(RES["realized_hits_by_jury"]))
for r in unit_rows:
    print(r["case_id"], "|", r["content_tokens_1"], "||", r["content_tokens_2"], "| J", r["jaccard"], "sub", r["strict_subset"],
          "| AT", r["answer_type_1"], r["trigger_1"], "/", r["answer_type_2"], r["trigger_2"], "| ENT", r["entities_1"], "/",
          r["entities_2"], r["entity_pass"], "|", r["final_recorded_L1_admission"])
