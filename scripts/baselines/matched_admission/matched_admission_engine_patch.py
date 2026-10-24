"""Matched-admission baselines. SCALM-style is our reimplementation of SCALM's admission policy (Li et al., IWQoS 2024), not the
complete published system; it omits SCALM's hierarchical cache structure and its eviction policy.

Baseline engine patcher: builds an in-memory patched copy of the frozen mixed_age_replay.py (the file on disk is never modified).
Baselines use only P1 (scalm_eqgate), P4/P6 (trace) and P5 (per-variant L2 floor); the P2/P3 variant sets stay empty. Patches (each asserted to apply once):
  P1 L1 mode 'scalm_eqgate': SCALM-style same-class candidates (similarity >= L1_SIM_THRESHOLD), filtered by FreshCache's semantic gate (entity match +
     semantic_equivalent: cos >= 0.80, Jaccard >= 0.30, answer type, subset) and -- unless the variant is in NO_TTL_VARIANTS -- FreshCache's calibrated
     answer-tier freshness window; selection by highest use frequency, then similarity (SCALM rule); frequency incremented on hit
  P2 per-tier temporal rule: for variants in RISK_L1_VARIANTS the ANSWER tier uses the calibrated risk window, other tiers keep the variant's cfg ttl
  P3 FreshCache L1 timestamp (oldest supporting page) for variants in OLDEST_EVIDENCE_VARIANTS, without changing their REAL_TIME behaviour
  P4 trace records the served evidence and generation query (observation only)
  P5 per-variant raw L2 similarity floor (L2_FLOOR_BY_VARIANT)
  P6 trace records the admitted L1/L2 similarity and L1 entry age (observation only)"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import pathlib, types
def build(me):
    src = open(me.__file__, encoding="utf-8").read()
    def rep(old, new):
        assert src.count(old) == 1, (src.count(old), old[:80]); return src.replace(old, new)
    src = rep('''                if l1m == "scalm":
                    same = np.fromiter((f == fc for f in l1_fc), bool, n1)''', '''                if l1m == "scalm_eqgate":
                    same = np.fromiter((f == fc for f in l1_fc), bool, n1)
                    cand = np.nonzero((sims >= L1T) & same)[0]
                    ent["l1_candidates"] += int(cand.size)
                    if cand.size and name not in NO_TTL_VARIANTS:
                        cand = cand[np.fromiter((temporal_ok(l1_fc[int(j)], t-l1_t[j], "answer", l1_q[int(j)]) for j in cand), bool, cand.size)]
                    cand = cand[sims[cand] >= exp._EQ_SIM_FLOOR]   # semantic_equivalent rejects sim < floor first; identical result, avoids NER on hopeless pairs
                    keep = [int(j) for j in cand if exp.semantic_equivalent(q, l1_q[int(j)], float(sims[j])) and exp._entity_match(q, l1_q[int(j)])]   # same conjunction, cheap test first
                    if keep:
                        kk = np.asarray(keep, dtype=np.int64); fr = np.fromiter((l1_freq[j] for j in kk), np.int64, kk.size)
                        hit1 = int(kk[np.lexsort((-sims[kk], -fr))[0]])
                elif l1m == "scalm":
                    same = np.fromiter((f == fc for f in l1_fc), bool, n1)''')
    src = rep('''            if l1m == "scalm":
                l1_freq[hit1] += 1''', '''            if l1m in ("scalm", "scalm_eqgate"):
                l1_freq[hit1] += 1''')
    old_t = '''        t = cfg["ttl"]\n'''; assert src.count(old_t) == 1
    src = src.replace(old_t, '''        t = cfg["ttl"]\n        if name in RISK_L1_VARIANTS and tier == "answer": t = None\n''')
    old_te = "if te_oldest and (use_l2 or use_l3):"; assert src.count(old_te) == 1
    src = src.replace(old_te, "if (te_oldest or name in OLDEST_EVIDENCE_VARIANTS) and (use_l2 or use_l3):")
    old_l1 = '''            if trace is not None: trace[qid] = {"tier": "L1", "fc": fc, "entry": int(hit1), "entry_fc": l1_fc[hit1], "reused": []}   # [INSTR]'''
    assert src.count(old_l1) == 1
    src = src.replace(old_l1, old_l1 + '''\n            if trace is not None: trace[qid]["ev"] = list(l1_ev[hit1]); trace[qid]["gen_query"] = l1_q[hit1]''')
    lines = src.split("\n"); idx = [i for i, l in enumerate(lines) if l.strip().startswith('if trace is not None: trace[qid] = {"tier": ("L2" if use_l2 and hit2 >= 0 else "miss")')]
    assert len(idx) == 1; ind = lines[idx[0]][:len(lines[idx[0]]) - len(lines[idx[0]].lstrip())]
    lines.insert(idx[0] + 1, ind + 'if trace is not None: trace[qid]["ev"] = list(evid); trace[qid]["gen_query"] = q')
    src = "\n".join(lines)
    old_f = """    L2FLOOR = (max(exp.L2_SIM_THRESHOLD, exp._L2_EQ_SIM_FLOOR)
               if l2m.startswith("eq") else exp.L2_SIM_THRESHOLD)
"""
    assert src.count(old_f) == 1
    src = src.replace(old_f, old_f + "    if name in L2_FLOOR_BY_VARIANT: L2FLOOR = L2_FLOOR_BY_VARIANT[name]\n")
    lines = src.split("\n"); idx = [i for i, l in enumerate(lines) if l.strip() == 'if trace is not None: trace[qid]["ev"] = list(evid); trace[qid]["gen_query"] = q']
    assert len(idx) == 1; ind = lines[idx[0]][:len(lines[idx[0]]) - len(lines[idx[0]].lstrip())]
    lines.insert(idx[0] + 1, ind + 'if trace is not None and use_l2 and hit2 >= 0: trace[qid]["l2_sim"] = float(sims2[hit2])   # [BL-P6] observation only')
    src = "\n".join(lines)
    old_l1 = 'if trace is not None: trace[qid]["ev"] = list(l1_ev[hit1]); trace[qid]["gen_query"] = l1_q[hit1]'
    assert src.count(old_l1) == 1
    src = src.replace(old_l1, old_l1 + '; trace[qid]["l1_sim"] = float(row[l1_qi[hit1]]); trace[qid]["l1_age"] = float(t - l1_t[hit1])   # [BL-P6] observation only')
    mod = types.ModuleType("mixed_engine_baselines"); mod.__file__ = me.__file__
    mod.NO_TTL_VARIANTS, mod.RISK_L1_VARIANTS, mod.OLDEST_EVIDENCE_VARIANTS, mod.L2_FLOOR_BY_VARIANT = set(), set(), set(), {}
    exec(compile(src, "<Baselines in-memory patched copy of " + me.__file__ + ">", "exec"), mod.__dict__)
    assert mod.exp is me.exp and mod.ea is me.ea
    return mod
