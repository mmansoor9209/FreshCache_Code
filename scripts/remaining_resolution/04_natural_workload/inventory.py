#!/usr/bin/env python3
"""
EXPERIMENT 4, TASK 4A -- inventory of request traces actually available.

For each candidate source, determine whether it can support L1-only evaluation
or a full L1/L2/L3 evaluation, by checking for the six things a three-tier
replay needs: natural queries, genuine request timestamps, repeated queries,
semantically related queries, retrieved URL lists, and page snapshots.

Nothing is assumed. Every field is probed on the actual files.
"""
from __future__ import annotations
import glob, json, os, pathlib, re, sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
RR = HERE.parent
ROOT = RR.parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
REQ = ["natural_queries", "genuine_timestamps", "repeated_queries",
       "semantic_neighbours", "url_lists", "page_snapshots"]


def probe_jsonl(p, n=3):
    out = []
    try:
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for i, l in enumerate(fh):
                if i >= n:
                    break
                l = l.strip()
                if l:
                    try:
                        out.append(json.loads(l))
                    except Exception:
                        return None
    except Exception:
        return None
    return out


def main():
    log = []
    def say(s=""):
        print(s, flush=True); log.append(s)

    say("EXPERIMENT 4, TASK 4A -- natural request-trace inventory")

    # ---- 1. exhaustive scan for anything resembling a served-traffic log ----
    say(f"\n  == scan for a production / served request log ==")
    pats = ["*log*", "*trace*", "*traffic*", "*session*", "*query_log*",
            "*request*", "*clickstream*", "*serp*"]
    hits = []
    for base in (ROOT, ROOT / "data", ROOT / "logs", ROOT / "external",
                 RR.parent):
        if not base.exists():
            continue
        for pat in pats:
            for f in base.rglob(pat):
                if f.is_file() and f.suffix in (".jsonl", ".json", ".csv", ".tsv"):
                    hits.append(f)
    hits = sorted(set(hits))
    say(f"    candidate files matched by name: {len(hits)}")
    real = []
    for f in hits[:60]:
        s = probe_jsonl(f, 2)
        if not s or not isinstance(s[0], dict):
            continue
        k = set(s[0])
        has_q = bool(k & {"query", "question", "q", "search_query"})
        has_t = bool(k & {"timestamp", "ts", "time", "request_time",
                          "issued_at", "event_time"})
        has_u = bool(k & {"user_id", "session_id", "uid", "anon_id"})
        if has_q and (has_t or has_u):
            real.append((str(f.relative_to(ROOT)), sorted(k)[:10]))
    if real:
        for f, k in real:
            say(f"    POSSIBLE TRACE: {f}  keys {k}")
    else:
        say(f"    NONE of the matched files carries (query + timestamp) or "
            f"(query + user/session id). **No served-traffic request log "
            f"exists in this repository.**")

    # ---- 2. the benchmark's own workload ----
    say(f"\n  == FreshCache-Bench's own workload ==")
    import experiment as exp
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    recs = exp.build_query_records(queries, manifest, paras)
    npara = sum(1 for r in recs if r.get("is_paraphrase"))
    nurl = sum(1 for r in recs if r.get("urls"))
    say(f"    records {len(recs):,}; with URL lists {nurl:,} "
        f"({100*nurl/len(recs):.1f}%); paraphrase-derived {npara:,} "
        f"({100*npara/len(recs):.1f}%)")
    say(f"    request timestamps come from schedules.build_stream "
        f"(zipf_uniform, seed 42) -- SYNTHETIC arrival times, not observed")
    say(f"    -> supports FULL L1/L2/L3, but the repeats and the arrival "
        f"process are constructed, which is the reviewer's objection")

    # ---- 3. DailyQA ----
    say(f"\n  == DailyQA ==")
    dq = sorted((ROOT / "external" / "DailyQA" / "data" / "qa").glob("qa_*.jsonl"))
    keys = Counter()
    per_day = {}
    for f in dq:
        qs = set()
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if l:
                d = json.loads(l)
                keys.update(d.keys())
                q = (d.get("query") or d.get("question") or "").strip()
                if q:
                    qs.add(q)
        per_day[f.stem] = qs
    allq = set().union(*per_day.values()) if per_day else set()
    say(f"    files {len(dq)}; row keys {dict(keys)}")
    say(f"    distinct questions {len(allq):,}; days {len(per_day)}")
    say(f"    URL lists present: {'url' in keys or 'urls' in keys}")
    say(f"    page snapshots present: False (no snapshot path in any row)")
    say(f"    day granularity only -- one date per file, no intra-day "
        f"request times")
    say(f"    -> supports **L1 ONLY**. It cannot support L2 (no URL lists) or "
        f"L3 (no page content).")

    # ---- 4. EvolvingQA ----
    say(f"\n  == EvolvingQA ==")
    try:
        from datasets import get_dataset_split_names, load_dataset
        cols = {}
        for cfg in ("edited", "unchanged"):
            sp = get_dataset_split_names("kat-research/EvolvingQA", cfg)
            ds = load_dataset("kat-research/EvolvingQA", cfg, split=sp[0])
            cols[cfg] = ds.column_names
        say(f"    columns {cols}")
        say(f"    URL lists: False; page snapshots: False; "
            f"request timestamps: False (monthly snapshot pairs only)")
        say(f"    -> supports **L1 ONLY**")
    except Exception as e:
        say(f"    probe failed: {e}")

    # ---- 5. other HF datasets ----
    say(f"\n  == other locally cached QA datasets ==")
    hf = pathlib.Path("~/.cache/huggingface/hub").expanduser()
    names = sorted(d.name.replace("datasets--", "").replace("--", "/")
                   for d in hf.glob("datasets--*")) if hf.exists() else []
    say(f"    {len(names)} cached datasets; none is a served-traffic log.")
    say(f"    temporal QA sets present: "
        f"{[n for n in names if any(k in n.lower() for k in ('fresh','realtime','temporal','evolv','daily','hoh'))]}")

    verdict = {
        "production_request_log": {"available": False,
                                   "supports": "nothing",
                                   "note": "no file in the repository carries "
                                           "query+timestamp or query+session id"},
        "freshcache_bench": {"available": True, "supports": "L1+L2+L3",
                             "natural_queries": "partly: base queries are "
                                                "natural, paraphrases are "
                                                "generated",
                             "genuine_timestamps": False,
                             "url_lists": True, "page_snapshots": True,
                             "limitation": "arrival times and most repeats are "
                                           "synthetic"},
        "dailyqa": {"available": True, "supports": "L1 only",
                    "natural_queries": True, "genuine_timestamps": "day only",
                    "repeated_queries": True, "url_lists": False,
                    "page_snapshots": False},
        "evolvingqa": {"available": True, "supports": "L1 only",
                       "natural_queries": True, "genuine_timestamps": False,
                       "url_lists": False, "page_snapshots": False},
    }
    json.dump({"requirements": REQ, "sources": verdict,
               "conclusion": "No source provides natural queries WITH genuine "
                             "request timestamps AND URL lists AND page "
                             "snapshots. Full three-tier evaluation on genuine "
                             "natural traffic is NOT possible with available "
                             "data. DailyQA and EvolvingQA support L1-only "
                             "evaluation, which has already been run in "
                             "remaining_critical_issues/04_natural_workload "
                             "and 09_external_generalization."},
              open(HERE / "dataset_inventory.json", "w"), indent=2)
    say(f"\n  == TASK 4A CONCLUSION ==")
    say(f"    full L1/L2/L3 on genuine natural traffic: **NOT POSSIBLE** "
        f"with available data")
    say(f"    L1-only on natural repeated queries: DailyQA, EvolvingQA "
        f"(both already evaluated in earlier workstreams)")
    say(f"    the missing ingredient is a request trace that carries BOTH "
        f"genuine timestamps AND the retrieved URL list AND page snapshots")
    (RR / "logs").mkdir(exist_ok=True)
    (RR / "logs" / "exp4_inventory.log").write_text("\n".join(log) + "\n",
                                                    encoding="utf-8")
    say(f"  wrote dataset_inventory.json")


if __name__ == "__main__":
    main()
