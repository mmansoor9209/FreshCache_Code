#!/usr/bin/env python3
"""
v13_corrected/a1_dataset_audit.py — audit every snapshot and emit an immutable
corrected benchmark manifest.

WHAT IS WRONG WITH THE EXISTING LABELS
  `experiment.py: build_stale_sets` returns a SET of changed url_hashes. Every
  URL not in that set is treated as UNCHANGED by every simulator. That set is
  built only from change_log.jsonl, so a URL is silently "unchanged" when it
  was never fetched in that window, when the fetch failed, when the page came
  back a bot-block interstitial, and when only a title was extracted. Absence of
  evidence is recorded as evidence of absence.

THE CORRECTED LABEL SET
  Every (url_hash, window) transition run_00 -> W gets exactly one of:
    CHANGED       both sides observable and the content signal differs
    UNCHANGED     both sides observable and the content signal is identical
    UNOBSERVABLE  either side missing, non-200, or without a substantive body
  UNOBSERVABLE is never collapsed into UNCHANGED, at any stage, by anything
  downstream. The manifest records the reason so the class can be audited.

OBSERVABILITY RULE (unchanged from v9/build_version_timeline.py)
  substantive := extracted text >= 400 chars after stripping, and the first 400
  chars do not match the bot-block / error interstitial pattern.

TWO CONTENT SIGNALS, BOTH RECORDED
  hash_label  uses the snapshot's own `content_hash`, which is what the paper's
              drift metric uses. It is NOT a hash of the extracted body.
  text_label  uses md5 of the extracted body text.
  The manifest carries both. hash_label is primary, because the object of this
  exercise is to establish the correct results for the EXISTING methodology, not
  to change its change-detection signal. text_label is reported as a sensitivity
  so the gap between "the HTML changed" and "the readable content changed" is
  visible rather than assumed away.

SAFETY  Read-only on data/. Writes only v13_corrected/. The emitted manifest is
        written once and then set read-only; a rerun refuses to overwrite it.
"""
from __future__ import annotations

import hashlib, json, multiprocessing as mp, os, pathlib, re, stat, sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent
SNAP = ROOT/"data"/"snapshots"
RUNS = ["run_00", "rerun_1h", "rerun_12h", "rerun_24h", "rerun_48h", "rerun_7d"]
WINDOWS = ["rerun_1h", "rerun_12h", "rerun_24h", "rerun_48h", "rerun_7d"]
RUN_AGE = {"rerun_1h": 3_600.0, "rerun_12h": 43_200.0, "rerun_24h": 86_400.0,
           "rerun_48h": 172_800.0, "rerun_7d": 604_800.0}
MIN_BODY = 400
BLOCK = re.compile(r"pardon our interruption|just a moment|enable javascript|"
                   r"access denied|are you a robot|captcha|cloudflare|"
                   r"403 forbidden|404 not found|page not found", re.I)
MANIFEST = OUT/"corrected_benchmark_manifest.jsonl"


def substantive(t: str) -> bool:
    if not t or len(t.strip()) < MIN_BODY:
        return False
    return not BLOCK.search(t[:400])


def _scan(args):
    run, paths = args
    out = []
    for p in paths:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as ex:
            out.append({"run": run, "url_hash": pathlib.Path(p).stem,
                        "read_error": str(ex)[:80]})
            continue
        t = d.get("extracted_text") or ""
        st = d.get("status_code")
        sub = substantive(t)
        why = None
        if st != 200:
            why = f"status_{st}"
        elif not t or len(t.strip()) < MIN_BODY:
            why = "body_too_short"
        elif BLOCK.search(t[:400]):
            why = "block_page"
        out.append({
            "run": run, "url_hash": d.get("url_hash") or pathlib.Path(p).stem,
            "url": d.get("url", ""), "status_code": st,
            "fetched_at": d.get("fetched_at"),
            "content_hash": d.get("content_hash"),
            "text_hash": hashlib.md5(t.encode("utf-8")).hexdigest(),
            "n_chars": len(t), "content_length": d.get("content_length"),
            "substantive": sub, "unobservable_reason": why})
    return out


def main():
    if MANIFEST.exists():
        print(f"  {MANIFEST.name} already exists and is immutable; "
              f"refusing to overwrite. Delete it explicitly to rebuild.")
        return

    print("="*104); print("  STEP 1 — SCAN EVERY SNAPSHOT"); print("="*104)
    tasks = []
    for run in RUNS:
        d = SNAP/run
        if not d.exists():
            print(f"  {run:<12} directory missing"); continue
        ps = sorted(str(p) for p in d.glob("*.json"))
        for i in range(0, len(ps), 400):
            tasks.append((run, ps[i:i+400]))
    with mp.Pool(min(16, os.cpu_count() or 8)) as pool:
        chunks = pool.map(_scan, tasks)
    snaps = defaultdict(dict)
    n_read_err = 0
    for ch in chunks:
        for r in ch:
            if r.get("read_error"):
                n_read_err += 1
                continue
            snaps[r["url_hash"]][r["run"]] = r
    print(f"  URLs seen: {len(snaps):,}   unreadable snapshot files: {n_read_err}")
    print(f"  {'run':<12}{'snapshots':>11}{'status 200':>12}{'substantive':>13}"
          f"{'block page':>12}{'short body':>12}{'non-200':>9}")
    per_run = {}
    for run in RUNS:
        rs = [v[run] for v in snaps.values() if run in v]
        if not rs:
            continue
        c = Counter(r["unobservable_reason"] for r in rs)
        per_run[run] = {
            "snapshots": len(rs),
            "status_200": sum(1 for r in rs if r["status_code"] == 200),
            "substantive": sum(1 for r in rs if r["substantive"]),
            "block_page": c.get("block_page", 0),
            "body_too_short": c.get("body_too_short", 0),
            "non_200": sum(1 for r in rs if r["status_code"] != 200)}
        p = per_run[run]
        print(f"  {run:<12}{p['snapshots']:>11,}{p['status_200']:>12,}"
              f"{p['substantive']:>13,}{p['block_page']:>12,}"
              f"{p['body_too_short']:>12,}{p['non_200']:>9,}")

    # ---------- previous labels ----------
    print(f"\n{'='*104}\n  STEP 2 — WHAT THE PREVIOUS PIPELINE LABELLED\n{'='*104}")
    changes = [json.loads(l) for l in open(ROOT/"data"/"change_log.jsonl")]
    prev_changed = defaultdict(set)
    prev_noise = defaultdict(set)
    for c in changes:
        (prev_noise if c.get("noise") else prev_changed)[c["run_id"]].add(c["url_hash"])
    print(f"  change_log.jsonl rows {len(changes):,}")
    for w in WINDOWS:
        print(f"    {w:<12} changed {len(prev_changed[w]):>6,}   "
              f"noise-flagged {len(prev_noise[w]):>6,}")
    print(f"\n  build_stale_sets() keeps only non-noise rows and folds rerun_12h")
    print(f"  into the 24h stale set as a proxy. Everything absent from that set")
    print(f"  is treated as UNCHANGED by every simulator.")

    # ---------- corrected transitions ----------
    print(f"\n{'='*104}\n  STEP 3 — CORRECTED TRANSITION LABELS\n{'='*104}")
    rows, summary = [], {}
    for w in WINDOWS:
        cnt = Counter(); why = Counter(); mism = Counter()
        for u, per in snaps.items():
            a, b = per.get("run_00"), per.get(w)
            if a is None or b is None:
                lab, reason = "UNOBSERVABLE", ("missing_run_00" if a is None
                                               else f"missing_{w}")
            elif not a["substantive"] or not b["substantive"]:
                lab = "UNOBSERVABLE"
                reason = (f"run_00_{a['unobservable_reason']}" if not a["substantive"]
                          else f"{w}_{b['unobservable_reason']}")
            else:
                reason = None
                lab = "CHANGED" if a["content_hash"] != b["content_hash"] else "UNCHANGED"
            if a is not None and b is not None:
                tl = ("CHANGED" if a["text_hash"] != b["text_hash"] else "UNCHANGED")
            else:
                tl = None
            text_label = tl if lab != "UNOBSERVABLE" else "UNOBSERVABLE"
            was_prev_changed = u in prev_changed[w]
            prev_label = "CHANGED" if was_prev_changed else "UNCHANGED"
            cnt[lab] += 1
            if reason:
                why[reason] += 1
            mism[(prev_label, lab)] += 1
            rows.append({
                "url_hash": u, "window": w, "age_seconds": RUN_AGE[w],
                "label": lab, "reason": reason,
                "text_label": text_label,
                "previous_label": prev_label,
                "content_hash_run_00": a["content_hash"] if a else None,
                "content_hash_window": b["content_hash"] if b else None,
                "n_chars_run_00": a["n_chars"] if a else None,
                "n_chars_window": b["n_chars"] if b else None,
                "status_run_00": a["status_code"] if a else None,
                "status_window": b["status_code"] if b else None,
                "fetched_at_run_00": a["fetched_at"] if a else None,
                "fetched_at_window": b["fetched_at"] if b else None,
                "url": (a or b or {}).get("url", "")})
        tot = sum(cnt.values())
        summary[w] = {"total": tot, **{k: cnt.get(k, 0) for k in
                                       ("CHANGED", "UNCHANGED", "UNOBSERVABLE")},
                      "unobservable_reasons": dict(why.most_common()),
                      "vs_previous": {f"{a}->{b}": v for (a, b), v in
                                      sorted(mism.items())}}
        print(f"\n  {w}   ({tot:,} URL transitions)")
        for k in ("CHANGED", "UNCHANGED", "UNOBSERVABLE"):
            print(f"    {k:<14}{cnt.get(k,0):>8,}{100*cnt.get(k,0)/tot:>8.1f}%")
        print(f"    unobservable reasons: " +
              ", ".join(f"{k} {v:,}" for k, v in why.most_common(5)))

    print(f"\n{'='*104}\n  STEP 4 — WHERE UNOBSERVABLE WAS PREVIOUSLY CALLED "
          f"UNCHANGED\n{'='*104}")
    print(f"  {'window':<12}{'prev UNCHANGED':>16}{'of which truly':>16}"
          f"{'truly':>10}{'truly':>14}")
    print(f"  {'':<12}{'':>16}{'UNOBSERVABLE':>16}{'UNCHANGED':>10}{'CHANGED':>14}")
    mislabel = {}
    for w in WINDOWS:
        v = summary[w]["vs_previous"]
        pu = sum(x for k, x in v.items() if k.startswith("UNCHANGED->"))
        uo = v.get("UNCHANGED->UNOBSERVABLE", 0)
        un = v.get("UNCHANGED->UNCHANGED", 0)
        ch = v.get("UNCHANGED->CHANGED", 0)
        mislabel[w] = {"previously_unchanged": pu, "truly_unobservable": uo,
                       "truly_unchanged": un, "truly_changed": ch,
                       "share_unobservable": round(uo/pu, 4) if pu else None}
        print(f"  {w:<12}{pu:>16,}{uo:>16,}{un:>10,}{ch:>14,}")
    print(f"\n  Every URL in the 'truly UNOBSERVABLE' column was scored as a")
    print(f"  clean, non-stale cache hit by every simulator in the paper.")

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    os.chmod(MANIFEST, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    json.dump({"observability_rule": {"min_body_chars": MIN_BODY,
                                      "block_pattern": BLOCK.pattern,
                                      "requires_status_200": True},
               "content_signal": {"primary": "snapshot content_hash (the signal "
                                             "the published drift metric uses; "
                                             "NOT a hash of the extracted body)",
                                  "secondary": "md5 of extracted_text"},
               "runs_scanned": per_run, "n_urls": len(snaps),
               "unreadable_files": n_read_err,
               "transitions": summary, "previously_mislabelled": mislabel,
               "manifest": MANIFEST.name, "manifest_rows": len(rows),
               "immutable": True},
              open(OUT/"a1_dataset_audit.json", "w"), indent=2)
    print(f"\n  wrote {MANIFEST.name} ({len(rows):,} transitions, read-only) "
          f"and a1_dataset_audit.json")


if __name__ == "__main__":
    main()
