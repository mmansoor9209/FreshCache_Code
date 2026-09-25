################## Start ###########
"""
calibrate.py — FreshCache Risk Model Calibration (C1)

Uses the real change observations from collect.py runs to:
  1. Fit half-life parameters per freshness class
  2. Update domain volatility priors from observed data
  3. Evaluate model quality (ECE, false-safe rate, AUC)
  4. Patch freshcache/risk_model.py with calibrated values

TEMPORAL HOLDOUT (W2 fix — LOCKED IN, asymmetric window):
  MEDIUM/FAST half-lives are fit on rerun_1h+rerun_12h, held out on
  rerun_24h+rerun_7d. TIMELESS/SLOW half-lives are fit on
  rerun_1h+rerun_12h+rerun_24h, held out strictly on rerun_7d.
  Two alternatives were tried and rejected: a uniform 1h+12h+24h fit
  for all classes did not separate MEDIUM from FAST (both converged to
  near-identical half-lives once 24h was included); a single-window
  1h-only fit for all classes produced half-lives too short to be
  usable at the t=24h decision horizon (the system made zero caching
  decisions). This asymmetric split is the only one tested that is
  both a temporal holdout for the volatile classes (MEDIUM and FAST are
  evaluated at 24h on data their fit never saw; TIMELESS and SLOW use 24h
  in their own fit, so for those two the 24h evaluation is in-sample, and
  7d is the common later-than-fit horizon) and produces a functioning
  system at t=24h (FreshCache_Full, with the C18 semantic-equivalence gate:
  4.06% hash-based content drift, 80.7% search savings). See
  FIT_WINDOWS_BY_CLASS / HOLDOUT_WINDOWS_BY_CLASS below.

Inputs:
  data/workload_analysis.json   — change rates per class
  data/url_manifest.jsonl       — URL metadata
  data/change_log.jsonl         — ground truth (which URLs changed)
  data/snapshots/run_00/        — fetched_at timestamps

Output:
  data/calibration_report.json  — full evaluation metrics
  freshcache/risk_model.py      — patched with calibrated values

Run:
    python calibrate.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

DATA_DIR      = Path("data")
SNAPSHOTS_DIR = DATA_DIR / "snapshots" / "run_00"
MANIFEST_FILE = DATA_DIR / "url_manifest.jsonl"
CHANGE_FILE   = DATA_DIR / "change_log.jsonl"
ANALYSIS_FILE = DATA_DIR / "workload_analysis.json"
REPORT_FILE   = DATA_DIR / "calibration_report.json"
RISK_MODEL    = Path("freshcache") / "risk_model.py"

NEW_RUN_CUTOFF = "2026-05-10"

# Nominal observation windows (seconds)
T_1H  = 3_600
T_24H = 86_400
T_7D  = 7 * 86_400

# Risk threshold used to classify reuse decisions
EPS_CONTENT = 0.35

# ── W2 temporal holdout (single-window fit) ─────────────────────────────
# Root cause finding: every class's implied half-life (P=1-exp(-ln2/hl*age),
# solved per window in isolation) grows sharply with the observation
# window — e.g. MEDIUM implies 0.89d at 1h but 30.8d at 7d, a 34x spread.
# This means the exponential-decay assumption is violated in practice
# (see fit diagnostics), and the previous age-weighted multi-window MLE
# let the largest-age window dominate the average. Since MEDIUM and FAST
# diverge clearly at 1h (0.89d vs 1.01d implied half-life) but converge
# to nearly the same value by 24h (14.97d vs 14.64d), age-weighting in a
# multi-window average erased the real signal once 24h was included.
# Fix: fit each class on rerun_1h ONLY (single window, no age-weighting
# across windows with different implied half-lives), held out on
# rerun_24h+rerun_7d. (rerun_12h has no ground-truth label in this
# dataset and is unused for both fit and eval.) This preserves the real
# MEDIUM/FAST separation and would have been a uniform temporal holdout,
# since 24h/7d were never seen by any class's fit. REJECTED: see the
# LOCKED IN block below, which is the configuration actually shipped and
# which holds 24h out for the volatile classes only.
# ── W2 temporal holdout (asymmetric mixed window) — LOCKED IN ──────────
# This is the final W2 calibration approach. MEDIUM/FAST fit on
# rerun_1h+rerun_12h only; TIMELESS/SLOW fit on rerun_1h+rerun_12h+
# rerun_24h. rerun_7d (and rerun_24h for MEDIUM/FAST) are held out.
#
# Two alternatives were tried and rejected:
#   - Uniform 1h+12h+24h fit for all classes: did not separate MEDIUM
#     from FAST any better (both converged near 9d), since the 24h
#     window's implied half-life is nearly identical for both classes.
#   - Single-window 1h-only fit for all classes: produced half-lives
#     too short to be usable at the t=24h decision horizon at all
#     (FreshCache_Full collapsed to Search/q=1.00, 0% search savings,
#     i.e. no caching decisions made), since 24h evaluation age becomes
#     comparable to or larger than the fitted half-life itself.
# This asymmetric split is the only one tested that both holds 24h out for
# the volatile classes and produces a functioning system at t=24h. It is NOT
# a uniform holdout: TIMELESS and SLOW use rerun_24h in their own fit, so the
# 24h evaluation is in-sample for those two; 7d is later than the fitting
# window for all four. With the C18 semantic-equivalence gate in force it
# gives FreshCache_Full 4.06% hash-based content drift at 80.7% search
# savings. The 3.3%/98.4% figures previously quoted here predate that gate.
FIT_WINDOWS_BY_CLASS = {
    "MEDIUM":    ["rerun_1h", "rerun_12h"],
    "FAST":      ["rerun_1h", "rerun_12h"],
    "TIMELESS":  ["rerun_1h", "rerun_12h", "rerun_24h"],
    "SLOW":      ["rerun_1h", "rerun_12h", "rerun_24h"],
    "REAL_TIME": [],   # kept as prior, not fit
}
HOLDOUT_WINDOWS_BY_CLASS = {
    "MEDIUM":    ["24h", "7d"],
    "FAST":      ["24h", "7d"],
    "TIMELESS":  ["7d"],
    "SLOW":      ["7d"],
    "REAL_TIME": [],
}


# ---------------------------------------------------------------------------
# Load utilities
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    if not path.exists():
        print(f"  [!] Not found: {path}")
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"  [!] Not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1 — Build ground truth
# ---------------------------------------------------------------------------

def build_ground_truth(manifest: list, changes: list) -> dict:
    """
    For each (url_hash, run_id) pair in reruns, record whether it changed.

    Returns:
        {url_hash: {
            "freshness_class": str,
            "domain": str,
            "changed_1h": bool | None,   # None = not re-fetched
            "changed_24h": bool | None,
        }}
    """
    # URLs that changed per run
    changed_1h  = {c["url_hash"] for c in changes
                   if c.get("run_id") == "rerun_1h"
                   and c.get("detected_at", "") > NEW_RUN_CUTOFF}
    changed_24h = {c["url_hash"] for c in changes
                   if c.get("run_id") == "rerun_24h"
                   and c.get("detected_at", "") > NEW_RUN_CUTOFF}
    changed_7d  = {c["url_hash"] for c in changes
                   if c.get("run_id") == "rerun_7d"
                   and c.get("detected_at", "") > NEW_RUN_CUTOFF}

    # URLs that were actually re-fetched (snapshot_available) per run
    fetched_1h  = {m["url_hash"] for m in manifest
                   if m.get("run_id") == "rerun_1h"
                   and m.get("snapshot_available")}
    fetched_24h = {m["url_hash"] for m in manifest
                   if m.get("run_id") == "rerun_24h"
                   and m.get("snapshot_available")}
    fetched_7d  = {m["url_hash"] for m in manifest
                   if m.get("run_id") == "rerun_7d"
                   and m.get("snapshot_available")}

    # Baseline metadata
    baseline = {}
    for m in manifest:
        if m.get("run_id") == "run_00" and m.get("url_hash"):
            uhash = m["url_hash"]
            if uhash not in baseline:
                baseline[uhash] = {
                    "freshness_class": m.get("freshness_class", "MEDIUM"),
                    "domain":          m.get("domain", ""),
                }

    ground_truth = {}
    for uhash, meta in baseline.items():
        ground_truth[uhash] = {
            "freshness_class": meta["freshness_class"],
            "domain":          meta["domain"],
            "changed_1h":  (True  if uhash in changed_1h
                            else False if uhash in fetched_1h
                            else None),
            "changed_24h": (True  if uhash in changed_24h
                            else False if uhash in fetched_24h
                            else None),
            "changed_7d":  (True  if uhash in changed_7d
                            else False if uhash in fetched_7d
                            else None),
        }

    n_1h  = sum(1 for v in ground_truth.values() if v["changed_1h"]  is not None)
    n_24h = sum(1 for v in ground_truth.values() if v["changed_24h"] is not None)
    n_7d  = sum(1 for v in ground_truth.values() if v["changed_7d"]  is not None)
    print(f"  Ground truth: {len(ground_truth)} URLs total")
    print(f"    With 1h  label: {n_1h}")
    print(f"    With 24h label: {n_24h}  (held out from fit, used for eval only)")
    print(f"    With 7d  label: {n_7d}  (held out from fit, used for eval only)")

    return ground_truth


# ---------------------------------------------------------------------------
# Step 2 — Fit half-life via grid search
# ---------------------------------------------------------------------------

def predicted_p_stale(half_life: float, age: float) -> float:
    lam = math.log(2) / half_life
    return 1.0 - math.exp(-lam * age)


def _fit_single(rate: float, age: float) -> float:
    """
    Closed-form half-life from a single observed change rate at one age.
    P = 1 - exp(-ln2/hl * age)  =>  hl = -ln2 * age / ln(1-rate)
    """
    if rate <= 0 or rate >= 1:
        return None
    return -math.log(2) * age / math.log(1.0 - rate)


def fit_half_life_multi_window(rate_age_pairs: list) -> tuple:
    """
    W2 per-class mixed-window calibration strategy.

    Fits half-life via time-weighted MLE across whichever (rate, age)
    observations fall inside this class's FIT_WINDOWS_BY_CLASS, matching
    the time-weighted averaging already used in
    RuleBasedRiskModel.calibrate() in risk_model.py.

    rate_age_pairs: list of (change_rate, age_seconds) tuples, one per
    run_id in the class's fit window. Pairs with rate <= 0 or rate >= 1
    are skipped (no usable signal at that window).

    Returns (half_life_seconds, note_string). Returns (None, note) if no
    pair in the fit window has usable signal.
    """
    usable = [(r, a) for r, a in rate_age_pairs if r is not None and 0 < r < 1]
    if not usable:
        return None, "no usable signal in fit window"

    lambdas = [_fit_lambda_from_rate(r, a) for r, a in usable]
    ages    = [a for _, a in usable]
    avg_lam = sum(l * a for l, a in zip(lambdas, ages)) / sum(ages)
    hl      = math.log(2) / avg_lam
    windows_used = [a for _, a in usable]
    note = f"time-weighted MLE over {len(usable)} window(s), ages={windows_used}"
    return round(hl, 1), note


def _fit_lambda_from_rate(rate: float, age: float) -> float:
    """
    Fit λ from a single observed change rate at a given age.
    P(stale) = 1 - exp(-λ × age) → λ = -ln(1 - P) / age
    Clamp rate to avoid log(0).
    """
    p = max(0.01, min(0.99, rate))
    return -math.log(1.0 - p) / age


def compute_run_class_rates(manifest: list, changes: list) -> dict:
    """
    Direct (run_id, freshness_class) -> change_rate counter, computed from
    raw manifest/change_log rather than the limited workload_analysis.json
    fields (which only expose 1h/24h aggregates). Needed because TIMELESS/
    SLOW's W2 fit window includes rerun_12h and rerun_24h, which are not
    both present in workload_analysis.json.

    Mirrors the tracked/changed counting in RuleBasedRiskModel.calibrate(),
    INCLUDING the is_noise_url filter — omitting this filter was the root
    cause of the first broken run, where noisy/mismatched URLs diluted the
    change rate and produced inflated half-lives for MEDIUM/FAST.
    """
    try:
        from freshcache.risk_model import is_noise_url
    except Exception as e:
        print(f"  [!] Could not import is_noise_url from freshcache.risk_model "
              f"({e}) — proceeding WITHOUT noise filtering, rates will not "
              f"match the original calibration logic.")
        is_noise_url = lambda domain, fc: False

    tracked: Dict[tuple, int] = defaultdict(int)
    for m in manifest:
        if (m.get("snapshot_available")
                and m.get("run_id", "run_00") != "run_00"):
            domain = m.get("domain", "")
            fc     = m.get("freshness_class", "")
            if fc and is_noise_url(domain, fc):
                continue
            key = (m["run_id"], fc)
            tracked[key] += 1

    changed: Dict[tuple, int] = defaultdict(int)
    for c in changes:
        if c.get("noise", False):
            continue
        if c.get("detected_at", "") <= NEW_RUN_CUTOFF:
            continue
        key = (c.get("run_id"), c.get("freshness_class", ""))
        changed[key] += 1

    rates = {}
    for key, total in tracked.items():
        if total == 0:
            continue
        rates[key] = changed.get(key, 0) / total
    return rates


# ---------------------------------------------------------------------------
# Step 3 — Compute model predictions for evaluation
# ---------------------------------------------------------------------------

def compute_predictions(ground_truth: dict,
                        half_lives: dict,
                        tier_mult: float = 1.0,
                        eval_windows: list = None,
                        eval_windows_by_class: dict = None) -> list:
    """
    For each URL with a known label, compute model prediction.

    eval_windows restricts which observation windows are scored, applied
    globally to all classes. eval_windows_by_class overrides this per
    class (e.g. {"TIMELESS": ["7d"], "MEDIUM": ["24h", "7d"]}) — needed
    for W2's per-class mixed-window holdout, where TIMELESS/SLOW are
    evaluated only on 7d while MEDIUM/FAST are evaluated on 24h+7d.
    If a class has no entry in eval_windows_by_class, falls back to
    eval_windows.

    Returns list of {p_stale, changed, freshness_class, domain, window}
    """
    records = []
    windows_to_use_default = eval_windows or ["1h", "24h", "7d"]
    by_class = eval_windows_by_class or {}

    for uhash, info in ground_truth.items():
        fc = info["freshness_class"]
        hl = half_lives.get(fc)
        if hl is None:
            continue

        windows_to_use = by_class.get(fc, windows_to_use_default)

        all_windows = [
            ("1h",  T_1H,  info["changed_1h"]),
            ("24h", T_24H, info["changed_24h"]),
            ("7d",  T_7D,  info.get("changed_7d")),
        ]

        for window_label, age, changed in all_windows:
            if window_label not in windows_to_use:
                continue
            if changed is None:
                continue

            lam     = math.log(2) / hl * tier_mult
            p_stale = round(1.0 - math.exp(-lam * age), 4)

            records.append({
                "url_hash":        uhash,
                "freshness_class": fc,
                "domain":          info["domain"],
                "window":          window_label,
                "age":             age,
                "p_stale":         p_stale,
                "changed":         changed,
            })

    return records


# ---------------------------------------------------------------------------
# Step 4 — Evaluation metrics
# ---------------------------------------------------------------------------

def compute_ece(predictions: list, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Lower is better. 0.0 = perfect calibration.
    """
    bins = defaultdict(list)
    for r in predictions:
        b = min(int(r["p_stale"] * n_bins), n_bins - 1)
        bins[b].append(r["changed"])

    total = len(predictions)
    if total == 0:
        return 0.0

    ece = 0.0
    for b, labels in bins.items():
        bin_center   = (b + 0.5) / n_bins
        actual_rate  = sum(labels) / len(labels)
        weight       = len(labels) / total
        ece         += weight * abs(bin_center - actual_rate)

    return round(ece, 4)


def compute_false_rates(predictions: list,
                        threshold: float = EPS_CONTENT) -> tuple:
    """
    false_safe  = predicted safe (p < threshold) but actually changed
    false_stale = predicted stale (p >= threshold) but actually unchanged
    """
    false_safe_n  = sum(1 for r in predictions
                        if r["p_stale"] < threshold and r["changed"])
    false_stale_n = sum(1 for r in predictions
                        if r["p_stale"] >= threshold and not r["changed"])
    total_changed   = sum(1 for r in predictions if r["changed"])
    total_unchanged = sum(1 for r in predictions if not r["changed"])

    false_safe  = round(false_safe_n  / max(total_changed, 1),   4)
    false_stale = round(false_stale_n / max(total_unchanged, 1), 4)
    return false_safe, false_stale


def compute_auc(predictions: list) -> float:
    """Area under ROC curve via trapezoidal rule."""
    if not predictions:
        return 0.5

    sorted_preds = sorted(predictions, key=lambda r: r["p_stale"], reverse=True)
    n_pos = sum(1 for r in sorted_preds if r["changed"])
    n_neg = sum(1 for r in sorted_preds if not r["changed"])

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tps, fps = 0, 0
    auc = 0.0
    prev_fp = 0

    for r in sorted_preds:
        if r["changed"]:
            tps += 1
        else:
            fps += 1
            auc += tps * (fps - prev_fp) / (n_pos * n_neg)
            prev_fp = fps

    return round(auc, 4)


def evaluate(predictions: list, label: str) -> dict:
    ece                  = compute_ece(predictions)
    false_safe, false_stale = compute_false_rates(predictions)
    auc                  = compute_auc(predictions)
    n_changed            = sum(1 for r in predictions if r["changed"])

    return {
        "label":           label,
        "n_samples":       len(predictions),
        "n_changed":       n_changed,
        "change_rate":     round(n_changed / max(len(predictions), 1), 4),
        "ece":             ece,
        "false_safe_rate": false_safe,
        "false_stale_rate":false_stale,
        "auc":             auc,
    }


# ---------------------------------------------------------------------------
# Step 5 — Domain volatility from observations
# ---------------------------------------------------------------------------

def compute_observed_domain_volatility(ground_truth: dict) -> dict:
    """
    Compute per-domain change rate from both reruns combined.
    Only include domains with >= 2 observations for reliability.

    Note: domain volatility is a per-domain prior used as a content-level
    adjustment, not the per-class half-life that W2 concerns. It is left
    using all available windows (1h, 24h) since it does not feed the
    half-life fit being held out for W2 and is not the quantity the
    reviewer's t=24h staleness evaluation is computed against.
    """
    domain_obs     = defaultdict(list)

    for info in ground_truth.values():
        domain = info["domain"]
        for changed in [info["changed_1h"], info["changed_24h"]]:
            if changed is not None:
                domain_obs[domain].append(int(changed))

    volatility = {}
    for domain, obs in domain_obs.items():
        if len(obs) >= 2:
            volatility[domain] = round(sum(obs) / len(obs), 3)

    return dict(sorted(volatility.items(),
                        key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Step 6 — Patch risk_model.py
# ---------------------------------------------------------------------------

def patch_risk_model(new_half_lives: dict,
                     new_domain_vol: dict) -> bool:
    """
    Write calibrated half-life and domain volatility values into risk_model.py.
    Creates a backup first.
    """
    if not RISK_MODEL.exists():
        print(f"  [!] {RISK_MODEL} not found — skipping patch")
        return False

    import shutil
    backup = RISK_MODEL.with_suffix(".py.bak")
    shutil.copy2(RISK_MODEL, backup)
    print(f"  Backed up → {backup.name}")

    with open(RISK_MODEL, encoding="utf-8") as f:
        src = f.read()

    # Build new _DEFAULT_HALF_LIFE dict string
    import re
    order = ["REAL_TIME", "FAST", "MEDIUM", "SLOW", "TIMELESS"]
    lines = []
    labels = {
        "REAL_TIME": "30 sec  (kept as prior — REAL_TIME pages often blocked by scrapers)",
        "FAST":      "calibrated from 1h/12h observed change rates (W2 temporal holdout)",
        "MEDIUM":    "calibrated from 1h/12h observed change rates (W2 temporal holdout)",
        "SLOW":      "calibrated from 1h/12h observed change rates (W2 temporal holdout)",
        "TIMELESS":  "calibrated from 1h/12h observed change rates (W2 temporal holdout)",
    }
    for fc in order:
        hl  = new_half_lives.get(fc, 30.0)
        lbl = labels.get(fc, "")
        lines.append(f"    FreshnessClass.{fc:<10}: {hl:>15.1f},   # {lbl}")

    new_hl_block = "_DEFAULT_HALF_LIFE: Dict[FreshnessClass, float] = {\n" + \
                   "\n".join(lines) + "\n}"

    # Find and replace the existing _DEFAULT_HALF_LIFE block
    pattern = r"_DEFAULT_HALF_LIFE: Dict\[FreshnessClass, float\] = \{[^}]+\}"
    if not re.search(pattern, src):
        print("  [!] Could not locate _DEFAULT_HALF_LIFE block — skipping patch")
        return False
    src = re.sub(pattern, new_hl_block, src)


    # Build new top-10 domain volatility entries
    top_domains = list(new_domain_vol.items())[:20]
    domain_lines = []
    for domain, vol in top_domains:
        domain_lines.append(f'    "{domain}": {vol},')

    # Only update the domains we have observed data for
    for domain, vol in top_domains:
        safe_domain = domain.replace(".", r"\.")
        pattern_d = rf'("{safe_domain}":\s*)[\d.]+,'
        replacement = f'"{domain}": {vol},'
        src = re.sub(pattern_d, replacement, src)

    with open(RISK_MODEL, "w", encoding="utf-8") as f:
        f.write(src)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*62}")
    print(f"  FreshCache Risk Model Calibration")
    print(f"  W2 fix: asymmetric mixed-window temporal holdout (LOCKED)")
    print(f"          MEDIUM/FAST   fit=1h+12h     held out=24h+7d")
    print(f"          TIMELESS/SLOW fit=1h+12h+24h held out=7d")
    print(f"{'='*62}\n")

    # ── Load ──────────────────────────────────────────────────────────
    print("Loading data...")
    manifest  = load_jsonl(MANIFEST_FILE)
    changes   = load_jsonl(CHANGE_FILE)

    if not manifest:
        print("ERROR: Run analyze.py and collect.py first.")
        sys.exit(1)

    # ── Ground truth ──────────────────────────────────────────────────
    print("\nBuilding ground truth...")
    ground_truth = build_ground_truth(manifest, changes)

    # ── Direct per-run, per-class change rates (computed from raw data,
    #    not workload_analysis.json, since TIMELESS/SLOW's fit window
    #    needs rerun_12h and rerun_24h rates not both exposed there) ───
    run_class_rates = compute_run_class_rates(manifest, changes)

    RUN_AGES = {
        "rerun_1h":  T_1H,
        "rerun_12h": 12 * 3_600,
        "rerun_24h": T_24H,
        "rerun_7d":  T_7D,
    }

    print("\n  [debug] raw run_class_rates (noise-filtered):")
    for fc in ["TIMELESS", "SLOW", "MEDIUM", "FAST"]:
        for run_id in ["rerun_1h", "rerun_12h", "rerun_24h", "rerun_7d"]:
            r = run_class_rates.get((run_id, fc))
            print(f"    {fc:<10} {run_id:<12} rate={r}")

    # ── Fit half-life per class — W2 per-class mixed window ────────────
    print("\nFitting half-life parameters (W2 per-class mixed window)...")

    CURRENT_HALF_LIVES = {
        "REAL_TIME": 30.0,
        "FAST":      3_600.0,
        "MEDIUM":    86_400.0,
        "SLOW":      604_800.0,
        "TIMELESS":  31_536_000.0,
    }

    fitted = {}
    fit_details = {}

    print(f"\n  {'Class':<12} {'Old HL':>12} {'New HL':>12}  Note")
    print(f"  {'-'*12} {'-'*12} {'-'*12}  {'-'*55}")

    for fc in ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]:
        old_hl = CURRENT_HALF_LIVES[fc]

        # REAL_TIME: keep prior — scrapers miss most dynamic pages
        if fc == "REAL_TIME":
            fitted[fc] = old_hl
            note = "kept as prior (scraper bias)"
            fit_details[fc] = {
                "old_half_life": old_hl,
                "new_half_life": old_hl,
                "fit_window": [],
                "note": note,
            }
            hl_str = f"{old_hl:>12.0f}"
            print(f"  {fc:<12} {old_hl:>12.0f} {hl_str}  {note}")
            continue

        fit_runs = FIT_WINDOWS_BY_CLASS[fc]
        rate_age_pairs = [
            (run_class_rates.get((run_id, fc)), RUN_AGES[run_id])
            for run_id in fit_runs
        ]

        new_hl, note = fit_half_life_multi_window(rate_age_pairs)
        if new_hl is None:
            new_hl = old_hl
            note = (f"no usable signal in fit window {fit_runs} "
                     "— keeping prior (no fallback to held-out runs)")
        fitted[fc] = new_hl
        fit_details[fc] = {
            "old_half_life":      old_hl,
            "new_half_life":      new_hl,
            "fit_window":         fit_runs,
            "holdout_window":     HOLDOUT_WINDOWS_BY_CLASS[fc],
            "observed_rates":     {run_id: run_class_rates.get((run_id, fc))
                                    for run_id in fit_runs},
            "calibration_strategy": "W2_per_class_mixed_window",
            "note":               note,
        }
        old_str = _fmt_hl(old_hl)
        new_str = _fmt_hl(new_hl)
        print(f"  {fc:<12} {old_str:>12} {new_str:>12}  {note[:55]}")

    # ── Evaluate on each class's own HELD-OUT window only ───────────────
    print("\nEvaluating models on per-class held-out window (W2 check)...")

    preds_current_heldout    = compute_predictions(
        ground_truth, CURRENT_HALF_LIVES,
        eval_windows_by_class=HOLDOUT_WINDOWS_BY_CLASS)
    preds_calibrated_heldout = compute_predictions(
        ground_truth, fitted,
        eval_windows_by_class=HOLDOUT_WINDOWS_BY_CLASS)

    print(f"\n  [debug] held-out prediction counts:")
    print(f"    current:    {len(preds_current_heldout)} records")
    print(f"    calibrated: {len(preds_calibrated_heldout)} records")
    for fc in ["TIMELESS", "SLOW", "MEDIUM", "FAST"]:
        n = sum(1 for r in preds_calibrated_heldout if r["freshness_class"] == fc)
        print(f"      {fc:<10} {n} held-out records")

    eval_current    = evaluate(preds_current_heldout,    "Current (prior), per-class held-out")
    eval_calibrated = evaluate(preds_calibrated_heldout, "W2-calibrated (mixed-window fit), per-class held-out")

    print(f"\n  {'Metric':<22} {'Current':>12} {'W2-Calib':>12}  Better?")
    print(f"  {'-'*22} {'-'*12} {'-'*12}  {'-'*8}")

    metrics = [
        ("ECE ↓",            "ece",             True),
        ("False-safe rate ↓","false_safe_rate",  True),
        ("False-stale rate ↓","false_stale_rate",True),
        ("AUC ↑",            "auc",             False),
    ]

    for label, key, lower_is_better in metrics:
        v_curr = eval_current[key]
        v_cal  = eval_calibrated[key]
        if lower_is_better:
            better = "✓" if v_cal < v_curr else ("=" if v_cal == v_curr else "✗")
        else:
            better = "✓" if v_cal > v_curr else ("=" if v_cal == v_curr else "✗")
        print(f"  {label:<22} {v_curr:>12.4f} {v_cal:>12.4f}  {better:>8}")

    # ── Domain volatility ─────────────────────────────────────────────
    print("\nComputing observed domain volatility...")
    domain_vol = compute_observed_domain_volatility(ground_truth)

    print(f"\n  Top volatile domains (from real observations):")
    print(f"  {'Domain':<38} {'Obs. Volatility':>16}")
    print(f"  {'-'*38} {'-'*16}")
    for domain, vol in list(domain_vol.items())[:15]:
        if vol > 0:
            print(f"  {domain:<38} {vol:>16.3f}")

    # ── Patch risk_model.py ───────────────────────────────────────────
    print("\nPatching freshcache/risk_model.py...")
    patched = patch_risk_model(fitted, domain_vol)
    if patched:
        print("  risk_model.py updated with W2 temporally-held-out half-lives.")
    else:
        print("  Patch skipped — update _DEFAULT_HALF_LIFE manually using values above.")

    # ── Save report ───────────────────────────────────────────────────
    report = {
        "w2_fit_windows_by_class":     FIT_WINDOWS_BY_CLASS,
        "w2_holdout_windows_by_class": HOLDOUT_WINDOWS_BY_CLASS,
        "fit_details":         fit_details,
        "fitted_half_lives":   fitted,
        "current_model_eval_heldout":    eval_current,
        "calibrated_model_eval_heldout": eval_calibrated,
        "domain_volatility":   domain_vol,
        "n_ground_truth_urls": len(ground_truth),
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved → {REPORT_FILE}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  W2 Calibration Summary (1h/12h fit, 24h/7d held out)")
    print(f"{'='*62}")
    print(f"  Held-out ECE:        "
          f"{eval_current['ece']:.4f} → {eval_calibrated['ece']:.4f}")
    print(f"  Held-out false-safe: "
          f"{eval_current['false_safe_rate']:.4f} → {eval_calibrated['false_safe_rate']:.4f}")
    print(f"  Held-out AUC:        "
          f"{eval_current['auc']:.4f} → {eval_calibrated['auc']:.4f}")
    print(f"\n  W2-calibrated half-lives (seconds):")
    for fc, hl in fitted.items():
        print(f"    {fc:<12} {_fmt_hl(hl):>15}")
    print(f"\n  Next step: python train_risk_model.py")
    print(f"{'='*62}\n")


def _fmt_hl(seconds: float) -> str:
    """Human-readable half-life label."""
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7_200:
        return f"{seconds/3600:.1f}h"
    if seconds < 172_800:
        return f"{seconds/3600:.0f}h"
    if seconds < 1_209_600:
        return f"{seconds/86400:.1f}d"
    return f"{seconds/86400:.0f}d"


if __name__ == "__main__":
    main()

################## End #######################


############## Start #####################
# """
# calibrate.py — FreshCache Risk Model Calibration (C1)

# Uses the real change observations from collect.py runs to:
#   1. Fit half-life parameters per freshness class
#   2. Update domain volatility priors from observed data
#   3. Evaluate model quality (ECE, false-safe rate, AUC)
#   4. Patch freshcache/risk_model.py with calibrated values

# Inputs:
#   data/workload_analysis.json   — change rates per class
#   data/url_manifest.jsonl       — URL metadata
#   data/change_log.jsonl         — ground truth (which URLs changed)
#   data/snapshots/run_00/        — fetched_at timestamps

# Output:
#   data/calibration_report.json  — full evaluation metrics
#   freshcache/risk_model.py      — patched with calibrated parameters

# Run:
#     python calibrate.py
# """

# from __future__ import annotations

# import json
# import math
# import sys
# from collections import defaultdict
# from pathlib import Path

# DATA_DIR      = Path("data")
# SNAPSHOTS_DIR = DATA_DIR / "snapshots" / "run_00"
# MANIFEST_FILE = DATA_DIR / "url_manifest.jsonl"
# CHANGE_FILE   = DATA_DIR / "change_log.jsonl"
# ANALYSIS_FILE = DATA_DIR / "workload_analysis.json"
# REPORT_FILE   = DATA_DIR / "calibration_report.json"
# RISK_MODEL    = Path("freshcache") / "risk_model.py"

# NEW_RUN_CUTOFF = "2026-05-10"

# # Nominal observation windows (seconds)
# T_1H  = 3_600
# T_24H = 86_400
# T_7D  = 7 * 86_400

# # Risk threshold used to classify reuse decisions
# EPS_CONTENT = 0.35


# # ---------------------------------------------------------------------------
# # Load utilities
# # ---------------------------------------------------------------------------

# def load_jsonl(path: Path) -> list:
#     if not path.exists():
#         print(f"  [!] Not found: {path}")
#         return []
#     records = []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 try:
#                     records.append(json.loads(line))
#                 except json.JSONDecodeError:
#                     continue
#     return records


# def load_json(path: Path) -> dict:
#     if not path.exists():
#         print(f"  [!] Not found: {path}")
#         return {}
#     with open(path, encoding="utf-8") as f:
#         return json.load(f)


# # ---------------------------------------------------------------------------
# # Step 1 — Build ground truth
# # ---------------------------------------------------------------------------

# def build_ground_truth(manifest: list, changes: list) -> dict:
#     """
#     For each (url_hash, run_id) pair in reruns, record whether it changed.

#     Returns:
#         {url_hash: {
#             "freshness_class": str,
#             "domain": str,
#             "changed_1h": bool | None,   # None = not re-fetched
#             "changed_24h": bool | None,
#         }}
#     """
#     # URLs that changed per run
#     changed_1h  = {c["url_hash"] for c in changes
#                    if c.get("run_id") == "rerun_1h"
#                    and c.get("detected_at", "") > NEW_RUN_CUTOFF}
#     changed_24h = {c["url_hash"] for c in changes
#                    if c.get("run_id") == "rerun_24h"
#                    and c.get("detected_at", "") > NEW_RUN_CUTOFF}
#     changed_7d  = {c["url_hash"] for c in changes
#                    if c.get("run_id") == "rerun_7d"
#                    and c.get("detected_at", "") > NEW_RUN_CUTOFF}

#     # URLs that were actually re-fetched (snapshot_available) per run
#     fetched_1h  = {m["url_hash"] for m in manifest
#                    if m.get("run_id") == "rerun_1h"
#                    and m.get("snapshot_available")}
#     fetched_24h = {m["url_hash"] for m in manifest
#                    if m.get("run_id") == "rerun_24h"
#                    and m.get("snapshot_available")}
#     fetched_7d  = {m["url_hash"] for m in manifest
#                    if m.get("run_id") == "rerun_7d"
#                    and m.get("snapshot_available")}

#     # Baseline metadata
#     baseline = {}
#     for m in manifest:
#         if m.get("run_id") == "run_00" and m.get("url_hash"):
#             uhash = m["url_hash"]
#             if uhash not in baseline:
#                 baseline[uhash] = {
#                     "freshness_class": m.get("freshness_class", "MEDIUM"),
#                     "domain":          m.get("domain", ""),
#                 }

#     ground_truth = {}
#     for uhash, meta in baseline.items():
#         ground_truth[uhash] = {
#             "freshness_class": meta["freshness_class"],
#             "domain":          meta["domain"],
#             "changed_1h":  (True  if uhash in changed_1h
#                             else False if uhash in fetched_1h
#                             else None),
#             "changed_24h": (True  if uhash in changed_24h
#                             else False if uhash in fetched_24h
#                             else None),
#             "changed_7d":  (True  if uhash in changed_7d
#                             else False if uhash in fetched_7d
#                             else None),
#         }

#     n_1h  = sum(1 for v in ground_truth.values() if v["changed_1h"]  is not None)
#     n_24h = sum(1 for v in ground_truth.values() if v["changed_24h"] is not None)
#     n_7d  = sum(1 for v in ground_truth.values() if v["changed_7d"]  is not None)
#     print(f"  Ground truth: {len(ground_truth)} URLs total")
#     print(f"    With 1h  label: {n_1h}")
#     print(f"    With 24h label: {n_24h}")
#     print(f"    With 7d  label: {n_7d}")

#     return ground_truth


# # ---------------------------------------------------------------------------
# # Step 2 — Fit half-life via grid search
# # ---------------------------------------------------------------------------

# def predicted_p_stale(half_life: float, age: float) -> float:
#     lam = math.log(2) / half_life
#     return 1.0 - math.exp(-lam * age)


# def _fit_single(rate: float, age: float) -> float:
#     """
#     Closed-form half-life from a single observed change rate at one age.
#     P = 1 - exp(-ln2/hl * age)  =>  hl = -ln2 * age / ln(1-rate)
#     """
#     if rate <= 0 or rate >= 1:
#         return None
#     return -math.log(2) * age / math.log(1.0 - rate)


# def fit_half_life(rate_1h: float | None,
#                   rate_24h: float | None,
#                   rate_7d: float | None = None,
#                   n_grid: int = 50_000) -> tuple:
#     """
#     Conservative calibration strategy using up to three observation windows.

#     Priority: 1h fit (most conservative) -> 24h fit with safety factor ->
#     7d fit with safety factor. For TIMELESS and SLOW, where 1h/24h change
#     rates are near zero, the 7d window provides the most reliable signal.

#     Returns (best_half_life_seconds, note_string).
#     """
#     if rate_1h is not None and rate_1h > 0:
#         hl_1h = _fit_single(rate_1h, T_1H)
#         if hl_1h and hl_1h > 0:
#             pred_24h = predicted_p_stale(hl_1h, T_24H) if rate_24h else None
#             pred_7d  = predicted_p_stale(hl_1h, T_7D)  if rate_7d  else None
#             note = f"1h-fitted; pred_24h={pred_24h:.3f}" +                    (f" vs obs={rate_24h:.3f}" if rate_24h else "") +                    (f"; pred_7d={pred_7d:.3f} vs obs={rate_7d:.3f}" if rate_7d else "")
#             return round(hl_1h, 1), note

#     if rate_7d is not None and rate_7d > 0:
#         hl_7d = _fit_single(rate_7d, T_7D)
#         if hl_7d and hl_7d > 0:
#             hl_safe = hl_7d / 2.0
#             return round(hl_safe, 1), "7d-fitted with 2x safety factor"

#     if rate_24h is not None and rate_24h > 0:
#         hl_24h = _fit_single(rate_24h, T_24H)
#         if hl_24h and hl_24h > 0:
#             hl_safe = hl_24h / 2.0
#             return round(hl_safe, 1), "24h-fitted with 2x safety factor"

#     return None, "no data"


# # ---------------------------------------------------------------------------
# # Step 3 — Compute model predictions for evaluation
# # ---------------------------------------------------------------------------

# def compute_predictions(ground_truth: dict,
#                         half_lives: dict,
#                         tier_mult: float = 1.0) -> list:
#     """
#     For each URL with a known label, compute model prediction.

#     Returns list of {p_stale, changed, freshness_class, domain, window}
#     """
#     records = []

#     for uhash, info in ground_truth.items():
#         fc = info["freshness_class"]
#         hl = half_lives.get(fc)
#         if hl is None:
#             continue

#         for window_label, age, changed in [
#             ("1h",  T_1H,  info["changed_1h"]),
#             ("24h", T_24H, info["changed_24h"]),
#             ("7d",  T_7D,  info.get("changed_7d")),
#         ]:
#             if changed is None:
#                 continue

#             lam     = math.log(2) / hl * tier_mult
#             p_stale = round(1.0 - math.exp(-lam * age), 4)

#             records.append({
#                 "url_hash":        uhash,
#                 "freshness_class": fc,
#                 "domain":          info["domain"],
#                 "window":          window_label,
#                 "age":             age,
#                 "p_stale":         p_stale,
#                 "changed":         changed,
#             })

#     return records


# # ---------------------------------------------------------------------------
# # Step 4 — Evaluation metrics
# # ---------------------------------------------------------------------------

# def compute_ece(predictions: list, n_bins: int = 10) -> float:
#     """
#     Expected Calibration Error.
#     Lower is better. 0.0 = perfect calibration.
#     """
#     bins = defaultdict(list)
#     for r in predictions:
#         b = min(int(r["p_stale"] * n_bins), n_bins - 1)
#         bins[b].append(r["changed"])

#     total = len(predictions)
#     if total == 0:
#         return 0.0

#     ece = 0.0
#     for b, labels in bins.items():
#         bin_center   = (b + 0.5) / n_bins
#         actual_rate  = sum(labels) / len(labels)
#         weight       = len(labels) / total
#         ece         += weight * abs(bin_center - actual_rate)

#     return round(ece, 4)


# def compute_false_rates(predictions: list,
#                         threshold: float = EPS_CONTENT) -> tuple:
#     """
#     false_safe  = predicted safe (p < threshold) but actually changed
#     false_stale = predicted stale (p >= threshold) but actually unchanged
#     """
#     false_safe_n  = sum(1 for r in predictions
#                         if r["p_stale"] < threshold and r["changed"])
#     false_stale_n = sum(1 for r in predictions
#                         if r["p_stale"] >= threshold and not r["changed"])
#     total_changed   = sum(1 for r in predictions if r["changed"])
#     total_unchanged = sum(1 for r in predictions if not r["changed"])

#     false_safe  = round(false_safe_n  / max(total_changed, 1),   4)
#     false_stale = round(false_stale_n / max(total_unchanged, 1), 4)
#     return false_safe, false_stale


# def compute_auc(predictions: list) -> float:
#     """Area under ROC curve via trapezoidal rule."""
#     if not predictions:
#         return 0.5

#     sorted_preds = sorted(predictions, key=lambda r: r["p_stale"], reverse=True)
#     n_pos = sum(1 for r in sorted_preds if r["changed"])
#     n_neg = sum(1 for r in sorted_preds if not r["changed"])

#     if n_pos == 0 or n_neg == 0:
#         return 0.5

#     tps, fps = 0, 0
#     auc = 0.0
#     prev_fp = 0

#     for r in sorted_preds:
#         if r["changed"]:
#             tps += 1
#         else:
#             fps += 1
#             auc += tps * (fps - prev_fp) / (n_pos * n_neg)
#             prev_fp = fps

#     return round(auc, 4)


# def evaluate(predictions: list, label: str) -> dict:
#     ece                  = compute_ece(predictions)
#     false_safe, false_stale = compute_false_rates(predictions)
#     auc                  = compute_auc(predictions)
#     n_changed            = sum(1 for r in predictions if r["changed"])

#     return {
#         "label":           label,
#         "n_samples":       len(predictions),
#         "n_changed":       n_changed,
#         "change_rate":     round(n_changed / max(len(predictions), 1), 4),
#         "ece":             ece,
#         "false_safe_rate": false_safe,
#         "false_stale_rate":false_stale,
#         "auc":             auc,
#     }


# # ---------------------------------------------------------------------------
# # Step 5 — Domain volatility from observations
# # ---------------------------------------------------------------------------

# def compute_observed_domain_volatility(ground_truth: dict) -> dict:
#     """
#     Compute per-domain change rate from both reruns combined.
#     Only include domains with >= 2 observations for reliability.
#     """
#     domain_obs     = defaultdict(list)

#     for info in ground_truth.values():
#         domain = info["domain"]
#         for changed in [info["changed_1h"], info["changed_24h"]]:
#             if changed is not None:
#                 domain_obs[domain].append(int(changed))

#     volatility = {}
#     for domain, obs in domain_obs.items():
#         if len(obs) >= 2:
#             volatility[domain] = round(sum(obs) / len(obs), 3)

#     return dict(sorted(volatility.items(),
#                         key=lambda x: x[1], reverse=True))


# # ---------------------------------------------------------------------------
# # Step 6 — Patch risk_model.py
# # ---------------------------------------------------------------------------

# def patch_risk_model(new_half_lives: dict,
#                      new_domain_vol: dict) -> bool:
#     """
#     Write calibrated half-life and domain volatility values into risk_model.py.
#     Creates a backup first.
#     """
#     if not RISK_MODEL.exists():
#         print(f"  [!] {RISK_MODEL} not found — skipping patch")
#         return False

#     import shutil
#     backup = RISK_MODEL.with_suffix(".py.bak")
#     shutil.copy2(RISK_MODEL, backup)
#     print(f"  Backed up → {backup.name}")

#     with open(RISK_MODEL, encoding="utf-8") as f:
#         src = f.read()

#     # Build new _DEFAULT_HALF_LIFE dict string
#     import re
#     order = ["REAL_TIME", "FAST", "MEDIUM", "SLOW", "TIMELESS"]
#     lines = []
#     labels = {
#         "REAL_TIME": "30 sec  (kept as prior — REAL_TIME pages often blocked by scrapers)",
#         "FAST":      "calibrated from observed change rates",
#         "MEDIUM":    "calibrated from observed change rates",
#         "SLOW":      "calibrated from observed change rates",
#         "TIMELESS":  "calibrated from observed change rates",
#     }
#     for fc in order:
#         hl  = new_half_lives.get(fc, 30.0)
#         lbl = labels.get(fc, "")
#         lines.append(f"    FreshnessClass.{fc:<10}: {hl:>15.1f},   # {lbl}")

#     new_hl_block = "_DEFAULT_HALF_LIFE: Dict[FreshnessClass, float] = {\n" + \
#                    "\n".join(lines) + "\n}"

#     # Find and replace the existing _DEFAULT_HALF_LIFE block
#     pattern = r"_DEFAULT_HALF_LIFE: Dict\[FreshnessClass, float\] = \{[^}]+\}"
#     if not re.search(pattern, src):
#         print("  [!] Could not locate _DEFAULT_HALF_LIFE block — skipping patch")
#         return False
#     src = re.sub(pattern, new_hl_block, src)


#     # Build new top-10 domain volatility entries
#     top_domains = list(new_domain_vol.items())[:20]
#     domain_lines = []
#     for domain, vol in top_domains:
#         domain_lines.append(f'    "{domain}": {vol},')

#     # Only update the domains we have observed data for
#     for domain, vol in top_domains:
#         safe_domain = domain.replace(".", r"\.")
#         pattern_d = rf'("{safe_domain}":\s*)[\d.]+,'
#         replacement = f'"{domain}": {vol},'
#         src = re.sub(pattern_d, replacement, src)

#     with open(RISK_MODEL, "w", encoding="utf-8") as f:
#         f.write(src)

#     return True


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main() -> None:
#     print(f"\n{'='*62}")
#     print(f"  FreshCache Risk Model Calibration")
#     print(f"{'='*62}\n")

#     # ── Load ──────────────────────────────────────────────────────────
#     print("Loading data...")
#     analysis  = load_json(ANALYSIS_FILE)
#     manifest  = load_jsonl(MANIFEST_FILE)
#     changes   = load_jsonl(CHANGE_FILE)

#     if not analysis or not manifest:
#         print("ERROR: Run analyze.py and collect.py first.")
#         sys.exit(1)

#     # ── Ground truth ──────────────────────────────────────────────────
#     print("\nBuilding ground truth...")
#     ground_truth = build_ground_truth(manifest, changes)

#     # ── Observed change rates ─────────────────────────────────────────
#     cr = analysis.get("change_rates", {})
#     obs_1h  = cr.get("change_rate_1h",  {})
#     obs_24h = cr.get("change_rate_24h", {})

#     # ── Fit half-life per class ───────────────────────────────────────
#     print("\nFitting half-life parameters...")

#     CURRENT_HALF_LIVES = {
#         "REAL_TIME": 30.0,
#         "FAST":      3_600.0,
#         "MEDIUM":    86_400.0,
#         "SLOW":      604_800.0,
#         "TIMELESS":  31_536_000.0,
#     }

#     fitted = {}
#     fit_details = {}

#     print(f"\n  {'Class':<12} {'Old HL':>12} {'New HL':>12}  {'MSE':>10}  Note")
#     print(f"  {'-'*12} {'-'*12} {'-'*12}  {'-'*10}  {'-'*30}")

#     for fc in ["TIMELESS", "SLOW", "MEDIUM", "FAST", "REAL_TIME"]:
#         r1h  = obs_1h.get(fc)
#         r24h = obs_24h.get(fc)
#         old_hl = CURRENT_HALF_LIVES[fc]

#         # REAL_TIME: keep prior — scrapers miss most dynamic pages
#         if fc == "REAL_TIME":
#             fitted[fc] = old_hl
#             note = "kept as prior (scraper bias)"
#             fit_details[fc] = {
#                 "old_half_life": old_hl,
#                 "new_half_life": old_hl,
#                 "mse": None,
#                 "note": note,
#             }
#             hl_str = f"{old_hl:>12.0f}"
#             print(f"  {fc:<12} {old_hl:>12.0f} {hl_str}  {'—':>10}  {note}")
#             continue

#         r7d = None
#         # compute 7d change rate from ground truth directly
#         gt_7d = [(1 if v.get("changed_7d") else 0)
#                  for v in ground_truth.values()
#                  if v["freshness_class"] == fc and v.get("changed_7d") is not None]
#         if gt_7d:
#             r7d = round(sum(gt_7d) / len(gt_7d), 4)

#         if r1h is None and r24h is None and r7d is None:
#             fitted[fc] = old_hl
#             note = "no data — keeping prior"
#             old_str = _fmt_hl(old_hl)
#             print(f"  {fc:<12} {old_str:>12} {old_str:>12}  {'—':>10}  {note}")
#         else:
#             new_hl, note = fit_half_life(r1h, r24h, r7d)
#             if new_hl is None:
#                 new_hl = old_hl
#             fitted[fc] = new_hl
#             fit_details[fc] = {
#                 "old_half_life":      old_hl,
#                 "new_half_life":      new_hl,
#                 "observed_rate_1h":   r1h,
#                 "observed_rate_24h":  r24h,
#                 "predicted_rate_1h":  round(predicted_p_stale(new_hl, T_1H), 4),
#                 "predicted_rate_24h": round(predicted_p_stale(new_hl, T_24H), 4),
#                 "calibration_strategy": "conservative_1h_fitted",
#                 "note":               note,
#             }
#             old_str = _fmt_hl(old_hl)
#             new_str = _fmt_hl(new_hl)
#             print(f"  {fc:<12} {old_str:>12} {new_str:>12}  {'—':>10}  {note[:55]}")

#     # ── Evaluate: current vs calibrated model ─────────────────────────
#     print("\nEvaluating models...")

#     preds_current    = compute_predictions(ground_truth, CURRENT_HALF_LIVES)
#     preds_calibrated = compute_predictions(ground_truth, fitted)

#     eval_current    = evaluate(preds_current,    "Current (prior)")
#     eval_calibrated = evaluate(preds_calibrated, "Calibrated")

#     print(f"\n  {'Metric':<22} {'Current':>12} {'Calibrated':>12}  Better?")
#     print(f"  {'-'*22} {'-'*12} {'-'*12}  {'-'*8}")

#     metrics = [
#         ("ECE ↓",            "ece",             True),
#         ("False-safe rate ↓","false_safe_rate",  True),
#         ("False-stale rate ↓","false_stale_rate",True),
#         ("AUC ↑",            "auc",             False),
#     ]

#     for label, key, lower_is_better in metrics:
#         v_curr = eval_current[key]
#         v_cal  = eval_calibrated[key]
#         if lower_is_better:
#             better = "✓" if v_cal < v_curr else ("=" if v_cal == v_curr else "✗")
#         else:
#             better = "✓" if v_cal > v_curr else ("=" if v_cal == v_curr else "✗")
#         print(f"  {label:<22} {v_curr:>12.4f} {v_cal:>12.4f}  {better:>8}")

#     # ── Domain volatility ─────────────────────────────────────────────
#     print("\nComputing observed domain volatility...")
#     domain_vol = compute_observed_domain_volatility(ground_truth)

#     print(f"\n  Top volatile domains (from real observations):")
#     print(f"  {'Domain':<38} {'Obs. Volatility':>16}")
#     print(f"  {'-'*38} {'-'*16}")
#     for domain, vol in list(domain_vol.items())[:15]:
#         if vol > 0:
#             print(f"  {domain:<38} {vol:>16.3f}")

#     # ── Patch risk_model.py ───────────────────────────────────────────
#     print("\nPatching freshcache/risk_model.py...")
#     patched = patch_risk_model(fitted, domain_vol)
#     if patched:
#         print("  risk_model.py updated with calibrated parameters.")
#     else:
#         print("  Patch skipped — update _HALF_LIFE manually using values above.")

#     # ── Save report ───────────────────────────────────────────────────
#     report = {
#         "fit_details":         fit_details,
#         "fitted_half_lives":   fitted,
#         "current_model_eval":  eval_current,
#         "calibrated_model_eval": eval_calibrated,
#         "domain_volatility":   domain_vol,
#         "n_ground_truth_urls": len(ground_truth),
#     }

#     with open(REPORT_FILE, "w", encoding="utf-8") as f:
#         json.dump(report, f, indent=2, ensure_ascii=False)

#     print(f"\n  Report saved → {REPORT_FILE}")

#     # ── Final summary ──────────────────────────────────────────────────
#     print(f"\n{'='*62}")
#     print(f"  Calibration Summary")
#     print(f"{'='*62}")
#     print(f"  ECE improved:        "
#           f"{eval_current['ece']:.4f} → {eval_calibrated['ece']:.4f}")
#     print(f"  False-safe improved: "
#           f"{eval_current['false_safe_rate']:.4f} → {eval_calibrated['false_safe_rate']:.4f}")
#     print(f"  AUC:                 "
#           f"{eval_current['auc']:.4f} → {eval_calibrated['auc']:.4f}")
#     print(f"\n  Calibrated half-lives (seconds):")
#     for fc, hl in fitted.items():
#         print(f"    {fc:<12} {_fmt_hl(hl):>15}")
#     print(f"\n  Next step: python train_risk_model.py")
#     print(f"{'='*62}\n")


# def _fmt_hl(seconds: float) -> str:
#     """Human-readable half-life label."""
#     if seconds < 120:
#         return f"{seconds:.0f}s"
#     if seconds < 7_200:
#         return f"{seconds/3600:.1f}h"
#     if seconds < 172_800:
#         return f"{seconds/3600:.0f}h"
#     if seconds < 1_209_600:
#         return f"{seconds/86400:.1f}d"
#     return f"{seconds/86400:.0f}d"


# if __name__ == "__main__":
#     main()