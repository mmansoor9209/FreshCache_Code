"""
vcache_implementation.py — Reference-based reimplementation of vCache's VerifiedDecisionPolicy
for FreshCache-Bench.

Ported against the reference implementation:
    vcache/vcache_policy/strategies/verified.py  (class _Algorithm)
    https://github.com/vcache-project/vCache
and the paper (Schroeder et al., arXiv:2502.03771v4, ICLR 2026),
Algorithms 1-2 and Equations 9-11.

WHY THIS EXISTS
---------------
The vCache baseline in experiment.py (sim_vcache / _vcache_threshold) is a
static similarity-threshold rule, not the vCache policy. Two defects made
the error bound delta inert:

  1. _vcache_threshold iterates candidate thresholds ascending and
     overwrites best_t, so it returns the LARGEST threshold meeting the
     error constraint. At the largest candidate the subset is a single
     observation whose error rate is 0 or 1, so the constraint almost
     always passes and best_t collapses to the maximum observed
     similarity, independent of delta.
  2. Observations are appended on the EXPLOIT branch. Algorithm 1 (lines
     7-9) and the reference (__perform_cache_update, reached only from the
     _Action.EXPLORE branch) append them on EXPLORE. Under the original
     code an entry can only learn once it is already being served from
     cache, so most entries hold zero observations and the threshold falls
     back to the L1_SIM_THRESHOLD floor.

A sweep of delta from 0.01 to 0.90 consequently produced byte-identical
results: 7.22% stale, 47.36% saved, 14,778 L1 hits at every value.

WHAT THE REFERENCE DOES (and this file reproduces)
--------------------------------------------------
  * Fewer than MIN_OBSERVATIONS = 6 observations -> EXPLORE, no fit.
  * Logistic regression of label on [1, s], unregularised
    (penalty=None, lbfgs, tol=1e-8, max_iter=1000, fit_intercept=False
    with an explicit constant column). coef_[0] = (intercept, gamma).
  * gamma = max(gamma, 1e-6); t_hat = clip(-intercept / gamma, 0, 1);
    both rounded to 3 decimals. Similarities rounded to 3 decimals on
    entry and at decision time.
  * var_t: if the observations are perfectly separable, i.e.
    min(s | label=1) > max(s | label=0), var_t is read from an empirical
    variance_map keyed on observation count (ported verbatim, clamped to
    the largest key beyond its range). Otherwise var_t comes from the
    delta method on t = -intercept/gamma using the inverse Fisher
    information.
  * epsilon grid = linspace(1e-6, 1 - 1e-6, 50).
    t'(eps) = clip(t_hat + z_{1-eps} * sqrt(var_t), 0, 1).
    alpha(eps) = (1 - eps) * L(s, t'(eps), gamma).
    tau = min over the grid of 1 - delta / (1 - alpha).   [Eq 11]
  * u ~ Uniform(0,1); EXPLORE iff u <= tau, else EXPLOIT.   [Alg 2]
  * On explore: record (s, c) against the nearest neighbour, and insert
    the new prompt into the cache only when the cached response was
    wrong (reference: `if not should_have_exploited: self.cache.add(...)`;
    paper: Alg 1 lines 10-11).

tau is not clamped below zero, matching the reference. A negative tau
simply means u <= tau never holds, so the policy exploits.

TWO ADAPTATIONS, STATED EXPLICITLY
----------------------------------
(a) Correctness label. vCache defines c(x) = 1 iff r(nn(x)) == r(x):
    would serving the cached response to THIS prompt have been correct.
    In FreshCache-Bench, serving the cached answer to x is correct only
    if both hold:
      (i)  the cached entry answers the same question, i.e. its URL set
           is the one x would itself have retrieved, and
      (ii) that evidence has not gone stale.
    So c(x) = 1 iff entry.url_set == x.url_set and no URL of the entry
    is in the stale set.

    Condition (i) matters. An earlier version of this file defined c(x)
    from staleness alone, which made c a property of the ENTRY rather
    than of the pair (x, entry). Every observation recorded against a
    given entry then carried the same label, the observation set was
    single-class, the logistic regression was unidentifiable, and the
    policy explored on every request: 0 hits, 0% savings, tau = 1.0 at
    every delta. It was also unfair to the baseline in the other
    direction, since an entry with fresh URLs would count as "correct"
    for an arbitrary unrelated query. Label variation within an entry is
    exactly the signal vCache's sigmoid is meant to learn, namely that
    higher similarity predicts response match.

    Note that the reported stale-error metric is unchanged: the exploit
    branch still counts an error iff a served entry has a stale URL,
    which is the criterion experiment.py uses for every other method.
    Condition (i) affects only the labels the policy learns from.

(b) REAL_TIME bypass. Kept from the original sim_vcache: REAL_TIME is
    never cached by any method in this benchmark, so vCache is not
    charged for a class no one caches. This is generous to the baseline.

Everything else follows the reference. In particular there is no
similarity floor: the nearest neighbour is used whatever its similarity,
and the policy alone decides. This is deliberate, and is the main reason
delta now binds.

Run:
    python -u vcache_implementation.py
"""

from __future__ import annotations
import json
import math
import random
import multiprocessing as mp
from pathlib import Path

import numpy as np
import experiment as exp

DATA_DIR = Path("data")
RESULTS_FILE = DATA_DIR / "vcache_results.json"

# vCache Appendix F suggests 0.5% for high-accuracy uses and 2-3% where
# cost matters. We bracket well past both ends so the baseline has every
# opportunity to reach a high hit rate.
DELTA_VALUES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
PAPER_DELTA = 0.10   # the value experiment.py reports vCache at

RANDOM_SEED = 42
N_WORKERS = 7

# Reference: `if len(similarities) < 6 or len(labels) < 6: return EXPLORE`
MIN_OBSERVATIONS = 6

# Reference: np.linspace(1e-6, 1 - 1e-6, 50)
EPSILON_GRID = np.linspace(1e-6, 1 - 1e-6, 50)

# Reference _Algorithm.variance_map, ported verbatim. Used only when the
# observations are perfectly separable, where the MLE covariance is not
# usable because the coefficients diverge.
VARIANCE_MAP = {
    6: 0.035445, 7: 0.028285, 8: 0.026436, 9: 0.021349, 10: 0.019371,
    11: 0.012615, 12: 0.011433, 13: 0.010228, 14: 0.009963, 15: 0.009253,
    16: 0.011674, 17: 0.013015, 18: 0.010897, 19: 0.011841, 20: 0.013081,
    21: 0.010585, 22: 0.014255, 23: 0.012058, 24: 0.013002, 25: 0.011715,
    26: 0.00839, 27: 0.008839, 28: 0.010628, 29: 0.009899, 30: 0.008033,
    31: 0.00457, 32: 0.007335, 33: 0.008932, 34: 0.00729, 35: 0.007445,
    36: 0.00761, 37: 0.011423, 38: 0.011233, 39: 0.006783, 40: 0.005233,
    41: 0.00872, 42: 0.010005, 43: 0.01199, 44: 0.00977, 45: 0.01891,
    46: 0.01513, 47: 0.02109, 48: 0.01531,
}
_VAR_MAP_MAX_KEY = max(VARIANCE_MAP)

_RECORDS: list | None = None
_STALE: set | None = None
_SIM_AGE: float | None = None
_NC_SEARCH: int | None = None


# ---------------------------------------------------------------------------
# norm.ppf without a scipy dependency (Acklam's rational approximation,
# refined once by Halley's method).
# ---------------------------------------------------------------------------

_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -30.0
    if p >= 1.0:
        return 30.0
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
            ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
            ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((_A[0]*r+_A[1])*r+_A[2])*r+_A[3])*r+_A[4])*r+_A[5])*q / \
            (((((_B[0]*r+_B[1])*r+_B[2])*r+_B[3])*r+_B[4])*r+1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


_Z_GRID = np.array([_norm_ppf(1.0 - e) for e in EPSILON_GRID])


# ---------------------------------------------------------------------------
# Eq 9 and Eq 10
# ---------------------------------------------------------------------------

def _likelihood(s: float, t, gamma: float):
    """Eq 9: L(s, t, gamma) = sigmoid(gamma * (s - t)). Vectorised over t."""
    return 1.0 / (1.0 + np.exp(-np.clip(gamma * (s - np.asarray(t)), -35, 35)))


def _fit_logistic(sims: np.ndarray, labels: np.ndarray):
    """
    Unregularised logistic regression of labels on [1, s] by Newton-Raphson
    with step damping, standing in for the reference's lbfgs call.

    Returns (intercept, gamma, cov_beta) or None. cov_beta is the inverse
    Fisher information, used for the delta method.
    """
    X = np.column_stack([np.ones_like(sims), sims])
    y = labels.astype(np.float64)
    b = np.zeros(2, dtype=np.float64)

    for _ in range(1000):
        eta = np.clip(X @ b, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-12, None)
        grad = X.T @ (y - p)
        H = (X.T * w) @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        # Damp the step so separable data does not overflow before the
        # iteration cap is reached.
        norm_step = float(np.max(np.abs(step)))
        if norm_step > 10.0:
            step = step * (10.0 / norm_step)
        b = b + step
        if norm_step < 1e-8:
            break

    if not np.all(np.isfinite(b)):
        return None

    eta = np.clip(X @ b, -35, 35)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1.0 - p), 1e-12, None)
    H = (X.T * w) @ X
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = None

    return float(b[0]), float(b[1]), cov


def _estimate_parameters(sims: np.ndarray, labels: np.ndarray):
    """
    Reference _estimate_parameters. Returns (t_hat, gamma, var_t) or None,
    where None corresponds to the reference's t_hat == -1 sentinel.
    """
    if len(set(labels.tolist())) < 2:
        # sklearn raises on a single class; the reference catches this and
        # returns the sentinel, which maps to EXPLORE.
        return None

    fit = _fit_logistic(sims, labels)
    if fit is None:
        return None
    intercept, gamma, cov = fit

    gamma = max(gamma, 1e-6)
    t_hat = float(np.clip(-intercept / gamma, 0.0, 1.0))

    ones = sims[labels == 1]
    zeros = sims[labels == 0]
    perfect_separation = bool(ones.min() > zeros.max())

    if perfect_separation:
        n = len(sims)
        var_t = VARIANCE_MAP.get(n, VARIANCE_MAP[_VAR_MAP_MAX_KEY])
    else:
        if cov is None:
            return None
        grad = np.array([-1.0 / gamma, intercept / (gamma ** 2)])
        var_t = float(grad @ cov @ grad)
        if not np.isfinite(var_t) or var_t < 0:
            return None

    return round(t_hat, 3), round(gamma, 3), var_t


def _get_tau(s: float, t_hat: float, gamma: float, var_t: float,
             delta: float) -> float:
    """
    Eq 11 via the reference's _get_tau: sweep the epsilon grid, take the
    minimum tau. Not clamped below zero, matching the reference.
    """
    sd = math.sqrt(max(var_t, 0.0))
    t_primes = np.clip(t_hat + _Z_GRID * sd, 0.0, 1.0)
    likelihoods = _likelihood(s, t_primes, gamma)
    alpha = (1.0 - EPSILON_GRID) * likelihoods
    taus = 1.0 - delta / (1.0 - alpha)
    return float(np.min(taus))


def _select_action_explore(s: float, observations: list, delta: float,
                           rng: random.Random) -> tuple:
    """
    Reference select_action. Returns (explore: bool, fitted: bool, tau).
    fitted reports whether a sigmoid fit was actually used, for diagnostics.
    """
    s = round(s, 3)
    if len(observations) < MIN_OBSERVATIONS:
        return True, False, 1.0

    sims = np.array([o[0] for o in observations], dtype=np.float64)
    labels = np.array([o[1] for o in observations], dtype=np.int64)

    est = _estimate_parameters(sims, labels)
    if est is None:
        return True, False, 1.0

    t_hat, gamma, var_t = est
    tau = _get_tau(s, t_hat, gamma, var_t, delta)
    u = rng.uniform(0, 1)
    return (u <= tau), True, tau


# ---------------------------------------------------------------------------
# Algorithm 1, adapted to the FreshCache pipeline
# ---------------------------------------------------------------------------

def sim_vcache_reference(records: list, stale_urls: set, sim_age: float,
                        delta: float = 0.10, seed: int = RANDOM_SEED) -> dict:
    """
    Latency accounting and the returned dict match sim_vcache in
    experiment.py so results are directly comparable.
    """
    rng = random.Random(seed)

    n = len(records)
    cache = []            # D: {query, fc, url_hashes, observations}
    search_calls = 0
    fetches = 0
    l1_hits = 0
    latency_list = []
    stale_errors = 0
    base_search_calls = 0
    base_stale_errors = 0
    base_n = 0

    n_decisions = 0
    n_fitted = 0
    tau_sum = 0.0

    for r in records:
        q = r["query"]
        fc = r["freshness_class"]
        urls = r["urls"]
        url_hashes = [u["url_hash"] for u in urls]
        url_set_new = frozenset(url_hashes)
        is_base = not r.get("is_paraphrase")
        if is_base:
            base_n += 1

        def _full_pipeline(add_entry: bool) -> float:
            nonlocal search_calls, fetches, base_search_calls
            search_calls += 1
            if is_base:
                base_search_calls += 1
            nf = len(urls)
            fetches += nf
            lat = (exp.LATENCY["search_api"] +
                   nf * exp.LATENCY["web_fetch"] +
                   exp.LATENCY["llm_generate"])
            if add_entry:
                cache.append({"query": q, "fc": fc,
                              "url_hashes": url_hashes,
                              "url_set": url_set_new,
                              "observations": []})
            return lat

        if fc == "REAL_TIME":
            latency_list.append(_full_pipeline(add_entry=True))
            continue

        # Alg 1 lines 1-3
        best_sim = -1.0
        nn_entry = None
        for entry in cache:
            sim = exp.cosine_sim(q, entry["query"])
            if sim > best_sim:
                best_sim = sim
                nn_entry = entry

        if nn_entry is None:
            latency_list.append(_full_pipeline(add_entry=True))
            continue

        explore, fitted, tau = _select_action_explore(
            best_sim, nn_entry["observations"], delta, rng)
        n_decisions += 1
        tau_sum += tau
        if fitted:
            n_fitted += 1

        if not explore:
            # Alg 1 line 5: exploit
            l1_hits += 1
            lat = exp.LATENCY["l1_lookup"] + exp.LATENCY["llm_generate"]
            if any(uh in stale_urls for uh in nn_entry["url_hashes"]):
                stale_errors += 1
                if is_base:
                    base_stale_errors += 1
            latency_list.append(lat)
            continue

        # Alg 1 lines 7-12: explore
        entry_stale = any(uh in stale_urls for uh in nn_entry["url_hashes"])
        same_question = (nn_entry["url_set"] == url_set_new)
        c = 1 if (same_question and not entry_stale) else 0
        nn_entry["observations"].append((round(best_sim, 3), c))
        latency_list.append(_full_pipeline(add_entry=(c == 0)))

    base_lats = [lat for r, lat in zip(records, latency_list)
                 if not r.get("is_paraphrase")]
    ordered = sorted(latency_list)
    p50 = ordered[int(n * 0.50)]
    p95 = ordered[int(n * 0.95)]
    p50_base = sorted(base_lats)[int(len(base_lats) * 0.50)] if base_lats else p50

    return {
        "search_calls":           search_calls,
        "fetches":                fetches,
        "l1_hits":                l1_hits,
        "l2_hits":                0,
        "l3_hits":                0,
        "validated_reuses":       0,
        "total_latency_ms":       sum(latency_list),
        "p50_ms":                 p50,
        "p95_ms":                 p95,
        "p50_base_ms":            p50_base,
        "cache_stale_errors":     stale_errors,
        "cache_stale_error_rate": stale_errors / max(l1_hits, 1),
        "n_queries":              n,
        "base_search_calls":      base_search_calls,
        "base_stale_errors":      base_stale_errors,
        "base_n":                 base_n,
        "cache_entries":          len(cache),
        "decisions":              n_decisions,
        "decisions_with_fit":     n_fitted,
        "mean_tau":               (tau_sum / n_decisions) if n_decisions else 0.0,
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def _one_delta(delta: float) -> dict:
    m = sim_vcache_reference(_RECORDS, _STALE, _SIM_AGE, delta=delta)
    saved = (1 - m["search_calls"] / max(_NC_SEARCH, 1)) * 100
    return {
        "scale":              delta,
        "swept_param":        "DELTA",
        "search_calls":       m["search_calls"],
        "n_queries":          m["n_queries"],
        "stale_error_rate":   m["cache_stale_error_rate"],
        "search_saved_pct":   round(saved, 2),
        "l1_hits":            m["l1_hits"],
        "l2_hits":            0,
        "l3_hits":            0,
        "cache_entries":      m["cache_entries"],
        "decisions":          m["decisions"],
        "decisions_with_fit": m["decisions_with_fit"],
        "mean_tau":           round(m["mean_tau"], 4),
        "overall_error_rate": m["cache_stale_errors"] / max(m["n_queries"], 1),
    }


def main() -> None:
    global _RECORDS, _STALE, _SIM_AGE, _NC_SEARCH

    print(f"\n{'='*84}")
    print(f"  vCache (Schroeder et al., arXiv:2502.03771v4) at t=24h")
    print(f"  Ported from vcache/vcache_policy/strategies/verified.py")
    print(f"  delta in {DELTA_VALUES}   (experiment.py reports delta={PAPER_DELTA})")
    print(f"{'='*84}\n", flush=True)

    queries     = exp.load_jsonl(exp.QUERIES_FILE)
    manifest    = exp.load_jsonl(exp.MANIFEST_FILE)
    changes     = exp.load_jsonl(exp.CHANGE_FILE)
    paraphrases = exp.load_jsonl(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else []

    if not queries or not manifest:
        print("ERROR: Run collect.py and build_queries.py first.")
        return

    records = exp.build_query_records(queries, manifest, paraphrases)

    print("  Loading cached similarity matrix (must already exist)...", flush=True)
    exp.precompute_similarity_matrix(records)
    print("  Matrix ready.\n", flush=True)

    stale_map = exp.build_stale_sets(changes)
    stale_24h = stale_map["rerun_24h"]
    sim_age_24h = exp.SIM_AGES["24h"]

    nc = exp.sim_nocache(records, stale_24h, sim_age_24h)

    _RECORDS   = records
    _STALE     = stale_24h
    _SIM_AGE   = sim_age_24h
    _NC_SEARCH = nc["search_calls"]

    print(f"  Stale URL set at t=24h: {len(stale_24h)} URLs\n", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(N_WORKERS, len(DELTA_VALUES))) as pool:
        rows = pool.map(_one_delta, DELTA_VALUES)

    rows.sort(key=lambda d: d["scale"])

    print(f"  {'delta':>7} {'Stale':>9} {'Saved':>9} {'L1 hits':>10} "
          f"{'Entries':>9} {'Fitted':>9} {'mean tau':>9}")
    print(f"  {'-'*7} {'-'*9} {'-'*9} {'-'*10} {'-'*9} {'-'*9} {'-'*9}")
    for d in rows:
        mark = "  <- paper's delta" if d["scale"] == PAPER_DELTA else ""
        fitted_pct = 100.0 * d["decisions_with_fit"] / max(d["decisions"], 1)
        print(f"  {d['scale']:>7.2f} {d['stale_error_rate']*100:>8.2f}% "
              f"{d['search_saved_pct']:>8.2f}% {d['l1_hits']:>10,} "
              f"{d['cache_entries']:>9,} {fitted_pct:>8.1f}% "
              f"{d['mean_tau']:>9.3f}{mark}", flush=True)

    results = {"t=24h": {"vCache": rows}}
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # vCache bounds FP / n over all queries (Section 5, Metrics), not the
    # error rate among cache hits, since an explore is correct by design.
    print(f"\n  Guarantee check (vCache bounds FP / n over ALL queries):")
    for d in rows:
        ok = d["overall_error_rate"] <= d["scale"]
        print(f"    delta={d['scale']:.2f}: overall error "
              f"{d['overall_error_rate']*100:.3f}%  -> "
              f"{'within bound' if ok else 'EXCEEDS BOUND'}")

    # Appendix F's claim, restated against this implementation.
    fc_stale, fc_saved = 3.35, 98.39
    dominating = [d for d in rows
                  if d["stale_error_rate"] * 100 <= fc_stale
                  and d["search_saved_pct"] >= fc_saved]
    print()
    if dominating:
        print("  DOMINANCE: these vCache settings match or beat FreshCache_Full")
        print("  on BOTH axes. Appendix F must be revised.")
        for d in dominating:
            print(f"    delta={d['scale']}: {d['stale_error_rate']*100:.2f}% stale, "
                  f"{d['search_saved_pct']:.2f}% saved")
    else:
        print("  DOMINANCE: no vCache setting beats FreshCache_Full on")
        print("  both axes. Appendix F's claim holds against the real policy.")

    print(f"\n  Saved: {RESULTS_FILE}")
    print(f"{'='*84}\n")


if __name__ == "__main__":
    main()