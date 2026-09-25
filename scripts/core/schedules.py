#!/usr/bin/env python3
"""
v16_exp12/schedules.py — request arrival schedules for the mixed-age robustness
study.

Every schedule places the SAME 31,201 requests on the same 7-day horizon and
differs only in when each one arrives. Cluster membership is never broken and no
request is added or dropped, so the workload composition is identical across
schedules; only the ordering and the resulting cache-entry ages change.

  zipf_uniform   the current published schedule, copied verbatim from
                 v9/mixed_age_v2.py: a cluster's base time is a Zipf-weighted
                 mixture of two uniforms over [0, H/2], and its remaining members
                 land uniformly in [base, H]. Popular clusters start earlier.
  uniform        every request drawn independently uniform on [0, H]. No
                 clustering in time at all: repeats are maximally separated, so
                 cache entries are old when they are reused.
  bursty_poisson cluster onsets are a Poisson process over [0, H]; a cluster's
                 members then arrive in a tight burst with exponential gaps
                 (mean BURST_MEAN). Repeats are maximally close together, so
                 cache entries are young when they are reused.

The two new schedules bracket the published one: `uniform` is the worst case for
a cache (oldest entries at reuse) and `bursty_poisson` the best (youngest).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

HORIZON = 7*86_400
ZIPF_A = 1.0
BURST_MEAN = 900.0          # 15 min mean gap inside a burst
CLUSTER_RATE_SCALE = 1.0    # Poisson onsets spread over the whole horizon
SCHEDULES = ["zipf_uniform", "uniform", "bursty_poisson"]


def _clusters(records):
    by = defaultdict(list)
    for r in records:
        by[r.get("cluster_id") or r["query_id"]].append(r)
    return [(c, sorted(by[c], key=lambda r: (1 if r.get("is_paraphrase") else 0,
                                             r["query_id"])))
            for c in sorted(by)]


def build_stream(records, schedule="zipf_uniform", seed=42):
    """Return [(arrival_time, record), ...] sorted by time. Deterministic in seed."""
    rng = random.Random(seed)
    cl = _clusters(records)

    if schedule == "zipf_uniform":
        # verbatim from v9/mixed_age_v2.build_stream, seed parameterised
        weights = [1.0/((i+1)**ZIPF_A) for i in range(len(cl))]
        tot = sum(weights)
        stream = []
        for (cid, rows), w in zip(cl, weights):
            base = (rng.uniform(0, HORIZON/2)*(1-w/tot)
                    + rng.uniform(0, HORIZON/2)*(w/tot))
            for i, r in enumerate(rows):
                stream.append((base if i == 0 else rng.uniform(base, HORIZON), r))

    elif schedule == "uniform":
        stream = [(rng.uniform(0, HORIZON), r) for _, rows in cl for r in rows]

    elif schedule == "bursty_poisson":
        n_cl = len(cl)
        # Poisson onsets: exponential gaps with mean H/n_cl
        gap_mean = HORIZON/max(n_cl, 1)*CLUSTER_RATE_SCALE
        t = 0.0
        stream = []
        for cid, rows in cl:
            t += rng.expovariate(1.0/gap_mean)
            onset = t if t < HORIZON else rng.uniform(0, HORIZON)
            cur = onset
            for i, r in enumerate(rows):
                if i:
                    cur += rng.expovariate(1.0/BURST_MEAN)
                stream.append((min(cur, HORIZON), r))

    else:
        raise ValueError(schedule)

    stream.sort(key=lambda x: x[0])
    return stream


def describe(stream):
    """Summary statistics a reader can use to tell the schedules apart."""
    import statistics
    ts = [t for t, _ in stream]
    by = defaultdict(list)
    for t, r in stream:
        by[r.get("cluster_id") or r["query_id"]].append(t)
    spans = [max(v)-min(v) for v in by.values() if len(v) > 1]
    gaps = []
    for v in by.values():
        v = sorted(v)
        gaps.extend(v[i+1]-v[i] for i in range(len(v)-1))
    return {"n": len(stream),
            "t_median_h": round(statistics.median(ts)/3600, 2),
            "cluster_span_median_h": round(statistics.median(spans)/3600, 2) if spans else 0.0,
            "intra_cluster_gap_median_h": round(statistics.median(gaps)/3600, 2) if gaps else 0.0,
            "intra_cluster_gap_p95_h": round(sorted(gaps)[int(.95*len(gaps))]/3600, 2) if gaps else 0.0}
