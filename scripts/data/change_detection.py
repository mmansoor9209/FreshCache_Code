"""
change_detection.py

Re-applies normalize_for_hashing() to all stored snapshots,
recomputes content hashes, and rebuilds change_log.jsonl from scratch.

Includes domain volatility filtering:
URLs whose domain volatility is too high for their assigned query class
are excluded from class-level calibration counts. This removes search
quality noise automatically regardless of dataset size.

Run after any change to the web_collection.py normalization. No re-fetching needed.

Usage:
    python change_detection.py
"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys; sys.path.insert(0, str(REPO_ROOT)); import json; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from web_collection import normalize_for_hashing, content_hash, url_hash, canonical_url
from freshcache.risk_model import _DEFAULT_DOMAIN_VOLATILITY, is_noise_url, CLASS_VOL_CEILING

DATA_DIR    = (DATA_ROOT)
SNAP_DIR    = DATA_DIR / "snapshots"
MANIFEST    = DATA_DIR / "url_manifest.jsonl"
CHANGE_LOG  = DATA_DIR / "change_log.jsonl"
BASELINE_ID = "run_00"

RERUN_IDS = [
    "rerun_1h", "rerun_12h",
    "rerun_24h", "rerun_7d",
]

# ── Domain volatility ceiling per freshness class ──────────────────────────
# If a URL's domain volatility exceeds this ceiling for its assigned class,
# the URL is search quality noise and is excluded from calibration counts.
# These thresholds are intentionally generous to avoid over-filtering.
# CLASS_VOL_CEILING = {
#     "TIMELESS":  0.25,
#     "SLOW":      0.50,
#     "MEDIUM":    0.70,
#     "FAST":      0.90,
#     "REAL_TIME": 1.00,
# }

# DEFAULT_DOMAIN_VOL = 0.40   # fallback for unknown domains


# def get_domain_volatility(domain: str) -> float:
#     """Look up domain volatility from the risk model's prior table."""
#     if not domain:
#         return DEFAULT_DOMAIN_VOL
#     domain = domain.lower().removeprefix("www.")
#     if domain in _DEFAULT_DOMAIN_VOLATILITY:
#         return _DEFAULT_DOMAIN_VOLATILITY[domain]
# Partial match (e.g. subdomain of a known domain)
#     for known, vol in _DEFAULT_DOMAIN_VOLATILITY.items():
#         if domain.endswith("." + known):
#             return vol
#     return DEFAULT_DOMAIN_VOL

def get_domain_volatility(domain: str) -> float:
    domain = (domain or "").lower().removeprefix("www.")
    if domain in _DEFAULT_DOMAIN_VOLATILITY:
        return _DEFAULT_DOMAIN_VOLATILITY[domain]
    for known, vol in _DEFAULT_DOMAIN_VOLATILITY.items():
        if domain.endswith("." + known):
            return vol
    return 0.40

# def is_noise_url(domain: str, freshness_class: str) -> bool:
#     """
#     Return True if this URL's domain volatility is too high for its
#     assigned query class. These are search quality mismatches and should
#     not affect class-level calibration.
#     """
#     vol     = get_domain_volatility(domain)
#     ceiling = CLASS_VOL_CEILING.get(freshness_class, 0.50)
#     return vol > ceiling


def recompute_hash(snap_path: Path) -> str | None:
    """Re-normalise stored extracted_text and recompute content hash."""
    try:
        with open(snap_path, encoding="utf-8") as f:
            data = json.load(f)
        text = data.get("extracted_text", "")
        if not text:
            return None
        new_hash = content_hash(normalize_for_hashing(text))
        # Write updated hash back into snapshot file
        data["content_hash"] = new_hash
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return new_hash
    except Exception as e:
        print(f"  ERROR reading {snap_path.name}: {e}")
        return None


def load_manifest_index(run_id: str) -> dict:
    """Build url_hash → manifest record for a given run_id."""
    index = {}
    if not MANIFEST.exists():
        return index
    for line in MANIFEST.open(encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("run_id") == run_id and r.get("snapshot_available"):
                index[r["url_hash"]] = r
        except Exception:
            continue
    return index


def main():
    print("\n=== Recomputing content hashes (with domain noise filtering) ===\n")

    # ── Step 1: Recompute baseline hashes ─────────────────────────────────
    baseline_dir = SNAP_DIR / BASELINE_ID
    if not baseline_dir.exists():
        print(f"ERROR: {baseline_dir} not found. Run run_00 first.")
        return

    baseline_hashes = {}
    for snap_file in sorted(baseline_dir.glob("*.json")):
        new_hash = recompute_hash(snap_file)
        if new_hash:
            baseline_hashes[snap_file.stem] = new_hash

    print(f"Baseline ({BASELINE_ID}): recomputed {len(baseline_hashes)} hashes\n")

    # ── Step 2: Recompute rerun hashes and build change log ────────────────
    new_change_log  = []
    noise_skipped   = defaultdict(int)   # class → count of noise URLs skipped
    class_totals    = defaultdict(int)   # for summary

    for run_id in RERUN_IDS:
        rerun_dir = SNAP_DIR / run_id
        if not rerun_dir.exists():
            continue

        manifest_index = load_manifest_index(run_id)
        changed = 0
        total   = 0
        noise   = 0

        for snap_file in sorted(rerun_dir.glob("*.json")):
            u_hash   = snap_file.stem
            new_hash = recompute_hash(snap_file)
            if not new_hash:
                continue

            old_hash = baseline_hashes.get(u_hash)
            if old_hash is None:
                continue

            rec              = manifest_index.get(u_hash, {})
            domain           = rec.get("domain", "")
            freshness_class  = rec.get("freshness_class", "")

            # ── Domain noise filter ─────────────────────────────────────
            if freshness_class and is_noise_url(domain, freshness_class):
                noise += 1
                noise_skipped[freshness_class] += 1
                # Still write a noise-flagged record for audit purposes
                # but mark it so calibrate() can optionally skip it
                if old_hash != new_hash:
                    new_change_log.append({
                        "run_id":          run_id,
                        "url":             rec.get("url", ""),
                        "url_hash":        u_hash,
                        "domain":          domain,
                        "freshness_class": freshness_class,
                        "detected_at":     datetime.now(timezone.utc).isoformat(),
                        "baseline_hash":   old_hash,
                        "new_hash":        new_hash,
                        "noise":           True,   # ← flagged, excluded from calibration
                        "domain_vol":      round(get_domain_volatility(domain), 3),
                    })
                continue

            total += 1
            class_totals[freshness_class] += 1

            if old_hash != new_hash:
                changed += 1
                new_change_log.append({
                    "run_id":          run_id,
                    "url":             rec.get("url", ""),
                    "url_hash":        u_hash,
                    "domain":          domain,
                    "freshness_class": freshness_class,
                    "detected_at":     datetime.now(timezone.utc).isoformat(),
                    "baseline_hash":   old_hash,
                    "new_hash":        new_hash,
                    "noise":           False,
                    "domain_vol":      round(get_domain_volatility(domain), 3),
                })

        print(f"{run_id}: {changed}/{total} clean changes  |  {noise} noise URLs skipped")

    # ── Step 3: Write new change_log.jsonl ────────────────────────────────
    with open(CHANGE_LOG, "w", encoding="utf-8") as f:
        for record in new_change_log:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_records = len(new_change_log)
    clean_records = sum(1 for r in new_change_log if not r.get("noise"))
    noise_records = total_records - clean_records

    print(f"\nchange_log.jsonl rebuilt:")
    print(f"  Clean change records : {clean_records}")
    print(f"  Noise records (flagged, not used in calibration): {noise_records}")

    if noise_skipped:
        print(f"\nNoise URLs skipped per class (search quality mismatches):")
        for cls, count in sorted(noise_skipped.items()):
            vol_ceil = CLASS_VOL_CEILING.get(cls, "?")
            print(f"  {cls:<12}: {count} URLs  (domain vol > {vol_ceil})")

    print(f"\nNext: run risk_model.calibrate() to get new half-lives.\n")


if __name__ == "__main__":
    main()