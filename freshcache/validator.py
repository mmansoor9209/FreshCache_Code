"""
freshcache/validator.py
Lightweight staleness validation (C3).

Signal hierarchy (cheapest → most expensive):
  1. Metadata TTL check   — free, uses age + freshness class
  2. HEAD request         — one round-trip, no body download
  3. Conditional GET      — ETag / Last-Modified based; returns 304 if fresh
  4. UNKNOWN              — escalate to full refetch in core.py
"""

from __future__ import annotations

from typing import Optional

from .models import FreshnessClass, L3Entry, QueryFeatures, ValidationResult

_TIMEOUT = 6  # seconds for any HTTP request

# Safe-reuse windows (seconds) per freshness class.
# If age < 10% of window  → definitely FRESH (no HTTP needed).
# If age > 200% of window → definitely STALE (skip validation).
_SAFE_WINDOW: dict = {
    FreshnessClass.TIMELESS:  30 * 86_400,   # 30 days
    FreshnessClass.SLOW:       7 * 86_400,   #  7 days
    FreshnessClass.MEDIUM:     1 * 86_400,   #  1 day
    FreshnessClass.FAST:       3_600,         #  1 hour
    FreshnessClass.REAL_TIME:  30,            # 30 seconds
}


def _metadata_ttl_check(entry: L3Entry, features: QueryFeatures) -> ValidationResult:
    """
    Free check: decide based purely on age vs. expected safe window.
    Returns FRESH or STALE only when confident; UNKNOWN otherwise.
    """
    age = entry.age()
    window = _SAFE_WINDOW.get(features.freshness_class, 3_600)

    if age < window * 0.10:
        return ValidationResult.FRESH   # Well within the safe window
    if age > window * 2.00:
        return ValidationResult.STALE   # Far past the safe window
    return ValidationResult.UNKNOWN     # Ambiguous — try HTTP


def _head_check(url: str, entry: L3Entry) -> ValidationResult:
    """
    Issue a HEAD request and compare Last-Modified header.
    Returns UNKNOWN if the server doesn't send comparable headers.
    """
    try:
        import requests  # type: ignore
    except ImportError:
        return ValidationResult.UNKNOWN

    try:
        resp = requests.head(url, timeout=_TIMEOUT, allow_redirects=True)
    except Exception:
        return ValidationResult.ERROR

    if resp.status_code not in (200, 204):
        return ValidationResult.UNKNOWN

    server_lm = resp.headers.get("Last-Modified") or resp.headers.get("last-modified")
    if server_lm and entry.last_modified:
        return (ValidationResult.FRESH
                if server_lm == entry.last_modified
                else ValidationResult.STALE)

    return ValidationResult.UNKNOWN


def _conditional_get(url: str, entry: L3Entry) -> ValidationResult:
    """
    Issue a conditional GET using ETag or Last-Modified.
    304 Not Modified → FRESH; 200 → STALE; anything else → UNKNOWN.
    """
    try:
        import requests  # type: ignore
    except ImportError:
        return ValidationResult.UNKNOWN

    headers: dict = {}
    if entry.etag:
        headers["If-None-Match"] = entry.etag
    elif entry.last_modified:
        headers["If-Modified-Since"] = entry.last_modified
    else:
        return ValidationResult.UNKNOWN  # No validator tokens available

    try:
        resp = requests.get(
            url, headers=headers, timeout=_TIMEOUT, allow_redirects=True
        )
    except Exception:
        return ValidationResult.ERROR

    if resp.status_code == 304:
        return ValidationResult.FRESH
    if resp.status_code == 200:
        return ValidationResult.STALE
    return ValidationResult.UNKNOWN


class LightweightValidator:
    """
    Implements the C3 validation pipeline.

    Call validate() with a URL, the cached L3Entry, and the QueryFeatures
    for the incoming query. It walks the signal hierarchy and stops as soon
    as a definitive FRESH or STALE verdict is reached.

    If no HTTP is requested (use_http=False) it only applies the metadata
    TTL heuristic — useful for offline / replay evaluation.
    """

    def validate(
        self,
        url: str,
        entry: L3Entry,
        features: QueryFeatures,
        use_http: bool = True,
    ) -> ValidationResult:
        """
        Walk the validation hierarchy from cheap to expensive.

        Returns:
            FRESH   — safe to reuse the cached content.
            STALE   — must refetch.
            UNKNOWN — validator could not decide; caller decides (typically refetch).
            ERROR   — HTTP error; treat as UNKNOWN.
        """
        # Step 1: Free metadata check
        result = _metadata_ttl_check(entry, features)
        if result in (ValidationResult.FRESH, ValidationResult.STALE):
            return result

        if not use_http:
            return ValidationResult.UNKNOWN

        # Step 2: HEAD request (no body download)
        result = _head_check(url, entry)
        if result in (ValidationResult.FRESH, ValidationResult.STALE):
            return result

        # Step 3: Conditional GET
        return _conditional_get(url, entry)
