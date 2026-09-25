"""
collect.py — FreshCache Real Data Collection Pipeline

Output (written to ./data/):
    queries.jsonl                   — every query with timestamp + freshness label
    url_manifest.jsonl              — every (query, URL) pair with run_id
    change_log.jsonl                — content changes with run_id for filtering
    snapshots/run_00/<hash>.json    — baseline snapshots (first run)
    snapshots/rerun_1h/<hash>.json  — snapshots from 1-hour re-run
    snapshots/rerun_24h/<hash>.json — snapshots from 24-hour re-run

Requirements:
    pip install requests beautifulsoup4

Setup:
    export TAVILY_API_KEY=your_serper_key_here

First run (baseline):
    python collect.py --run-id run_00

Re-run after 1 hour:
    python collect.py --run-id rerun_1h

Re-run after 24 hours:
    python collect.py --run-id rerun_24h
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR      = Path("data")
QUERIES_FILE  = DATA_DIR / "queries.jsonl"
MANIFEST_FILE = DATA_DIR / "url_manifest.jsonl"
CHANGE_FILE   = DATA_DIR / "change_log.jsonl"

BASELINE_RUN_ID = "run_00"
VALID_RUN_IDS   = {"run_00", "rerun_1h", "rerun_6h", "rerun_12h", "rerun_24h", "rerun_48h", "rerun_72h", "rerun_7d"}

FETCH_TIMEOUT      = 20
FETCH_DELAY        = 1.5
SEARCH_DELAY       = 1.0
MAX_URLS_PER_QUERY = 2
MAX_TEXT_CHARS     = 5000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    "Accept-Encoding": "gzip, deflate, br"
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().lower().encode()).hexdigest()[:24]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def canonical_url(url: str) -> str:
    try:
        p = urlparse(url.strip())
        if p.query:
            kept = [q for q in p.query.split("&")
                    if not q.startswith(("utm_", "ref=", "fbclid=", "source="))]
            query = "&".join(kept)
            return f"{p.scheme}://{p.netloc}{p.path}{'?' + query if kept else ''}".lower().rstrip("/")
        return f"{p.scheme}://{p.netloc}{p.path}".lower().rstrip("/")
    except Exception:
        return url.strip().lower()


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def snapshots_dir(run_id: str) -> Path:
    d = DATA_DIR / "snapshots" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_baseline_hashes() -> dict:
    """Load url_hash → content_hash from run_00 snapshots."""
    baseline_dir = DATA_DIR / "snapshots" / BASELINE_RUN_ID
    result = {}
    if not baseline_dir.exists():
        return result
    for snap_file in baseline_dir.glob("*.json"):
        try:
            with open(snap_file, encoding="utf-8") as f:
                data = json.load(f)
            u_hash = data.get("url_hash")
            c_hash = data.get("content_hash")
            if u_hash and c_hash:
                result[u_hash] = c_hash
        except Exception:
            continue
    return result


def load_existing_run_hashes(run_id: str) -> set:
    """Return set of url_hashes already fetched in this specific run."""
    d = DATA_DIR / "snapshots" / run_id
    if not d.exists():
        return set()
    return {p.stem for p in d.glob("*.json")}


def load_baseline_manifest() -> list:
    """Load all run_00 manifest records for reuse in reruns."""
    if not MANIFEST_FILE.exists():
        return []
    records = []
    for line in open(MANIFEST_FILE, encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("run_id") == BASELINE_RUN_ID and r.get("snapshot_available"):
                records.append(r)
        except Exception:
            continue
    return records


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

# _DYNAMIC_PATTERNS = [
#     "ad", "advertisement", "banner", "popup", "cookie", "consent",
#     "trending", "recommend", "related", "sidebar", "widget",
#     "comment", "discussion", "share", "social", "newsletter",
#     "notification", "promo", "sponsor", "footer", "header",
#     "nav", "menu", "breadcrumb", "pagination", "view-count",
#     "viewcount", "hit-count", "hitcount", "timestamp", "byline",
#     "blog_visitor", "blog_count", "pcol", "u_cbox",
# ]

_DYNAMIC_PATTERNS = [
    # Ads and monetisation
    "ad", "ads", "advert", "advertisement", "banner", "sponsor", "promo",
    # Dynamic widgets and overlays
    "sidebar", "widget", "popup", "modal", "overlay",
    # Social and sharing
    "share", "social", "like-count", "likecount",
    "comment-count", "commentcount", "reaction",
    # Navigation and structure
    "nav", "menu", "breadcrumb", "pagination", "toolbar",
    # View and hit counters
    "view-count", "viewcount", "hit-count", "hitcount",
    "read-count", "readcount", "visit", "counter",
    # Timestamps embedded in class/id names
    "timestamp", "byline", "dateline", "pubdate",
    # Recommendation feeds
    "trending", "recommend", "related", "most-read", "mostread",
    "popular", "suggested",
    # Notifications and banners
    "notification", "alert", "toast", "cookie", "consent",
    # Newsletter
    "newsletter", "subscribe",
    # Korean-specific dynamic elements
    "blog_visitor", "blog_count", "pcol", "u_cbox", "ly_visitor",
    # Korean view count without 수 suffix (e.g. "조회 1,234")
    r"조회\s+[\d,]+",
    # Namu.wiki edit request counter (e.g. "9 편집 요청")
    r"\d+\s*편집\s*요청",
    # "Data as of [date/time]" stamp (boxofficemojo, similar sites)
    r"data as of[^\.]{0,80}",
]


_NORMALIZE_PATTERNS = [
    # Edit and update timestamps
    r"last\s+edited[^\.]{0,80}\.",
    r"last\s+modified[^\.]{0,80}\.",
    r"last\s+updated[^\.]{0,80}\.",
    r"last\s+reviewed[^\.]{0,80}\.",
    r"updated\s+on[^\.]{0,60}\.",
    r"published[:\s]+[^\.]{0,60}\.",
    r"modified[:\s]+[^\.]{0,60}\.",
    # Relative timestamps
    r"\d+\s+seconds?\s+ago",
    r"\d+\s+minutes?\s+ago",
    r"\d+\s+hours?\s+ago",
    r"\d+\s+days?\s+ago",
    r"\d+\s+weeks?\s+ago",
    r"\d+\s+months?\s+ago",
    r"\d+\s+years?\s+ago",
    r"just\s+now",
    r"moments?\s+ago",
    # Absolute dates — English
    r"\b(january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}\s+(january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\s+\d{4}\b",
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
    r"\b\d{2}[-/]\d{2}[-/]\d{4}\b",
    r"\b\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\s*(utc|gmt|est|pst|kst|cst)?\b",
    # Korean dates and counters
    r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일",
    r"\d{1,2}시간?\s*전",
    r"\d{1,2}분\s*전",
    r"조회수\s*[\d,]+",
    r"방문자\s*[\d,]+",
    r"댓글\s*[\d,]+",
    # View / hit / engagement counts
    r"view[s]?\s*:?\s*[\d,]+",
    r"hit[s]?\s*:?\s*[\d,]+",
    r"read[s]?\s*:?\s*[\d,]+",
    r"share[s]?\s*:?\s*[\d,]+",
    r"like[s]?\s*:?\s*[\d,]+",
    r"comment[s]?\s*:?\s*[\d,]+",
    r"[\d,]+\s+view[s]?",
    r"[\d,]+\s+share[s]?",
    r"[\d,]+\s+like[s]?",
    r"[\d,]+\s+comment[s]?",
    # Wikipedia-specific
    r"available in \d+ languages?",
    r"\d+\s+languages?",
    r"retrieved\s+\d{1,2}\s+\w+\s+\d{4}",
]
_NORMALIZE_RE = re.compile("|".join(_NORMALIZE_PATTERNS), flags=re.IGNORECASE)


def _is_dynamic_tag(tag) -> bool:
    for attr in ("class", "id", "role"):
        values = tag.get(attr, [])
        if isinstance(values, str):
            values = [values]
        for val in values:
            if any(pat in val.lower() for pat in _DYNAMIC_PATTERNS):
                return True
    return False


def _main_content(soup, url: str) -> str:
    domain = urlparse(url).netloc.lower()

    if "wikipedia.org" in domain:
        node = soup.find(id="mw-content-text")
        if node:
            return node.get_text(separator=" ", strip=True)

    if "britannica.com" in domain:
        node = soup.find("div", class_=lambda c: c and "topic-content" in c)
        if node:
            return node.get_text(separator=" ", strip=True)

    if "blog.naver.com" in domain or "m.blog.naver.com" in domain:
        node = soup.find("div", id="postViewArea") or \
               soup.find("div", class_=lambda c: c and "se-main-container" in (c or ""))
        if node:
            return node.get_text(separator=" ", strip=True)

    # ── NEW RULES ─────────────────────────────────────────────────────────

    if "namu.wiki" in domain:
        # Strip the edit request counter block and recent changes rail
        for tag in soup.find_all(class_=lambda c: c and any(
                p in (c or "") for p in ["edit-request", "recent", "toolbar"])):
            tag.decompose()
        node = soup.find("div", class_=lambda c: c and "wiki-content" in (c or ""))
        if node:
            return node.get_text(separator=" ", strip=True)

    if "apnews.com" in domain:
        # Remove the TOP STORIES rail which injects live headlines into article pages
        for tag in soup.find_all(class_=lambda c: c and any(
                p in (c or "").lower() for p in ["hub", "top-stories", "promo"])):
            tag.decompose()
        node = soup.find("div", class_=lambda c: c and "article" in (c or "").lower())
        if node:
            return node.get_text(separator=" ", strip=True)

    if "newsbytesapp.com" in domain:
        # Strip the "In the news" trending widget
        for tag in soup.find_all(class_=lambda c: c and any(
                p in (c or "").lower() for p in ["trending", "in-the-news", "inthenews"])):
            tag.decompose()

    if "brainly.com" in domain:
        # Target the answer content directly, skip surrounding ads and widgets
        node = soup.find("div", class_=lambda c: c and "brainly-content" in (c or "")) or \
               soup.find("article")
        if node:
            return node.get_text(separator=" ", strip=True)

    return soup.get_text(separator=" ", strip=True)


def normalize_for_hashing(text: str) -> str:
    text = text.lower()
    text = _NORMALIZE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text(html: str, url: str = "") -> tuple:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "svg",
                 "canvas", "form", "header", "footer", "nav", "aside"]):
            tag.decompose()
        for tag in list(soup.find_all(True)):
            try:
                if _is_dynamic_tag(tag):
                    tag.decompose()
            except Exception:
                continue
        raw = _main_content(soup, url)
        display_text = re.sub(r"\s+", " ", raw).strip()[:MAX_TEXT_CHARS]
        hash_text    = normalize_for_hashing(display_text)
        return display_text, hash_text
    except Exception:
        return "", ""


# ---------------------------------------------------------------------------
# Search API (Serper — baseline only)
# ---------------------------------------------------------------------------

def search_tavily(query: str, api_key: str) -> list:
    try:
        headers  = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json={"q": query, "num": MAX_URLS_PER_QUERY},
            timeout=10,
        )
        data = response.json()
        return [
            {
                "url":         r.get("link", ""),
                "title":       r.get("title", ""),
                "snippet":     r.get("snippet", "")[:300],
                "raw_content": "",
            }
            for r in data.get("organic", [])
            if r.get("link")
        ]
    except Exception as e:
        print(f"    [Search ERROR] {e}")
        return []


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_url(url: str) -> Optional[dict]:
    try:
        resp = requests.get(
            url,
            headers=HEADERS,
            timeout=FETCH_TIMEOUT,
            allow_redirects=True,
            verify=False,
        )
        if resp.status_code not in (200, 203, 206):
            print(f"    [Fetch] HTTP {resp.status_code} — {url[:60]}")
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            print(f"    [Fetch] SKIP non-HTML ({content_type[:30]}) — {url[:60]}")
            return None

        # Force UTF-8 re-decode if the library guessed wrong
        if resp.encoding and resp.encoding.upper() not in ("UTF-8", "UTF8"):
            try:
                raw_text = resp.content.decode("utf-8", errors="replace")
            except Exception:
                raw_text = resp.text
        else:
            raw_text = resp.text

        display_text, hash_text = extract_text(resp.text, url)
        if not display_text:
            return None

        c_hash = content_hash(hash_text)
        h      = {k.lower(): v for k, v in resp.headers.items()}

        return {
            "url":            url,
            "canonical_url":  canonical_url(url),
            "url_hash":       url_hash(canonical_url(url)),
            "status_code":    resp.status_code,
            "fetched_at":     now_iso(),
            "final_url":      resp.url,
            "content_hash":   c_hash,
            "content_length": len(display_text),
            "extracted_text": display_text,
            "etag":           h.get("etag"),
            "last_modified":  h.get("last-modified"),
            "content_type":   h.get("content-type", ""),
            "cache_control":  h.get("cache-control", ""),
            "server":         h.get("server", ""),
        }

    except requests.exceptions.Timeout:
        print(f"    [Fetch] TIMEOUT — {url[:60]}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"    [Fetch] CONNECTION ERROR — {url[:60]}")
        return None
    except Exception as e:
        print(f"    [Fetch] ERROR {type(e).__name__} — {url[:60]}")
        return None


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

MAX_CREDITS_PER_KEY = 2400  # rotate before hitting 2500 limit

def collect(api_keys: list, run_id: str) -> None:
    api_key       = api_keys[0]
    key_index     = 0
    key_use_count = 0

    def next_key():
        nonlocal api_key, key_index, key_use_count
        key_index += 1
        if key_index >= len(api_keys):
            print("\nERROR: All API keys exhausted. Add more keys via --api-keys.\n")
            sys.exit(1)
        api_key = api_keys[key_index]
        key_use_count = 0
        print(f"  [KEY ROTATION] Switched to key {key_index+1}/{len(api_keys)}")
    is_baseline = (run_id == BASELINE_RUN_ID)
    is_rerun    = not is_baseline

    DATA_DIR.mkdir(exist_ok=True)
    snap_dir = snapshots_dir(run_id)

    # Load baseline data for reruns
    # Load baseline data for reruns
    baseline_hashes   = load_baseline_hashes() if is_rerun else {}
    baseline_manifest = load_baseline_manifest() if is_rerun else []
    existing_in_run   = load_existing_run_hashes(run_id)

    # Load already-collected query_ids for baseline (skip Serper call)
    collected_query_ids: set = set()
    if is_baseline and MANIFEST_FILE.exists():
        for line in open(MANIFEST_FILE, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("run_id") == BASELINE_RUN_ID:
                    collected_query_ids.add(r["query_id"])
            except Exception:
                continue
        print(f"  Skipping {len(collected_query_ids)} already-collected query_ids")

    if is_rerun and not baseline_hashes:
        print(f"\nERROR: No baseline found at data/snapshots/{BASELINE_RUN_ID}/")
        print(f"  Run 'python collect.py --run-id run_00' first.\n")
        sys.exit(1)

    # Load queries from queries.jsonl (built by build_queries.py)
    if not QUERIES_FILE.exists():
        print(f"\nERROR: {QUERIES_FILE} not found. Run build_queries.py first.\n")
        sys.exit(1)

    SEED_QUERIES  = [json.loads(l) for l in open(QUERIES_FILE, encoding="utf-8")]
    total_queries = len(SEED_QUERIES)
    total_fetches = 0
    total_changes = 0
    total_errors  = 0

    print(f"\n{'='*60}")
    print(f"  FreshCache Data Collection")
    print(f"  Run ID : {run_id}")
    print(f"  Mode   : {'BASELINE' if is_baseline else 'RE-RUN — comparing against run_00'}")
    print(f"  Queries: {total_queries}")
    print(f"  Output : {snap_dir.resolve()}")
    print(f"{'='*60}\n")

    for i, q_rec in enumerate(SEED_QUERIES, 1):
        query    = q_rec["query"]
        fc_class = q_rec["freshness_class"]
        lang     = q_rec["language"]

        print(f"[{i:02d}/{total_queries}] [{fc_class:<10}] {query}")

        # Search — baseline only; reruns reuse URLs from run_00 manifest
        if is_baseline:
            if q_rec.get("query_id") in collected_query_ids:
                print(f"    [SKIP] Already collected in run_00")
                continue
            if key_use_count >= MAX_CREDITS_PER_KEY:
                next_key()
            search_results = search_tavily(query, api_key)
            key_use_count += 1
            if not search_results:
                print(f"    [!] No search results — skipping")
                total_errors += 1
                time.sleep(SEARCH_DELAY)
                continue
            print(f"    → {len(search_results)} URLs from search")
            time.sleep(SEARCH_DELAY)
        else:
            search_results = [
                {
                    "url":         r["url"],
                    "title":       r.get("title", ""),
                    "snippet":     r.get("snippet", ""),
                    "raw_content": "",
                }
                for r in baseline_manifest
                if r.get("query") == query and r.get("snapshot_available")
            ]
            if not search_results:
                print(f"    [!] No baseline URLs found — skipping")
                continue
            print(f"    → {len(search_results)} URLs from run_00 manifest")

        # Fetch each URL
        for rank, sr in enumerate(search_results, 1):
            url    = sr["url"]
            c_url  = canonical_url(url)
            u_hash = url_hash(c_url)

            manifest_record = {
                "run_id":              run_id,
                "query_id":            q_rec.get("query_id", f"q_{i:06d}"),
                "query":               query,
                "freshness_class":     fc_class,
                "retrieval_timestamp": now_iso(),
                "rank":                rank,
                "url":                 url,
                "canonical_url":       c_url,
                "url_hash":            u_hash,
                "title":               sr.get("title", ""),
                "snippet":             sr.get("snippet", ""),
                "domain":              urlparse(url).netloc.lower().removeprefix("www."),
                "snapshot_available":  False,
                "content_hash":        None,
                "changed":             None,
            }

            # Skip if already fetched in this run
            if u_hash in existing_in_run:
                print(f"    [{rank}] SKIP (already in {run_id}) — {url[:50]}")
                manifest_record["snapshot_available"] = True
                append_jsonl(MANIFEST_FILE, manifest_record)
                continue

            print(f"    [{rank}] Fetching — {url[:50]}")
            fetch_result = fetch_url(url)
            time.sleep(FETCH_DELAY)
            if fetch_result is None:
                total_errors += 1
                append_jsonl(MANIFEST_FILE, manifest_record)
                continue

            new_hash = fetch_result["content_hash"]

            # Change detection — compare against run_00 baseline
            if is_rerun and u_hash in baseline_hashes:
                old_hash = baseline_hashes[u_hash]
                changed  = (old_hash != new_hash)
                manifest_record["changed"] = changed
                if changed:
                    append_jsonl(CHANGE_FILE, {
                        "run_id":          run_id,
                        "url":             url,
                        "url_hash":        u_hash,
                        "domain":          manifest_record["domain"],
                        "freshness_class": fc_class,
                        "detected_at":     now_iso(),
                        "baseline_hash":   old_hash,
                        "new_hash":        new_hash,
                    })
                    total_changes += 1
                    print(f"         *** CONTENT CHANGED ***")
                else:
                    print(f"         (no change)")

            # Save snapshot
            snap_path = snap_dir / f"{u_hash}.json"
            fetch_result["run_id"] = run_id
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(fetch_result, f, ensure_ascii=False, indent=2)

            manifest_record["snapshot_available"] = True
            manifest_record["content_hash"]       = new_hash
            manifest_record["etag"]               = fetch_result.get("etag")
            manifest_record["last_modified"]       = fetch_result.get("last_modified")
            manifest_record["content_length"]      = fetch_result.get("content_length")
            manifest_record["snapshot_path"]       = str(snap_path)

            append_jsonl(MANIFEST_FILE, manifest_record)
            existing_in_run.add(u_hash)
            total_fetches += 1

    # Summary
    snap_count = len(list(snap_dir.glob("*.json")))
    print(f"\n{'='*60}")
    print(f"  Run complete : {run_id}")
    print(f"  Queries      : {total_queries}")
    print(f"  URLs fetched : {total_fetches}")
    print(f"  Errors       : {total_errors}")
    if is_rerun:
        print(f"  Changes vs run_00 : {total_changes}")
    print(f"\n  Snapshots : data/snapshots/{run_id}/ ({snap_count} files)")
    print(f"  Manifest  : {MANIFEST_FILE}")
    if is_rerun and total_changes > 0:
        print(f"  Changes   : {CHANGE_FILE}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreshCache data collection")
    parser.add_argument(
        "--run-id",
        type=str,
        default=BASELINE_RUN_ID,
        choices=list(VALID_RUN_IDS),
        help="run_00 = baseline, rerun_1h = after 1 hour, rerun_24h = after 24 hours",
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        default="",
        help="Comma-separated Serper API keys. Rotates after 2400 uses each.",
    )
    args = parser.parse_args()

    # Support multiple API keys via --api-keys or TAVILY_API_KEY env var
    if args.api_keys:
        api_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    else:
        single = os.environ.get("TAVILY_API_KEY", "").strip()
        if not single:
            print("\nERROR: Provide API keys via --api-keys key1,key2 or TAVILY_API_KEY env var.\n")
            sys.exit(1)
        api_keys = [single]

    collect(api_keys=api_keys, run_id=args.run_id)