#!/usr/bin/env python3
"""
Deterministic high-precision structural guards for the FreshCache L1 gate.

These run ONLINE inside the L1 admission test. They are pure string
operations: no LLM, no embedding, no model of any kind, no network, no
randomness. They are applied to L1 ONLY; L2 and L3 are untouched.

Each guard answers one question about a candidate pair (incoming query,
cached query) and returns True to ADMIT and False to REJECT.

This file is hashed into l1_precision_prereg.json BEFORE the validation grid
runs, so the guard definitions are frozen before any result is seen.
"""
from __future__ import annotations
import re
import unicodedata

# ───────────────────────────── A. numeric / date guard ─────────────────────
# Ordered most-specific first; each pattern yields a canonical string.
_CURRENCY = "$€£¥₩₹"
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_ORDINAL_SUFFIX = re.compile(r"^(\d+)(st|nd|rd|th)$")
_VERSION = re.compile(r"^\d+(?:\.\d+){1,3}$")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PCT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:%|percent|pct)\b")
_MONEY = re.compile(r"[" + _CURRENCY + r"]\s*(\d[\d,]*(?:\.\d+)?)"
                    r"|(\d[\d,]*(?:\.\d+)?)\s*(?:usd|eur|gbp|jpy|krw|inr|"
                    r"dollars?|euros?|pounds?|yen|won|rupees?)\b")
_ISO_DATE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_DMY_DATE = re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b")
_MONTH_DAY_YEAR = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) +
    r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?\b", re.I)
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b")
# Scale words that multiply a bare number; normalised so "5 million" and
# "5,000,000" do not look like different quantities.
_SCALE = {"hundred": 100, "thousand": 10 ** 3, "k": 10 ** 3,
          "million": 10 ** 6, "m": 10 ** 6, "billion": 10 ** 9,
          "bn": 10 ** 9, "b": 10 ** 9, "trillion": 10 ** 12,
          "lakh": 10 ** 5, "crore": 10 ** 7}
_SCALE_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(" + "|".join(sorted(_SCALE, key=len, reverse=True))
    + r")\b", re.I)


def _canon_number(s: str) -> str:
    """'1,200.00' -> '1200'; '3.50' -> '3.5'; keeps integers integral."""
    s = s.replace(",", "").strip()
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else repr(f)


def numeric_tokens(text: str) -> set:
    """Canonical numeric / date / quantity tokens carried by a query.

    Deliberately normalising, so that pure formatting differences do NOT
    create a mismatch:
        '2024'      and 'year 2024'        -> {'Y:2024'}
        '$1,200'    and '1200 dollars'     -> {'MONEY:1200'}
        '5 million' and '5,000,000'        -> {'N:5000000'}
        '3rd'       and '3'                -> {'N:3'}
    while genuine differences survive:
        'top 3' vs 'top 5'  -> {'N:3'} vs {'N:5'}
        'in 2024' vs 'in 2025' -> {'Y:2024'} vs {'Y:2025'}
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    out, consumed = set(), []

    def take(m, tag, val):
        out.add(f"{tag}:{val}")
        consumed.append((m.start(), m.end()))

    for m in _ISO_DATE.finditer(t):
        take(m, "D", f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}")
    for m in _DMY_DATE.finditer(t):
        take(m, "D", f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}")
    for m in _MONTH_DAY_YEAR.finditer(t):
        mo = _MONTHS[m.group(1).lower().rstrip(".")]
        day = int(m.group(2))
        yr = m.group(3)
        take(m, "D", (f"{int(yr):04d}-{mo:02d}-{day:02d}" if yr
                      else f"????-{mo:02d}-{day:02d}"))
    for m in _PCT.finditer(t):
        take(m, "PCT", _canon_number(m.group(1)))
    for m in _MONEY.finditer(t):
        take(m, "MONEY", _canon_number(m.group(1) or m.group(2)))
    for m in _SCALE_RE.finditer(t):
        base = float(m.group(1).replace(",", ""))
        take(m, "N", _canon_number(str(base * _SCALE[m.group(2).lower()])))
    for m in _YEAR.finditer(t):
        take(m, "Y", m.group(1))

    # bare numbers, version strings and ordinals that no earlier rule consumed
    for m in _NUM.finditer(t):
        if any(a <= m.start() < b or a < m.end() <= b for a, b in consumed):
            continue
        raw = m.group(0)
        tail = t[m.end():m.end() + 2]
        if _VERSION.match(raw):
            out.add(f"V:{raw}")
        elif _ORDINAL_SUFFIX.match(raw + tail.strip()):
            out.add(f"N:{_canon_number(raw)}")
        else:
            out.add(f"N:{_canon_number(raw)}")
    return out


def numeric_guard(q1: str, q2: str) -> bool:
    """Admit only if the two queries carry the SAME numeric/date information.

    If neither query carries any, the guard is silent (admits)."""
    a, b = numeric_tokens(q1), numeric_tokens(q2)
    if not a and not b:
        return True
    return a == b


# ───────────────────────────── B. negation guard ───────────────────────────
_NEG_WORDS = {"not", "never", "without", "except", "excluding", "cannot",
              "didn", "doesn", "isn", "wasn", "aren", "weren", "don", "won",
              "can", "hasn", "haven", "hadn", "shouldn", "wouldn", "couldn"}
_NEG_CONTRACTION = re.compile(
    r"\b(?:did|does|do|is|was|are|were|has|have|had|should|would|could|will|can)"
    r"\s*n[’']?t\b|\bcan\s*not\b|\bcannot\b", re.I)
_BARE_NO = re.compile(r"\bno\b(?!\s*[.:#]|\s*\d)", re.I)


def has_negation(text: str) -> bool:
    """Conservative: an explicit negation cue that changes the proposition.

    'no' is counted only when it is a standalone word and not the abbreviation
    for 'number' ('no. 1', 'no 5'), which is the common false positive."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    if _NEG_CONTRACTION.search(t):
        return True
    if _BARE_NO.search(t):
        return True
    toks = set(re.findall(r"[a-z’']+", t))
    return bool(toks & {"not", "never", "without", "except", "excluding",
                        "cannot", "neither", "nor"})


def negation_guard(q1: str, q2: str) -> bool:
    """Admit only if both queries agree on the presence of explicit negation."""
    return has_negation(q1) == has_negation(q2)


# ──────────────────── C. comparative / superlative guard ───────────────────
# Each family is an antonym pair. A query's relation signature is the set of
# (family, pole) markers it carries.
_FAMILIES = {
    "size":     (["largest", "biggest", "greatest", "maximum", "max"],
                 ["smallest", "tiniest", "minimum", "min"]),
    "height":   (["highest", "tallest", "top", "upper"],
                 ["lowest", "shortest", "bottom", "lower"]),
    "order":    (["first", "earliest", "initial", "opening"],
                 ["last", "latest", "final", "closing"]),
    "quantity": (["more", "most", "greater than", "over", "above", "at least"],
                 ["less", "least", "fewer", "under", "below", "at most"]),
    "time":     (["before", "prior to", "preceding", "earlier than"],
                 ["after", "following", "later than", "since"]),
    "age":      (["older", "oldest", "elder"],
                 ["newer", "newest", "younger", "youngest", "recent"]),
    "trend":    (["increase", "increased", "rise", "rose", "growth", "grew",
                  "gain", "up"],
                 ["decrease", "decreased", "fall", "fell", "decline",
                  "declined", "drop", "down", "loss"]),
    "quality":  (["best", "finest", "strongest", "winner", "won"],
                 ["worst", "weakest", "loser", "lost"]),
}
# Phrases where a marker word is not a comparative at all.
_EXCLUDE = [
    "first name", "first lady", "first aid", "first class", "first world",
    "last name", "top level", "top secret", "up to date", "down under",
    "most of", "no. 1",
]


def relation_signature(text: str) -> frozenset:
    t = unicodedata.normalize("NFKC", text or "").lower()
    for ph in _EXCLUDE:
        t = t.replace(ph, " ")
    toks = set(re.findall(r"[a-z]+", t))
    sig = set()
    for fam, (pos, neg) in _FAMILIES.items():
        for pole, words in (("+", pos), ("-", neg)):
            for w in words:
                if (" " in w and w in t) or (" " not in w and w in toks):
                    sig.add(f"{fam}{pole}")
                    break
    return frozenset(sig)


def comparative_guard(q1: str, q2: str) -> bool:
    """Admit only if the two queries carry the SAME comparative/superlative
    relation signature. A differing pole (largest vs smallest) and a
    present-vs-absent relation both count as a difference."""
    return relation_signature(q1) == relation_signature(q2)


# ───────────────────────────── combined ────────────────────────────────────
def structural_guards(q1: str, q2: str) -> bool:
    """All three guards, conjunctive. True = admit."""
    return (numeric_guard(q1, q2)
            and negation_guard(q1, q2)
            and comparative_guard(q1, q2))


def guard_detail(q1: str, q2: str) -> dict:
    """Per-guard verdicts and the evidence, for the per-hit record."""
    return {
        "numeric_tokens_incoming": "|".join(sorted(numeric_tokens(q1))),
        "numeric_tokens_cached": "|".join(sorted(numeric_tokens(q2))),
        "numeric_guard_pass": numeric_guard(q1, q2),
        "negation_incoming": has_negation(q1),
        "negation_cached": has_negation(q2),
        "negation_guard_pass": negation_guard(q1, q2),
        "relation_incoming": "|".join(sorted(relation_signature(q1))),
        "relation_cached": "|".join(sorted(relation_signature(q2))),
        "comparative_guard_pass": comparative_guard(q1, q2),
    }


if __name__ == "__main__":
    CASES = [
        ("Who won in 2024?", "Who won in 2025?", False, "year differs"),
        ("How much did X cost in 2023?", "How much did X cost in 2024?", False, "year differs"),
        ("What is the top 3 teams?", "What is the top 5 teams?", False, "quantity differs"),
        ("Who won in 2024?", "Who won in year 2024?", True, "formatting only"),
        ("What did it cost? $1,200", "What did it cost? 1200 dollars", True, "money formatting"),
        ("How many users? 5 million", "How many users? 5,000,000", True, "scale formatting"),
        ("Who is the CEO?", "Who is the CEO?", True, "identical"),
        ("Who did not attend?", "Who attended?", False, "negation differs"),
        ("Which is the largest state?", "Which is the smallest state?", False, "pole differs"),
        ("Which is the largest state?", "Which state is it?", False, "relation present vs absent"),
        ("What is his first name?", "What is his first name?", True, "excluded phrase"),
        ("Who is the president?", "Who is the president now?", True, "no numerics or relations"),
    ]
    ok = True
    for a, b, want, why in CASES:
        got = structural_guards(a, b)
        flag = "ok " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] admit={got!s:<5} want={want!s:<5} {why:<32} "
              f"{a!r} vs {b!r}")
    print("\n  ALL GUARD SELF-TESTS PASS" if ok else "\n  GUARD SELF-TESTS FAILED")
    raise SystemExit(0 if ok else 1)
