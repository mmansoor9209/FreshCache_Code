"""
generate_paraphrases.py

Generates 4 paraphrases per base query using:
  English : humarin/chatgpt_paraphraser_on_T5_base (~900 MB, CPU ~30 min)
  Korean  : regex-based sentence-ending variation (instant)

Output: data/paraphrase_clusters.jsonl
Run once, then:  rm generate_paraphrases.py

Usage:
    python generate_paraphrases.py
"""

import json
import re
import uuid
from pathlib import Path
from collections import defaultdict

DATA_DIR        = Path("data")
QUERIES_FILE    = DATA_DIR / "queries.jsonl"
MANIFEST_FILE   = DATA_DIR / "url_manifest.jsonl"
PARAPHRASE_FILE = DATA_DIR / "paraphrase_clusters.jsonl"

NUM_PARAPHRASES = 4

# ── Korean sentence-ending variations ─────────────────────────────────────
# Each tuple: (pattern_to_match, list_of_replacements)
KO_ENDINGS = [
    (r"는 무엇입니까\?$",   ["은 뭐입니까?", "에 대해 알려주세요.", "을 설명해 주세요.", "은 무엇인가요?"]),
    (r"은 무엇입니까\?$",   ["는 뭐입니까?", "에 대해 알려주세요.", "를 설명해 주세요.", "은 무엇인가요?"]),
    (r"는 얼마입니까\?$",   ["가 얼마예요?", "의 가격을 알려주세요.", "는 얼마인가요?", "의 현재 가격은 얼마입니까?"]),
    (r"은 얼마입니까\?$",   ["이 얼마예요?", "의 가격을 알려주세요.", "은 얼마인가요?", "의 현재 가격은 얼마입니까?"]),
    (r"는 어떻습니까\?$",   ["은 어때요?", "의 현황을 알려주세요.", "은 어떤가요?", "에 대해 말해 주세요."]),
    (r"은 어떻습니까\?$",   ["는 어때요?", "의 현황을 알려주세요.", "는 어떤가요?", "에 대해 말해 주세요."]),
    (r"는 어디에 위치합니까\?$", ["은 어디 있나요?", "의 위치를 알려주세요.", "은 어디에 있습니까?", "의 위치는 어디입니까?"]),
    (r"란 무엇입니까\?$",   ["이란 뭔가요?", "에 대해 설명해 주세요.", "이 무엇인지 알려주세요.", "의 개념을 설명해 주세요."]),
    (r"의 원리는 무엇입니까\?$", ["는 어떻게 작동합니까?", "의 원리를 설명해 주세요.", "의 원리는 무엇인가요?", "이 어떻게 이루어집니까?"]),
    (r"알려주세요\.$",      ["알려주실 수 있나요?", "설명해 주세요.", "알고 싶습니다.", "말씀해 주세요."]),
    (r"입니까\?$",          ["인가요?", "이에요?", "예요?", "인지 알려주세요."]),
    (r"합니까\?$",          ["하나요?", "해요?", "하는지 알려주세요.", "하는지요?"]),
]


def ko_paraphrase(text: str, n: int = NUM_PARAPHRASES) -> list[str]:
    results = []
    for pattern, replacements in KO_ENDINGS:
        if re.search(pattern, text):
            stem = re.sub(pattern, "", text).rstrip()
            for repl in replacements[:n]:
                candidate = stem + repl if not stem.endswith(" ") else stem.rstrip() + repl
                if candidate != text and candidate not in results:
                    results.append(candidate)
            if len(results) >= n:
                break
    # Fallback: append generic suffix if not enough variations
    fallbacks = [
        f"{text.rstrip('?.')} (알고 싶습니다.)",
        f"{text.rstrip('?.')}에 대한 정보를 알려주세요.",
        f"현재 {text.lstrip('현재 ')}",
        f"최신 {text.lstrip('최신 ')}",
    ]
    for fb in fallbacks:
        if fb != text and fb not in results and len(results) < n:
            results.append(fb)
    return results[:n]


def load_covered_query_ids() -> set:
    """Return query_ids that have at least one URL in run_00."""
    covered = set()
    if not MANIFEST_FILE.exists():
        return covered
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("run_id") == "run_00" and r.get("snapshot_available"):
                covered.add(r["query_id"])
    return covered


def main():
    queries = []
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if not q.get("is_paraphrase"):    # base queries only
                queries.append(q)

    covered = load_covered_query_ids()
    base_queries = [q for q in queries if q["query_id"] in covered]
    print(f"Base queries with URL coverage: {len(base_queries)}")

    en_queries = [q for q in base_queries if q.get("language", "en") == "en"]
    ko_queries = [q for q in base_queries if q.get("language") == "ko"]
    print(f"  English: {len(en_queries)}  Korean: {len(ko_queries)}")

    # ── Korean paraphrases (template-based, instant) ───────────────────────
    print("\nGenerating Korean paraphrases (template-based)...")
    ko_records = []
    ko_skipped = 0
    for q in ko_queries:
        variants = ko_paraphrase(q["query"])
        if not variants:
            ko_skipped += 1
            continue
        for i, variant in enumerate(variants):
            ko_records.append({
                "query_id":      f"p_{q['query_id']}_{i}",
                "cluster_id":    q["query_id"],
                "base_query_id": q["query_id"],
                "query":         variant,
                "freshness_class": q["freshness_class"],
                "language":      "ko",
                "source":        "ko_paraphrase_template",
                "is_paraphrase": True,
            })
    print(f"  Korean records generated: {len(ko_records)}  (skipped: {ko_skipped})")

    # ── English paraphrases (T5 model) ────────────────────────────────────
    print("\nLoading English paraphrase model (humarin/chatgpt_paraphraser_on_T5_base)...")
    print("First run downloads ~900 MB — this is a one-time download.\n")

    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch

    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer  = T5Tokenizer.from_pretrained(model_name)
    model      = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"  Running on: {device}")

    def en_paraphrase(text: str, n: int = NUM_PARAPHRASES) -> list[str]:
        input_ids = tokenizer(
            f"paraphrase: {text}",
            return_tensors="pt",
            padding="longest",
            max_length=128,
            truncation=True,
        ).input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=128,
                num_return_sequences=n,
                do_sample=True,
                temperature=1.5,
                top_k=120,
                top_p=0.98,
                no_repeat_ngram_size=2,
                repetition_penalty=2.5,
            )
        results = []
        for o in outputs:
            decoded = tokenizer.decode(o, skip_special_tokens=True).strip()
            if decoded and decoded.lower() != text.lower() and decoded not in results:
                results.append(decoded)
        return results[:n]

    en_records  = []
    en_skipped  = 0
    total_en    = len(en_queries)

    for idx, q in enumerate(en_queries, 1):
        if idx % 50 == 0 or idx == 1:
            print(f"  [{idx}/{total_en}] Processing: {q['query'][:60]}...")
        try:
            variants = en_paraphrase(q["query"])
            if not variants:
                en_skipped += 1
                continue
            for i, variant in enumerate(variants):
                en_records.append({
                    "query_id":      f"p_{q['query_id']}_{i}",
                    "cluster_id":    q["query_id"],
                    "base_query_id": q["query_id"],
                    "query":         variant,
                    "freshness_class": q["freshness_class"],
                    "language":      "en",
                    "source":        "en_paraphrase_t5",
                    "is_paraphrase": True,
                })
        except Exception as e:
            print(f"  WARN: failed for query {q['query_id']}: {e}")
            en_skipped += 1

    print(f"\n  English records generated: {len(en_records)}  (skipped: {en_skipped})")

    # ── Write output ───────────────────────────────────────────────────────
    all_paraphrases = ko_records + en_records
    with open(PARAPHRASE_FILE, "w", encoding="utf-8") as f:
        for r in all_paraphrases:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTotal paraphrase records : {len(all_paraphrases)}")
    print(f"Output                   : {PARAPHRASE_FILE}")
    print(f"\nDelete this file when done:  rm generate_paraphrases.py")


if __name__ == "__main__":
    main()