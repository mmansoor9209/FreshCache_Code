import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def encode(text: str) -> np.ndarray:
    return _get_model().encode(text, convert_to_numpy=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def query_similarity(q1: str, q2: str) -> float:
    if not q1.strip() or not q2.strip():
        return 0.0
    return cosine_similarity(encode(q1), encode(q2))

def query_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]

def tokenize(text: str) -> list:
    return text.lower().split()




# """
# freshcache/embeddings.py
# Lightweight TF-IDF cosine similarity — zero external dependencies.
# In production, replace query_similarity() with a proper sentence-encoder.
# """

# from __future__ import annotations

# import hashlib
# import math
# import re
# from collections import Counter
# from typing import Dict, List


# def tokenize(text: str) -> List[str]:
#     """Word tokenizer supporting Korean and English."""
#     return re.findall(r"[a-zA-Z0-9]+|[가-힣]+", text.lower())


# def _tf_vector(tokens: List[str]) -> Dict[str, float]:
#     count = Counter(tokens)
#     total = len(tokens) or 1
#     return {term: freq / total for term, freq in count.items()}


# def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
#     """Cosine similarity between two sparse float vectors."""
#     if not v1 or not v2:
#         return 0.0
#     common = set(v1) & set(v2)
#     if not common:
#         return 0.0
#     dot   = sum(v1[k] * v2[k] for k in common)
#     norm1 = math.sqrt(sum(x * x for x in v1.values()))
#     norm2 = math.sqrt(sum(x * x for x in v2.values()))
#     if norm1 == 0.0 or norm2 == 0.0:
#         return 0.0
#     return dot / (norm1 * norm2)


# def query_similarity(q1: str, q2: str) -> float:
#     """Compute TF-cosine similarity between two query strings."""
#     return cosine_similarity(
#         _tf_vector(tokenize(q1)),
#         _tf_vector(tokenize(q2)),
#     )


# def query_hash(query: str) -> str:
#     """Stable 16-char hex hash for a normalized query string."""
#     normalized = query.strip().lower()
#     return hashlib.sha256(normalized.encode()).hexdigest()[:16]
