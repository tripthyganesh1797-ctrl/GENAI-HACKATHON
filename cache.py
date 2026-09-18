"""
cache.py — Fast-path semantic cache.

v1 (this file): keyword-overlap matching against a normalised technical_query.
Simple, fully understandable, debuggable by reading printed output — good fit
for a Python-only team.

v2 (upgrade later, optional): swap `_similarity()` to use sentence-transformers
embeddings + cosine similarity for real semantic matching. Everything else
in this file stays the same — that's the whole point of isolating it here.
"""

import json
import os
import re
import time
from typing import Optional

CACHE_FILE = "cache_store.json"
SIMILARITY_THRESHOLD = 0.6  # tune this based on testing


def _normalise(text: str) -> set:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return set(text.split())


def _similarity(query_a: str, query_b: str) -> float:
    """Jaccard similarity over word sets. Cheap, no external dependency."""
    set_a, set_b = _normalise(query_a), _normalise(query_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def _load_store() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r") as f:
        return json.load(f)


def _save_store(store: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(store, f, indent=2)


def get_cached(technical_query: str) -> Optional[dict]:
    """Returns cached response dict if a similar enough query exists, else None."""
    store = _load_store()
    best_match, best_score = None, 0.0
    for cached_query, entry in store.items():
        score = _similarity(technical_query, cached_query)
        if score > best_score:
            best_score, best_match = score, entry
    if best_score >= SIMILARITY_THRESHOLD:
        return best_match
    return None


def set_cached(technical_query: str, response: dict) -> None:
    store = _load_store()
    store[technical_query] = response
    store[technical_query]["_cached_at"] = time.time()
    _save_store(store)
