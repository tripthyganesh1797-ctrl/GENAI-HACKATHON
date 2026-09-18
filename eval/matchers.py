"""
matchers.py — Three deeplink-matching architectures, so the ablation study
in metrics.md is backed by real, run code rather than opinions.

Variant A (rules-based): deterministic fuzzy keyword matching, no ranking
    model -- deeplink_matching.RulesDeeplinkIndex. This is also the
    service's own fallback if index construction ever fails.
Variant B (hybrid): BM25 + dense (sentence-transformers if available,
    else TF-IDF/SVD) -- deeplink_matching.HybridDeeplinkIndex. This is
    what the shipped pipeline.py actually uses by default.
Variant C (full-LLM): asks the LLM itself to pick the best catalog entry
    by ID, given the full catalog. Most flexible, highest cost and
    latency -- and not used in the shipped pipeline for that reason.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import call_llm_json  # noqa: E402
from deeplink_matching import RulesDeeplinkIndex, HybridDeeplinkIndex  # noqa: E402

_rules_index_cache = {}
_hybrid_index_cache = {}


def _index_key(catalog):
    return id(catalog)


def match_rules_based(query: str, catalog: list) -> str:
    key = _index_key(catalog)
    if key not in _rules_index_cache:
        _rules_index_cache[key] = RulesDeeplinkIndex(catalog)
    entry, _score = _rules_index_cache[key].best_match(query)
    return entry.deeplink if entry else None


def match_hybrid(query: str, catalog: list) -> str:
    key = _index_key(catalog)
    if key not in _hybrid_index_cache:
        _hybrid_index_cache[key] = HybridDeeplinkIndex(catalog)
    entry, _score = _hybrid_index_cache[key].best_match(query)
    return entry.deeplink if entry else None


def match_llm(query: str, catalog: list) -> str:
    catalog_lines = "\n".join(
        f"{i}: {entry.description} | {entry.message}" for i, entry in enumerate(catalog)
    )
    prompt = f"""Given this user need: "{query}"

Pick the SINGLE best matching entry from this catalog by its number.
Catalog:
{catalog_lines}

Output ONLY valid JSON: {{"best_match_index": <number>}}
If nothing matches well, use -1.
"""
    try:
        result = call_llm_json(prompt, max_tokens=500)
        idx = result.get("best_match_index", -1)
        if 0 <= idx < len(catalog):
            return catalog[idx].deeplink
    except Exception:
        pass
    return None
