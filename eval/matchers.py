"""
matchers.py — Three deeplink-matching architectures, so the ablation study
in metrics.md is backed by real, run code rather than opinions.

Variant A (rules-based): deterministic substring match on the single most
    distinctive word in the query. Zero LLM cost, fastest, least flexible.
Variant B (hybrid): keyword-overlap scoring (Jaccard + coverage) — this is
    what the shipped pipeline.py actually uses.
Variant C (full-LLM): asks the LLM itself to pick the best catalog entry by
    ID, given the full catalog. Most flexible, highest cost and latency.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import call_llm_json  # noqa: E402

STOPWORDS = {
    "tap", "on", "the", "a", "an", "to", "and", "for", "of", "settings",
    "open", "your", "will", "it", "in", "under", "check", "choose", "app",
    "per", "at", "once", "device", "or",
}


def _keywords(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


# ---------------------------------------------------------------------------
# Variant A: Pure rules-based
# ---------------------------------------------------------------------------

def match_rules_based(query: str, catalog: list) -> str:
    """Picks the catalog entry whose description shares the SINGLE most
    distinctive (longest) keyword with the query. No scoring, no ranking —
    just a deterministic first-match rule."""
    query_kw = _keywords(query)
    if not query_kw:
        return None
    # Sort keywords longest-first: a rules engineer would prioritise the
    # most specific/rare term as the strongest signal.
    for keyword in sorted(query_kw, key=len, reverse=True):
        for entry in catalog:
            haystack = (entry["description"] + " " + entry.get("message", "")).lower()
            if keyword in haystack:
                return entry["deeplink"]
    return None


# ---------------------------------------------------------------------------
# Variant B: Hybrid keyword-overlap (matches production pipeline.py)
# ---------------------------------------------------------------------------

def match_hybrid(query: str, catalog: list) -> str:
    query_kw = _keywords(query)
    best_id, best_score = None, 0.0
    for entry in catalog:
        haystack_kw = _keywords(entry["description"] + " " + entry.get("message", ""))
        if not query_kw or not haystack_kw:
            continue
        overlap = len(query_kw & haystack_kw)
        union = len(query_kw | haystack_kw)
        jaccard = overlap / union if union else 0.0
        coverage = overlap / min(len(query_kw), len(haystack_kw))
        score = 0.6 * jaccard + 0.4 * coverage
        if score > best_score:
            best_score, best_id = score, entry["deeplink"]
    return best_id if best_score >= 0.4 else None


# ---------------------------------------------------------------------------
# Variant C: Full-LLM mapping
# ---------------------------------------------------------------------------

def match_llm(query: str, catalog: list) -> str:
    catalog_lines = "\n".join(
        f"{i}: {entry['description']}" for i, entry in enumerate(catalog)
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
            return catalog[idx]["deeplink"]
    except Exception:
        pass
    return None
