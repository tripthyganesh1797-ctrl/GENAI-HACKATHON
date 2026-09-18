"""
pipeline.py — Orchestrates the full flow. This is the file you'll iterate on
most. Run it directly (`python pipeline.py`) to test against sample_queries.json
without needing the API server running.
"""

import json
import time
import re
from difflib import SequenceMatcher

from llm_client import call_llm_json, MODEL, LAST_USAGE, estimate_cost_usd
from prompts import STAGE0_ENRICHMENT_PROMPT, STAGE1_EXTRACTION_PROMPT
from validators import validate_goal_object, strip_urls
from cache import get_cached, set_cached
from request_log import append_log

with open("deeplinks.json") as f:
    DEEPLINK_CATALOG = json.load(f)


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------

def stage0_enrich(raw_complaint: str) -> dict:
    prompt = STAGE0_ENRICHMENT_PROMPT.format(raw_complaint=raw_complaint)
    result = call_llm_json(prompt, max_tokens=2000)
    return result  # {"technical_query": ..., "query_variations": [...]}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def _guess_topic(technical_query: str) -> str:
    """Very simple topic extraction for the {topic} slot in the goal string.
    Upgrade with an LLM call if needed, but keyword rules work for most cases."""
    q = technical_query.lower()
    if "battery" in q:
        return "Battery"
    if "camera" in q:
        return "Camera"
    if "swipe" in q or "gesture" in q or "screen" in q or "display" in q:
        return "Display"
    if "slow" in q or "lag" in q or "performance" in q:
        return "Performance"
    return "Device"


def stage1_extract(technical_query: str, siis_response: str = "") -> dict:
    topic = _guess_topic(technical_query)
    prompt = STAGE1_EXTRACTION_PROMPT.format(
        technical_query=technical_query,
        siis_response=siis_response or "(no reference text provided)",
        topic=topic,
    )
    result = call_llm_json(prompt, max_tokens=4000)
    return result  # {"contexts": [Goal, ...]}  or {"contexts": []}


# ---------------------------------------------------------------------------
# Stage 2 — Deeplink matching (done in CODE, never trust the LLM with real URIs)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "tap", "on", "the", "a", "an", "to", "and", "for", "of", "settings",
    "open", "your", "will", "it", "in", "under", "check", "choose", "app",
}


def _keywords(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _match_score(action_name: str, steps: list, catalog_entry: dict) -> float:
    """Meaningful-keyword overlap (Jaccard on domain words), NOT raw string
    similarity. Raw SequenceMatcher was matching on filler words like
    'Tap'/'Settings'/'Open' shared by every entry, causing wrong matches
    (e.g. a Battery action matching a Camera deeplink). This version only
    scores on the words that actually carry meaning."""
    haystack_text = catalog_entry["description"] + " " + catalog_entry.get("message", "")
    needle_text = action_name + " " + " ".join(steps)

    needle_kw = _keywords(needle_text)
    haystack_kw = _keywords(haystack_text)

    if not needle_kw or not haystack_kw:
        return 0.0

    overlap = len(needle_kw & haystack_kw)
    union = len(needle_kw | haystack_kw)
    jaccard = overlap / union if union else 0.0

    # Small bonus if the exact overlap count is high relative to the shorter set
    # (helps short action names like "Camera App Storage" match confidently)
    coverage = overlap / min(len(needle_kw), len(haystack_kw))

    return 0.6 * jaccard + 0.4 * coverage


def match_deeplink(action_name: str, steps: list) -> dict:
    """Returns the best-matching deeplink entry, or the dummy_positive fallback.
    Threshold raised from 0.25 -> 0.4 after testing showed low thresholds
    let clearly-wrong matches through (see pipeline docstring above)."""
    best_entry, best_score = None, 0.0
    for entry in DEEPLINK_CATALOG:
        score = _match_score(action_name, steps, entry)
        if score > best_score:
            best_score, best_entry = score, entry

    if best_entry and best_score >= 0.4:
        return {
            "deeplink": best_entry["deeplink"],
            "description": best_entry["description"],
            "message": best_entry.get("message", ""),
        }

    # Reserved fallback for a valid but unindexed screen (per theme guide)
    return {
        "deeplink": "bixby://dummy_positive",
        "description": f"Open settings related to {action_name}",
        "message": f"Navigate manually to {action_name}",
    }


def enrich_with_deeplinks(goal_dict: dict) -> dict:
    for action in goal_dict.get("actions", []):
        if action.get("category") == "critical":
            continue  # critical actions cannot carry an actionable deeplink
        for step_group in action.get("stepGroups", []):
            step_group["actionableDeeplink"] = match_deeplink(
                action["actionName"], step_group["steps"]
            )
    return goal_dict


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(raw_complaint: str, siis_response: str = "") -> dict:
    start = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Stage 0
    enrichment = stage0_enrich(raw_complaint)
    total_prompt_tokens += LAST_USAGE["prompt_tokens"]
    total_completion_tokens += LAST_USAGE["completion_tokens"]
    technical_query = enrichment["technical_query"]

    # Fast-path cache check
    cached = get_cached(technical_query)
    if cached:
        cached["meta"]["cache_hit"] = True
        cached["meta"]["latency_ms"] = round((time.time() - start) * 1000, 1)
        append_log({
            "domain_guess": _guess_topic(technical_query),
            "cache_hit": True,
            "latency_ms": cached["meta"]["latency_ms"],
            "total_tokens": 0,
            "cost_usd": 0.0,
            "fallback": cached["meta"].get("fallback"),
        })
        return cached

    # Stage 1
    extraction = stage1_extract(technical_query, siis_response)
    total_prompt_tokens += LAST_USAGE["prompt_tokens"]
    total_completion_tokens += LAST_USAGE["completion_tokens"]
    contexts = extraction.get("contexts", [])

    # Validate + fix each Goal
    all_errors = []
    for goal in contexts:
        topic = _guess_topic(technical_query)
        errors = validate_goal_object(goal, topic)
        if errors:
            all_errors.extend(errors)
        # Hard-scrub any URL that slipped through, regardless of validation result
        for action in goal.get("actions", []):
            action["description"] = strip_urls(action["description"])
            for sg in action.get("stepGroups", []):
                sg["steps"] = [strip_urls(s) for s in sg["steps"]]

    # Stage 2: attach deeplinks
    contexts = [enrich_with_deeplinks(g) for g in contexts]

    latency_ms = round((time.time() - start) * 1000, 1)
    real_cost = estimate_cost_usd(total_prompt_tokens, total_completion_tokens)

    response = {
        "query": raw_complaint,
        "query_variations": enrichment.get("query_variations", []),
        "response": {"contexts": contexts},
        "meta": {
            "latency_ms": latency_ms,
            "cache_hit": False,
            "model": MODEL,
            "cost_usd": real_cost,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "fallback": "no_match" if not contexts else None,
            "validation_errors": all_errors,  # remove/log separately before final submission
        },
    }

    if contexts:
        set_cached(technical_query, response)

    # Log every request (cache hits AND misses) for the /stats endpoint
    append_log({
        "domain_guess": _guess_topic(technical_query),
        "cache_hit": False,
        "latency_ms": latency_ms,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "cost_usd": real_cost,
        "fallback": response["meta"]["fallback"],
    })

    return response


if __name__ == "__main__":
    # Quick manual test loop against sample_queries.json
    with open("sample_queries.json") as f:
        samples = json.load(f)

    for sample in samples[:3]:  # test first 3 by default; change slice as needed
        print(f"\n{'='*70}\nDOMAIN: {sample['domain']}\nCOMPLAINT: {sample['complaint']}\n{'='*70}")
        try:
            result = run_pipeline(sample["complaint"])
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")
