"""
pipeline.py — Orchestrates the full flow. This is the file you'll iterate on
most. Run it directly (`python pipeline.py`) to test against
sample_queries_real.json without needing the API server running.

Two execution paths, chosen automatically per request:

  1. LLM path (primary, best quality): Stage 0 + Stage 1 call the configured
     LLM (llm_client.py / prompts.py). Requires LLM_API_KEY.

  2. Offline fallback path (offline_fallback.py): fully rule-based, zero
     API key, zero network, zero cost. Used automatically when LLM_API_KEY
     isn't set, or when the LLM call/JSON-parse fails after retries — the
     service degrades gracefully instead of returning a 500.

Stage 2 (deeplink matching) is identical on both paths — it always runs in
code against deeplink_matching.py, never trusting the LLM with real URIs.
"""

import json
import time

from llm_client import call_llm_json, MODEL, API_KEY, LAST_USAGE, estimate_cost_usd
from prompts import STAGE0_ENRICHMENT_PROMPT, STAGE1_EXTRACTION_PROMPT
from validators import validate_goal_object, strip_urls
from cache import get_cached, set_cached
from request_log import append_log
from deeplink_matching import match_and_build_deeplink
import offline_fallback

_PLACEHOLDER_KEYS = {None, "", "your_key_here"}


def llm_available() -> bool:
    return API_KEY not in _PLACEHOLDER_KEYS


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------

def stage0_enrich(raw_complaint: str) -> tuple[dict, bool]:
    """Returns (result, used_fallback)."""
    if llm_available():
        try:
            prompt = STAGE0_ENRICHMENT_PROMPT.format(raw_complaint=raw_complaint)
            result = call_llm_json(prompt, max_tokens=2000)
            if result.get("technical_query"):
                return result, False
        except Exception:
            pass  # fall through to offline path
    return offline_fallback.offline_enrich(raw_complaint), True


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def _guess_topic(technical_query: str) -> str:
    """Simple topic extraction for the {topic} slot in the goal string and
    for the request-log domain breakdown."""
    q = technical_query.lower()
    if "battery" in q:
        return "Battery"
    if "camera" in q:
        return "Camera"
    if any(k in q for k in ("swipe", "gesture", "screen", "display", "touch",
                             "flicker", "crack", "blank", "black")):
        return "Display"
    if "slow" in q or "lag" in q or "performance" in q:
        return "Performance"
    return "Device"


def stage1_extract(technical_query: str, siis_response: str = "",
                    force_offline: bool = False) -> tuple[dict, bool]:
    """Returns (result, used_fallback)."""
    if not force_offline and llm_available():
        topic = _guess_topic(technical_query)
        try:
            prompt = STAGE1_EXTRACTION_PROMPT.format(
                technical_query=technical_query,
                siis_response=siis_response or "(no reference text provided)",
                topic=topic,
            )
            result = call_llm_json(prompt, max_tokens=4000)
            if "contexts" in result:
                return result, False
        except Exception:
            pass  # fall through to offline path
    return offline_fallback.offline_extract(technical_query, siis_response), True


# ---------------------------------------------------------------------------
# Stage 2 — Deeplink matching (done in CODE, never trust the LLM with real URIs)
# ---------------------------------------------------------------------------

def enrich_with_deeplinks(goal_dict: dict, variant: str = "hybrid") -> dict:
    """Stage 2. The offline fallback path already attaches deeplinks
    per-step while it groups steps (it needs the match result to decide
    grouping in the first place) -- so this only fills in actions that
    don't have one yet, which is exactly the LLM path's output shape."""
    for action in goal_dict.get("actions", []):
        if action.get("category") == "critical":
            continue  # critical actions cannot carry an actionable deeplink
        for step_group in action.get("stepGroups", []):
            if step_group.get("actionableDeeplink") is not None:
                continue  # already resolved (offline path)
            actionable, validation = match_and_build_deeplink(
                action["actionName"], step_group["steps"], variant=variant,
            )
            step_group["actionableDeeplink"] = actionable
            if validation is not None:
                step_group["validationDeeplink"] = validation
    return goal_dict


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(raw_complaint: str, siis_response: str = "") -> dict:
    start = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    used_fallback_any = False

    # Stage 0
    enrichment, fb0 = stage0_enrich(raw_complaint)
    used_fallback_any = used_fallback_any or fb0
    if not fb0:
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
            "used_offline_fallback": cached["meta"].get("used_offline_fallback", False),
        })
        return cached

    # Stage 1
    extraction, fb1 = stage1_extract(technical_query, siis_response)
    used_fallback_any = used_fallback_any or fb1
    if not fb1:
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
        for action in goal.get("actions", []):
            action["description"] = strip_urls(action["description"])
            for sg in action.get("stepGroups", []):
                sg["steps"] = [strip_urls(s) for s in sg["steps"]]

    # Stage 2: attach deeplinks (same code path regardless of Stage0/1 source)
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
            "model": "offline-rule-based" if used_fallback_any else MODEL,
            "cost_usd": real_cost,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "fallback": "no_match" if not contexts else None,
            "used_offline_fallback": used_fallback_any,
            "validation_errors": all_errors,
        },
    }

    if contexts:
        set_cached(technical_query, response)

    append_log({
        "domain_guess": _guess_topic(technical_query),
        "cache_hit": False,
        "latency_ms": latency_ms,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "cost_usd": real_cost,
        "fallback": response["meta"]["fallback"],
        "used_offline_fallback": used_fallback_any,
    })

    return response


if __name__ == "__main__":
    with open("sample_queries_real.json") as f:
        samples = json.load(f)

    for sample in samples[:3]:
        print(f"\n{'='*70}\nDOMAIN: {sample['domain']}\nCOMPLAINT: {sample['complaint']}\n{'='*70}")
        try:
            result = run_pipeline(sample["complaint"], sample.get("siis_response", ""))
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")
