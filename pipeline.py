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
from typing import Optional

from llm_client import call_llm_json, MODEL, API_KEY, LAST_USAGE, estimate_cost_usd
from prompts import STAGE0_ENRICHMENT_PROMPT, STAGE1_EXTRACTION_PROMPT
from validators import validate_goal_object, strip_urls
from cache import get_cached, set_cached
from request_log import append_log
from deeplink_matching import match_and_build_deeplink
from escalation import build_escalation_recommendation, LLM_SCORE_ESCALATION_THRESHOLD
from device_signals import apply_device_context
from session_memory import get_avoid_set
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


# Task 37: finer-grained than _guess_topic()'s 5 domain buckets. Powers the
# /stats "trending issues" breakdown -- a real recurring symptom ("battery
# draining fast", "screen flickering") is far more actionable for a team
# triaging complaints than a coarse domain count, which request_log.py
# already surfaces separately as requests_by_domain. Deliberately simple
# keyword matching (same style as _guess_topic above, same offline-first
# philosophy as the rest of this codebase) rather than an LLM call -- this
# runs on every request, including ones served entirely offline.
def _guess_issue_phrase(technical_query: str) -> str:
    """Best-effort human-readable issue label for a technical_query. Falls
    back to '<domain> issue' (never crashes, never returns empty) so every
    logged request still counts toward some trending-issues bucket."""
    q = technical_query.lower()
    if "battery" in q and any(k in q for k in ("drain", "die", "dying", "dies")):
        return "Battery draining fast"
    if "battery" in q and "charg" in q:
        return "Battery not charging"
    if "camera" in q and ("blur" in q or "focus" in q):
        return "Camera blurry / out of focus"
    if "camera" in q and "crash" in q:
        return "Camera app crashing"
    if "wifi" in q or "wi-fi" in q:
        return "Wi-Fi connectivity"
    if "bluetooth" in q:
        return "Bluetooth connectivity"
    if "overheat" in q or "too hot" in q or "overheating" in q:
        return "Device overheating"
    if "flicker" in q:
        return "Screen flickering"
    if "crack" in q:
        return "Screen cracked / physical damage"
    if "black" in q or "blank" in q:
        return "Black / blank screen"
    if "touch" in q and any(k in q for k in ("unresponsive", "not working", "not respond", "stuck")):
        return "Touchscreen unresponsive"
    if "storage" in q and "full" in q:
        return "Storage full"
    if "app" in q and "crash" in q:
        return "App crashing"
    if "update" in q:
        return "Software update issue"
    if "slow" in q or "lag" in q or "performance" in q:
        return "Device running slow"
    if "speaker" in q or "audio" in q or "sound" in q:
        return "Speaker / audio issue"
    if "microphone" in q or "mic " in q or q.endswith("mic"):
        return "Microphone issue"
    if "signal" in q or "network" in q:
        return "Network / signal issue"
    return f"{_guess_topic(technical_query)} issue"


def _format_device_context_block(device: Optional[dict]) -> str:
    """Renders schema.DeviceContext's fields (already validated, plain
    dict) into the "Known device state" block STAGE1_EXTRACTION_PROMPT
    expects -- an empty string (no block at all) when there's no device
    context, or none of its fields are set, so the prompt is byte-
    identical to before Task 35 for every caller that doesn't pass one."""
    if not device:
        return ""
    lines = []
    if device.get("battery_pct") is not None:
        lines.append(f"- Battery level: {device['battery_pct']}%")
    if device.get("storage_free_pct") is not None:
        lines.append(f"- Free storage: {device['storage_free_pct']}%")
    if device.get("os_version"):
        lines.append(f"- OS version: {device['os_version']}")
    if device.get("uptime_hours") is not None:
        lines.append(f"- Uptime since last restart: {device['uptime_hours']} hours")
    if device.get("last_restart_hours_ago") is not None:
        lines.append(f"- Hours since last restart: {device['last_restart_hours_ago']}")
    if not lines:
        return ""
    return (
        "Known device state (use per rule 10 below -- context only, never a "
        "license to invent a fix):\n" + "\n".join(lines)
    )


def stage1_extract(technical_query: str, siis_response: str = "",
                    force_offline: bool = False, device: Optional[dict] = None,
                    avoid_deeplinks=None) -> tuple[dict, bool]:
    """Returns (result, used_fallback). `device` (Task 35) is only used to
    enrich the LLM prompt here -- the offline path ignores it entirely,
    and either way the actual device-aware reordering/notes happen once,
    uniformly, in run_pipeline()/run_pipeline_streaming() via
    apply_device_context() -- see device_signals.py for why that's done
    centrally instead of inside each path separately.

    `avoid_deeplinks` (Task 36, session_memory.py) is the opposite: the
    LLM path never picks a real deeplink itself (Stage 2 always resolves
    that in code, see enrich_with_deeplinks()), so there's nothing to pass
    into the LLM prompt for it -- it only matters on the offline path,
    where offline_extract() resolves deeplinks directly while building
    each Goal's actions."""
    if not force_offline and llm_available():
        topic = _guess_topic(technical_query)
        try:
            prompt = STAGE1_EXTRACTION_PROMPT.format(
                technical_query=technical_query,
                siis_response=siis_response or "(no reference text provided)",
                topic=topic,
                device_context_block=_format_device_context_block(device),
            )
            result = call_llm_json(prompt, max_tokens=4000)
            if "contexts" in result:
                return result, False
        except Exception:
            pass  # fall through to offline path
    return offline_fallback.offline_extract(technical_query, siis_response,
                                             avoid_deeplinks=avoid_deeplinks), True


# ---------------------------------------------------------------------------
# Stage 2 — Deeplink matching (done in CODE, never trust the LLM with real URIs)
# ---------------------------------------------------------------------------

def _maybe_attach_llm_escalation(goal: dict) -> None:
    """LLM-path counterpart to offline_fallback.py's own escalation gate
    (see escalation.py's module docstring for why the two paths use
    different signals). Only called for goals that came from the LLM path
    -- an offline-path goal already carries its own escalation object
    (attached inside offline_fallback._build_single_goal, using relevance
    rather than this self-reported score), and re-checking it here with a
    threshold calibrated for the LLM's 0-1 confidence scale would be
    comparing incompatible numbers."""
    score = goal.get("score")
    if isinstance(score, (int, float)) and score < LLM_SCORE_ESCALATION_THRESHOLD:
        goal["escalation"] = build_escalation_recommendation(
            f"The model's own confidence in this match was {score:.2f} (below our "
            f"{LLM_SCORE_ESCALATION_THRESHOLD:.2f} comfort threshold) -- it may not be "
            f"the right fix. Consider running a full device diagnostic, or contacting "
            f"Samsung Support if these steps don't help."
        )


def enrich_with_deeplinks(goal_dict: dict, variant: str = "hybrid", avoid_deeplinks=None) -> dict:
    """Stage 2. The offline fallback path already attaches deeplinks
    per-step while it groups steps (it needs the match result to decide
    grouping in the first place) -- so this only fills in actions that
    don't have one yet, which is exactly the LLM path's output shape.
    `avoid_deeplinks` (Task 36) passes straight through to the matcher."""
    for action in goal_dict.get("actions", []):
        if action.get("category") == "critical":
            continue  # critical actions cannot carry an actionable deeplink
        for step_group in action.get("stepGroups", []):
            if step_group.get("actionableDeeplink") is not None:
                continue  # already resolved (offline path)
            actionable, validation = match_and_build_deeplink(
                action["actionName"], step_group["steps"], variant=variant,
                avoid_deeplinks=avoid_deeplinks,
            )
            step_group["actionableDeeplink"] = actionable
            if validation is not None:
                step_group["validationDeeplink"] = validation
    return goal_dict


# ---------------------------------------------------------------------------
# Task 36 — session-scoped "already tried this" summary. Scans the final
# contexts for the per-action flags deeplink_matching.py's
# best_match_explained() may have set (session_avoid_skipped /
# already_tried_this_session) and turns them into a couple of plain-
# language sentences for meta.session_notes, mirroring
# device_signals.py's meta.device_context_notes in shape and purpose.
# ---------------------------------------------------------------------------

def _summarize_session_avoidance(contexts: list) -> list[str]:
    skipped_count = 0
    repeated_action_names: set[str] = set()
    for goal in contexts:
        for action in goal.get("actions", []):
            for sg in action.get("stepGroups", []):
                dl = sg.get("actionableDeeplink") or {}
                exp = dl.get("matchExplanation") or {}
                if exp.get("session_avoid_skipped"):
                    skipped_count += 1
                if exp.get("already_tried_this_session"):
                    repeated_action_names.add(action.get("actionName") or "this action")

    notes = []
    if skipped_count:
        plural = "es" if skipped_count != 1 else ""
        notes.append(
            f"Skipped {skipped_count} previously-tried match{plural} from this session "
            f"that you already marked unhelpful, in favor of a different suggestion."
        )
    if repeated_action_names:
        names = ", ".join(sorted(repeated_action_names))
        notes.append(
            f"{names} is suggested again because it's still the best available match for "
            f"this complaint -- you told us it didn't help last time, so this may be worth "
            f"escalating instead of repeating."
        )
    return notes


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(raw_complaint: str, siis_response: str = "", device: Optional[dict] = None,
                  session_id: Optional[str] = None) -> dict:
    start = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    used_fallback_any = False

    # Task 36: which deeplinks (if any) THIS session already tried and
    # marked unhelpful. Empty for no session_id / a session with no
    # negative feedback yet -- the common case, and a complete no-op.
    avoid_deeplinks = get_avoid_set(session_id)

    # Stage 0
    enrichment, fb0 = stage0_enrich(raw_complaint)
    used_fallback_any = used_fallback_any or fb0
    if not fb0:
        total_prompt_tokens += LAST_USAGE["prompt_tokens"]
        total_completion_tokens += LAST_USAGE["completion_tokens"]
    technical_query = enrichment["technical_query"]

    # Fast-path cache check. The cache stores the base plan only (no device
    # data baked in, since the cache key is technical_query alone) -- so
    # device context is applied fresh on EVERY request, cache hit or not,
    # never persisted into the cached entry. See device_signals.py.
    #
    # Task 36: a cached entry's Stage 2 deeplink choices were already
    # baked in by WHOEVER first populated that cache slot -- possibly a
    # different session, possibly no session at all. Serving it to a
    # session with active avoidance data would silently reintroduce
    # exactly the deeplink this feature exists to steer away from, so the
    # cache is bypassed entirely (read AND write) whenever avoid_deeplinks
    # is non-empty. See session_memory.py's module docstring.
    cached = get_cached(technical_query) if not avoid_deeplinks else None
    if cached:
        cached["meta"]["cache_hit"] = True
        cached["meta"]["latency_ms"] = round((time.time() - start) * 1000, 1)
        cached["meta"]["device_context_notes"] = apply_device_context(
            cached["response"]["contexts"], device
        )
        cached["meta"]["session_notes"] = []  # avoid_deeplinks is empty on every cache hit
        append_log({
            "domain_guess": _guess_topic(technical_query),
            "issue_guess": _guess_issue_phrase(technical_query),
            "cache_hit": True,
            "latency_ms": cached["meta"]["latency_ms"],
            "total_tokens": 0,
            "cost_usd": 0.0,
            "fallback": cached["meta"].get("fallback"),
            "used_offline_fallback": cached["meta"].get("used_offline_fallback", False),
        })
        return cached

    # Stage 1
    extraction, fb1 = stage1_extract(technical_query, siis_response, device=device,
                                      avoid_deeplinks=avoid_deeplinks)
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
        if not fb1:
            _maybe_attach_llm_escalation(goal)

    # Stage 2: attach deeplinks (same code path regardless of Stage0/1 source)
    contexts = [enrich_with_deeplinks(g, avoid_deeplinks=avoid_deeplinks) for g in contexts]

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
            "detected_language": enrichment.get("detected_language", "en" if used_fallback_any else None),
            "validation_errors": all_errors,
        },
    }

    if contexts and not avoid_deeplinks:
        # Cache the BASE plan (no device data, no session-specific deeplink
        # substitution) before applying device context below -- set_cached()
        # serialises to disk synchronously (cache.py's json.dump), so the
        # in-place reordering that follows can never leak into what's
        # persisted for the next caller.
        set_cached(technical_query, response)

    response["meta"]["device_context_notes"] = apply_device_context(contexts, device)
    response["meta"]["session_notes"] = _summarize_session_avoidance(contexts)

    append_log({
        "domain_guess": _guess_topic(technical_query),
        "issue_guess": _guess_issue_phrase(technical_query),
        "cache_hit": False,
        "latency_ms": latency_ms,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "cost_usd": real_cost,
        "fallback": response["meta"]["fallback"],
        "used_offline_fallback": used_fallback_any,
    })

    return response


# ---------------------------------------------------------------------------
# Streaming variant — same pipeline, yields one event per stage so a client
# (the demo UI, or `curl -N`) can render live progress instead of waiting on
# one big response. Reuses every helper above; the final "complete" event
# carries the exact same payload run_pipeline() would return, so the two
# paths can never silently drift apart in their actual troubleshooting logic
# -- only in how/when the result is delivered.
# ---------------------------------------------------------------------------

def run_pipeline_streaming(raw_complaint: str, siis_response: str = "", device: Optional[dict] = None,
                            session_id: Optional[str] = None):
    """Generator of small dicts: {"stage": ..., "status": "running"|"done"|"error", "data": {...}}.
    Caller (main.py's SSE route) is responsible for JSON-encoding each one."""
    start = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    used_fallback_any = False
    avoid_deeplinks = get_avoid_set(session_id)  # Task 36, see run_pipeline()'s comments

    try:
        yield {"stage": "start", "status": "done", "data": {"query": raw_complaint}}

        # Stage 0
        yield {"stage": "enrich", "status": "running"}
        enrichment, fb0 = stage0_enrich(raw_complaint)
        used_fallback_any = used_fallback_any or fb0
        if not fb0:
            total_prompt_tokens += LAST_USAGE["prompt_tokens"]
            total_completion_tokens += LAST_USAGE["completion_tokens"]
        technical_query = enrichment["technical_query"]
        yield {
            "stage": "enrich",
            "status": "done",
            "data": {
                "technical_query": technical_query,
                "query_variations": enrichment.get("query_variations", []),
                "detected_language": enrichment.get("detected_language"),
                "used_offline_fallback": fb0,
            },
        }

        # Fast-path cache check (Task 36: bypassed when this session has
        # active avoidance data -- see run_pipeline()'s comments)
        cached = get_cached(technical_query) if not avoid_deeplinks else None
        if cached:
            cached["meta"]["cache_hit"] = True
            cached["meta"]["latency_ms"] = round((time.time() - start) * 1000, 1)
            cached["meta"]["device_context_notes"] = apply_device_context(
                cached["response"]["contexts"], device
            )
            cached["meta"]["session_notes"] = []
            yield {"stage": "cache", "status": "hit"}
            append_log({
                "domain_guess": _guess_topic(technical_query),
                "issue_guess": _guess_issue_phrase(technical_query),
                "cache_hit": True,
                "latency_ms": cached["meta"]["latency_ms"],
                "total_tokens": 0,
                "cost_usd": 0.0,
                "fallback": cached["meta"].get("fallback"),
                "used_offline_fallback": cached["meta"].get("used_offline_fallback", False),
            })
            yield {"stage": "complete", "status": "done", "data": cached}
            return
        yield {"stage": "cache", "status": "miss"}

        # Stage 1
        yield {"stage": "extract", "status": "running"}
        extraction, fb1 = stage1_extract(technical_query, siis_response, device=device,
                                          avoid_deeplinks=avoid_deeplinks)
        used_fallback_any = used_fallback_any or fb1
        if not fb1:
            total_prompt_tokens += LAST_USAGE["prompt_tokens"]
            total_completion_tokens += LAST_USAGE["completion_tokens"]
        contexts = extraction.get("contexts", [])
        yield {
            "stage": "extract",
            "status": "done",
            "data": {"num_contexts": len(contexts), "used_offline_fallback": fb1},
        }

        # Validate + fix each Goal
        yield {"stage": "validate", "status": "running"}
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
            if not fb1:
                _maybe_attach_llm_escalation(goal)
        yield {"stage": "validate", "status": "done", "data": {"validation_errors": all_errors}}

        # Stage 2: attach deeplinks
        yield {"stage": "deeplink_match", "status": "running"}
        contexts = [enrich_with_deeplinks(g, avoid_deeplinks=avoid_deeplinks) for g in contexts]
        yield {"stage": "deeplink_match", "status": "done"}

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
                "detected_language": enrichment.get("detected_language", "en" if used_fallback_any else None),
                "validation_errors": all_errors,
            },
        }

        if contexts and not avoid_deeplinks:
            set_cached(technical_query, response)

        response["meta"]["device_context_notes"] = apply_device_context(contexts, device)
        response["meta"]["session_notes"] = _summarize_session_avoidance(contexts)

        append_log({
            "domain_guess": _guess_topic(technical_query),
            "issue_guess": _guess_issue_phrase(technical_query),
            "cache_hit": False,
            "latency_ms": latency_ms,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cost_usd": real_cost,
            "fallback": response["meta"]["fallback"],
            "used_offline_fallback": used_fallback_any,
        })

        yield {"stage": "complete", "status": "done", "data": response}

    except Exception as e:
        yield {"stage": "error", "status": "error", "data": {"message": str(e)}}


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
