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
from safety import detect_physical_hazard, build_safety_goal
from clarify import detect_vague_complaint, build_clarifying_question, CLARIFYING_TOPIC_OPTIONS
from answer_source import classify_answer_source
from related_issues import suggest_related_issues
from image_analysis import describe_image, NO_IMAGE_ANALYSIS
from ambiguity import refine_needs_clarification
import recovery
from topic_manager import update_topic_stack
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
                    avoid_deeplinks=None) -> tuple[dict, bool, dict]:
    """Returns (result, used_fallback, recovery_info). `device` (Task 35) is
    only used to enrich the LLM prompt here -- the offline path ignores it
    entirely, and either way the actual device-aware reordering/notes
    happen once, uniformly, in run_pipeline()/run_pipeline_streaming() via
    apply_device_context() -- see device_signals.py for why that's done
    centrally instead of inside each path separately.

    `avoid_deeplinks` (Task 36, session_memory.py) is the opposite: the
    LLM path never picks a real deeplink itself (Stage 2 always resolves
    that in code, see enrich_with_deeplinks()), so there's nothing to pass
    into the LLM prompt for it -- it only matters on the offline path,
    where offline_extract() resolves deeplinks directly while building
    each Goal's actions.

    `recovery_info` (recovery.py, Guided-Retry inspired) is recovery.
    NO_RECOVERY for the common cases (no LLM configured, or the first
    attempt just worked) or a dict describing an LLM-call failure and
    whether one structured guided-retry attempt recovered it before
    falling through to the offline path. See recovery.py's module
    docstring for the full rationale and why this deliberately does NOT
    retry a legitimate empty-contexts result."""
    if not force_offline and llm_available():
        topic = _guess_topic(technical_query)
        base_prompt = STAGE1_EXTRACTION_PROMPT.format(
            technical_query=technical_query,
            siis_response=siis_response or "(no reference text provided)",
            topic=topic,
            device_context_block=_format_device_context_block(device),
        )
        try:
            result = call_llm_json(base_prompt, max_tokens=4000)
            if "contexts" in result:
                return result, False, dict(recovery.NO_RECOVERY)
        except Exception as e:
            failure = recovery.classify_llm_failure(True, e)
            # Guided-Retry: one explicit, structured retry with a recovery
            # instruction appended to the SAME prompt -- distinct from
            # call_llm_json()'s own internal retries=2, which only retries
            # raw JSON-parse noise on a byte-identical prompt. This is a
            # higher-level retry after THAT has already been exhausted.
            try:
                retry_prompt = base_prompt + recovery.build_guided_retry_suffix(failure["failure_type"])
                result = call_llm_json(retry_prompt, max_tokens=4000, retries=1)
                if "contexts" in result:
                    failure["guided_retry_attempted"] = True
                    failure["guided_retry_succeeded"] = True
                    return result, False, failure
            except Exception:
                pass  # guided retry also failed -- fall through to offline
            failure["guided_retry_attempted"] = True
            failure["guided_retry_succeeded"] = False
            return offline_fallback.offline_extract(
                technical_query, siis_response, avoid_deeplinks=avoid_deeplinks,
            ), True, failure
    return offline_fallback.offline_extract(technical_query, siis_response,
                                             avoid_deeplinks=avoid_deeplinks), True, dict(recovery.NO_RECOVERY)


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

def _build_safety_response(raw_complaint: str, hazard_reason: str, start: float) -> dict:
    """A physical-hazard complaint (safety.py) short-circuits everything
    else -- no LLM/offline extraction attempt, no cache read/write, no
    device-context reordering. See safety.py's module docstring for why
    this has to be a hard override rather than an extra caveat on a
    normal plan."""
    contexts = [build_safety_goal(hazard_reason)]
    # Same validation every other Goal in this pipeline goes through
    # before shipping -- belt-and-suspenders, not because this
    # hand-built goal is expected to fail it (it's covered by
    # tests/test_safety.py), but because nothing in this codebase skips
    # the compliance check just because a human wrote the source dict
    # instead of the LLM/offline extractor.
    validation_errors = validate_goal_object(contexts[0])
    latency_ms = round((time.time() - start) * 1000, 1)

    append_log({
        "domain_guess": "Safety",
        "issue_guess": f"Physical hazard: {hazard_reason}",
        "cache_hit": False,
        "latency_ms": latency_ms,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "fallback": None,
        "used_offline_fallback": False,
    })

    return {
        "query": raw_complaint,
        # Deterministic, zero-cost, zero-LLM paraphrase generator (same one
        # the offline path uses) -- keeps this response schema/contract
        # compliant (8-10 query_variations) even on the safety short-circuit.
        "query_variations": offline_fallback.generate_paraphrases(raw_complaint),
        "response": {"contexts": contexts},
        "meta": {
            "latency_ms": latency_ms,
            "cache_hit": False,
            "model": "safety-rule-based",
            "cost_usd": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "fallback": None,
            "used_offline_fallback": False,
            "used_builtin_reference": False,
            "detected_language": "en",
            "validation_errors": validation_errors,
            "device_context_notes": [],
            "session_notes": [],
            "safety_alert": True,
            "safety_reason": hazard_reason,
            # A physical-hazard complaint always contains enough signal to
            # act on (that's what detect_physical_hazard() matched on) --
            # never worth also asking "could you clarify?" here.
            "needs_clarification": False,
            "clarifying_question": None,
            "clarifying_topic_options": [],
            # A hazard short-circuit never reaches the ambiguity check or
            # Stage 1, so there's nothing for either signal to report --
            # None/NO_RECOVERY rather than omitting the keys, so callers
            # never have to special-case the safety response shape.
            "ambiguity": None,
            "recovery": dict(recovery.NO_RECOVERY),
            # Deliberately always None here, even with an active session --
            # a physical-hazard interrupt isn't a "topic" a user would want
            # topic_manager.py to push/resume; it doesn't touch the stack.
            "topic_stack": None,
            "answer_source": classify_answer_source(
                safety_alert=True, used_offline_fallback=False,
                used_builtin_reference=False, siis_response_provided=False,
                has_contexts=True,
            ),
            # A safety short-circuit already IS the single, certain answer
            # (a hazard warning) -- there's no "borderline confidence" for
            # related_issues.py to react to here; always [].
            "related_possibilities": [],
            # Placeholder -- the real, request-specific value (whether an
            # image was attached, and what it showed) is spliced in by
            # run_pipeline() right after this call, same reasoning as
            # needs_clarification elsewhere; this default only matters if
            # this helper is ever called standalone.
            "image_analysis": dict(NO_IMAGE_ANALYSIS),
        },
    }


def run_pipeline(raw_complaint: str, siis_response: str = "", device: Optional[dict] = None,
                  session_id: Optional[str] = None, image_data_url: Optional[str] = None) -> dict:
    start = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    used_fallback_any = False

    # image_analysis.py (point 1): a complete no-op when image_data_url is
    # None (the overwhelming majority of requests) -- describe_image()
    # returns NO_IMAGE_ANALYSIS instantly with no LLM call. When an image
    # IS supplied and successfully analyzed, its factual description is
    # folded into `effective_complaint`, which -- and ONLY which -- feeds
    # every downstream stage below (hazard detection, clarify, Stage 0/1).
    # `raw_complaint` itself is left untouched so response["query"] still
    # shows exactly what the user typed, not the image analysis text.
    image_analysis = describe_image(image_data_url)
    effective_complaint = raw_complaint
    if image_analysis["analyzed"] and image_analysis["description"]:
        effective_complaint = f"{raw_complaint}\n\n[What the attached photo shows: {image_analysis['description']}]"

    # Physical-hazard short-circuit (safety.py) -- runs before ANYTHING
    # else, including the cache lookup, so it can never be skipped by a
    # stale cache entry and never pollutes the cache for this query's
    # normal technical_query key. See _build_safety_response()/safety.py.
    # Checked against effective_complaint so a photo showing e.g. visible
    # smoke or a swollen battery can trigger this even if the typed text
    # alone wouldn't have -- no separate image-specific safety path needed.
    hazard_reason = detect_physical_hazard(effective_complaint)
    if hazard_reason:
        response = _build_safety_response(raw_complaint, hazard_reason, start)
        response["meta"]["image_analysis"] = image_analysis
        return response

    # clarify.py: purely additive (see its module docstring) -- computed
    # once here from the (image-informed) complaint text itself (not
    # technical_query, and not dependent on cache/Stage0/1 at all) so it's
    # cheap, always available, and applies identically whether this
    # request is served from cache or extracted fresh below.
    heuristic_needs_clarification = detect_vague_complaint(effective_complaint)
    # ambiguity.py (CLAM-inspired): only ever RAISES the heuristic's verdict,
    # never lowers it -- see ambiguity.py's module docstring for why. A
    # complete no-op (ambiguity_info stays None) whenever no LLM is
    # configured, so this can never change offline/gate-G3 behavior.
    needs_clarification, ambiguity_info = refine_needs_clarification(
        heuristic_needs_clarification, effective_complaint
    )
    clarifying_question = build_clarifying_question(effective_complaint) if needs_clarification else None

    # Task 36: which deeplinks (if any) THIS session already tried and
    # marked unhelpful. Empty for no session_id / a session with no
    # negative feedback yet -- the common case, and a complete no-op.
    avoid_deeplinks = get_avoid_set(session_id)

    # Stage 0 -- runs against effective_complaint (raw text + any folded-in
    # image description), so an attached photo can genuinely change what
    # technical_query comes out, and therefore which cache entry/plan this
    # request lands on. See image_analysis.py's module docstring for why
    # that's the right behavior rather than something to special-case.
    enrichment, fb0 = stage0_enrich(effective_complaint)
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
        # Recomputed fresh, not read from the cached entry: it depends on
        # THIS request's raw_complaint wording, not on the technical_query
        # cache key (two differently-worded complaints can share a cache
        # entry via cache.py's fuzzy match, but "my phone's broken" and a
        # detailed complaint that landed on the same entry deserve
        # different clarification verdicts).
        cached["meta"]["needs_clarification"] = needs_clarification
        cached["meta"]["clarifying_question"] = clarifying_question
        cached["meta"]["clarifying_topic_options"] = CLARIFYING_TOPIC_OPTIONS if needs_clarification else []
        # ambiguity.py: same "request-specific, recompute fresh" reasoning
        # as needs_clarification -- this request's own complaint wording,
        # not anything about the cache slot it happened to land on.
        cached["meta"]["ambiguity"] = ambiguity_info
        # topic_manager.py (DiagGPT-inspired): also request/session-specific
        # -- recorded fresh on every request, same as session_notes below,
        # never baked into the cached base plan.
        cached["meta"]["topic_stack"] = update_topic_stack(
            session_id, _guess_issue_phrase(technical_query)
        )
        # recovery.py (Guided-Retry inspired) DOES reflect how the cached
        # plan was actually produced, so it's baked in at cache-write time
        # like answer_source/related_possibilities below -- .setdefault()
        # backfills a cache entry written before this field existed.
        cached["meta"].setdefault("recovery", dict(recovery.NO_RECOVERY))
        # Same "request-specific, recompute fresh" treatment as
        # needs_clarification above: whether THIS request attached a photo
        # (and what it showed) has nothing to do with which cache slot the
        # resulting technical_query happened to land on.
        cached["meta"]["image_analysis"] = image_analysis
        # answer_source (and related_possibilities, below) DO reflect how the
        # cached plan was actually produced, so they're baked in at cache-write
        # time and read as-is here -- except a cache entry written before this
        # field existed, where .setdefault() backfills a safe classification
        # rather than silently omitting it.
        cached["meta"].setdefault("answer_source", classify_answer_source(
            safety_alert=False,
            used_offline_fallback=cached["meta"].get("used_offline_fallback", False),
            used_builtin_reference=cached["meta"].get("used_builtin_reference", False),
            siis_response_provided=bool(siis_response and siis_response.strip()),
            has_contexts=bool(cached["response"]["contexts"]),
        ))
        # Same "computed once, not request-specific" treatment as
        # answer_source above: whether the cached plan was borderline was
        # already decided when it was written (a Goal either carries an
        # escalation object or it doesn't), and the symptom bucket comes
        # from technical_query -- the cache key itself -- not anything
        # about THIS particular request. .setdefault() backfills entries
        # cached before this field existed.
        cached["meta"].setdefault("related_possibilities", suggest_related_issues(
            _guess_issue_phrase(technical_query)
        ) if any(g.get("escalation") for g in cached["response"]["contexts"]) else [])
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
    extraction, fb1, recovery_info = stage1_extract(technical_query, siis_response, device=device,
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
            "used_builtin_reference": extraction.get("used_builtin_reference", False),
            "detected_language": enrichment.get("detected_language", "en" if used_fallback_any else None),
            "validation_errors": all_errors,
            "safety_alert": False,
            "safety_reason": None,
            "needs_clarification": needs_clarification,
            "clarifying_question": clarifying_question,
            "clarifying_topic_options": CLARIFYING_TOPIC_OPTIONS if needs_clarification else [],
            # ambiguity.py (CLAM-inspired): the LLM-based signal that fed
            # into needs_clarification above, surfaced honestly (None
            # whenever no LLM was available to compute it).
            "ambiguity": ambiguity_info,
            # recovery.py (Guided-Retry inspired): whether Stage 1's LLM
            # call failed and, if so, whether the structured guided-retry
            # attempt recovered it before falling through to the offline
            # path. recovery.NO_RECOVERY for the common no-failure case.
            "recovery": recovery_info,
            # Computed once here, not request-specific (unlike device/session
            # notes below) -- flows through untouched on a cache hit exactly
            # like used_offline_fallback/used_builtin_reference already do.
            "answer_source": classify_answer_source(
                safety_alert=False, used_offline_fallback=used_fallback_any,
                used_builtin_reference=extraction.get("used_builtin_reference", False),
                siis_response_provided=bool(siis_response and siis_response.strip()),
                has_contexts=bool(contexts),
            ),
            # related_issues.py: only surfaced when this plan's own
            # confidence was borderline enough to already carry an
            # escalation recommendation (see escalation.py) -- when the
            # engine is confident, a list of "or maybe it's this instead"
            # alternatives would just undermine a correct answer.
            "related_possibilities": suggest_related_issues(_guess_issue_phrase(technical_query))
            if any(g.get("escalation") for g in contexts) else [],
            # image_analysis.py (point 1): request-specific, same as
            # needs_clarification above -- always the CURRENT request's own
            # image result, never something a cache entry could bake in.
            "image_analysis": image_analysis,
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
    # topic_manager.py (DiagGPT-inspired): session-scoped, recomputed fresh
    # on every request -- see the cache-hit branch above for why this is
    # never baked into the cached base plan either.
    response["meta"]["topic_stack"] = update_topic_stack(session_id, _guess_issue_phrase(technical_query))

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
    Caller (main.py's SSE route) is responsible for JSON-encoding each one.

    Deliberately does NOT accept image_data_url (see run_pipeline()'s
    image_analysis.py wiring for the non-streaming path): main.py's SSE
    route is GET-only (the browser EventSource API has no POST variant),
    and a base64-encoded photo is far too large for a URL query string.
    Every meta dict below still carries image_analysis, fixed at
    NO_IMAGE_ANALYSIS, so a caller reading meta.image_analysis.provided
    never has to special-case which endpoint it called.

    Also deliberately skips ambiguity.py's LLM-based ambiguity refinement
    (meta.ambiguity is always None here) -- that's an extra LLM round trip
    before the "clarify" stage event could even be emitted, which would
    either delay the first event or require restructuring the stage
    ordering for a purely additive confidence signal. meta.needs_clarification
    still reflects clarify.py's zero-cost heuristic exactly as before.
    recovery.py and topic_manager.py ARE fully wired here -- both are cheap
    (no extra LLM call, or one that Stage 1 already had to make anyway)."""
    start = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    used_fallback_any = False
    avoid_deeplinks = get_avoid_set(session_id)  # Task 36, see run_pipeline()'s comments

    try:
        yield {"stage": "start", "status": "done", "data": {"query": raw_complaint}}

        # Physical-hazard short-circuit -- see run_pipeline()'s identical
        # check and _build_safety_response()/safety.py for why this comes
        # before anything else, streaming included.
        hazard_reason = detect_physical_hazard(raw_complaint)
        if hazard_reason:
            safety_response = _build_safety_response(raw_complaint, hazard_reason, start)
            yield {"stage": "safety_alert", "status": "done", "data": {"reason": hazard_reason}}
            yield {"stage": "complete", "status": "done", "data": safety_response}
            return

        # clarify.py -- see run_pipeline()'s identical check for the full
        # rationale. Emitted as its own stage event so the UI can show the
        # "could you tell us more?" hint as soon as it's known, without
        # waiting for the full plan to finish extracting.
        needs_clarification = detect_vague_complaint(raw_complaint)
        clarifying_question = build_clarifying_question(raw_complaint) if needs_clarification else None
        if needs_clarification:
            yield {"stage": "clarify", "status": "done", "data": {"clarifying_question": clarifying_question}}

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
            cached["meta"]["needs_clarification"] = needs_clarification
            cached["meta"]["clarifying_question"] = clarifying_question
            cached["meta"]["clarifying_topic_options"] = CLARIFYING_TOPIC_OPTIONS if needs_clarification else []
            # ambiguity.py deliberately NOT run on the streaming path (see
            # run_pipeline_streaming()'s docstring) -- always None here.
            cached["meta"]["ambiguity"] = None
            cached["meta"]["topic_stack"] = update_topic_stack(
                session_id, _guess_issue_phrase(technical_query)
            )
            cached["meta"].setdefault("recovery", dict(recovery.NO_RECOVERY))
            cached["meta"].setdefault("answer_source", classify_answer_source(
                safety_alert=False,
                used_offline_fallback=cached["meta"].get("used_offline_fallback", False),
                used_builtin_reference=cached["meta"].get("used_builtin_reference", False),
                siis_response_provided=bool(siis_response and siis_response.strip()),
                has_contexts=bool(cached["response"]["contexts"]),
            ))
            cached["meta"].setdefault("related_possibilities", suggest_related_issues(
                _guess_issue_phrase(technical_query)
            ) if any(g.get("escalation") for g in cached["response"]["contexts"]) else [])
            cached["meta"]["image_analysis"] = dict(NO_IMAGE_ANALYSIS)  # streaming never accepts images
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
        extraction, fb1, recovery_info = stage1_extract(technical_query, siis_response, device=device,
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
                "used_builtin_reference": extraction.get("used_builtin_reference", False),
                "detected_language": enrichment.get("detected_language", "en" if used_fallback_any else None),
                "validation_errors": all_errors,
                "safety_alert": False,
                "safety_reason": None,
                "needs_clarification": needs_clarification,
                "clarifying_question": clarifying_question,
                "clarifying_topic_options": CLARIFYING_TOPIC_OPTIONS if needs_clarification else [],
                # ambiguity.py deliberately NOT run on the streaming path --
                # see this function's docstring.
                "ambiguity": None,
                "recovery": recovery_info,
                "answer_source": classify_answer_source(
                    safety_alert=False, used_offline_fallback=used_fallback_any,
                    used_builtin_reference=extraction.get("used_builtin_reference", False),
                    siis_response_provided=bool(siis_response and siis_response.strip()),
                    has_contexts=bool(contexts),
                ),
                "related_possibilities": suggest_related_issues(_guess_issue_phrase(technical_query))
                if any(g.get("escalation") for g in contexts) else [],
                "image_analysis": dict(NO_IMAGE_ANALYSIS),  # streaming never accepts images, see above
            },
        }

        if contexts and not avoid_deeplinks:
            set_cached(technical_query, response)

        response["meta"]["device_context_notes"] = apply_device_context(contexts, device)
        response["meta"]["session_notes"] = _summarize_session_avoidance(contexts)
        response["meta"]["topic_stack"] = update_topic_stack(session_id, _guess_issue_phrase(technical_query))

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
