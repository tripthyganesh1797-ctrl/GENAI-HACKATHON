"""
schema.py — Data contract for the Smart Guided Troubleshooting Engine (Theme 2)

This mirrors Appendix A of the theme guide almost verbatim. DO NOT loosen these
rules — the judges' automated gates check exact conformance (word counts,
enums, deeplink verbatim match, etc.).
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BaseDeeplink(BaseModel):
    deeplink: str


class Deeplink(BaseDeeplink):
    description: str
    message: Optional[str] = ""
    classes: Optional[Dict[str, str]] = None
    originalType: Optional[str] = None


class Condition(str, Enum):
    greater = "greater"
    equal = "equal"
    less = "less"


class ResultTypes(str, Enum):
    boolean = "boolean"
    intNum = "integer"
    string = "str"
    floatNum = "float"


class ActionCategory(str, Enum):
    auto = "auto"        # standard config screens reachable via deeplink
    manual = "manual"    # physical interventions (cleaning ports, service centre)
    critical = "critical"  # disruptive/irreversible (factory reset, restart) — ordered LAST


class ValidationDeepLink(BaseDeeplink):
    key: str
    resultType: Optional[ResultTypes] = None
    condition: Optional[Condition] = None
    value: Optional[str] = None


class StepGroup(BaseModel):
    steps: List[str]  # imperative UI steps, one physical interaction per step, no URLs
    validationDeeplink: Optional[ValidationDeepLink] = None
    actionableDeeplink: Optional[Deeplink] = None


class Action(BaseModel):
    actionName: str          # Title Case, represents ONE physical screen/feature
    description: str         # EXACTLY 5-7 words, MUST start with "It will"
    stepGroups: List[StepGroup]
    category: Optional[ActionCategory] = ActionCategory.manual


class EscalationAction(BaseModel):
    """A fallback action recommended alongside (never instead of) a
    low-confidence Goal -- see escalation.py."""
    deeplink: str
    message: str
    description: str
    originalType: Optional[str] = None


class EscalationRecommendation(BaseModel):
    """Attached to a Goal when the execution path that produced it judged
    its own confidence too low to hand over a fix with certainty. The
    plan is still returned in full -- this is an additive caveat, not a
    replacement. See escalation.py for the full design rationale and why
    the offline and LLM paths use different confidence signals to decide
    whether to attach one."""
    recommended: bool
    reason: str
    action: EscalationAction


class Goal(BaseModel):
    goal: str     # Exact syntax: "Follow these steps to perform this <Topic> Troubleshooting"
    title: str    # 2-3 words, sentence case (e.g. "Battery fast drain")
    actions: List[Action]
    score: float  # 0.0 - 1.0 confidence
    escalation: Optional[EscalationRecommendation] = None


class ContextDeeplinkResponse(BaseModel):
    """RAG response containing a list of Goal objects. Empty list = no_match fallback."""
    contexts: List[Goal] = []


# ---- API-level wrapper models (not in the original appendix, added for main.py) ----

class DeviceContext(BaseModel):
    """Optional, purely factual device-state signals (Task 35). Every field
    is optional and there's no "required" combination -- when a caller
    supplies none of this, device_signals.py's apply_device_context() is a
    no-op and the response is byte-identical to before this feature
    existed. When present, these numbers are used to (a) reorder same-
    category actions within a Goal so the step most relevant to the
    reported state runs first, and (b) add a short, honest advisory note
    to meta.device_context_notes -- never to invent a new action or claim
    a fix the catalog/reference text doesn't actually support. See
    device_signals.py for the full rationale and thresholds."""
    battery_pct: Optional[float] = Field(None, ge=0, le=100)
    storage_free_pct: Optional[float] = Field(None, ge=0, le=100)
    os_version: Optional[str] = None
    uptime_hours: Optional[float] = Field(None, ge=0)
    last_restart_hours_ago: Optional[float] = Field(None, ge=0)


class TroubleshootRequest(BaseModel):
    query: str
    siis_response: Optional[str] = None
    device: Optional[DeviceContext] = None
    # Task 36 (session_memory.py): entirely caller-supplied and opt-in --
    # an arbitrary caller-chosen string identifying "the same troubleshooting
    # conversation" across multiple requests, so a later call in the same
    # session can avoid re-suggesting an action this session already tried
    # and marked unhelpful via POST /v1/feedback. No session_id anywhere ->
    # this feature is a complete no-op.
    session_id: Optional[str] = None
    # Point 1 (image_analysis.py): an optional base64 data URL
    # (data:<mime>;base64,<data>) for a photo of the problem -- "so that
    # the problem will be very clear", the user's own words. Entirely
    # opt-in and additive: with this omitted, request handling is
    # byte-identical to before this feature existed. Deliberately NOT
    # format-validated here -- a malformed/corrupted value degrades
    # honestly inside image_analysis.py (meta.image_analysis.reason)
    # rather than failing the whole request with a 422.
    image_data_url: Optional[str] = None


class Meta(BaseModel):
    latency_ms: float
    cache_hit: bool
    model: str
    cost_usd: float
    fallback: Optional[str] = None
    device_context_notes: List[str] = []
    session_notes: List[str] = []
    # safety.py (physical-hazard short-circuit)
    safety_alert: bool = False
    safety_reason: Optional[str] = None
    # clarify.py (vague-complaint detection) -- additive only, never
    # affects `response.contexts`; see clarify.py's module docstring.
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    clarifying_topic_options: List[str] = []
    # answer_source.py -- where the plan's content actually came from
    # (official Samsung reference, generic built-in knowledge, AI general
    # knowledge, or the safety rule), surfaced for the UI's "where did this
    # come from?" disclosure.
    answer_source: Optional[Dict] = None
    # related_issues.py -- when this plan's own confidence was borderline
    # (it already carries an escalation recommendation, see escalation.py),
    # 2-3 other plausible root causes for the same reported symptom. Empty
    # whenever the engine is confident in its single best match.
    related_possibilities: List[str] = []
    # image_analysis.py (point 1) -- always present, shaped as
    # {"provided": bool, "analyzed": bool, "description": str|None, "reason": str|None}.
    # `reason` explains an honest degrade (no key, non-vision model, a
    # failed call) whenever provided=True but analyzed=False.
    image_analysis: Optional[Dict] = None
    # ambiguity.py (CLAM-inspired) -- {"ambiguous", "confidence", "reason",
    # "method": "llm"} when an LLM was available to compute a second
    # opinion on clarify.py's heuristic verdict, else None. Never lowers
    # needs_clarification, only ever able to raise it -- see the module.
    ambiguity: Optional[Dict] = None
    # recovery.py (Guided-Retry inspired, "When the Database Fails") --
    # whether Stage 1's LLM call failed and, if so, whether one structured
    # guided-retry attempt recovered it before falling through to the
    # offline path. recovery.NO_RECOVERY-shaped for the common no-failure
    # case, never absent.
    recovery: Optional[Dict] = None
    # topic_manager.py (DiagGPT-inspired) -- session-scoped topic-stack
    # tracking: {"action", "current_topic", "stack"}, or None for no
    # session_id / a hazard short-circuit (see topic_manager.py).
    topic_stack: Optional[Dict] = None


class TroubleshootResponse(BaseModel):
    query: str
    query_variations: List[str] = []
    response: ContextDeeplinkResponse
    meta: Meta


class BatchTroubleshootItem(BaseModel):
    query: str
    siis_response: Optional[str] = None
    device: Optional[DeviceContext] = None
    session_id: Optional[str] = None
    image_data_url: Optional[str] = None


class BatchTroubleshootRequest(BaseModel):
    """POST /v1/troubleshoot/batch -- runs several complaints through the
    same pipeline as /v1/troubleshoot in one HTTP round trip (e.g. a device
    health-check screen that wants results for 5-10 known symptom probes
    at once, without 5-10 separate requests). Capped at 20 items/request --
    this is a convenience batching endpoint, not a bulk-import job queue;
    see main.py's batch_limiter for why the per-request rate limit is
    tighter here than on the single-item endpoint."""
    items: List[BatchTroubleshootItem] = Field(..., min_length=1, max_length=20)


class ReportRequest(BaseModel):
    """POST /v1/report (Task 38) -- packages an already-computed
    troubleshooting result (the exact response body a prior
    /v1/troubleshoot or /v1/troubleshoot/stream call returned) into a
    compact, shareable report. Takes the result rather than re-running the
    pipeline: the caller already paid for that computation once, and
    formatting is a pure function of data it already has -- see report.py.
    `result` is intentionally typed as a plain dict rather than
    TroubleshootResponse: it's round-tripping a response this same service
    already produced and validated once, so re-validating it strictly here
    would only reject a caller's minor extra/missing field for no benefit."""
    result: Dict = Field(..., description="The exact response body from /v1/troubleshoot(/stream)")
    format: str = Field("markdown", description="\"markdown\" or \"html\"")


class ResolutionDeeplink(BaseModel):
    """One real matched deeplink from a Goal, paired with the action it
    belongs to -- see ResolutionRequest below."""
    deeplink: str
    action_name: str


class ResolutionRequest(BaseModel):
    """POST /v1/resolution (point 5) -- the goal-level "did this fix it?"
    Yes/No signal, asked once after a plan has been shown and attempted.
    Distinct from FeedbackRequest/POST /v1/feedback, which rates ONE
    deeplink's match quality and can be answered before ever trying it;
    this is about the outcome of the whole plan. See resolution.py for how
    a "No" is fanned out into the existing per-deeplink feedback/avoidance
    machinery rather than duplicating it."""
    goal_title: str                          # the Goal.title this resolution is about
    deeplinks: List[ResolutionDeeplink] = []  # every real deeplink this Goal's actions matched
    resolved: bool                           # True = "yes, fixed", False = "no, still broken"
    query: Optional[str] = None
    session_id: Optional[str] = None


class InvestigateStartRequest(BaseModel):
    """POST /v1/investigate/start -- begins an SIA-inspired interactive
    diagnosis (investigator.py): instead of committing to the engine's
    first guess, ask a short sequence of targeted questions to narrow down
    which of several candidate root causes is actually correct before
    running the normal troubleshooting pipeline. Entirely opt-in -- a
    caller that never calls this endpoint sees no behavior change
    whatsoever anywhere else in the service."""
    complaint: str
    session_id: Optional[str] = None


class InvestigateAnswerRequest(BaseModel):
    """POST /v1/investigate/answer -- submits the user's free-text answer
    to the most recent question from POST /v1/investigate/start (or a
    previous call to this same endpoint). See investigator.py."""
    investigation_id: str
    answer: str


class FeedbackRequest(BaseModel):
    """POST /v1/feedback -- thumbs up/down on one matched deeplink from a
    previous /v1/troubleshoot(/stream) response. Drives the adaptive
    re-ranking in deeplink_matching.py (see feedback.py), and -- when
    session_id is supplied and helpful=False -- also feeds session_memory.py
    (Task 36) so THIS session's later requests can avoid re-suggesting the
    exact same deeplink."""
    deeplink: str                    # the actionableDeeplink.deeplink string returned earlier
    action_name: str                 # the actionName it was attached to, for readability in logs
    helpful: bool
    query: Optional[str] = None      # original complaint, for traceability
    comment: Optional[str] = None
    session_id: Optional[str] = None
