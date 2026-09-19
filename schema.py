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

class TroubleshootRequest(BaseModel):
    query: str
    siis_response: Optional[str] = None


class Meta(BaseModel):
    latency_ms: float
    cache_hit: bool
    model: str
    cost_usd: float
    fallback: Optional[str] = None


class TroubleshootResponse(BaseModel):
    query: str
    query_variations: List[str] = []
    response: ContextDeeplinkResponse
    meta: Meta


class BatchTroubleshootItem(BaseModel):
    query: str
    siis_response: Optional[str] = None


class BatchTroubleshootRequest(BaseModel):
    """POST /v1/troubleshoot/batch -- runs several complaints through the
    same pipeline as /v1/troubleshoot in one HTTP round trip (e.g. a device
    health-check screen that wants results for 5-10 known symptom probes
    at once, without 5-10 separate requests). Capped at 20 items/request --
    this is a convenience batching endpoint, not a bulk-import job queue;
    see main.py's batch_limiter for why the per-request rate limit is
    tighter here than on the single-item endpoint."""
    items: List[BatchTroubleshootItem] = Field(..., min_length=1, max_length=20)


class FeedbackRequest(BaseModel):
    """POST /v1/feedback -- thumbs up/down on one matched deeplink from a
    previous /v1/troubleshoot(/stream) response. Drives the adaptive
    re-ranking in deeplink_matching.py (see feedback.py)."""
    deeplink: str                    # the actionableDeeplink.deeplink string returned earlier
    action_name: str                 # the actionName it was attached to, for readability in logs
    helpful: bool
    query: Optional[str] = None      # original complaint, for traceability
    comment: Optional[str] = None
