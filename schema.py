"""
schema.py — Data contract for the Smart Guided Troubleshooting Engine (Theme 2)

This mirrors Appendix A of the theme guide almost verbatim. DO NOT loosen these
rules — the judges' automated gates check exact conformance (word counts,
enums, deeplink verbatim match, etc.).
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel


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


class Goal(BaseModel):
    goal: str     # Exact syntax: "Follow these steps to perform this <Topic> Troubleshooting"
    title: str    # 2-3 words, sentence case (e.g. "Battery fast drain")
    actions: List[Action]
    score: float  # 0.0 - 1.0 confidence


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


class FeedbackRequest(BaseModel):
    """POST /v1/feedback -- thumbs up/down on one matched deeplink from a
    previous /v1/troubleshoot(/stream) response. Drives the adaptive
    re-ranking in deeplink_matching.py (see feedback.py)."""
    deeplink: str                    # the actionableDeeplink.deeplink string returned earlier
    action_name: str                 # the actionName it was attached to, for readability in logs
    helpful: bool
    query: Optional[str] = None      # original complaint, for traceability
    comment: Optional[str] = None
