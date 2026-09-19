"""
safety.py — physical-hazard short-circuit.

Why this exists: a complaint like "my battery looks swollen and the back
cover is bulging out" is not a normal troubleshooting scenario. A swollen
lithium-ion battery is a real fire/chemical-burn risk, and offering
software steps for it (even reasonable-sounding ones like "check battery
usage in Settings") is actively unsafe advice -- it implicitly tells the
person to keep using or handling a device that should be set down and left
alone. This module detects that narrow class of complaint and short-
circuits the pipeline to a single, unambiguous safety instruction instead
of ever attempting to derive troubleshooting steps for it.

Deliberately NOT the same mechanism as escalation.py. escalation.py adds a
caveat to a plan the system is genuinely unsure about (low relevance
score); this module fires on a much narrower, high-precision set of literal
danger signals (swelling, smoke, fire, sparks, chemical smell, leaking
battery, burns) and, when it fires, REPLACES the plan entirely -- there is
no safe "also try these software steps" version of this response.

Deliberately keyword-based, not LLM-based: this has to work identically
and instantly on the offline path (no API key, no network) and the LLM
path both, and a safety check that only sometimes fires depending on
which execution path served the request is worse than a simple, narrow,
always-on rule. False negatives are always possible with any keyword
list (a symptom described in unusual words won't match) -- this is a
floor, not a substitute for a person's own judgement, and the keyword
list is intentionally narrow (specific hazard phrases only, never a
generic word like "hot") to avoid false-positiving on ordinary
overheating/performance complaints that already have real, safe
self-service troubleshooting paths.
"""
from __future__ import annotations

import re

# Each entry is (compiled pattern, short human-readable reason). Patterns
# are intentionally specific phrases/word-forms, not single generic words
# like "hot" or "leak" alone -- those already describe ordinary complaints
# (Device Overheating, a headphone jack, etc.) with real self-service
# troubleshooting paths, and treating them as a hazard would wrongly
# block a normal answer. Every pattern here is a phrase a person would
# only realistically use to describe an actual physical hazard.
_HAZARD_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bswoll(?:en|ing)\b"), "battery swelling"),
    (re.compile(r"\bbulg(?:e|ing|ed)\b"), "battery/case bulging"),
    (re.compile(r"\bpuffy\b"), "battery swelling (puffy)"),
    (re.compile(r"batter\w*\D{0,20}(?:expand|expanding|expanded|pushing|deform)"
                r"|(?:expand|expanding|expanded|deform)\w*\D{0,20}batter"),
     "battery physically expanding/deforming"),
    (re.compile(r"\bsmok(?:e|ing)\b"), "smoke"),
    (re.compile(r"\b(?:on fire|caught fire|catching fire|is on fire)\b"), "fire"),
    (re.compile(r"\bspark(?:s|ing)?\b"), "sparking"),
    (re.compile(r"burn(?:ing|t)?\s+smell|smell\w*\s+(?:of|like)\s+burning|burning\s+odor"),
     "burning smell"),
    (re.compile(r"chemical\s+smell"), "chemical smell"),
    (re.compile(r"hiss(?:ing)?\s+(?:sound|noise)"), "hissing sound"),
    (re.compile(r"batter\w*\D{0,20}leak|leak\w*\D{0,20}batter"), "battery leaking"),
    (re.compile(r"pop(?:ping)?\s+(?:sound|noise)\D{0,20}batter"
                r"|batter\D{0,20}pop(?:ping)?\s+(?:sound|noise)"),
     "battery popping sound"),
    (re.compile(r"burn(?:ed|t)?\s+my\s+(?:hand|finger|skin)|burned\s+me\b"), "burn injury"),
)


def detect_physical_hazard(raw_complaint: str) -> str | None:
    """Returns a short human-readable hazard reason if `raw_complaint`
    describes a genuine physical safety hazard, else None. Case-
    insensitive, runs on the raw complaint text directly -- deliberately
    independent of Stage 0 normalization/translation, so it can't be
    skipped or altered by anything downstream."""
    text = raw_complaint.lower()
    for pattern, reason in _HAZARD_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def build_safety_goal(hazard_reason: str) -> dict:
    """A single, schema-compliant Goal instructing the person to stop
    using the device and contact Samsung Support -- no software steps,
    no actionable deeplink (matches the same manual/no-deeplink pattern
    already used for real hardware issues, e.g. the official
    sample_output.json's "Schedule Screen Repair Service" action).
    category is "critical": this must always sort last and must never be
    mistaken for a routine auto action (see validators.py's
    validate_category_ordering)."""
    return {
        "goal": "Follow these steps to perform this Battery Safety Troubleshooting",
        "title": "Battery safety hazard",
        "score": 0.99,
        "actions": [{
            "actionName": "Stop Use And Contact Support",
            "description": "It will connect you with Samsung support",
            "category": "critical",
            "stepGroups": [{
                "steps": [
                    "Stop using the device immediately.",
                    "Do not charge the device or attempt to power it on.",
                    "Move the device away from flammable materials and people, "
                    "ideally outdoors or on a non-flammable surface.",
                    "Contact Samsung Support or visit an authorized Samsung service "
                    "center immediately -- do not attempt the usual software "
                    "troubleshooting steps for this issue.",
                ],
                "actionableDeeplink": None,
                "validationDeeplink": None,
            }],
        }],
    }
