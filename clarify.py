"""
clarify.py — Detects when a complaint is too vague to troubleshoot well and
proposes a clarifying question, instead of silently guessing at a plan the
user probably didn't ask for.

Why this exists: a real "smart guided troubleshooting engine" (the theme's
own name) should behave the way a good support agent does -- when someone
opens with "my phone isn't working" or "it's broken", a good agent's first
move is "tell me more: is this about the battery, the screen, Wi-Fi...?",
not a guess dressed up as a confident diagnosis. Nothing in this codebase
did that before this module -- every complaint, however thin, went straight
into Stage 0/1 extraction and got back *something*, even when that
something was a low-relevance guess with barely any signal behind it.

Design constraints (why this is purely ADDITIVE, unlike safety.py's hard
short-circuit):
  - The official 20-query eval set (results.jsonl, gate G3's >=95% coverage
    requirement) must never lose a non-empty `contexts` response because of
    this feature. So detect_vague_complaint() only ever *flags* a complaint
    via meta -- it never blocks or replaces normal Stage 0/1/2 extraction.
    The pipeline still runs exactly as before and still returns whatever
    plan it would have returned; this just adds an honest "you may want to
    tell us more" signal alongside it. Verified empirically against every
    complaint in sample_queries_real.json (all 20 are long, specific,
    screen-related complaints) -- none of them trip this heuristic; see
    tests/test_clarify.py's TestNoFalsePositivesOnOfficialQueries.
  - Deliberately keyword-based and conservative (same philosophy as
    safety.py and offline_fallback.py's relevance gate): a complaint is
    only flagged when it BOTH (a) contains no recognizable device-symptom
    keyword, AND (b) has very few meaningful words once generic filler
    ("my", "phone", "is", "having", ...) is stripped out. Both conditions
    together catch "my phone isn't working" / "it's broken" / "having some
    issues" without catching genuinely detailed complaints that simply use
    unusual phrasing -- a false "needs clarification" flag isn't harmful
    (it's additive, see above), but a noisy one would just be a bad UX
    signal a caller learns to ignore.
"""
from __future__ import annotations

import re
import string
from typing import Optional

# Same broad symptom vocabulary as pipeline.py's _guess_issue_phrase(), plus
# a handful more (charging/volume/notification/call/sim/fingerprint/lock) --
# any of these appearing anywhere in the complaint means there's a concrete
# symptom to go on, so we never flag it as vague regardless of length.
_SYMPTOM_KEYWORDS = (
    "battery", "charg", "camera", "screen", "display", "touch", "gesture",
    "flicker", "crack", "blank", "black screen", "slow", "lag", "performance",
    "wifi", "wi-fi", "bluetooth", "overheat", "hot", "storage", "app crash",
    "crash", "update", "speaker", "audio", "sound", "microphone", "mic",
    "signal", "network", "volume", "notification", "call", "sim", "gps",
    "location", "fingerprint", "face unlock", "lock screen", "keyboard",
    "freeze", "frozen", "restart", "reboot", "shut down", "shutting down",
    "connect", "pairing", "sync", "backup", "gallery", "contact", "message",
)

# Filler words that carry no diagnostic signal on their own -- stripped
# before counting "meaningful" words. Deliberately includes "phone"/
# "device"/"samsung"/"galaxy" since nearly every complaint mentions the
# device itself without that adding any specificity about WHAT is wrong.
_FILLER_WORDS = {
    "my", "the", "a", "an", "is", "are", "was", "were", "i", "it", "its",
    "it's", "this", "that", "phone", "device", "samsung", "galaxy",
    "having", "have", "has", "with", "on", "in", "of", "to", "and", "not",
    "please", "help", "me", "some", "there's", "there", "isn't", "doesn't",
    "won't", "wont", "just", "keeps", "keep", "getting", "seems", "seem",
    "up", "out", "again", "suddenly", "randomly", "still",
}

_VAGUE_MEANINGFUL_WORD_LIMIT = 3

CLARIFYING_TOPIC_OPTIONS = [
    "Battery / charging",
    "Screen / display",
    "Camera",
    "Wi-Fi / Bluetooth / signal",
    "Performance (slow, lag, freezing)",
    "Storage / software update",
    "Something else",
]


def _meaningful_word_count(text: str) -> int:
    normalized = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = normalized.split()
    return len([w for w in words if w not in _FILLER_WORDS])


def _has_symptom_keyword(text: str) -> bool:
    normalized = text.lower()
    return any(kw in normalized for kw in _SYMPTOM_KEYWORDS)


def detect_vague_complaint(raw_complaint: str) -> bool:
    """True when `raw_complaint` gives us too little to go on: no
    recognizable symptom keyword AND very few meaningful words once filler
    is stripped. Both conditions have to hold -- see module docstring."""
    if not raw_complaint or not raw_complaint.strip():
        return True
    if _has_symptom_keyword(raw_complaint):
        return False
    return _meaningful_word_count(raw_complaint) <= _VAGUE_MEANINGFUL_WORD_LIMIT


def build_clarifying_question(raw_complaint: str) -> str:
    return (
        "That doesn't mention a specific symptom yet, so this plan is a "
        "best-effort guess. For a more accurate fix, could you say more "
        "about what's actually happening -- for example, is it the "
        "battery/charging, the screen, the camera, Wi-Fi/Bluetooth, "
        "performance, or something else?"
    )
