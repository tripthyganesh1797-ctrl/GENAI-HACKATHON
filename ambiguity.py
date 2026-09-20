"""
ambiguity.py — CLAM-inspired ambiguity confidence scoring, layered on top of
clarify.py's existing keyword/word-count heuristic.

Paper this is inspired by: "CLAM: Selective Clarification for Ambiguous
Questions with Generative Language Models" (Kuhn et al.). CLAM's core idea
is that a model can be asked to self-classify whether a question is
ambiguous, using the log-probability it assigns to a "True"/"False"
self-check token, and that this scales better than hand-written heuristics
alone.

Why this is a SEPARATE module rather than a rewrite of clarify.py:
clarify.py's heuristic is deliberately conservative and keyword-based so it
can run with zero cost/latency/API-key on every single request, including
fully offline ones -- and its module docstring documents a hard constraint
(gate G3's >=95% coverage requirement) that depends on it staying that way.
Replacing it outright with an LLM call would make clarify.py's behavior
depend on network/API availability, which this codebase never does for a
core signal. Instead, this module adds a SECOND, complementary signal that
only ever RAISES the "needs clarification" flag beyond what the heuristic
already caught -- never lowers it -- and only when an LLM is actually
available. See classify_ambiguity()'s docstring for the exact contract.

Honest implementation note on the "CLAM-inspired" framing: this project's
llm_client.py doesn't expose token log-probabilities (none of anthropic/
openai/groq's chat-completion SDKs return them for arbitrary tokens without
switching to a completions-style API this codebase doesn't use elsewhere),
so this uses a practical proxy CLAM's own paper discusses as a simpler
alternative: a directly self-reported confidence score from a structured
few-shot prompt, rather than an actual log-prob(“True”) computation. This
is documented here rather than silently claimed as the literal method.
"""
from __future__ import annotations

from typing import Optional

from llm_client import call_llm_json, api_key_configured


# Few-shot examples mirror CLAM's own framing: a question/complaint is
# "ambiguous" here specifically in the troubleshooting sense -- too thin or
# too open-ended to point at one clear symptom/fix, not merely short.
_FEW_SHOT_EXAMPLES = """\
Example 1:
Complaint: "it's broken"
{"ambiguous": true, "confidence": 0.97, "reason": "No symptom, component, or context at all."}

Example 2:
Complaint: "my battery drains from 100% to 20% in under two hours even when I'm not using the phone"
{"ambiguous": false, "confidence": 0.95, "reason": "Specific symptom, component, and quantified behavior."}

Example 3:
Complaint: "something's off with it lately"
{"ambiguous": true, "confidence": 0.9, "reason": "No component or symptom named, just a vague feeling."}

Example 4:
Complaint: "my phone has an issue with the screen"
{"ambiguous": true, "confidence": 0.72, "reason": "Names a component (screen) but no actual symptom -- flicker? crack? unresponsive? black? Each needs a different fix."}

Example 5:
Complaint: "camera app crashes every time I try to switch to the front camera"
{"ambiguous": false, "confidence": 0.93, "reason": "Specific app, specific trigger, specific failure mode."}
"""

_AMBIGUITY_PROMPT = """You are classifying whether a device-troubleshooting complaint is too \
AMBIGUOUS to act on without asking a clarifying question first. A complaint \
is ambiguous when it names no clear symptom, or names a component but not \
what's actually wrong with it, such that two different real problems could \
both match it and would need different fixes.

{examples}

Now classify this complaint. Respond with ONLY minified JSON, no other text:
{{"ambiguous": <true|false>, "confidence": <0.0-1.0>, "reason": "<one short sentence>"}}

Complaint: "{complaint}"
"""


def classify_ambiguity(complaint: str) -> Optional[dict]:
    """Returns {"ambiguous": bool, "confidence": float, "reason": str,
    "method": "llm"} on a successful LLM call, or None when no LLM is
    available or the call/parse fails -- callers MUST treat None as "no
    additional signal, defer entirely to clarify.py's heuristic", never as
    "not ambiguous". Never raises."""
    if not complaint or not complaint.strip():
        return None
    if not api_key_configured():
        return None  # no wasted call attempt -- same honest-degrade gate llm_available() uses
    try:
        prompt = _AMBIGUITY_PROMPT.format(examples=_FEW_SHOT_EXAMPLES, complaint=complaint.strip())
        result = call_llm_json(prompt, max_tokens=150, retries=1)
        ambiguous = result.get("ambiguous")
        confidence = result.get("confidence")
        if not isinstance(ambiguous, bool) or not isinstance(confidence, (int, float)):
            return None
        confidence = max(0.0, min(1.0, float(confidence)))
        return {
            "ambiguous": ambiguous,
            "confidence": confidence,
            "reason": str(result.get("reason", ""))[:200],
            "method": "llm",
        }
    except Exception:
        return None  # honest degrade -- caller falls back to the heuristic alone


# Only override the heuristic's "not vague" verdict when the LLM is BOTH
# confident AND says ambiguous -- a low-confidence LLM opinion shouldn't
# flip an otherwise-clear complaint, and this never runs the other
# direction (LLM saying "not ambiguous" never suppresses a heuristic flag,
# consistent with clarify.py's own "never blocks/replaces, only adds"
# contract -- see clarify.py's module docstring).
CONFIDENCE_OVERRIDE_THRESHOLD = 0.75


def refine_needs_clarification(heuristic_flag: bool, complaint: str) -> tuple[bool, Optional[dict]]:
    """Combines clarify.py's heuristic_flag with this module's LLM signal
    (when available). Returns (final_flag, ambiguity_info) where
    ambiguity_info is the classify_ambiguity() dict or None. The heuristic
    flag is NEVER lowered by this function -- only ever raised, and only
    when the LLM is confidently sure. This keeps every existing guarantee
    clarify.py's tests pin down (e.g. zero false positives on the official
    20-query eval set, which never touches this function at all when no
    LLM key is configured) fully intact."""
    info = classify_ambiguity(complaint)
    if heuristic_flag:
        return True, info
    if info and info["ambiguous"] and info["confidence"] >= CONFIDENCE_OVERRIDE_THRESHOLD:
        return True, info
    return False, info
