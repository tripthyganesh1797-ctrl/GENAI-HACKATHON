"""
recovery.py — Guided-Retry style recovery when the LLM path fails mid-Stage-1,
inspired by "When the Database Fails: Prompting LLM Dialogue Agents for Safe
Recovery in Task-Oriented Dialogue".

That paper's core finding: when a task-oriented dialogue agent's backend
call fails, how you tell the model about it matters enormously. A "Naive"
strategy (say nothing, let the model improvise) produces the most
hallucination; an "Inform" strategy (just say it failed) is better; a
"Guided-Retry" strategy -- an explicit, structured, per-failure-type
recovery procedure -- produces the safest, least-hallucinated recovery.

Where this project's own "backend" analog lives: Stage 1 (pipeline.py's
stage1_extract()) is the step that actually produces a plan's content by
calling the LLM. Before this module existed, a Stage-1 LLM
exception/malformed-JSON was handled the "Naive" way this paper's baseline
uses -- a bare `except Exception: pass` that silently drops straight to the
offline fallback with zero acknowledgement of what happened or any
structured second attempt. That's honest (the offline fallback never
hallucinates a Samsung-official answer, see answer_source.py), but it wastes
a real, likely-recoverable failure (a transient network blip, a model that
returned malformed JSON once) without ever giving the LLM path a real,
guided second chance -- and it gives an operator/judge zero visibility into
whether failures are happening at all.

This module owns:
  1. classify_llm_failure() -- names what happened (currently only
     "llm_call_failed": the LLM path was available and attempted, but
     raised -- a real API/parse failure, not a "not applicable" case).
     Deliberately does NOT try to distinguish empty-result/wrong-domain the
     way the paper's fault-injection setup does: this project has no live
     backend database to inject those specific faults into (Stage 1 either
     succeeds, or the whole call fails) -- see the module-level scope note
     below for why "empty result" is intentionally left alone.
  2. build_guided_retry_suffix() -- the explicit structured recovery
     instruction appended to the SAME Stage-1 prompt for exactly one retry
     attempt, per failure type -- never a second blind identical retry
     (call_llm_json() already retries transient JSON-parse noise
     internally; this is a higher-level retry AFTER that's exhausted).
  3. NO_RECOVERY -- the default meta.recovery shape for the overwhelming
     common case (LLM unavailable at all, or the first attempt just
     worked) so every response has a consistent, honest shape.

Scope note on "empty result": Stage 1 returning `{"contexts": []}` (a
confident, honest "no match") is NOT treated as a failure here and never
triggers a guided retry -- that's clarify.py/offline_fallback.py's
existing "never invent, degrade honestly" behavior working exactly as
intended, not something to retry away. Guided-Retry only fires for an
actual failure (an exception -- a real error, not an honest empty answer),
consistent with every other module in this codebase never second-guessing
a legitimate no_match.
"""
from __future__ import annotations

from typing import Optional

FAILURE_LLM_CALL_FAILED = "llm_call_failed"

# The common, expected, NOT-a-failure case: no LLM configured at all (the
# offline fallback is simply the primary path for this request), or the
# first Stage-1 attempt succeeded outright. Every response carries a
# meta.recovery object with this shape (or the failure variant below) so
# a caller never has to special-case a missing field.
NO_RECOVERY = {
    "failure_detected": False,
    "failure_type": None,
    "detail": None,
    "guided_retry_attempted": False,
    "guided_retry_succeeded": None,
}

_RETRY_SUFFIXES = {
    FAILURE_LLM_CALL_FAILED: (
        "\n\n---\nRECOVERY INSTRUCTION (this is an automatic retry -- your "
        "previous response either failed or could not be parsed as valid "
        "JSON): Return ONLY minified, valid JSON matching the schema above. "
        "No markdown code fences, no prose before or after the JSON, no "
        "commentary. If you genuinely cannot produce a confident, "
        "well-grounded match for this complaint, return exactly "
        '{"contexts": []} -- do NOT invent a plausible-sounding Goal just '
        "to have something to return; an honest empty result is always "
        "better than a fabricated one.\n---"
    ),
}


def classify_llm_failure(llm_was_available: bool, exception: Optional[Exception]) -> dict:
    """`llm_was_available` mirrors pipeline.llm_available() at the moment
    Stage 1 was attempted. Returns NO_RECOVERY when there's genuinely
    nothing to report (no LLM configured, or no exception was raised),
    or a failure dict (failure_detected=True, failure_type=..., detail=...,
    plus the same guided_retry_* keys defaulted to their "not yet
    attempted" values so the caller can fill them in after actually
    attempting the retry)."""
    if not llm_was_available or exception is None:
        return dict(NO_RECOVERY)
    return {
        "failure_detected": True,
        "failure_type": FAILURE_LLM_CALL_FAILED,
        "detail": str(exception)[:300],
        "guided_retry_attempted": False,
        "guided_retry_succeeded": None,
    }


def build_guided_retry_suffix(failure_type: Optional[str]) -> str:
    """Returns the structured recovery instruction to append to the
    ORIGINAL Stage-1 prompt for one retry attempt. Empty string for an
    unrecognized/None failure_type (defensive default -- callers only ever
    pass what classify_llm_failure() itself returned, so this should
    always hit)."""
    return _RETRY_SUFFIXES.get(failure_type, "")
