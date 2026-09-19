"""
session_memory.py — session-scoped "don't re-suggest what already failed"
memory (Task 36).

This is deliberately a DIFFERENT mechanism from feedback.py's global
adaptive re-ranking. feedback.py's adjustment is shared across every
caller: a deeplink that keeps getting thumbs-down from MANY people sinks
for EVERYONE, permanently, because it's probably just a bad match. This
module is about ONE troubleshooting session: if a user in session S
already tried "Battery Settings -> power saving mode" via an earlier
/v1/troubleshoot call and told us via /v1/feedback it didn't help, THEIR
next call in the SAME session (a follow-up, a rephrased description of
the same problem, a related symptom) shouldn't casually recommend the
identical deeplink again as if it were a fresh idea -- even though that
exact deeplink might still be perfectly good advice for a different user
who's never tried it. Nothing here touches feedback.py's global scores.

Design, to stay consistent with the rest of this project's stance on never
silently withholding a real answer:

  - Session identity is entirely caller-supplied and OPT-IN (schema.py's
    new `session_id` field on TroubleshootRequest/BatchTroubleshootItem/
    FeedbackRequest). No session_id anywhere in a request -> this module
    is a complete no-op and behavior is unchanged from before Task 36.

  - deeplink_matching.py's best_match_explained() (both index variants)
    accepts an `avoid_deeplinks` set and, when the single best-ranked
    candidate is in it, walks DOWN the same ranking to the next candidate
    that (a) still clears the acceptance threshold and (b) isn't itself
    avoided -- see its own docstring for why this never happens by
    touching the threshold decision itself. If no such alternative
    exists, the avoided deeplink is still returned (never a spurious "no
    match"), just flagged `already_tried_this_session: true` in
    matchExplanation so a caller can render "you said this didn't help
    last time" instead of pretending it's new advice.

  - pipeline.py bypasses the semantic cache (both read and write) for any
    request carrying a non-empty avoid set -- Stage 2's deeplink choice is
    baked into the cached response, so serving a cache entry that was
    resolved under a DIFFERENT session's avoidance state (or none at all)
    would silently reintroduce exactly the deeplink this feature exists
    to steer away from. See pipeline.py's run_pipeline() for the exact
    bypass logic and tests/test_pipeline.py::TestSessionMemory for the
    regression test pinning this down.

Persisted the same simple way feedback.py/request_log.py are (a small
JSON file) -- no database needed for a hackathon-scale demo.
"""
from __future__ import annotations

import json
from pathlib import Path

TRIED_UNHELPFUL_PATH = Path("session_tried_unhelpful.json")

_cache: dict[str, list[str]] | None = None


def _load() -> dict[str, list[str]]:
    global _cache
    if _cache is not None:
        return _cache
    if TRIED_UNHELPFUL_PATH.exists():
        try:
            _cache = json.loads(TRIED_UNHELPFUL_PATH.read_text())
        except Exception:
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save() -> None:
    if _cache is not None:
        TRIED_UNHELPFUL_PATH.write_text(json.dumps(_cache, indent=2))


def record_unhelpful(session_id: str | None, deeplink: str | None) -> None:
    """Called from feedback.record_feedback() whenever helpful=False AND
    the caller supplied a session_id -- a no-op for either missing piece,
    since session tracking is strictly opt-in, never inferred."""
    if not session_id or not deeplink:
        return
    store = _load()
    tried = store.setdefault(session_id, [])
    if deeplink not in tried:
        tried.append(deeplink)
    _save()


def get_avoid_set(session_id: str | None) -> set[str]:
    """What pipeline.py passes down as deeplink_matching.py's
    avoid_deeplinks for this one request. Empty for no session_id, or a
    session_id that's never had unhelpful feedback recorded -- so a
    request without an active avoidance history behaves exactly like
    before this feature existed."""
    if not session_id:
        return set()
    store = _load()
    return set(store.get(session_id, []))
