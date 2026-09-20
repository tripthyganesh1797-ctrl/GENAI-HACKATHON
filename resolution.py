"""
resolution.py — the goal-level "did this fix it?" signal (point 5): once a
full plan (a Goal, with all its actions) has been shown and attempted, the
UI asks a single Yes/No question about the OUTCOME -- did the reported
problem actually get fixed? -- not about any one deeplink match's quality
in isolation.

Why this is a separate signal from feedback.py's existing thumbs up/down:
a per-deeplink 👍/👎 answers "was THIS specific step useful", which a user
can judge the moment it's shown. "Did this fix it?" can only be answered
honestly AFTER attempting the plan, and it's about the whole troubleshooting
outcome for this complaint, not a single action -- so record_resolution()
takes a Goal's title and its full set of matched deeplinks together, once,
rather than one deeplink-quality rating at a time.

Design, same "reuse existing signals, never duplicate a mechanism" pattern
used throughout this codebase: a "No" (not resolved) is folded, one at a
time, into feedback.py's EXISTING record_feedback() call for every real
deeplink this Goal actually matched (skipping the DUMMY_POSITIVE_DEEPLINK
placeholder -- same guard main.py's POST /v1/feedback route already
applies, since there's no real catalog match to score or avoid). That
means the exact same bounded per-deeplink score adjustment
(feedback.py's _adjustment_for) and the exact same session-scoped
avoidance list (session_memory.py, Task 36) that already exist for point 2
apply here automatically -- a "No" on this screen makes the NEXT request
in this session (whether the user retries the same complaint or clicks one
of related_issues.py's suggested alternatives) already steer away from the
deeplink that just failed. This module's own job stays narrow: log the
goal-level resolution event for its own honest record (append-only, same
pattern as feedback_log.jsonl/request_log.jsonl) and fan the signal out to
feedback.py -- nothing more.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import feedback as feedback_module
from deeplink_matching import DUMMY_POSITIVE_DEEPLINK

RESOLUTION_LOG_PATH = Path("resolution_log.jsonl")


def record_resolution(goal_title: str, deeplinks: list[dict], resolved: bool,
                       query: str = "", session_id: str = "") -> dict:
    """`deeplinks` is a list of {"deeplink": ..., "action_name": ...} for
    every action in this Goal that had a real (non-placeholder) matched
    deeplink -- the caller (main.py) builds this straight from the Goal
    object the earlier /v1/troubleshoot(/stream) response already returned.

    Never raises on an empty/placeholder-only deeplinks list: a Goal with
    no real catalog match can still be logged as resolved/unresolved, it
    just has nothing for feedback.py to score or avoid."""
    event = {
        "timestamp": time.time(),
        "goal_title": goal_title,
        "resolved": bool(resolved),
        "query": query,
        "session_id": session_id,
        "num_deeplinks": len(deeplinks),
    }
    with RESOLUTION_LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")

    deeplinks_updated = []
    for dl in deeplinks:
        deeplink = (dl or {}).get("deeplink", "")
        if not deeplink or deeplink == DUMMY_POSITIVE_DEEPLINK:
            continue  # no real catalog match -- nothing meaningful to score/avoid
        feedback_module.record_feedback(
            deeplink=deeplink,
            action_name=(dl or {}).get("action_name", ""),
            helpful=resolved,
            query=query,
            comment="auto-recorded from goal-level \"did this fix it?\" check",
            session_id=session_id,
        )
        deeplinks_updated.append(deeplink)

    return {"goal_title": goal_title, "resolved": bool(resolved), "deeplinks_updated": deeplinks_updated}


def resolution_summary() -> dict:
    """Powers /stats -- same shape/spirit as feedback.feedback_summary(),
    but at the "did the overall plan actually fix it" level rather than
    per-deeplink match quality. Reads the append-only log directly (there's
    no running aggregate to maintain here, unlike feedback.py's per-
    deeplink scores -- this is a single global rate, cheap to recompute)."""
    if not RESOLUTION_LOG_PATH.exists():
        return {"total_resolution_events": 0, "resolved": 0, "unresolved": 0, "resolved_rate": None}
    resolved = 0
    unresolved = 0
    with RESOLUTION_LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than crash
            if event.get("resolved"):
                resolved += 1
            else:
                unresolved += 1
    total = resolved + unresolved
    return {
        "total_resolution_events": total,
        "resolved": resolved,
        "unresolved": unresolved,
        "resolved_rate": round(resolved / total, 3) if total else None,
    }
