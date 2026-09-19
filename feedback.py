"""
feedback.py — Human-in-the-loop feedback + adaptive re-ranking.

A judge/user can tell the API "this deeplink match was right" or "this was
wrong" for a given action via POST /v1/feedback (see main.py). We persist
every event (an append-only log, same pattern as request_log.py) AND keep a
running per-deeplink aggregate that deeplink_matching.py consults on every
future search -- so a deeplink that keeps getting thumbs-down quietly sinks
in the ranking, and one that keeps getting thumbs-up quietly rises, without
retraining anything. This is a small, honest form of online learning that's
easy to explain in a demo: "watch the score change after I click thumbs
down twice."

Deliberately NOT a black box: the adjustment is a bounded, logged,
explainable additive term (see _adjustment_for), never a silent multiplier
that could flip a match on one stray click.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

FEEDBACK_LOG_PATH = Path("feedback_log.jsonl")
FEEDBACK_SCORES_PATH = Path("feedback_scores.json")

# Bounds on how much accumulated feedback can move a match score. Kept small
# relative to the ~0.12-0.45 score range the matchers operate in, so
# feedback nudges ranking rather than overriding real semantic/lexical
# signal outright.
MAX_ADJUSTMENT = 0.15

_scores_cache: dict[str, dict] | None = None


def _load_scores() -> dict[str, dict]:
    global _scores_cache
    if _scores_cache is not None:
        return _scores_cache
    if FEEDBACK_SCORES_PATH.exists():
        try:
            _scores_cache = json.loads(FEEDBACK_SCORES_PATH.read_text())
        except Exception:
            _scores_cache = {}
    else:
        _scores_cache = {}
    return _scores_cache


def _save_scores() -> None:
    if _scores_cache is not None:
        FEEDBACK_SCORES_PATH.write_text(json.dumps(_scores_cache, indent=2))


def record_feedback(deeplink: str, action_name: str, helpful: bool,
                     query: str = "", comment: str = "") -> dict:
    """Appends the raw event to the log and updates the running aggregate
    for this deeplink. Returns the updated aggregate for that deeplink."""
    event = {
        "timestamp": time.time(),
        "deeplink": deeplink,
        "action_name": action_name,
        "helpful": bool(helpful),
        "query": query,
        "comment": comment,
    }
    with FEEDBACK_LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")

    scores = _load_scores()
    agg = scores.setdefault(deeplink, {"helpful": 0, "unhelpful": 0, "action_names": []})
    if helpful:
        agg["helpful"] += 1
    else:
        agg["unhelpful"] += 1
    if action_name and action_name not in agg["action_names"]:
        agg["action_names"].append(action_name)
    _save_scores()
    return {"deeplink": deeplink, **agg, "adjustment": _adjustment_for(agg)}


def _adjustment_for(agg: dict) -> float:
    """Wilson-ish shrinkage: net signal scaled down for low vote counts so
    a single click can't swing a match, but a consistent pattern (5+ votes,
    lopsided) approaches the cap."""
    helpful, unhelpful = agg.get("helpful", 0), agg.get("unhelpful", 0)
    total = helpful + unhelpful
    if total == 0:
        return 0.0
    net_rate = (helpful - unhelpful) / total          # -1.0 .. 1.0
    confidence = min(1.0, math.log1p(total) / math.log1p(8))  # ramps up to ~8 votes
    return round(MAX_ADJUSTMENT * net_rate * confidence, 4)


def get_adjustment(deeplink: str) -> float:
    """What deeplink_matching.py calls per-candidate during ranking. O(1),
    safe to call in a hot loop -- reads from the in-memory cache."""
    if not deeplink:
        return 0.0
    scores = _load_scores()
    agg = scores.get(deeplink)
    if not agg:
        return 0.0
    return _adjustment_for(agg)


def feedback_summary() -> dict:
    """Powers /stats -- aggregate feedback health across the whole catalog."""
    scores = _load_scores()
    total_helpful = sum(v.get("helpful", 0) for v in scores.values())
    total_unhelpful = sum(v.get("unhelpful", 0) for v in scores.values())
    total = total_helpful + total_unhelpful
    boosted = sum(1 for v in scores.values() if _adjustment_for(v) > 0)
    demoted = sum(1 for v in scores.values() if _adjustment_for(v) < 0)
    return {
        "total_feedback_events": total,
        "helpful": total_helpful,
        "unhelpful": total_unhelpful,
        "helpful_rate": round(total_helpful / total, 3) if total else None,
        "deeplinks_with_feedback": len(scores),
        "deeplinks_boosted": boosted,
        "deeplinks_demoted": demoted,
    }
