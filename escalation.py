"""
escalation.py — shared confidence-gated escalation recommendation.

Both execution paths can attach the SAME structured `escalation` object to
a Goal when they judge their own match confidence too low to hand over a
fix with full certainty. Crucially, this never REPLACES the plan -- the
troubleshooting steps are still returned in full, just flagged with an
honest caveat and a backup action. Withholding an automated plan entirely
for a borderline-but-plausible match would be a worse user experience
than showing it with a clear "this one's less certain" signal; silently
presenting every match with equal confidence would be worse still.

The offline and LLM paths use genuinely different confidence signals (the
offline path's own bag-of-words `section_relevance()` in
offline_fallback.py vs. the LLM's self-reported `Goal.score` from
prompts.py's STAGE1 contract), so each path decides FOR ITSELF whether to
escalate, using whichever signal it actually has. This module only owns
the shared recommendation shape and the target action, so both paths
produce byte-identical `escalation` objects rather than two subtly
different ones.

Honest scope note: this is a genuinely different failure mode than the
"Assistive menu" grounding mismatch documented in eval/metrics.md's known
-limitations section. That case has comparatively HIGH computed relevance
(0.556) despite being topically wrong -- shared generic vocabulary fools
the bag-of-words scorer while looking confident, so a low-confidence gate
like this one does NOT catch it (verified directly: see
tests/test_escalation.py). This is a complementary safety net for
genuinely thin/uncertain matches, not a fix for topical grounding
mismatches -- that limitation is still exactly what it was before.
"""
from __future__ import annotations

# The LLM path (prompts.py's STAGE1 contract) asks the model to self-report
# a 0.0-1.0 confidence per Goal. 0.6 is a deliberately moderate bar: low
# enough that a reasonably-grounded match isn't second-guessed for no
# reason, high enough to catch a match the model itself wasn't sure about.
LLM_SCORE_ESCALATION_THRESHOLD = 0.6

# "View Device Details" -- runs Samsung Members' own full device
# diagnostic (DL-0478 in the real 578-entry catalog). Chosen deliberately
# from real official data, not invented: when the engine isn't confident
# in a specific fix, pointing at the official self-diagnostic tool is a
# more honest, in-catalog answer than fabricating a "contact support"
# deeplink that doesn't exist anywhere in deeplinks.json.
_ESCALATION_ACTION = {
    "deeplink": "bixby://masked/act/5930a08d3d",
    "message": "Run Full Device Diagnostic",
    "description": "Runs a full device status diagnosis in Samsung Members to detect hardware or software issues requiring attention.",
    "originalType": "onClickURL",
}


def build_escalation_recommendation(reason: str) -> dict:
    """Returns the shared `escalation` object shape both paths attach to a
    low-confidence Goal. `reason` is a short, specific, human-readable
    explanation of *why* this particular match was flagged -- never a
    generic "we're not sure" -- so the caveat is itself trustworthy."""
    return {
        "recommended": True,
        "reason": reason,
        "action": dict(_ESCALATION_ACTION),
    }
