"""
investigator.py — SIA-inspired interactive multi-turn diagnosis loop.

Paper this is inspired by: "LLM-as-an-Investigator: Evidence-First
Reasoning for Robust Interactive Problem Diagnosis". Its central finding is
"user-driven sycophancy" -- an LLM assistant tends to prematurely agree with
whatever cause the user first suggests (or the engine's own first guess)
instead of actually testing alternative explanations. Its fix, the
Solution Investigator Agent (SIA), keeps a probability distribution over
several candidate hypotheses, asks ONE targeted question chosen to best
discriminate between the leading candidates, updates the distribution from
the answer, and repeats until one hypothesis clears a confidence threshold
(tau) or a question budget runs out -- only THEN committing to an answer.

What this module reuses rather than reinvents (same "never duplicate a
mechanism" pattern as resolution.py/session_memory.py elsewhere in this
codebase):
  - pipeline.py's `_guess_issue_phrase()` symptom-bucket vocabulary, so a
    complaint's initial classification can never drift out of sync with
    what /stats and related_issues.py already use it for.
  - related_issues.py's hand-curated alternative-root-cause lists as the
    candidate HYPOTHESIS pool -- the exact same reviewed, zero-cost data
    this codebase already trusts for "or maybe it's this instead" caveats,
    reused here as the thing to actually discriminate between rather than
    just display.
  - pipeline.run_pipeline() for the FINAL answer once a hypothesis wins.
    This module's own job stops at "which hypothesis" -- it hands the
    winning hypothesis to the real pipeline by folding it into the
    complaint text (the exact same "a few more words in the complaint"
    trick image_analysis.py uses for photo descriptions), so Stage 0/1/2,
    safety.py, clarify.py, caching, device context, etc. all run
    completely unchanged and produce a real, schema-compliant, grounded
    plan -- never a plan this module invents or formats itself.

Dual-path, same philosophy as the rest of this codebase: when an LLM is
configured, it's used to (a) assign sensible initial priors, (b) generate a
genuinely targeted discriminative question each round, and (c) interpret
the user's free-text answer to update the probability vector. Every one of
those three steps has a fully deterministic, zero-cost offline fallback
(see the _offline_* functions below) so an investigation can run start to
finish with zero API key/cost/network -- degrading in QUALITY (a fixed
generic question sequence and a simple keyword-bucket probability update,
rather than genuinely tailored reasoning) but never in AVAILABILITY,
exactly like Stage 0/1 already do.

Session state: kept in an in-process dict, not persisted to disk. An
investigation is a short handful of request/response round trips (at most
QUESTION_BUDGET questions), never intended to survive a server restart the
way session_memory.py's avoidance data or feedback.py's scores are -- same
honestly-scoped tradeoff middleware.py's rate limiter documents for itself.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from llm_client import call_llm_json, api_key_configured
from pipeline import _guess_issue_phrase, run_pipeline
from related_issues import suggest_related_issues

# tau in the SIA paper: stop asking once the leading hypothesis's
# probability clears this bar.
CONFIDENCE_THRESHOLD = 0.90

# B in the SIA paper: never ask more than this many questions even if no
# hypothesis has cleared CONFIDENCE_THRESHOLD -- an investigation always
# terminates and hands back a real answer.
QUESTION_BUDGET = 3

# In-process only -- see module docstring's "Session state" note.
_STORE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Candidate hypotheses
# ---------------------------------------------------------------------------

# Used whenever related_issues.py has no curated entry for this symptom
# bucket -- a generic software/hardware/external three-way split that
# applies to virtually any symptom, so an investigation can always start
# with at least two genuinely different candidates to discriminate between,
# even fully offline.
_GENERIC_FALLBACK_HYPOTHESES = [
    "A software issue -- a recent update, an app conflict, or a temporary glitch",
    "A hardware issue specific to this device",
    "An external factor -- network, an accessory, or how the device is being used",
]


def _candidate_hypotheses(issue_phrase: str) -> list[str]:
    curated = suggest_related_issues(issue_phrase)
    return curated if curated else list(_GENERIC_FALLBACK_HYPOTHESES)


# ---------------------------------------------------------------------------
# Initial priors
# ---------------------------------------------------------------------------

def _llm_initial_priors(hypotheses: list[str], raw_complaint: str) -> Optional[dict[str, float]]:
    try:
        listed = "\n".join(f"- {h}" for h in hypotheses)
        prompt = (
            "A user reported this device problem:\n"
            f'"{raw_complaint}"\n\n'
            "Here are candidate root causes:\n"
            f"{listed}\n\n"
            "Assign each one an initial probability (0.0-1.0) that it's the real "
            "cause, based only on the complaint text so far -- probabilities must "
            "sum to 1.0. Respond with ONLY minified JSON:\n"
            '{"probabilities": {"<exact hypothesis text>": <number>, ...}}'
        )
        result = call_llm_json(prompt, max_tokens=300, retries=1)
        probs = result.get("probabilities")
        if not isinstance(probs, dict):
            return None
        return _validate_and_normalize(probs, hypotheses)
    except Exception:
        return None


def _initial_priors(hypotheses: list[str], raw_complaint: str) -> dict[str, float]:
    if api_key_configured():
        llm_priors = _llm_initial_priors(hypotheses, raw_complaint)
        if llm_priors:
            return llm_priors
    # Offline fallback: descending deterministic weights (n, n-1, ..., 1),
    # normalized -- a soft prior that favors earlier entries, since
    # related_issues.py's curated lists are informally hand-ordered by
    # likelihood, and _GENERIC_FALLBACK_HYPOTHESES lists software first as
    # simply the statistically most common category of device complaint.
    n = len(hypotheses)
    raw_weights = [n - i for i in range(n)]
    total = sum(raw_weights)
    return {h: round(w / total, 4) for h, w in zip(hypotheses, raw_weights)}


def _validate_and_normalize(probs: dict, hypotheses: list[str]) -> Optional[dict[str, float]]:
    """Every hypothesis must be present with a numeric value; anything else
    is treated as an unparseable/untrustworthy LLM response and the caller
    falls back to the offline path. Always renormalizes so the result sums
    to 1.0 even if the model's own numbers didn't quite."""
    try:
        values = {h: float(probs[h]) for h in hypotheses}
    except (KeyError, TypeError, ValueError):
        return None
    values = {h: max(0.01, v) for h, v in values.items()}
    total = sum(values.values())
    if total <= 0:
        return None
    return {h: round(v / total, 4) for h, v in values.items()}


# ---------------------------------------------------------------------------
# Targeted discriminative question
# ---------------------------------------------------------------------------

# Offline fallback question pool: deliberately generic (no symptom-specific
# wording) so it works for any hypothesis set, but each question still
# carries real discriminative value along the software / hardware /
# external axis most curated hypothesis lists (and the generic fallback
# triplet) naturally split along -- see _offline_update_probabilities().
_GENERIC_DISCRIMINATIVE_QUESTIONS = [
    "Did this start right after a specific event -- a software update, "
    "installing or using a particular app, a drop or physical impact, or "
    "exposure to water, heat, or cold? Or did it just start happening with "
    "no clear trigger?",
    "Does it happen every single time, consistently -- or only sometimes, "
    "unpredictably?",
    "Have you already tried restarting the device? If so, did that change "
    "anything at all, even temporarily?",
]


def _llm_next_question(hypotheses: list[str], probabilities: dict[str, float],
                        asked: list[str], raw_complaint: str) -> Optional[str]:
    try:
        ranked = sorted(hypotheses, key=lambda h: -probabilities.get(h, 0))
        listed = "\n".join(f"- {h} (current probability: {probabilities.get(h, 0):.2f})" for h in ranked)
        asked_block = ("\nAlready asked (do not repeat):\n" + "\n".join(f"- {q}" for q in asked)) if asked else ""
        prompt = (
            "You are diagnosing a device problem by asking ONE targeted "
            "question at a time, to distinguish between candidate causes.\n\n"
            f'Original complaint: "{raw_complaint}"\n\n'
            f"Candidate causes:\n{listed}\n"
            f"{asked_block}\n\n"
            "Write ONE new, short, specific question that would best help tell "
            "the top 2 candidates apart -- something the user can answer from "
            "direct observation, not technical knowledge. Respond with ONLY "
            'minified JSON: {"question": "<the question>"}'
        )
        result = call_llm_json(prompt, max_tokens=150, retries=1)
        question = result.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
        return None
    except Exception:
        return None


def _offline_next_question(asked: list[str]) -> Optional[str]:
    for q in _GENERIC_DISCRIMINATIVE_QUESTIONS:
        if q not in asked:
            return q
    return None  # pool exhausted -- caller resolves with the current best guess


def _next_question(hypotheses: list[str], probabilities: dict[str, float],
                    asked: list[str], raw_complaint: str) -> Optional[str]:
    if api_key_configured():
        q = _llm_next_question(hypotheses, probabilities, asked, raw_complaint)
        if q:
            return q
    return _offline_next_question(asked)


# ---------------------------------------------------------------------------
# Probability update
# ---------------------------------------------------------------------------

_SOFTWARE_SIGNAL_WORDS = {
    "update", "updated", "app", "apps", "software", "recent", "recently",
    "install", "installed", "sometimes", "occasionally", "randomly",
    "intermittent", "intermittently", "glitch",
}
_HARDWARE_SIGNAL_WORDS = {
    "drop", "dropped", "drops", "water", "wet", "heat", "hot", "crack",
    "cracked", "physical", "hardware", "always", "every", "consistently",
    "constant", "constantly", "old", "wear", "worn",
}
_EXTERNAL_SIGNAL_WORDS = {
    "network", "wifi", "wi-fi", "router", "accessory", "cable", "charger",
    "case", "protector", "other device", "carrier", "adapter",
}
_HYPOTHESIS_BUCKET_WORDS = {
    "software": _SOFTWARE_SIGNAL_WORDS | {"bug", "app"},
    "hardware": _HARDWARE_SIGNAL_WORDS | {"fault", "damage", "digitizer"},
    "external": _EXTERNAL_SIGNAL_WORDS,
}


def _text_signal_bucket(text: str) -> Optional[str]:
    """Best-effort classification of free text into "software"/"hardware"/
    "external" by keyword overlap, or None when no bucket clearly wins.
    Used on BOTH the user's answer and each hypothesis's own label text --
    same small, reviewable keyword-matching approach the rest of this
    codebase uses (clarify.py, safety.py, related_issues.py's own bucket
    keys), not a second hidden classifier."""
    lowered = text.lower()
    scores = {
        bucket: sum(1 for w in words if w in lowered)
        for bucket, words in _HYPOTHESIS_BUCKET_WORDS.items()
    }
    best_bucket = max(scores, key=scores.get)
    if scores[best_bucket] == 0:
        return None
    # A tie between buckets carries no real signal either.
    if list(scores.values()).count(scores[best_bucket]) > 1:
        return None
    return best_bucket


def _offline_update_probabilities(hypotheses: list[str], probabilities: dict[str, float],
                                   answer_text: str) -> dict[str, float]:
    """Deterministic Bayesian-ish update: nudges any hypothesis whose own
    bucket (software/hardware/external) matches the answer's bucket up,
    nudges the rest down slightly, then renormalizes. A genuinely simple
    approximation of SIA's own LLM-driven update -- documented here as
    exactly that, not dressed up as anything more rigorous."""
    answer_bucket = _text_signal_bucket(answer_text)
    if answer_bucket is None:
        return dict(probabilities)  # no usable signal -- leave priors untouched
    updated = {}
    for h in hypotheses:
        p = probabilities.get(h, 1.0 / len(hypotheses))
        h_bucket = _text_signal_bucket(h)
        if h_bucket == answer_bucket:
            p += 0.20
        else:
            p -= 0.05
        updated[h] = max(0.01, min(0.97, p))
    total = sum(updated.values())
    return {h: round(v / total, 4) for h, v in updated.items()}


def _llm_update_probabilities(hypotheses: list[str], probabilities: dict[str, float],
                               question: str, answer_text: str, raw_complaint: str) -> Optional[dict[str, float]]:
    try:
        listed = "\n".join(f"- {h} (current probability: {probabilities.get(h, 0):.2f})" for h in hypotheses)
        prompt = (
            "You are updating your belief about which of several candidate "
            "causes is correct, given a new piece of evidence.\n\n"
            f'Original complaint: "{raw_complaint}"\n'
            f"Candidate causes:\n{listed}\n\n"
            f'Question just asked: "{question}"\n'
            f'User\'s answer: "{answer_text}"\n\n'
            "Update each candidate's probability in light of this answer. "
            "Probabilities must sum to 1.0. Respond with ONLY minified JSON:\n"
            '{"probabilities": {"<exact hypothesis text>": <number>, ...}}'
        )
        result = call_llm_json(prompt, max_tokens=300, retries=1)
        probs = result.get("probabilities")
        if not isinstance(probs, dict):
            return None
        return _validate_and_normalize(probs, hypotheses)
    except Exception:
        return None


def _update_probabilities(hypotheses: list[str], probabilities: dict[str, float],
                           question: str, answer_text: str, raw_complaint: str) -> dict[str, float]:
    if api_key_configured():
        updated = _llm_update_probabilities(hypotheses, probabilities, question, answer_text, raw_complaint)
        if updated:
            return updated
    return _offline_update_probabilities(hypotheses, probabilities, answer_text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _public_view(state: dict, question: Optional[str] = None,
                  final_result: Optional[dict] = None) -> dict:
    ranked = sorted(state["probabilities"].items(), key=lambda kv: -kv[1])
    return {
        "investigation_id": state["investigation_id"],
        "status": state["status"],
        "hypotheses": [{"label": h, "probability": p} for h, p in ranked],
        "questions_asked": len(state["answers"]),
        "question_budget": QUESTION_BUDGET,
        "question": question,
        "winning_hypothesis": state.get("winning_hypothesis"),
        "final_result": final_result if final_result is not None else state.get("final_result"),
    }


def _resolve(state: dict) -> dict:
    """Commits to the current leading hypothesis, runs it through the REAL
    pipeline (see module docstring), and marks the investigation resolved."""
    top_hyp = max(state["probabilities"], key=state["probabilities"].get)
    state["status"] = "resolved"
    state["winning_hypothesis"] = top_hyp
    investigated_complaint = (
        f"{state['raw_complaint']}\n\n"
        f"[After a short diagnostic Q&A, the most likely cause was narrowed "
        f"down to: {top_hyp}]"
    )
    final_result = run_pipeline(investigated_complaint, session_id=state.get("session_id"))
    # Same guarantee image_analysis.py's folded-in photo description has
    # (see tests/test_image_pipeline.py's TestImageFoldedIntoComplaint):
    # the internal hint is allowed to influence technical_query/matching,
    # but response["query"] must still echo exactly what the user actually
    # typed, never this module's own internal bookkeeping text.
    final_result["query"] = state["raw_complaint"]
    state["final_result"] = final_result
    return _public_view(state, final_result=final_result)


def start_investigation(raw_complaint: str, session_id: Optional[str] = None) -> dict:
    """Begins a new investigation for `raw_complaint`, returning the first
    targeted question and the initial hypothesis/probability state. Never
    raises: every internal LLM call degrades to its offline counterpart."""
    issue_phrase = _guess_issue_phrase(raw_complaint)
    hypotheses = _candidate_hypotheses(issue_phrase)
    probabilities = _initial_priors(hypotheses, raw_complaint)

    investigation_id = uuid.uuid4().hex
    state = {
        "investigation_id": investigation_id,
        "session_id": session_id,
        "raw_complaint": raw_complaint,
        "issue_phrase": issue_phrase,
        "hypotheses": hypotheses,
        "probabilities": probabilities,
        "asked_questions": [],
        "answers": [],
        "status": "in_progress",
        "created_at": time.time(),
    }

    # An already-confident prior (e.g. only one real candidate exists)
    # skips straight to resolution rather than asking a pointless question.
    top_prob = max(probabilities.values())
    if top_prob >= CONFIDENCE_THRESHOLD or len(hypotheses) < 2:
        _STORE[investigation_id] = state
        return _resolve(state)

    question = _next_question(hypotheses, probabilities, [], raw_complaint)
    state["asked_questions"].append(question)
    _STORE[investigation_id] = state
    return _public_view(state, question=question)


def answer_investigation(investigation_id: str, answer_text: str) -> Optional[dict]:
    """Records an answer to the most recent question, updates the
    probability vector, and either asks another targeted question or
    resolves the investigation (confidence threshold reached, or the
    question budget is exhausted). Returns None when investigation_id is
    unknown -- main.py turns that into a 404. Re-answering an already-
    resolved investigation is a harmless idempotent no-op (returns its
    existing final result rather than erroring or double-running the
    pipeline)."""
    state = _STORE.get(investigation_id)
    if state is None:
        return None
    if state["status"] != "in_progress":
        return _public_view(state)

    last_question = state["asked_questions"][-1] if state["asked_questions"] else ""
    state["answers"].append(answer_text)
    state["probabilities"] = _update_probabilities(
        state["hypotheses"], state["probabilities"], last_question, answer_text, state["raw_complaint"]
    )

    top_prob = max(state["probabilities"].values())
    if top_prob >= CONFIDENCE_THRESHOLD or len(state["answers"]) >= QUESTION_BUDGET:
        return _resolve(state)

    next_q = _next_question(state["hypotheses"], state["probabilities"], state["asked_questions"], state["raw_complaint"])
    if next_q is None:
        return _resolve(state)  # offline question pool exhausted -- commit to the current best guess

    state["asked_questions"].append(next_q)
    return _public_view(state, question=next_q)


def investigation_stats() -> dict:
    """Powers /stats -- same spirit as feedback.feedback_summary()/
    resolution.resolution_summary(), but for this in-process-only store
    (see module docstring's "Session state" note -- these counts reset on
    restart, unlike the other two, which are honest about that here too)."""
    statuses = [s["status"] for s in _STORE.values()]
    return {
        "total_investigations": len(statuses),
        "in_progress": statuses.count("in_progress"),
        "resolved": statuses.count("resolved"),
    }


def get_investigation(investigation_id: str) -> Optional[dict]:
    """Read-only lookup, no state change -- for polling/introspection/tests.
    Re-surfaces the current pending question (if any) so a caller that lost
    track of it (e.g. after a dropped connection) can recover without
    re-answering anything."""
    state = _STORE.get(investigation_id)
    if state is None:
        return None
    pending_question = state["asked_questions"][-1] if (
        state["status"] == "in_progress" and state["asked_questions"]
    ) else None
    return _public_view(state, question=pending_question, final_result=state.get("final_result"))
