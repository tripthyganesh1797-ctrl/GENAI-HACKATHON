"""
conftest.py — shared fixtures.

cache.py / request_log.py / feedback.py / session_memory.py all persist to
plain relative-path files (cache_store.json, request_log.jsonl,
feedback_log.jsonl, feedback_scores.json, session_tried_unhelpful.json) so
the demo stays a "just run it" single-process service with zero external
infra. That's the right call for a hackathon service, but it means tests
MUST redirect those paths into a scratch directory -- otherwise running
the suite would read/write the developer's real cache and log files (and
could make tests order-dependent on whatever state a previous manual run
left behind).
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolated_state_files(tmp_path, monkeypatch):
    """Point every module's persisted-state file at a fresh tmp_path for
    every test, and reset feedback.py's in-memory cache so tests can't
    leak state into each other via that module-level cache variable."""
    import cache
    import request_log
    import feedback
    import session_memory
    import middleware

    monkeypatch.setattr(cache, "CACHE_FILE", str(tmp_path / "cache_store.json"))
    monkeypatch.setattr(request_log, "LOG_FILE", str(tmp_path / "request_log.jsonl"))
    monkeypatch.setattr(feedback, "FEEDBACK_LOG_PATH", tmp_path / "feedback_log.jsonl")
    monkeypatch.setattr(feedback, "FEEDBACK_SCORES_PATH", tmp_path / "feedback_scores.json")
    monkeypatch.setattr(feedback, "_scores_cache", None)
    monkeypatch.setattr(session_memory, "TRIED_UNHELPFUL_PATH",
                         tmp_path / "session_tried_unhelpful.json")
    monkeypatch.setattr(session_memory, "_cache", None)
    # The rate limiters in middleware.py are process-wide singletons (by
    # design -- they track real client traffic across the app's lifetime),
    # so reset them between tests or an earlier test's requests could trip
    # a later test's rate limit and make the suite flaky/order-dependent.
    middleware.troubleshoot_limiter.reset()
    middleware.feedback_limiter.reset()
    middleware.batch_limiter.reset()
    yield
    monkeypatch.setattr(feedback, "_scores_cache", None)
    monkeypatch.setattr(session_memory, "_cache", None)


@pytest.fixture(scope="session")
def real_samples():
    """The 20 official SIIS queries, loaded once per test session."""
    import json
    with open(REPO_ROOT / "sample_queries_real.json") as f:
        return json.load(f)
