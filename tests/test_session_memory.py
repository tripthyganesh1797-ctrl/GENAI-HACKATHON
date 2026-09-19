"""session_memory.py — session-scoped "don't re-suggest what already
failed" memory (Task 36). Distinct from feedback.py's GLOBAL adaptive
re-ranking, exercised separately in tests/test_feedback.py and
tests/test_deeplink_matching.py::TestFeedbackDrivenReranking."""
import session_memory as sm


def test_get_avoid_set_empty_for_no_session_id():
    assert sm.get_avoid_set(None) == set()
    assert sm.get_avoid_set("") == set()


def test_get_avoid_set_empty_for_unseen_session_id():
    assert sm.get_avoid_set("never-seen-this-session-before") == set()


def test_record_unhelpful_then_get_avoid_set_round_trips():
    sm.record_unhelpful("session-1", "bixby://masked/act/aaa")
    assert sm.get_avoid_set("session-1") == {"bixby://masked/act/aaa"}


def test_record_unhelpful_accumulates_multiple_deeplinks_for_same_session():
    sm.record_unhelpful("session-2", "bixby://masked/act/aaa")
    sm.record_unhelpful("session-2", "bixby://masked/act/bbb")
    assert sm.get_avoid_set("session-2") == {
        "bixby://masked/act/aaa", "bixby://masked/act/bbb",
    }


def test_record_unhelpful_is_idempotent_for_the_same_deeplink():
    sm.record_unhelpful("session-3", "bixby://masked/act/aaa")
    sm.record_unhelpful("session-3", "bixby://masked/act/aaa")
    assert sm.get_avoid_set("session-3") == {"bixby://masked/act/aaa"}


def test_sessions_are_isolated_from_each_other():
    sm.record_unhelpful("session-a", "bixby://masked/act/aaa")
    sm.record_unhelpful("session-b", "bixby://masked/act/bbb")
    assert sm.get_avoid_set("session-a") == {"bixby://masked/act/aaa"}
    assert sm.get_avoid_set("session-b") == {"bixby://masked/act/bbb"}


def test_record_unhelpful_is_a_noop_for_missing_session_id_or_deeplink():
    sm.record_unhelpful(None, "bixby://masked/act/aaa")
    sm.record_unhelpful("session-4", None)
    sm.record_unhelpful("session-4", "")
    assert sm.get_avoid_set("session-4") == set()
    assert sm.get_avoid_set(None) == set()


def test_persists_across_cache_reset_via_the_file():
    """The in-memory _cache is the hot path, but the file underneath it is
    what actually makes this survive across requests in a real process --
    reset the cache (as a fresh process/module import effectively would)
    and confirm the data is still there via a fresh read."""
    sm.record_unhelpful("session-5", "bixby://masked/act/aaa")
    sm._cache = None  # simulate a fresh module state, forcing a re-read from disk
    assert sm.get_avoid_set("session-5") == {"bixby://masked/act/aaa"}
