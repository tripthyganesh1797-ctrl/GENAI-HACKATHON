"""feedback.py — feedback logging + the adaptive re-ranking adjustment
formula. These are the load-bearing guarantees for the feedback loop:
a single click can never flip a ranking, but a consistent pattern can."""
import feedback
import session_memory


def test_no_feedback_means_zero_adjustment():
    assert feedback.get_adjustment("bixby://masked/act/unknown") == 0.0


def test_record_feedback_persists_and_is_readable():
    result = feedback.record_feedback("bixby://masked/act/aaa", "Battery Settings", helpful=True)
    assert result["helpful"] == 1
    assert result["unhelpful"] == 0
    assert feedback.get_adjustment("bixby://masked/act/aaa") > 0


def test_single_thumbs_down_has_bounded_small_effect():
    """A single vote must nudge, not swing, the ranking -- the confidence
    ramp (log1p(total)/log1p(8)) keeps low-count adjustments small."""
    feedback.record_feedback("bixby://masked/act/bbb", "Wifi Settings", helpful=False)
    adj = feedback.get_adjustment("bixby://masked/act/bbb")
    assert -feedback.MAX_ADJUSTMENT < adj < 0
    assert abs(adj) < feedback.MAX_ADJUSTMENT * 0.5  # single vote stays well under the cap


def test_consistent_negative_feedback_approaches_the_cap():
    for _ in range(10):
        feedback.record_feedback("bixby://masked/act/ccc", "Camera Settings", helpful=False)
    adj = feedback.get_adjustment("bixby://masked/act/ccc")
    assert adj < 0
    assert abs(adj - (-feedback.MAX_ADJUSTMENT)) < 0.02  # within 2% of the negative cap


def test_mixed_feedback_partially_cancels():
    for _ in range(5):
        feedback.record_feedback("bixby://masked/act/ddd", "X", helpful=True)
    for _ in range(5):
        feedback.record_feedback("bixby://masked/act/ddd", "X", helpful=False)
    assert feedback.get_adjustment("bixby://masked/act/ddd") == 0.0


def test_adjustment_never_exceeds_bounds():
    for _ in range(50):
        feedback.record_feedback("bixby://masked/act/eee", "X", helpful=True)
    adj = feedback.get_adjustment("bixby://masked/act/eee")
    assert adj <= feedback.MAX_ADJUSTMENT


def test_feedback_summary_counts_events():
    feedback.record_feedback("bixby://masked/act/f1", "A", helpful=True)
    feedback.record_feedback("bixby://masked/act/f2", "B", helpful=False)
    feedback.record_feedback("bixby://masked/act/f2", "B", helpful=False)
    summary = feedback.feedback_summary()
    assert summary["total_feedback_events"] == 3
    assert summary["helpful"] == 1
    assert summary["unhelpful"] == 2
    assert summary["deeplinks_with_feedback"] == 2
    assert summary["deeplinks_demoted"] == 1
    assert summary["deeplinks_boosted"] == 1


def test_empty_deeplink_key_is_safe():
    assert feedback.get_adjustment("") == 0.0
    assert feedback.get_adjustment(None) == 0.0


class TestSessionMemoryHook:
    """Task 36: record_feedback() is also the write path into
    session_memory.py's per-session avoid list -- a DIFFERENT, session-
    scoped mechanism from the global adjustment tested above. See
    session_memory.py's module docstring for why they're kept separate."""

    def test_unhelpful_feedback_with_session_id_records_in_session_memory(self):
        feedback.record_feedback("bixby://masked/act/sess1", "X", helpful=False,
                                  session_id="my-session")
        assert "bixby://masked/act/sess1" in session_memory.get_avoid_set("my-session")

    def test_helpful_feedback_with_session_id_does_not_get_avoided(self):
        feedback.record_feedback("bixby://masked/act/sess2", "X", helpful=True,
                                  session_id="my-session-2")
        assert session_memory.get_avoid_set("my-session-2") == set()

    def test_unhelpful_feedback_without_session_id_touches_no_session(self):
        feedback.record_feedback("bixby://masked/act/sess3", "X", helpful=False)
        # No session_id was given, so nothing should be recorded anywhere --
        # this is really just confirming record_feedback() never crashes
        # or invents a session when session_id="" (the default).
        assert session_memory.get_avoid_set("") == set()
        assert session_memory.get_avoid_set(None) == set()

    def test_session_avoidance_does_not_affect_the_global_feedback_score(self):
        """The two mechanisms are independent: a single session's negative
        feedback still only nudges the GLOBAL score by the normal single-
        vote amount, same as if no session_id had been supplied."""
        feedback.record_feedback("bixby://masked/act/sess4", "X", helpful=False,
                                  session_id="my-session-4")
        adj = feedback.get_adjustment("bixby://masked/act/sess4")
        assert -feedback.MAX_ADJUSTMENT < adj < 0
        assert abs(adj) < feedback.MAX_ADJUSTMENT * 0.5  # still a bounded single-vote nudge
