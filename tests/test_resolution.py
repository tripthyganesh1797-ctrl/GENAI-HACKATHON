"""tests/test_resolution.py -- resolution.py's goal-level "did this fix
it?" signal (point 5) and its wiring into the SAME per-deeplink feedback
and session-avoidance machinery point 2's thumbs up/down already drives.
See resolution.py's module docstring for why this is deliberately a thin
fan-out rather than a second scoring mechanism."""
import json

import feedback
import resolution
import session_memory
from deeplink_matching import DUMMY_POSITIVE_DEEPLINK


class TestRecordResolution:
    def test_resolved_true_records_positive_feedback_for_each_deeplink(self):
        result = resolution.record_resolution(
            goal_title="Battery fast drain",
            deeplinks=[
                {"deeplink": "bixby://masked/act/res-aaa", "action_name": "Battery Settings"},
                {"deeplink": "bixby://masked/act/res-bbb", "action_name": "Power Saving"},
            ],
            resolved=True,
        )
        assert result["resolved"] is True
        assert set(result["deeplinks_updated"]) == {
            "bixby://masked/act/res-aaa", "bixby://masked/act/res-bbb",
        }
        assert feedback.get_adjustment("bixby://masked/act/res-aaa") > 0
        assert feedback.get_adjustment("bixby://masked/act/res-bbb") > 0

    def test_resolved_false_records_negative_feedback_for_each_deeplink(self):
        result = resolution.record_resolution(
            goal_title="Camera crash",
            deeplinks=[{"deeplink": "bixby://masked/act/res-ccc", "action_name": "Camera Settings"}],
            resolved=False,
        )
        assert result["resolved"] is False
        assert feedback.get_adjustment("bixby://masked/act/res-ccc") < 0

    def test_resolved_false_with_session_id_feeds_session_avoidance(self):
        """The whole point of reusing feedback.record_feedback() rather
        than a bespoke mechanism: a "No" here must ALSO land in
        session_memory.py's avoid list, exactly like a thumbs-down does."""
        resolution.record_resolution(
            goal_title="Wifi issue",
            deeplinks=[{"deeplink": "bixby://masked/act/res-ddd", "action_name": "Wifi Settings"}],
            resolved=False,
            session_id="sess-resolution-1",
        )
        assert "bixby://masked/act/res-ddd" in session_memory.get_avoid_set("sess-resolution-1")

    def test_resolved_true_does_not_feed_session_avoidance(self):
        resolution.record_resolution(
            goal_title="Wifi issue",
            deeplinks=[{"deeplink": "bixby://masked/act/res-eee", "action_name": "Wifi Settings"}],
            resolved=True,
            session_id="sess-resolution-2",
        )
        assert "bixby://masked/act/res-eee" not in session_memory.get_avoid_set("sess-resolution-2")

    def test_placeholder_deeplink_is_skipped(self):
        """No real catalog match -- nothing meaningful to score or avoid,
        same guard main.py's /v1/feedback route applies for this deeplink."""
        result = resolution.record_resolution(
            goal_title="No match",
            deeplinks=[{"deeplink": DUMMY_POSITIVE_DEEPLINK, "action_name": "X"}],
            resolved=False,
        )
        assert result["deeplinks_updated"] == []

    def test_empty_deeplinks_list_still_logs_and_does_not_raise(self):
        result = resolution.record_resolution(goal_title="No real match at all", deeplinks=[], resolved=False)
        assert result["deeplinks_updated"] == []
        assert result["resolved"] is False

    def test_event_is_appended_to_the_log(self):
        resolution.record_resolution(
            goal_title="Logged goal", deeplinks=[], resolved=True, query="my thing is fixed now",
        )
        lines = resolution.RESOLUTION_LOG_PATH.read_text().strip().splitlines()
        events = [json.loads(l) for l in lines]
        assert any(e["goal_title"] == "Logged goal" and e["resolved"] is True for e in events)


class TestResolutionSummary:
    def test_no_log_file_returns_zeroed_summary(self):
        summary = resolution.resolution_summary()
        assert summary == {
            "total_resolution_events": 0, "resolved": 0, "unresolved": 0, "resolved_rate": None,
        }

    def test_summary_counts_resolved_and_unresolved(self):
        resolution.record_resolution(goal_title="A", deeplinks=[], resolved=True)
        resolution.record_resolution(goal_title="B", deeplinks=[], resolved=True)
        resolution.record_resolution(goal_title="C", deeplinks=[], resolved=False)
        summary = resolution.resolution_summary()
        assert summary["total_resolution_events"] == 3
        assert summary["resolved"] == 2
        assert summary["unresolved"] == 1
        assert summary["resolved_rate"] == round(2 / 3, 3)

    def test_corrupted_line_is_skipped_not_fatal(self):
        resolution.record_resolution(goal_title="Good", deeplinks=[], resolved=True)
        with resolution.RESOLUTION_LOG_PATH.open("a") as f:
            f.write("not valid json\n")
        summary = resolution.resolution_summary()
        assert summary["total_resolution_events"] == 1
