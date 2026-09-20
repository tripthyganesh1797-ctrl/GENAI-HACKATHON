"""tests/test_topic_manager.py -- topic_manager.py's DiagGPT-inspired
session topic-stack: no-op without a session_id, and the three core
actions (stay / create / jump) behave the way DiagGPT's own paper defines
them, deterministically, offline."""
import topic_manager
import pipeline


class TestNoSessionIdIsANoOp:
    def test_returns_none(self):
        assert topic_manager.update_topic_stack(None, "Battery draining fast") is None
        assert topic_manager.update_topic_stack("", "Battery draining fast") is None

    def test_get_topic_stack_empty_without_session(self):
        assert topic_manager.get_topic_stack(None) == []


class TestStayAtCurrentTopic:
    def test_same_topic_twice_in_a_row_stays(self):
        first = topic_manager.update_topic_stack("s1", "Battery draining fast")
        second = topic_manager.update_topic_stack("s1", "Battery draining fast")
        assert first["action"] == "create_a_new_topic"
        assert second["action"] == "stay_at_current_topic"
        assert second["stack"] == ["Battery draining fast"]


class TestCreateANewTopic:
    def test_different_topic_pushes_on_top(self):
        topic_manager.update_topic_stack("s1", "Battery draining fast")
        second = topic_manager.update_topic_stack("s1", "Wi-Fi connectivity")
        assert second["action"] == "create_a_new_topic"
        assert second["current_topic"] == "Wi-Fi connectivity"
        assert second["stack"] == ["Wi-Fi connectivity", "Battery draining fast"]

    def test_stack_depth_is_bounded(self):
        for i in range(topic_manager.MAX_STACK_DEPTH + 3):
            result = topic_manager.update_topic_stack("s1", f"Topic {i}")
        assert len(result["stack"]) == topic_manager.MAX_STACK_DEPTH
        assert result["stack"][0] == f"Topic {topic_manager.MAX_STACK_DEPTH + 2}"


class TestJumpToExistingTopic:
    def test_returning_to_an_older_topic_moves_it_to_top(self):
        topic_manager.update_topic_stack("s1", "Battery draining fast")
        topic_manager.update_topic_stack("s1", "Wi-Fi connectivity")
        result = topic_manager.update_topic_stack("s1", "Battery draining fast")
        assert result["action"] == "jump_to_existing_topic"
        assert result["stack"] == ["Battery draining fast", "Wi-Fi connectivity"]


class TestSessionsAreIndependent:
    def test_two_sessions_never_share_a_stack(self):
        topic_manager.update_topic_stack("s1", "Battery draining fast")
        topic_manager.update_topic_stack("s2", "Wi-Fi connectivity")
        assert topic_manager.get_topic_stack("s1") == ["Battery draining fast"]
        assert topic_manager.get_topic_stack("s2") == ["Wi-Fi connectivity"]


class TestPipelineIntegration:
    def test_meta_topic_stack_none_without_session_id(self):
        result = pipeline.run_pipeline("my battery drains fast, topic test no session")
        assert result["meta"]["topic_stack"] is None

    def test_meta_topic_stack_present_with_session_id(self):
        result = pipeline.run_pipeline(
            "my battery drains fast, topic test with session", session_id="topic-sess-1"
        )
        assert result["meta"]["topic_stack"] is not None
        assert result["meta"]["topic_stack"]["current_topic"] == "Battery draining fast"
        assert result["meta"]["topic_stack"]["stack"][0] == "Battery draining fast"

    def test_second_different_complaint_same_session_creates_new_topic(self):
        pipeline.run_pipeline("my battery drains fast, topic seq test", session_id="topic-sess-2")
        second = pipeline.run_pipeline("my wifi keeps dropping, topic seq test", session_id="topic-sess-2")
        assert second["meta"]["topic_stack"]["action"] == "create_a_new_topic"
        assert second["meta"]["topic_stack"]["stack"] == ["Wi-Fi connectivity", "Battery draining fast"]

    def test_safety_short_circuit_leaves_topic_stack_none(self):
        result = pipeline.run_pipeline("my phone is smoking, topic safety test", session_id="topic-sess-3")
        assert result["meta"]["topic_stack"] is None

    def test_cache_hit_still_recomputes_topic_stack_per_session(self):
        pipeline.run_pipeline("my battery drains fast, topic cache test", session_id="topic-sess-4")
        second = pipeline.run_pipeline("my battery drains fast, topic cache test", session_id="topic-sess-5")
        assert second["meta"]["cache_hit"] is True
        assert second["meta"]["topic_stack"]["current_topic"] == "Battery draining fast"
        # A DIFFERENT session should see its OWN fresh stack, not session-4's.
        assert topic_manager.get_topic_stack("topic-sess-5") == ["Battery draining fast"]
