"""tests/test_related_issues.py -- covers related_issues.py's curated
lookup and its wiring into pipeline.py's meta.related_possibilities. Users
should only see "it might also be X, Y, or Z" when the engine's own
confidence in its single best match is genuinely borderline -- these tests
pin both the lookup itself and the borderline-only wiring."""
import pipeline
from related_issues import suggest_related_issues


class TestSuggestRelatedIssues:
    def test_known_bucket_returns_two_or_three_alternatives(self):
        result = suggest_related_issues("Black / blank screen")
        assert 2 <= len(result) <= 3
        assert all(isinstance(item, str) and item for item in result)

    def test_matches_the_users_own_example(self):
        """The user's own example: a black screen can be software, battery,
        or display -- make sure the curated entry actually reflects that,
        not just that SOMETHING is returned."""
        result = suggest_related_issues("Black / blank screen")
        joined = " ".join(result).lower()
        assert "software" in joined
        assert "battery" in joined
        assert "display" in joined

    def test_unknown_bucket_returns_empty_list_not_none(self):
        assert suggest_related_issues("Some made-up bucket that does not exist") == []

    def test_every_bucket_returns_a_fresh_list_not_a_shared_reference(self):
        """Regression guard: mutating one caller's result must not corrupt
        the module-level curated map for the next caller."""
        first = suggest_related_issues("Battery draining fast")
        first.append("mutated")
        second = suggest_related_issues("Battery draining fast")
        assert "mutated" not in second

    def test_no_bucket_has_more_than_three_alternatives(self):
        from related_issues import _RELATED_CAUSES
        for bucket, alternatives in _RELATED_CAUSES.items():
            assert 2 <= len(alternatives) <= 3, bucket


class TestPipelineIntegration:
    def test_borderline_llm_goal_gets_related_possibilities(self, monkeypatch):
        """Same monkeypatch pattern as test_pipeline.py's LLM-escalation
        tests: a low self-reported score attaches escalation, and a
        borderline plan should now also carry related_possibilities for
        its symptom bucket."""
        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Battery Troubleshooting",
                "title": "Battery issue",
                "score": 0.35,
                "actions": [],
            }
            return {"contexts": [goal]}, False  # fb1=False -> LLM path

        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)
        result = pipeline.run_pipeline("my battery drains so fast", "some grounding text")
        goal = result["response"]["contexts"][0]
        assert goal.get("escalation") is not None
        assert result["meta"]["related_possibilities"] != []

    def test_confident_llm_goal_has_no_related_possibilities(self, monkeypatch):
        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Battery Troubleshooting",
                "title": "Battery issue",
                "score": 0.95,
                "actions": [],
            }
            return {"contexts": [goal]}, False

        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)
        result = pipeline.run_pipeline("my battery is fine actually", "some grounding text")
        goal = result["response"]["contexts"][0]
        assert "escalation" not in goal
        assert result["meta"]["related_possibilities"] == []

    def test_safety_short_circuit_has_no_related_possibilities(self):
        result = pipeline.run_pipeline("my phone is smoking", "")
        assert result["meta"]["related_possibilities"] == []

    def test_no_match_has_no_related_possibilities(self):
        result = pipeline.run_pipeline("how do I cook pasta at home", "")
        assert result["meta"]["related_possibilities"] == []

    def test_streaming_pipeline_matches_non_streaming(self, monkeypatch):
        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Battery Troubleshooting",
                "title": "Battery issue",
                "score": 0.35,
                "actions": [],
            }
            return {"contexts": [goal]}, False

        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)
        events = list(pipeline.run_pipeline_streaming("my battery drains so fast, streaming", "grounding text"))
        final = events[-1]["data"]
        assert final["meta"]["related_possibilities"] != []

    def test_cache_hit_carries_related_possibilities_through(self, monkeypatch):
        """The field is computed once at cache-write time and must survive
        a cache hit unchanged (via .setdefault(), same treatment as
        answer_source) -- not silently dropped on the second request."""
        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Battery Troubleshooting",
                "title": "Battery issue",
                "score": 0.35,
                "actions": [],
            }
            return {"contexts": [goal]}, False

        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)
        complaint = "my battery drains so fast, cache backfill test"
        first = pipeline.run_pipeline(complaint, "grounding text")
        assert first["meta"]["cache_hit"] is False

        second = pipeline.run_pipeline(complaint, "grounding text")
        assert second["meta"]["cache_hit"] is True
        assert second["meta"]["related_possibilities"] != []
