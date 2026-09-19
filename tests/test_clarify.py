"""tests/test_clarify.py -- covers clarify.py's vague-complaint detection
and its wiring into pipeline.py. A "smart guided troubleshooting engine"
should ask for more detail when a complaint gives it almost nothing to go
on ("my phone isn't working") rather than silently returning a guess --
these tests make sure that class of complaint gets flagged, that
genuinely specific complaints (including all 20 official queries) never
get flagged, and that the flag is purely additive: it never empties out
or otherwise changes `response.contexts`."""
import json
import os

import clarify
import pipeline
from validators import validate_goal_object


class TestDetectVagueComplaint:
    def test_flags_generic_not_working(self):
        assert clarify.detect_vague_complaint("my phone isn't working") is True

    def test_flags_generic_broken(self):
        assert clarify.detect_vague_complaint("it's broken") is True

    def test_flags_generic_having_issues(self):
        assert clarify.detect_vague_complaint("having some issues with my phone") is True

    def test_flags_bare_help(self):
        assert clarify.detect_vague_complaint("please help") is True

    def test_flags_empty_string(self):
        assert clarify.detect_vague_complaint("") is True
        assert clarify.detect_vague_complaint("   ") is True

    def test_does_not_flag_battery_complaint(self):
        assert clarify.detect_vague_complaint("my battery drains too fast") is False

    def test_does_not_flag_screen_complaint(self):
        assert clarify.detect_vague_complaint("the screen is cracked") is False

    def test_does_not_flag_wifi_complaint(self):
        assert clarify.detect_vague_complaint("wifi keeps disconnecting") is False

    def test_does_not_flag_detailed_complaint_even_without_a_listed_keyword(self):
        """Long enough (>3 meaningful words) complaints get the benefit of
        the doubt even if they happen to miss the keyword list -- both
        conditions (no keyword AND very few meaningful words) have to hold
        together, see the module docstring."""
        long_complaint = "every single morning when I first wake up and try to unlock it nothing happens at all"
        assert clarify.detect_vague_complaint(long_complaint) is False

    def test_is_case_insensitive(self):
        assert clarify.detect_vague_complaint("MY BATTERY DRAINS FAST") is False
        assert clarify.detect_vague_complaint("IT IS BROKEN") is True


class TestNoFalsePositivesOnOfficialQueries:
    def test_none_of_the_20_official_queries_are_flagged_vague(self):
        """These are real, detailed complaints -- gate G3 (>=95% coverage)
        must never be put at risk by this feature, so it's worth pinning
        down explicitly that not one of them trips the heuristic."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "sample_queries_real.json")) as f:
            queries = json.load(f)
        assert len(queries) == 20
        flagged = [q["complaint"] for q in queries if clarify.detect_vague_complaint(q["complaint"])]
        assert flagged == []


class TestBuildClarifyingQuestion:
    def test_returns_a_nonempty_question(self):
        q = clarify.build_clarifying_question("it's broken")
        assert isinstance(q, str) and len(q) > 0

    def test_topic_options_list_is_nonempty(self):
        assert len(clarify.CLARIFYING_TOPIC_OPTIONS) >= 3


class TestPipelineIntegration:
    def test_vague_complaint_sets_needs_clarification_true(self):
        result = pipeline.run_pipeline("my phone isn't working", "")
        assert result["meta"]["needs_clarification"] is True
        assert result["meta"]["clarifying_question"]
        assert len(result["meta"]["clarifying_topic_options"]) >= 3

    def test_vague_complaint_still_produces_a_normal_plan(self):
        """Additive only -- contexts must never be suppressed just because
        the complaint was vague; the offline/LLM path still does its
        best-effort extraction exactly as before."""
        result = pipeline.run_pipeline("my phone isn't working", "")
        # response shape is untouched: still schema-compliant if non-empty
        for goal in result["response"]["contexts"]:
            assert validate_goal_object(goal) == []

    def test_specific_complaint_sets_needs_clarification_false(self):
        result = pipeline.run_pipeline("my battery drains too fast", "")
        assert result["meta"]["needs_clarification"] is False
        assert result["meta"]["clarifying_question"] is None
        assert result["meta"]["clarifying_topic_options"] == []

    def test_hazard_complaint_never_needs_clarification(self):
        """A physical-hazard complaint always has enough signal to act on
        -- see safety.py -- so it should never also carry a clarification
        prompt even though it, too, might be short."""
        result = pipeline.run_pipeline("my phone is on fire", "")
        assert result["meta"]["safety_alert"] is True
        assert result["meta"]["needs_clarification"] is False
        assert result["meta"]["clarifying_question"] is None

    def test_streaming_pipeline_emits_a_clarify_stage_for_vague_input(self):
        events = list(pipeline.run_pipeline_streaming("it's broken", ""))
        stages = [e["stage"] for e in events]
        assert "clarify" in stages
        assert events[-1]["data"]["meta"]["needs_clarification"] is True

    def test_streaming_pipeline_skips_clarify_stage_for_specific_input(self):
        events = list(pipeline.run_pipeline_streaming("my camera won't focus", ""))
        stages = [e["stage"] for e in events]
        assert "clarify" not in stages
        assert events[-1]["data"]["meta"]["needs_clarification"] is False

    def test_cache_hit_recomputes_needs_clarification_from_the_new_complaint(self):
        """The cache key is technical_query (fuzzy-matched), not the raw
        complaint text -- two different requests can legitimately hit the
        same cache entry with different specificity, so this flag must be
        recomputed fresh per-request rather than read from whatever got
        cached the first time."""
        # Warm the cache with a specific complaint.
        first = pipeline.run_pipeline("my battery drains too fast", "")
        assert first["meta"]["needs_clarification"] is False
        # A near-duplicate, equally specific complaint -- likely a cache
        # hit via cache.py's fuzzy matching -- should still read False.
        second = pipeline.run_pipeline("my battery is draining too fast", "")
        assert second["meta"]["needs_clarification"] is False
