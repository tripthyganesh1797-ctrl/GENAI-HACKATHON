"""tests/test_safety.py -- covers safety.py's physical-hazard detection
and the hand-built safety Goal, plus its wiring into pipeline.py's
run_pipeline()/run_pipeline_streaming(). A swollen/smoking/sparking
battery is a real fire/burn risk, not a normal troubleshooting scenario
-- these tests exist to make sure that class of complaint always short-
circuits to the safety instruction (never software steps), and that
ordinary complaints (overheating, battery drain, a cracked screen) are
never mistaken for one."""
import pipeline
import safety
from validators import validate_goal_object


class TestDetectPhysicalHazard:
    def test_detects_battery_swelling_and_bulging(self):
        assert safety.detect_physical_hazard(
            "my phone battery looks swollen and the back cover is bulging out"
        ) is not None

    def test_detects_smoke(self):
        assert safety.detect_physical_hazard("I see smoke coming from my phone") is not None

    def test_detects_fire(self):
        assert safety.detect_physical_hazard("my phone caught fire while charging") is not None

    def test_detects_sparking(self):
        assert safety.detect_physical_hazard("the charger port is sparking") is not None

    def test_detects_burning_smell(self):
        assert safety.detect_physical_hazard("my phone smells like burning") is not None

    def test_detects_chemical_smell(self):
        assert safety.detect_physical_hazard("there's a strong chemical smell from the battery") is not None

    def test_detects_hissing_sound(self):
        assert safety.detect_physical_hazard("my phone is making a hissing sound and is hot") is not None

    def test_detects_battery_leak(self):
        assert safety.detect_physical_hazard("my battery is leaking some kind of fluid") is not None

    def test_detects_burn_injury(self):
        assert safety.detect_physical_hazard("the phone burned my hand when I picked it up") is not None

    def test_returns_none_for_ordinary_overheating(self):
        """Overheating alone is a normal, self-serviceable complaint --
        must NOT be treated as a hazard, or every legitimate Device
        Overheating query would wrongly lose its real troubleshooting
        steps."""
        assert safety.detect_physical_hazard("my phone is overheating a lot") is None

    def test_returns_none_for_ordinary_battery_drain(self):
        assert safety.detect_physical_hazard("my battery is draining too quickly") is None

    def test_returns_none_for_cracked_screen(self):
        assert safety.detect_physical_hazard("my screen is cracked and flickers") is None

    def test_returns_none_for_unrelated_case_popped_open(self):
        assert safety.detect_physical_hazard("the case popped open when I dropped the phone") is None

    def test_returns_none_for_nonsense_query(self):
        assert safety.detect_physical_hazard("how do I cook pasta at home") is None

    def test_is_case_insensitive(self):
        assert safety.detect_physical_hazard("MY BATTERY IS SWOLLEN") is not None


class TestBuildSafetyGoal:
    def test_produces_a_schema_compliant_goal(self):
        goal = safety.build_safety_goal("battery swelling")
        assert validate_goal_object(goal) == []

    def test_action_category_is_critical(self):
        goal = safety.build_safety_goal("battery swelling")
        assert goal["actions"][0]["category"] == "critical"

    def test_has_no_actionable_deeplink(self):
        """Never invent a deeplink for a hardware safety issue -- matches
        the same manual/no-deeplink pattern the official sample output
        uses for real hardware issues (e.g. screen replacement)."""
        goal = safety.build_safety_goal("battery swelling")
        step_group = goal["actions"][0]["stepGroups"][0]
        assert step_group["actionableDeeplink"] is None
        assert step_group["validationDeeplink"] is None

    def test_steps_tell_the_user_to_stop_and_contact_support(self):
        goal = safety.build_safety_goal("battery swelling")
        steps_text = " ".join(goal["actions"][0]["stepGroups"][0]["steps"]).lower()
        assert "stop using" in steps_text
        assert "samsung support" in steps_text or "service center" in steps_text


class TestPipelineIntegration:
    def test_hazard_complaint_short_circuits_to_safety_goal(self):
        result = pipeline.run_pipeline(
            "my phone battery looks swollen and the back cover is bulging out", ""
        )
        assert result["meta"]["safety_alert"] is True
        # Both "swollen" and "bulging" appear in this complaint; whichever
        # pattern matches first is fine -- what matters is that a hazard
        # was detected with SOME reason string, not which one won.
        assert result["meta"]["safety_reason"]
        assert len(result["response"]["contexts"]) == 1
        assert result["response"]["contexts"][0]["actions"][0]["category"] == "critical"

    def test_hazard_complaint_never_attempts_normal_extraction(self):
        """No software troubleshooting steps anywhere in the response --
        only the safety instruction."""
        result = pipeline.run_pipeline("my phone is smoking and smells like burning", "")
        assert len(result["response"]["contexts"]) == 1
        assert result["meta"]["used_offline_fallback"] is False
        assert result["meta"]["used_builtin_reference"] is False

    def test_hazard_complaint_bypasses_cache_entirely(self):
        """A hazard response must never be cached (so a later, unrelated
        query can't accidentally collide with it), and must never itself
        read a stale cached entry."""
        import cache
        query = "my battery is swollen and won't stop growing"
        pipeline.run_pipeline(query, "")
        # Nothing should have been cached under this query's technical_query
        from offline_fallback import core_problem_phrase
        assert cache.get_cached(core_problem_phrase(query)) is None

    def test_ordinary_complaint_carries_safety_alert_false(self):
        result = pipeline.run_pipeline("my phone is overheating a lot", "")
        assert result["meta"]["safety_alert"] is False
        assert result["meta"]["safety_reason"] is None

    def test_hazard_complaint_still_produces_query_variations(self):
        """gate A5 (8-10 query_variations) must hold even on the safety
        short-circuit -- this response doesn't go through Stage 0's
        normal enrichment, so query_variations has to be generated
        separately (see _build_safety_response())."""
        result = pipeline.run_pipeline("my phone caught fire while charging", "")
        assert 8 <= len(result["query_variations"]) <= 10

    def test_streaming_pipeline_also_short_circuits(self):
        events = list(pipeline.run_pipeline_streaming(
            "my phone battery is swollen", ""
        ))
        stages = [e["stage"] for e in events]
        assert "safety_alert" in stages
        assert stages[-1] == "complete"
        assert events[-1]["data"]["meta"]["safety_alert"] is True
        # None of the normal pipeline stages should have run.
        assert "enrich" not in stages
        assert "extract" not in stages
        assert "deeplink_match" not in stages
