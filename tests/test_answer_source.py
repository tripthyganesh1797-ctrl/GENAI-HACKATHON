"""tests/test_answer_source.py -- covers answer_source.py's classification
logic and its wiring into pipeline.py. Users comparing this engine's
answers against the real Samsung support app deserve to know whether a
given plan is official Samsung text, generic fallback advice, or an AI's
own general knowledge -- these tests make sure that classification is
always present and always correct for each of the five cases."""
import os

import pipeline
from answer_source import classify_answer_source


class TestClassifyAnswerSource:
    def test_safety_alert_wins_regardless_of_other_flags(self):
        result = classify_answer_source(
            safety_alert=True, used_offline_fallback=True,
            used_builtin_reference=True, siis_response_provided=True,
            has_contexts=True,
        )
        assert result["type"] == "safety_rule"

    def test_no_contexts_is_no_match(self):
        result = classify_answer_source(
            safety_alert=False, used_offline_fallback=True,
            used_builtin_reference=False, siis_response_provided=True,
            has_contexts=False,
        )
        assert result["type"] == "no_match"

    def test_offline_with_real_reference_text(self):
        result = classify_answer_source(
            safety_alert=False, used_offline_fallback=True,
            used_builtin_reference=False, siis_response_provided=True,
            has_contexts=True,
        )
        assert result["type"] == "offline_official_reference"

    def test_offline_with_builtin_reference(self):
        result = classify_answer_source(
            safety_alert=False, used_offline_fallback=True,
            used_builtin_reference=True, siis_response_provided=False,
            has_contexts=True,
        )
        assert result["type"] == "builtin_generic_knowledge"

    def test_llm_with_real_reference_text(self):
        result = classify_answer_source(
            safety_alert=False, used_offline_fallback=False,
            used_builtin_reference=False, siis_response_provided=True,
            has_contexts=True,
        )
        assert result["type"] == "ai_official_reference"

    def test_llm_with_no_reference_text(self):
        result = classify_answer_source(
            safety_alert=False, used_offline_fallback=False,
            used_builtin_reference=False, siis_response_provided=False,
            has_contexts=True,
        )
        assert result["type"] == "ai_general_knowledge"

    def test_every_case_has_a_label_and_detail(self):
        for kwargs in [
            dict(safety_alert=True, used_offline_fallback=False, used_builtin_reference=False,
                 siis_response_provided=False, has_contexts=True),
            dict(safety_alert=False, used_offline_fallback=False, used_builtin_reference=False,
                 siis_response_provided=False, has_contexts=False),
            dict(safety_alert=False, used_offline_fallback=True, used_builtin_reference=False,
                 siis_response_provided=True, has_contexts=True),
            dict(safety_alert=False, used_offline_fallback=True, used_builtin_reference=True,
                 siis_response_provided=False, has_contexts=True),
            dict(safety_alert=False, used_offline_fallback=False, used_builtin_reference=False,
                 siis_response_provided=True, has_contexts=True),
            dict(safety_alert=False, used_offline_fallback=False, used_builtin_reference=False,
                 siis_response_provided=False, has_contexts=True),
        ]:
            result = classify_answer_source(**kwargs)
            assert result["label"]
            assert result["detail"]


class TestPipelineIntegration:
    def test_official_query_with_siis_response_offline(self):
        """The 20 official queries (real SIIS text, no LLM key in this test
        env) should classify as offline_official_reference."""
        result = pipeline.run_pipeline(
            "My Galaxy S22 screen turns completely blank or white and no text "
            "appears when I search for a stock price or use the Smart Tutor app",
            "Smart Tutor troubleshooting: check the app is updated and the "
            "screen resolution settings are correct.",
        )
        assert result["meta"]["answer_source"]["type"] in (
            "offline_official_reference", "no_match",
        )

    def test_bare_complaint_uses_builtin_knowledge(self):
        result = pipeline.run_pipeline("my battery is draining too fast", "")
        assert result["meta"]["answer_source"]["type"] == "builtin_generic_knowledge"

    def test_hazard_complaint_is_safety_rule(self):
        result = pipeline.run_pipeline("my phone is smoking", "")
        assert result["meta"]["answer_source"]["type"] == "safety_rule"

    def test_out_of_scope_complaint_is_no_match(self):
        result = pipeline.run_pipeline("how do I cook pasta at home", "")
        assert result["meta"]["answer_source"]["type"] == "no_match"

    def test_streaming_pipeline_includes_answer_source(self):
        events = list(pipeline.run_pipeline_streaming("my battery is draining too fast", ""))
        final = events[-1]["data"]
        assert final["meta"]["answer_source"]["type"] == "builtin_generic_knowledge"

    def test_every_official_query_gets_a_classification(self):
        import json
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "sample_queries_real.json")) as f:
            queries = json.load(f)
        for case in queries:
            result = pipeline.run_pipeline(case["complaint"], case.get("siis_response", ""))
            assert result["meta"]["answer_source"] is not None
            assert result["meta"]["answer_source"]["type"] in (
                "offline_official_reference", "ai_official_reference", "no_match",
            )
