"""report.py — Task 38: packages an already-computed troubleshooting
result into a compact, shareable Markdown/HTML report. These tests cover
the pure extraction layer (build_report_sections), each renderer, and one
real end-to-end run through pipeline.py's offline path."""
import pytest

import report


def _sample_response(with_escalation=False, with_deeplink=True, category="auto"):
    step_group = {
        "steps": ["Open Settings", "Tap Battery", "Tap Optimize now"],
        "actionableDeeplink": (
            {"deeplink": "bixby://masked/act/example", "message": "Optimize Battery",
             "description": "Runs battery optimization."}
            if with_deeplink else None
        ),
    }
    goal = {
        "goal": "Follow these steps to perform this Battery Troubleshooting",
        "title": "Battery fast drain",
        "score": 0.42,
        "actions": [{
            "actionName": "Battery Optimization",
            "description": "It will optimize battery usage",
            "category": category,
            "stepGroups": [step_group],
        }],
    }
    if with_escalation:
        goal["escalation"] = {
            "recommended": True,
            "reason": "Match confidence was below the threshold",
            "action": {
                "deeplink": "bixby://masked/act/5930a08d3d",
                "message": "Run Full Device Diagnostic",
                "description": "Runs a full device status diagnosis.",
            },
        }
    return {
        "query": "my battery drains too fast",
        "response": {"contexts": [goal]},
        "meta": {
            "used_offline_fallback": True,
            "device_context_notes": ["Battery is at 8% -- charging steps were prioritized."],
            "session_notes": ["Skipped 1 previously-tried match."],
            "fallback": None,
        },
    }


class TestBuildReportSections:
    def test_extracts_query_and_goal_basics(self):
        sections = report.build_report_sections(_sample_response())
        assert sections["query"] == "my battery drains too fast"
        assert sections["no_match"] is False
        goal = sections["goals"][0]
        assert goal["title"] == "Battery fast drain"
        assert goal["score"] == 0.42
        assert goal["confidence_label"] == "Low"  # 0.42 < LLM_SCORE_ESCALATION_THRESHOLD (0.6)

    def test_confidence_labels_match_thresholds(self):
        import escalation as esc
        cases = [(0.95, "High"), (0.8, "High"), (0.79, "Medium"),
                 (esc.LLM_SCORE_ESCALATION_THRESHOLD, "Medium"), (0.1, "Low")]
        for score, expected in cases:
            resp = _sample_response()
            resp["response"]["contexts"][0]["score"] = score
            sections = report.build_report_sections(resp)
            assert sections["goals"][0]["confidence_label"] == expected, score

    def test_collects_steps_and_deeplink_from_action(self):
        sections = report.build_report_sections(_sample_response())
        action = sections["goals"][0]["actions"][0]
        assert action["steps"] == ["Open Settings", "Tap Battery", "Tap Optimize now"]
        assert action["deeplink"] == "bixby://masked/act/example"
        assert action["deeplink_message"] == "Optimize Battery"

    def test_action_with_no_deeplink_is_handled(self):
        sections = report.build_report_sections(_sample_response(with_deeplink=False))
        action = sections["goals"][0]["actions"][0]
        assert action["deeplink"] is None

    def test_escalation_is_extracted_only_when_recommended(self):
        sections = report.build_report_sections(_sample_response(with_escalation=True))
        esc = sections["goals"][0]["escalation"]
        assert esc["message"] == "Run Full Device Diagnostic"
        assert esc["deeplink"] == "bixby://masked/act/5930a08d3d"

        sections_no_esc = report.build_report_sections(_sample_response(with_escalation=False))
        assert sections_no_esc["goals"][0]["escalation"] is None

    def test_notes_are_carried_through(self):
        sections = report.build_report_sections(_sample_response())
        assert sections["device_context_notes"] == ["Battery is at 8% -- charging steps were prioritized."]
        assert sections["session_notes"] == ["Skipped 1 previously-tried match."]

    def test_no_match_response_yields_empty_goals(self):
        resp = {"query": "something unrelated", "response": {"contexts": []},
                "meta": {"fallback": "no_match"}}
        sections = report.build_report_sections(resp)
        assert sections["no_match"] is True
        assert sections["goals"] == []

    def test_missing_keys_do_not_crash(self):
        """A minimal/malformed-ish response dict must still produce a
        valid sections structure rather than raising."""
        sections = report.build_report_sections({})
        assert sections["query"] == ""
        assert sections["no_match"] is True

    def test_does_not_mutate_input(self):
        resp = _sample_response()
        import copy
        original = copy.deepcopy(resp)
        report.build_report_sections(resp)
        assert resp == original


class TestRenderMarkdown:
    def test_includes_query_and_goal_title(self):
        md = report.render_markdown(report.build_report_sections(_sample_response()))
        assert "my battery drains too fast" in md
        assert "Battery fast drain" in md
        assert "Confidence: Low" in md

    def test_includes_steps_and_deeplink(self):
        md = report.render_markdown(report.build_report_sections(_sample_response()))
        assert "Tap Optimize now" in md
        assert "bixby://masked/act/example" in md

    def test_includes_escalation_callout(self):
        md = report.render_markdown(report.build_report_sections(_sample_response(with_escalation=True)))
        assert "Run Full Device Diagnostic" in md
        assert "below the threshold" in md

    def test_includes_notes_section(self):
        md = report.render_markdown(report.build_report_sections(_sample_response()))
        assert "Battery is at 8%" in md
        assert "Skipped 1 previously-tried match." in md

    def test_no_match_message(self):
        sections = report.build_report_sections({"query": "q", "response": {"contexts": []}, "meta": {}})
        md = report.render_markdown(sections)
        assert "No matching troubleshooting plan" in md

    def test_action_with_no_deeplink_has_no_deeplink_line(self):
        resp = _sample_response(with_deeplink=False, category="critical")
        md = report.render_markdown(report.build_report_sections(resp))
        assert "bixby://" not in md


class TestRenderHtml:
    def test_is_well_formed_and_contains_content(self):
        html = report.render_html(report.build_report_sections(_sample_response()))
        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html and "</html>" in html
        assert "Battery fast drain" in html
        assert "bixby://masked/act/example" in html

    def test_escapes_user_supplied_text(self):
        resp = _sample_response()
        resp["query"] = "<script>alert(1)</script>"
        html = report.render_html(report.build_report_sections(resp))
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_confidence_css_class_matches_label(self):
        html = report.render_html(report.build_report_sections(_sample_response()))
        assert 'class="confidence Low"' in html

    def test_no_match_renders_a_message(self):
        sections = report.build_report_sections({"query": "q", "response": {"contexts": []}, "meta": {}})
        html = report.render_html(sections)
        assert "No matching troubleshooting plan" in html

    def test_includes_escalation_callout(self):
        html = report.render_html(report.build_report_sections(_sample_response(with_escalation=True)))
        assert "Run Full Device Diagnostic" in html
        assert "below the threshold" in html
        assert 'class="escalation"' in html


class TestGenerateReport:
    def test_markdown_is_the_default(self):
        content = report.generate_report(_sample_response())
        assert content.startswith("# Diagnostic Report")

    def test_html_format(self):
        content = report.generate_report(_sample_response(), fmt="html")
        assert content.startswith("<!DOCTYPE html>")

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError):
            report.generate_report(_sample_response(), fmt="pdf")


class TestEndToEndWithRealPipeline:
    """report.py must handle the exact dict shape pipeline.py actually
    produces (offline path, since no LLM_API_KEY is configured here) --
    not just the hand-built fixture above."""

    def test_real_offline_result_produces_a_readable_report(self):
        import pipeline
        result = pipeline.run_pipeline("my battery drains too fast and gets hot", "")
        md = report.generate_report(result, fmt="markdown")
        assert "# Diagnostic Report" in md
        assert "my battery drains too fast and gets hot" in md
        html = report.generate_report(result, fmt="html")
        assert "<!DOCTYPE html>" in html

    def test_real_no_match_result_produces_a_report(self):
        """A genuinely out-of-scope complaint (not just "no siis_response
        supplied" -- see builtin_knowledge.py, which now grounds a bare
        in-scope complaint in generic reference text instead) is what
        actually produces a no_match result on the offline path."""
        import pipeline
        result = pipeline.run_pipeline("how do I cook pasta at home", "")
        md = report.generate_report(result)
        assert "No matching troubleshooting plan" in md
