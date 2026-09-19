"""
Contract enforcement tests. These mirror the automated checks judges will
run (theme guide Appendix A) -- if these fail, the submission fails grading
regardless of how good the troubleshooting content looks to a human.
"""
import validators as v


class TestUrlLeak:
    def test_detects_http(self):
        assert v.has_url_leak("Visit http://example.com for help")

    def test_detects_https(self):
        assert v.has_url_leak("See https://samsung.com/support")

    def test_detects_bare_www(self):
        assert v.has_url_leak("Go to www.samsung.com")

    def test_clean_text_has_no_leak(self):
        assert not v.has_url_leak("Tap on Display, then tap Brightness.")

    def test_strip_removes_url_and_trims(self):
        assert v.strip_urls("Open Settings http://x.com now") == "Open Settings  now".strip() or \
            "http" not in v.strip_urls("Open Settings http://x.com now")


class TestDescriptionFormat:
    def test_valid_five_word(self):
        ok, msg = v.check_description_format("It will let you fix this")
        assert ok, msg

    def test_valid_seven_word(self):
        ok, msg = v.check_description_format("It will help you fix this issue")  # 7 words
        assert ok, msg

    def test_rejects_missing_it_will_prefix(self):
        ok, msg = v.check_description_format("You will fix your battery drain")
        assert not ok
        assert "It will" in msg

    def test_rejects_too_short(self):
        ok, msg = v.check_description_format("It will fix")
        assert not ok
        assert "5-7 words" in msg

    def test_rejects_too_long(self):
        ok, msg = v.check_description_format("It will let you configure many different display settings today")
        assert not ok
        assert "5-7 words" in msg


class TestGoalSyntax:
    def test_valid_troubleshooting_goal(self):
        ok, msg = v.check_goal_syntax("Follow these steps to perform this Battery Fast Drain Troubleshooting")
        assert ok, msg

    def test_valid_configuration_goal(self):
        ok, _ = v.check_goal_syntax("Follow these steps to perform this Display Configuration")
        assert ok

    def test_rejects_wrong_verb(self):
        ok, msg = v.check_goal_syntax("Complete these steps to fix this Battery Troubleshooting")
        assert not ok
        assert "syntax mismatch" in msg

    def test_rejects_missing_troubleshooting_suffix(self):
        ok, msg = v.check_goal_syntax("Follow these steps to perform this Battery Fix")
        assert not ok


class TestTitleFormat:
    def test_two_word_title_ok(self):
        ok, _ = v.check_title_format("Battery drain")
        assert ok

    def test_three_word_title_ok(self):
        ok, _ = v.check_title_format("Screen goes blank")
        assert ok

    def test_one_word_title_rejected(self):
        ok, msg = v.check_title_format("Battery")
        assert not ok
        assert "2-3 words" in msg

    def test_four_word_title_rejected(self):
        ok, msg = v.check_title_format("My phone battery drains")
        assert not ok


class TestCategoryOrdering:
    def test_all_auto_ok(self):
        ok, _ = v.validate_category_ordering(["auto", "auto"])
        assert ok

    def test_critical_last_ok(self):
        ok, _ = v.validate_category_ordering(["auto", "manual", "critical"])
        assert ok

    def test_critical_in_middle_rejected(self):
        ok, msg = v.validate_category_ordering(["auto", "critical", "manual"])
        assert not ok
        assert "critical" in msg


class TestCriticalSafetyOverride:
    def test_restart_forced_critical_even_if_llm_said_auto(self):
        goal = {"actions": [
            {"actionName": "Restart Device", "description": "It will restart your phone now",
             "category": "auto", "stepGroups": [{"steps": ["Press and hold power button."]}]},
        ]}
        v.enforce_critical_safety(goal)
        assert goal["actions"][0]["category"] == "critical"

    def test_benign_action_left_alone(self):
        goal = {"actions": [
            {"actionName": "Display Settings", "description": "It will let you adjust brightness",
             "category": "auto", "stepGroups": [{"steps": ["Tap Display."]}]},
        ]}
        v.enforce_critical_safety(goal)
        assert goal["actions"][0]["category"] == "auto"

    def test_factory_reset_keyword_detected_in_steps_not_just_name(self):
        goal = {"actions": [
            {"actionName": "Reset Options", "description": "It will reset your device settings",
             "category": "manual",
             "stepGroups": [{"steps": ["Tap Factory reset to erase all data."]}]},
        ]}
        v.enforce_critical_safety(goal)
        assert goal["actions"][0]["category"] == "critical"


class TestValidateGoalObjectIntegration:
    """The full validate_goal_object() path, including the re-sort bug fix:
    a restart action classified "auto" by the extractor must both be
    recategorized to "critical" AND moved to the end of the actions list."""

    def test_resorts_after_safety_override(self):
        goal = {
            "goal": "Follow these steps to perform this Device Troubleshooting",
            "title": "Device restart",
            "score": 0.9,
            "actions": [
                {"actionName": "Restart Device", "description": "It will restart your phone immediately",
                 "category": "auto", "stepGroups": [{"steps": ["Hold the power button."]}]},
                {"actionName": "Display Settings", "description": "It will let you adjust brightness level",
                 "category": "auto", "stepGroups": [{"steps": ["Tap Display."]}]},
            ],
        }
        errors = v.validate_goal_object(goal)
        assert errors == []
        assert [a["category"] for a in goal["actions"]] == ["auto", "critical"]
        assert goal["actions"][-1]["actionName"] == "Restart Device"

    def test_clean_goal_has_no_errors(self):
        goal = {
            "goal": "Follow these steps to perform this Battery Fast Drain Troubleshooting",
            "title": "Battery fast drain",
            "score": 0.85,
            "actions": [
                {"actionName": "Battery Settings", "description": "It will let you save battery power",
                 "category": "auto", "stepGroups": [{"steps": ["Tap Battery.", "Tap Power saving mode."]}]},
            ],
        }
        assert v.validate_goal_object(goal) == []

    def test_url_leak_flagged(self):
        goal = {
            "goal": "Follow these steps to perform this Battery Troubleshooting",
            "title": "Battery drain",
            "score": 0.8,
            "actions": [
                {"actionName": "Battery Settings", "description": "It will let you save battery power",
                 "category": "auto",
                 "stepGroups": [{"steps": ["See https://samsung.com/battery for details."]}]},
            ],
        }
        errors = v.validate_goal_object(goal)
        assert any("URL leak" in e for e in errors)
