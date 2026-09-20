"""tests/test_recovery.py -- recovery.py's Guided-Retry classification and
prompt-suffix helpers, plus pipeline.py's integration of it into
stage1_extract()."""
import recovery
import pipeline


class TestClassifyLLMFailure:
    def test_no_llm_available_is_not_a_failure(self):
        info = recovery.classify_llm_failure(False, None)
        assert info == recovery.NO_RECOVERY

    def test_no_exception_is_not_a_failure(self):
        info = recovery.classify_llm_failure(True, None)
        assert info == recovery.NO_RECOVERY

    def test_an_exception_while_available_is_classified(self):
        info = recovery.classify_llm_failure(True, ValueError("bad json"))
        assert info["failure_detected"] is True
        assert info["failure_type"] == recovery.FAILURE_LLM_CALL_FAILED
        assert "bad json" in info["detail"]
        assert info["guided_retry_attempted"] is False
        assert info["guided_retry_succeeded"] is None

    def test_detail_is_truncated(self):
        info = recovery.classify_llm_failure(True, ValueError("x" * 1000))
        assert len(info["detail"]) <= 300


class TestGuidedRetrySuffix:
    def test_known_failure_type_returns_nonempty_instruction(self):
        suffix = recovery.build_guided_retry_suffix(recovery.FAILURE_LLM_CALL_FAILED)
        assert "RECOVERY INSTRUCTION" in suffix
        assert "contexts" in suffix

    def test_unknown_failure_type_returns_empty_string(self):
        assert recovery.build_guided_retry_suffix("something_unrecognized") == ""
        assert recovery.build_guided_retry_suffix(None) == ""


class TestStage1ExtractIntegration:
    def test_first_attempt_success_reports_no_recovery(self, monkeypatch):
        monkeypatch.setattr(pipeline, "llm_available", lambda: True)
        monkeypatch.setattr(pipeline, "call_llm_json", lambda *a, **k: {"contexts": []})
        result, used_fallback, recovery_info = pipeline.stage1_extract("battery drains fast", "ref text")
        assert used_fallback is False
        assert recovery_info == recovery.NO_RECOVERY

    def test_llm_failure_then_successful_guided_retry(self, monkeypatch):
        """First call raises; the guided-retry attempt succeeds -- the LLM
        path is used after all, never falling to offline, and the recovery
        is honestly reported."""
        calls = {"n": 0}

        def flaky(prompt, max_tokens=4000, retries=2):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("LLM did not return valid JSON")
            assert "RECOVERY INSTRUCTION" in prompt  # the retry really used the guided suffix
            return {"contexts": []}

        monkeypatch.setattr(pipeline, "llm_available", lambda: True)
        monkeypatch.setattr(pipeline, "call_llm_json", flaky)
        result, used_fallback, recovery_info = pipeline.stage1_extract("battery drains fast", "ref text")
        assert used_fallback is False
        assert calls["n"] == 2
        assert recovery_info["failure_detected"] is True
        assert recovery_info["guided_retry_attempted"] is True
        assert recovery_info["guided_retry_succeeded"] is True

    def test_llm_failure_then_guided_retry_also_fails_degrades_to_offline(self, monkeypatch):
        def always_fails(prompt, max_tokens=4000, retries=2):
            raise ValueError("still broken")

        monkeypatch.setattr(pipeline, "llm_available", lambda: True)
        monkeypatch.setattr(pipeline, "call_llm_json", always_fails)
        result, used_fallback, recovery_info = pipeline.stage1_extract("battery drains fast", "ref text")
        assert used_fallback is True
        assert "contexts" in result  # offline_fallback.offline_extract() still answers
        assert recovery_info["failure_detected"] is True
        assert recovery_info["guided_retry_attempted"] is True
        assert recovery_info["guided_retry_succeeded"] is False

    def test_legitimate_empty_result_never_triggers_a_retry(self, monkeypatch):
        """An honest {"contexts": []} from the FIRST attempt is not a
        failure -- see recovery.py's module-level scope note."""
        calls = {"n": 0}

        def one_call_only(prompt, max_tokens=4000, retries=2):
            calls["n"] += 1
            return {"contexts": []}

        monkeypatch.setattr(pipeline, "llm_available", lambda: True)
        monkeypatch.setattr(pipeline, "call_llm_json", one_call_only)
        result, used_fallback, recovery_info = pipeline.stage1_extract("battery drains fast", "ref text")
        assert calls["n"] == 1
        assert used_fallback is False
        assert recovery_info == recovery.NO_RECOVERY

    def test_no_llm_available_at_all_reports_no_recovery(self, monkeypatch):
        monkeypatch.setattr(pipeline, "llm_available", lambda: False)
        result, used_fallback, recovery_info = pipeline.stage1_extract("battery drains fast", "ref text")
        assert used_fallback is True
        assert recovery_info == recovery.NO_RECOVERY


class TestRunPipelineSurfacesRecovery:
    def test_meta_recovery_present_on_a_clean_offline_run(self):
        """Zero LLM key configured in the test environment (conftest.py
        never sets one) -- the whole suite runs on the offline path, so
        meta.recovery should be the honest no-failure shape end to end."""
        result = pipeline.run_pipeline("my battery drains fast, recovery meta test")
        assert result["meta"]["recovery"] == recovery.NO_RECOVERY

    def test_safety_short_circuit_carries_no_recovery_too(self):
        result = pipeline.run_pipeline("my phone is smoking, recovery safety test")
        assert result["meta"]["recovery"] == recovery.NO_RECOVERY
        assert result["meta"]["safety_alert"] is True
