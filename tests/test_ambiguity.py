"""tests/test_ambiguity.py -- ambiguity.py's CLAM-inspired second opinion on
clarify.py's heuristic: must be a complete no-op with no LLM key, must only
ever RAISE the heuristic's verdict (never lower it), and must degrade
honestly on any malformed/failed LLM response."""
import ambiguity


class TestNoLLMKeyIsANoOp:
    def test_classify_ambiguity_returns_none_without_a_key(self, monkeypatch):
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: False)
        assert ambiguity.classify_ambiguity("it's broken") is None

    def test_refine_never_raises_without_a_key(self, monkeypatch):
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: False)
        flag, info = ambiguity.refine_needs_clarification(False, "my phone has an issue with the screen")
        assert flag is False
        assert info is None

    def test_refine_keeps_true_heuristic_flag_without_a_key(self, monkeypatch):
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: False)
        flag, info = ambiguity.refine_needs_clarification(True, "it's broken")
        assert flag is True
        assert info is None


class TestEmptyComplaint:
    def test_empty_string_returns_none(self, monkeypatch):
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: True)
        assert ambiguity.classify_ambiguity("") is None
        assert ambiguity.classify_ambiguity("   ") is None


class TestLLMPathHonestDegrade:
    def test_malformed_llm_response_returns_none(self, monkeypatch):
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: True)
        monkeypatch.setattr(ambiguity, "call_llm_json", lambda *a, **k: {"not_ambiguous_key": True})
        assert ambiguity.classify_ambiguity("my phone has an issue with the screen") is None

    def test_llm_exception_returns_none_not_raise(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("provider down")
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: True)
        monkeypatch.setattr(ambiguity, "call_llm_json", boom)
        assert ambiguity.classify_ambiguity("my phone has an issue with the screen") is None

    def test_confidence_clamped_to_0_1(self, monkeypatch):
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: True)
        monkeypatch.setattr(
            ambiguity, "call_llm_json",
            lambda *a, **k: {"ambiguous": True, "confidence": 5.0, "reason": "way over"},
        )
        info = ambiguity.classify_ambiguity("my phone has an issue with the screen")
        assert info["confidence"] == 1.0
        assert info["method"] == "llm"


class TestRefineOnlyEverRaisesTheFlag:
    def test_heuristic_already_true_ignores_llm_opinion(self, monkeypatch):
        """Even if the LLM confidently says NOT ambiguous, a True heuristic
        flag must never be lowered -- see ambiguity.py's module docstring."""
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: True)
        monkeypatch.setattr(
            ambiguity, "call_llm_json",
            lambda *a, **k: {"ambiguous": False, "confidence": 0.99, "reason": "very clear"},
        )
        flag, info = ambiguity.refine_needs_clarification(True, "it's broken")
        assert flag is True
        assert info["ambiguous"] is False  # info is still honestly reported

    def test_low_confidence_llm_opinion_does_not_flip_a_false_heuristic(self, monkeypatch):
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: True)
        monkeypatch.setattr(
            ambiguity, "call_llm_json",
            lambda *a, **k: {"ambiguous": True, "confidence": 0.5, "reason": "maybe"},
        )
        flag, info = ambiguity.refine_needs_clarification(False, "my phone has an issue with the screen")
        assert flag is False
        assert info["confidence"] == 0.5

    def test_high_confidence_ambiguous_opinion_flips_a_false_heuristic(self, monkeypatch):
        """The catch this module exists for: a complaint that names a
        component (so clarify.py's keyword gate lets it through) but no
        actual symptom -- the LLM catching that IS the value-add."""
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: True)
        monkeypatch.setattr(
            ambiguity, "call_llm_json",
            lambda *a, **k: {"ambiguous": True, "confidence": 0.9,
                              "reason": "names screen but not what's wrong with it"},
        )
        flag, info = ambiguity.refine_needs_clarification(False, "my phone has an issue with the screen")
        assert flag is True
        assert info["method"] == "llm"

    def test_confidence_exactly_at_threshold_flips(self, monkeypatch):
        monkeypatch.setattr(ambiguity, "api_key_configured", lambda: True)
        monkeypatch.setattr(
            ambiguity, "call_llm_json",
            lambda *a, **k: {"ambiguous": True, "confidence": ambiguity.CONFIDENCE_OVERRIDE_THRESHOLD,
                              "reason": "boundary"},
        )
        flag, _ = ambiguity.refine_needs_clarification(False, "something's off with the screen")
        assert flag is True
