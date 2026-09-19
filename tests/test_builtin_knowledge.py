"""tests/test_builtin_knowledge.py -- covers builtin_knowledge.py itself
(the generic offline-fallback reference text) and the `strict` matching
behaviour in offline_fallback.py that only ever applies to it (see
_build_single_goal()'s docstring). These are regression tests for two
real bugs found while verifying "can a real person actually use this
with no LLM key and no reference text": a single coincidental shared
word (e.g. "what", or "wifi" appearing once in an unrelated section)
could either produce a false-positive match on a meaningless query, or
tie two unrelated sections together into one garbled, mixed-topic plan.
Both are fixed structurally in offline_fallback.py (a minimum absolute
token-overlap count, plus sourcing steps from exactly one section in
strict mode) and reinforced here so they can't silently regress."""
import offline_fallback as of
from builtin_knowledge import BUILTIN_REFERENCE_TEXT, get_builtin_reference_text


class TestGetBuiltinReferenceText:
    def test_returns_the_module_constant(self):
        assert get_builtin_reference_text() == BUILTIN_REFERENCE_TEXT

    def test_parses_into_multiple_real_sections(self):
        sections = of.parse_sections(get_builtin_reference_text())
        headers = {s.header for s in sections}
        assert len(sections) >= 15
        assert "Battery Draining Quickly" in headers
        assert "Wifi And Bluetooth Connectivity" in headers

    def test_every_section_yields_at_least_two_extractable_steps(self):
        """A section that can't produce steps (e.g. every sentence gets
        filtered by _is_step_sentence()) would silently return an empty
        goal -- catch that at test time, not in front of a real user."""
        sections = of.parse_sections(get_builtin_reference_text())
        for sec in sections:
            steps = of.extract_steps(sec)
            assert len(steps) >= 2, f"section {sec.header!r} only yielded {steps!r}"


class TestStrictModeFalsePositives:
    """Bug 1: a query with no real diagnostic content ("what is the
    capital of France") must not match just because a common word like
    "what" or "some" happens to appear once, somewhere, in the broad
    built-in corpus."""

    def test_generic_question_words_never_cause_a_false_match(self):
        for junk_query in (
            "what is the capital of France",
            "some complaint with nothing to ground it",
            "how do I cook pasta at home",
            "why is the sky blue",
        ):
            result = of.offline_extract(junk_query, "")
            assert result["contexts"] == [], junk_query

    def test_what_how_why_which_who_some_are_not_significant_tokens(self):
        for w in ("what", "how", "why", "which", "who", "some"):
            assert of.significant_tokens(w) == set()


class TestStrictModeNoCrossTopicMixing:
    """Bug 2: a real, in-scope query ("wifi keeps disconnecting") must
    source its plan from exactly one topic section, never a blend of
    unrelated sections that happen to tie on a single shared word."""

    def _origins(self, query):
        sections = of.parse_sections(get_builtin_reference_text())
        step_to_headers = {}
        for sec in sections:
            for step in of.extract_steps(sec):
                step_to_headers.setdefault(step.strip().lower(), set()).add(sec.header)
        result = of.offline_extract(query, "")
        assert result["contexts"], query
        origins = set()
        for ctx in result["contexts"]:
            for action in ctx["actions"]:
                for group in action["stepGroups"]:
                    for step in group["steps"]:
                        origins |= step_to_headers.get(step.strip().lower(), {f"UNKNOWN:{step}"})
        return origins

    def test_wifi_query_matches_only_the_wifi_section(self):
        assert self._origins("wifi keeps disconnecting randomly") == {"Wifi And Bluetooth Connectivity"}

    def test_battery_query_does_not_pull_in_black_screen_section(self):
        assert self._origins("my battery is getting drained quickly") == {"Battery Draining Quickly"}

    def test_touchscreen_query_matches_only_the_touchscreen_section(self):
        assert self._origins("touchscreen is not responding") == {"Touchscreen Unresponsive"}


class TestStrictModeStemming:
    """Ordinary word-ending variation ("crashing" vs "crash", "working"
    vs "work") should still match in strict mode -- only real
    caller-supplied siis_response matching stays exact-token (untouched
    by `stem`, see significant_tokens())."""

    def test_gerund_form_matches_base_form_in_builtin_corpus(self):
        result = of.offline_extract("speaker is not working", "")
        assert result["contexts"]
        assert result["contexts"][0]["title"].lower().startswith("speaker")

    def test_stemming_is_off_by_default_for_real_siis_response(self):
        # A real siis_response is matched with exact tokens only -- this
        # is a sanity check that `stem` defaults to False everywhere it
        # isn't explicitly requested.
        assert of.significant_tokens("crashing") == {"crashing"}
        assert of.significant_tokens("crashing", stem=True) == {"crash"}


class TestUsedBuiltinReferenceFlag:
    def test_flagged_true_when_no_siis_response_supplied(self):
        result = of.offline_extract("my battery is draining fast", "")
        assert result["used_builtin_reference"] is True

    def test_flagged_false_when_real_siis_response_supplied(self, real_samples=None):
        result = of.offline_extract("anything", "# Some Real Doc\nRestart your phone.")
        assert result["used_builtin_reference"] is False
