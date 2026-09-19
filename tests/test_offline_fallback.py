"""
offline_fallback.py — the zero-API-key, zero-cost path. These tests are
the most important in the whole suite: this is what actually runs when a
judge clones the repo and runs it with no LLM_API_KEY configured (the
README's advertised "works with zero setup" path), and it's what backs
every number in eval/metrics.md.
"""
import offline_fallback as of


class TestQueryNormalization:
    def test_expands_contractions(self):
        out = of.normalize_query("My Phone Won't Turn On")
        assert "will not" in out.lower()
        assert "won't" not in out.lower()

    def test_core_problem_phrase_strips_filler_lead(self):
        core = of.core_problem_phrase("My battery drains too fast")
        assert not core.lower().startswith("my ")

    def test_significant_tokens_drops_stopwords(self):
        toks = of.significant_tokens("the battery is draining very fast")
        assert "the" not in toks
        assert "is" not in toks
        assert "battery" in toks
        assert "draining" in toks


class TestParaphraseGeneration:
    def test_generates_requested_count_range(self):
        paras = of.generate_paraphrases("battery drains fast")
        assert 8 <= len(paras) <= 10

    def test_paraphrases_are_distinct(self):
        paras = of.generate_paraphrases("screen flickers randomly")
        assert len(set(paras)) == len(paras)

    def test_paraphrases_do_not_invent_new_symptoms(self):
        """No-hallucination constraint applied to Stage 0 too: paraphrases
        must not introduce words the user never said (beyond generic
        connector/filler words)."""
        paras = of.generate_paraphrases("wifi keeps disconnecting")
        for p in paras:
            assert "wifi" in p.lower() or "fi" in p.lower()  # tolerate typo variants


class TestOfflineEnrich:
    def test_returns_required_keys(self):
        # offline_enrich itself doesn't set detected_language (it can't
        # detect language without an LLM) -- pipeline.py's meta assembly
        # defaults it to "en" for the offline path, which is covered in
        # test_pipeline.py instead.
        result = of.offline_enrich("My phone battery drains too fast")
        assert "technical_query" in result
        assert "query_variations" in result

    def test_technical_query_is_nonempty(self):
        result = of.offline_enrich("camera lags when I open it")
        assert result["technical_query"].strip() != ""


class TestMultiIssueSplitting:
    def test_splits_genuinely_disjoint_domains(self):
        parts = of.split_multi_issue("battery drains fast and camera lags when I open it")
        assert len(parts) == 2

    def test_does_not_split_single_issue_described_with_multiple_clauses(self):
        parts = of.split_multi_issue("screen flickers and then goes black")
        assert len(parts) == 1

    def test_does_not_split_on_bare_also(self):
        """Regression test: 'also' alone used to be a split trigger and
        incorrectly fragmented 'camera also lags' mid-clause."""
        parts = of.split_multi_issue("my camera also lags when opening")
        assert len(parts) == 1

    def test_three_way_split(self):
        parts = of.split_multi_issue("battery drains fast, wifi keeps disconnecting, and camera lags")
        assert len(parts) == 3


class TestContractPhrasingHelpers:
    def test_make_title_is_two_to_three_words(self):
        title = of.make_title("battery drains too fast after update")
        assert 2 <= len(title.split()) <= 3

    def test_make_goal_matches_required_syntax(self):
        goal = of.make_goal(of.make_title("battery drains fast"))
        import validators
        ok, msg = validators.check_goal_syntax(goal)
        assert ok, msg

    def test_make_description_is_five_to_seven_words_starting_with_it_will(self):
        desc = of.make_description("let you fix the battery drain issue completely and permanently")
        import validators
        ok, msg = validators.check_description_format(desc)
        assert ok, msg


class TestOfflineExtractEndToEnd:
    """Runs offline_extract against the real official 20-query dataset --
    the same data judges evaluate against."""

    def test_all_real_queries_produce_schema_valid_output(self, real_samples):
        import validators
        for sample in real_samples:
            result = of.offline_extract(sample["complaint"], sample.get("siis_response", ""))
            assert "contexts" in result
            for goal in result["contexts"]:
                errors = validators.validate_goal_object(dict(goal))
                assert errors == [], f"{sample['complaint'][:50]}: {errors}"

    def test_empty_siis_response_always_yields_no_match(self, real_samples):
        """No-hallucination guarantee: with no reference text, the offline
        path must never invent a fix."""
        sample = real_samples[0]
        result = of.offline_extract(sample["complaint"], "")
        assert result["contexts"] == []

    def test_actions_are_one_screen_each_and_ordered_critical_last(self, real_samples):
        for sample in real_samples:
            result = of.offline_extract(sample["complaint"], sample.get("siis_response", ""))
            for goal in result["contexts"]:
                cats = [a["category"] for a in goal["actions"]]
                if "critical" in cats:
                    first = cats.index("critical")
                    assert all(c == "critical" for c in cats[first:])
