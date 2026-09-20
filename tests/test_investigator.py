"""tests/test_investigator.py -- investigator.py's SIA-inspired multi-turn
diagnosis loop. The whole suite runs with no LLM_API_KEY configured (see
conftest.py / the rest of this repo's test philosophy), so every case here
exercises the fully offline, deterministic path -- exactly the path a
judge running this project with zero setup will actually see."""
import investigator


class TestCandidateHypotheses:
    def test_known_issue_phrase_uses_curated_related_issues(self):
        hyps = investigator._candidate_hypotheses("Battery draining fast")
        assert len(hyps) == 3
        assert "A specific app running heavily in the background" in hyps

    def test_unknown_issue_phrase_uses_generic_fallback(self):
        hyps = investigator._candidate_hypotheses("Some completely unmapped bucket")
        assert hyps == investigator._GENERIC_FALLBACK_HYPOTHESES


class TestOfflineInitialPriors:
    def test_priors_sum_to_one(self):
        hyps = ["a", "b", "c"]
        priors = investigator._initial_priors(hyps, "some complaint")
        assert abs(sum(priors.values()) - 1.0) < 1e-6

    def test_earlier_hypotheses_get_more_weight(self):
        hyps = ["first", "second", "third"]
        priors = investigator._initial_priors(hyps, "some complaint")
        assert priors["first"] > priors["second"] > priors["third"]


class TestOfflineQuestionPool:
    def test_first_call_returns_first_question(self):
        q = investigator._offline_next_question([])
        assert q == investigator._GENERIC_DISCRIMINATIVE_QUESTIONS[0]

    def test_skips_already_asked_questions(self):
        asked = [investigator._GENERIC_DISCRIMINATIVE_QUESTIONS[0]]
        q = investigator._offline_next_question(asked)
        assert q == investigator._GENERIC_DISCRIMINATIVE_QUESTIONS[1]

    def test_returns_none_once_pool_is_exhausted(self):
        q = investigator._offline_next_question(list(investigator._GENERIC_DISCRIMINATIVE_QUESTIONS))
        assert q is None


class TestOfflineProbabilityUpdate:
    def test_software_signal_boosts_software_bucketed_hypothesis(self):
        hyps = [
            "A software bug introduced by a recent update",
            "A hardware fault in the battery",
        ]
        priors = {hyps[0]: 0.5, hyps[1]: 0.5}
        updated = investigator._offline_update_probabilities(
            hyps, priors, "it started right after I updated the software"
        )
        assert updated[hyps[0]] > updated[hyps[1]]

    def test_hardware_signal_boosts_hardware_bucketed_hypothesis(self):
        hyps = [
            "A software bug introduced by a recent update",
            "A hardware fault in the battery",
        ]
        priors = {hyps[0]: 0.5, hyps[1]: 0.5}
        updated = investigator._offline_update_probabilities(
            hyps, priors, "it happens every single time, consistently, since I dropped it"
        )
        assert updated[hyps[1]] > updated[hyps[0]]

    def test_no_signal_leaves_probabilities_untouched(self):
        hyps = ["a mysterious cause", "another mysterious cause"]
        priors = {hyps[0]: 0.6, hyps[1]: 0.4}
        updated = investigator._offline_update_probabilities(hyps, priors, "I'm not sure, hard to say")
        assert updated == priors

    def test_probabilities_stay_normalized(self):
        hyps = ["A software bug introduced by a recent update", "A hardware fault"]
        priors = {hyps[0]: 0.5, hyps[1]: 0.5}
        updated = investigator._offline_update_probabilities(hyps, priors, "a recent app update caused it")
        assert abs(sum(updated.values()) - 1.0) < 1e-6


class TestStartInvestigation:
    def test_returns_in_progress_with_first_question(self):
        result = investigator.start_investigation("my battery drains really fast")
        assert result["status"] == "in_progress"
        assert result["question"] == investigator._GENERIC_DISCRIMINATIVE_QUESTIONS[0]
        assert len(result["hypotheses"]) == 3
        assert result["investigation_id"]
        assert result["final_result"] is None

    def test_hypotheses_sorted_by_probability_descending(self):
        result = investigator.start_investigation("my battery drains really fast")
        probs = [h["probability"] for h in result["hypotheses"]]
        assert probs == sorted(probs, reverse=True)

    def test_unmapped_complaint_still_starts_with_generic_hypotheses(self):
        result = investigator.start_investigation("my phone's alarm clock feature seems off")
        labels = {h["label"] for h in result["hypotheses"]}
        assert labels == set(investigator._GENERIC_FALLBACK_HYPOTHESES)

    def test_single_hypothesis_resolves_immediately(self, monkeypatch):
        monkeypatch.setattr(investigator, "_candidate_hypotheses", lambda issue_phrase: ["only one cause"])
        result = investigator.start_investigation("some complaint")
        assert result["status"] == "resolved"
        assert result["winning_hypothesis"] == "only one cause"
        assert result["final_result"] is not None
        assert "response" in result["final_result"]

    def test_already_confident_prior_resolves_immediately_without_a_question(self, monkeypatch):
        monkeypatch.setattr(
            investigator, "_initial_priors",
            lambda hyps, complaint: {hyps[0]: 0.95, **{h: 0.05 / (len(hyps) - 1) for h in hyps[1:]}},
        )
        result = investigator.start_investigation("my battery drains really fast")
        assert result["status"] == "resolved"
        assert result["question"] is None


class TestAnswerInvestigation:
    def test_unknown_investigation_id_returns_none(self):
        assert investigator.answer_investigation("does-not-exist", "some answer") is None

    def test_answering_advances_to_a_second_question(self):
        start = investigator.start_investigation("my battery drains really fast")
        result = investigator.answer_investigation(
            start["investigation_id"], "it happens every single time, consistently"
        )
        assert result["status"] == "in_progress"
        assert result["questions_asked"] == 1
        assert result["question"] == investigator._GENERIC_DISCRIMINATIVE_QUESTIONS[1]

    def test_budget_exhaustion_resolves_with_a_real_pipeline_result(self):
        start = investigator.start_investigation("my battery drains really fast")
        inv_id = start["investigation_id"]
        answers = [
            "it happens every single time, consistently",
            "no clear trigger that I've noticed",
            "restarting didn't change anything",
        ]
        result = None
        for answer in answers:
            result = investigator.answer_investigation(inv_id, answer)
        assert result["status"] == "resolved"
        assert result["questions_asked"] == investigator.QUESTION_BUDGET
        assert result["winning_hypothesis"] in [h["label"] for h in result["hypotheses"]]
        final = result["final_result"]
        # The winning hypothesis is folded into what the PIPELINE sees (so
        # it can influence matching), but response["query"] must still
        # echo exactly what the user typed -- same guarantee
        # image_analysis.py's folded-in photo description has.
        assert final["query"] == "my battery drains really fast"
        assert "response" in final and "meta" in final

    def test_confidence_threshold_can_resolve_before_budget_exhausted(self, monkeypatch):
        start = investigator.start_investigation("my battery drains really fast")
        inv_id = start["investigation_id"]

        def force_confident(hyps, probs, answer):
            top = hyps[0]
            return {h: (0.95 if h == top else 0.05 / (len(hyps) - 1)) for h in hyps}

        monkeypatch.setattr(investigator, "_offline_update_probabilities", force_confident)
        result = investigator.answer_investigation(inv_id, "clearly the first cause")
        assert result["status"] == "resolved"
        assert result["questions_asked"] == 1

    def test_reanswering_a_resolved_investigation_is_idempotent(self):
        start = investigator.start_investigation("my battery drains really fast")
        inv_id = start["investigation_id"]
        for answer in ["a", "b", "c"]:
            result = investigator.answer_investigation(inv_id, answer)
        assert result["status"] == "resolved"
        again = investigator.answer_investigation(inv_id, "does this do anything?")
        assert again["status"] == "resolved"
        assert again["winning_hypothesis"] == result["winning_hypothesis"]
        assert again["questions_asked"] == result["questions_asked"]  # no extra answer recorded


class TestGetInvestigation:
    def test_unknown_id_returns_none(self):
        assert investigator.get_investigation("does-not-exist") is None

    def test_known_id_returns_current_state_without_mutating_it(self):
        start = investigator.start_investigation("my battery drains really fast")
        inv_id = start["investigation_id"]
        first_read = investigator.get_investigation(inv_id)
        second_read = investigator.get_investigation(inv_id)
        assert first_read["questions_asked"] == second_read["questions_asked"] == 0
        assert first_read["question"] == start["question"]  # re-surfaces the pending question unchanged


class TestInvestigationStats:
    def test_counts_reflect_store_contents(self):
        before = investigator.investigation_stats()
        investigator.start_investigation("my battery drains really fast")
        after = investigator.investigation_stats()
        assert after["total_investigations"] == before["total_investigations"] + 1
        assert after["in_progress"] == before["in_progress"] + 1
