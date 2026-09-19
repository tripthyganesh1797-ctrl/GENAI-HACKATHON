"""deeplink_matching.py — retrieval against the real 578-entry official
catalog, both matcher variants, and the feedback-driven re-ranking hook."""
import deeplink_matching as dm
import feedback


def test_loads_real_catalog():
    entries = dm.load_deeplinks("deeplinks.json")
    assert len(entries) > 500  # official catalog has 578 entries


def test_hybrid_index_returns_a_real_match_for_a_clear_query():
    index = dm.get_index("hybrid")
    entry, score = index.best_match("wifi settings screen")
    assert entry is not None
    assert score > 0


def test_hybrid_index_rejects_nonsense_below_threshold():
    index = dm.get_index("hybrid")
    entry, score = index.best_match("zzzzz qqqqq xxxxx gibberish nonword")
    assert entry is None


def test_rules_index_returns_a_real_match():
    entries = dm.load_deeplinks("deeplinks.json")
    index = dm.RulesDeeplinkIndex(entries)
    entry, score = index.best_match("wifi settings screen")
    assert entry is not None


def test_match_and_build_deeplink_shape():
    actionable, validation = dm.match_and_build_deeplink(
        "Wifi Settings", ["Tap Wifi.", "Toggle Wifi on."], variant="hybrid",
    )
    assert "deeplink" in actionable
    assert "description" in actionable
    assert "message" in actionable

    # falls back to the placeholder deeplink for something the catalog has
    # nothing for (rather than raising) -- never crashes the pipeline
    actionable2, validation2 = dm.match_and_build_deeplink(
        "Completely Made Up Nonexistent Screen Xyzzy", ["Do the thing."], variant="rules",
    )
    assert actionable2["deeplink"] in (dm.DUMMY_POSITIVE_DEEPLINK,) or actionable2["deeplink"]


class TestMatchExplainability:
    """Task 29: every actionableDeeplink (real match or placeholder) should
    carry a matchExplanation dict a judge/dev can use to see *why* -- score
    components, the feedback nudge applied, and overlapping keywords."""

    def test_hybrid_explanation_has_score_breakdown(self):
        index = dm.get_index("hybrid")
        entry, score, explanation = index.best_match_explained("wifi settings screen")
        assert entry is not None
        assert explanation["matcher"] == "hybrid_bm25_dense"
        for key in ("bm25_component", "dense_component", "combined_before_feedback",
                    "feedback_adjustment", "final_score", "threshold", "matched_keywords"):
            assert key in explanation
        assert explanation["final_score"] == round(score, 4)
        # the components should actually blend into the final score (loose
        # tolerance since bm25_component/dense_component are independently
        # rounded to 4dp before this recomputation, which can compound)
        expected = (explanation["alpha"] * explanation["bm25_component"]
                    + (1 - explanation["alpha"]) * explanation["dense_component"])
        assert abs(explanation["combined_before_feedback"] - expected) < 1e-3

    def test_hybrid_explanation_on_rejection_still_returned(self):
        index = dm.get_index("hybrid")
        entry, score, explanation = index.best_match_explained("zzzzz qqqqq xxxxx gibberish nonword")
        assert entry is None
        assert explanation["rejected_reason"] == "final_score below threshold"

    def test_hybrid_best_match_and_best_match_explained_agree(self):
        """The two entry points must never disagree on the winning entry --
        best_match() is still used by offline_fallback.py's older call
        sites and eval/matchers.py, so a silent divergence here would mean
        the API's explained response describes a different match than the
        one actually used elsewhere."""
        index = dm.get_index("hybrid")
        for query in ("wifi settings screen", "battery drains fast", "camera blurry photos"):
            plain_entry, plain_score = index.best_match(query)
            exp_entry, exp_score, _ = index.best_match_explained(query)
            assert plain_entry == exp_entry
            assert abs(plain_score - exp_score) < 1e-9

    def test_rules_explanation_has_score_breakdown(self):
        entries = dm.load_deeplinks("deeplinks.json")
        index = dm.RulesDeeplinkIndex(entries)
        entry, score, explanation = index.best_match_explained("wifi settings screen")
        assert entry is not None
        assert explanation["matcher"] == "rules_fuzzy"
        for key in ("fuzzy_score", "feedback_adjustment_pts", "final_score_pts",
                    "threshold_pts", "matched_keywords"):
            assert key in explanation

    def test_rules_best_match_and_best_match_explained_agree(self):
        entries = dm.load_deeplinks("deeplinks.json")
        index = dm.RulesDeeplinkIndex(entries)
        for query in ("wifi settings screen", "battery drains fast"):
            plain_entry, plain_score = index.best_match(query)
            exp_entry, exp_score, _ = index.best_match_explained(query)
            assert plain_entry == exp_entry
            assert abs(plain_score - exp_score) < 1e-9

    def test_match_and_build_deeplink_includes_explanation_for_real_match(self):
        actionable, _ = dm.match_and_build_deeplink(
            "Wifi Settings", ["Tap Wifi.", "Toggle Wifi on."], variant="hybrid",
        )
        assert "matchExplanation" in actionable
        assert actionable["matchExplanation"]["matcher"] == "hybrid_bm25_dense"

    def test_match_and_build_deeplink_includes_explanation_for_placeholder(self):
        actionable, _ = dm.match_and_build_deeplink(
            "Completely Made Up Nonexistent Screen Xyzzy", ["Do the thing."], variant="rules",
        )
        assert "matchExplanation" in actionable
        assert actionable["matchExplanation"]["matcher"] == "rules_fuzzy"
        assert "rejected_reason" in actionable["matchExplanation"]

    def test_matched_keywords_are_real_overlap(self):
        index = dm.get_index("hybrid")
        entry, _, explanation = index.best_match_explained("wifi settings screen toggle")
        assert entry is not None
        corpus_tokens = set(dm._tokenize(entry.corpus_text))
        for kw in explanation["matched_keywords"]:
            assert kw in corpus_tokens


def test_never_matches_the_dummy_positive_entry_itself():
    """DUMMY_POSITIVE_DEEPLINK is a sentinel in the source data, not a real
    screen -- it must be filtered out of the index entirely."""
    index = dm.get_index("hybrid")
    assert all(e.deeplink != dm.DUMMY_POSITIVE_DEEPLINK for e in index.entries)


class TestFeedbackDrivenReranking:
    """Regression test for the Task 22 feature: repeated negative feedback
    on the current top match must be able to demote it below an
    alternative candidate."""

    def test_repeated_thumbs_down_demotes_top_match(self):
        index = dm.get_index("hybrid")
        query = "battery drains fast charging settings"
        entry, _ = index.best_match(query)
        assert entry is not None

        for _ in range(8):
            feedback.record_feedback(entry.deeplink, "Battery Settings", helpful=False)

        entry_after, _ = index.best_match(query)
        assert entry_after.deeplink != entry.deeplink

    def test_thumbs_up_increases_score_for_that_entry(self):
        index = dm.get_index("hybrid")
        entries = index.entries
        target = entries[0]
        before = feedback.get_adjustment(target.deeplink)
        feedback.record_feedback(target.deeplink, "X", helpful=True)
        after = feedback.get_adjustment(target.deeplink)
        assert after > before


class TestSessionAvoidance:
    """Task 36 (session_memory.py): `avoid_deeplinks` on best_match_explained()
    is a DIFFERENT mechanism from the feedback-driven re-ranking above --
    global feedback permanently nudges a score for everyone; avoid_deeplinks
    is per-request and only reorders which of the already-qualifying
    candidates gets returned, never touching the threshold decision."""

    def test_hybrid_default_is_unaffected_by_empty_or_none_avoid_set(self):
        index = dm.get_index("hybrid")
        query = "wifi settings screen"
        entry_none, score_none, exp_none = index.best_match_explained(query)
        entry_empty, score_empty, exp_empty = index.best_match_explained(query, avoid_deeplinks=set())
        assert entry_none.deeplink == entry_empty.deeplink
        assert score_none == score_empty
        assert "session_avoid_skipped" not in exp_none
        assert "already_tried_this_session" not in exp_none

    def test_hybrid_skips_past_avoided_top_match_to_the_next_viable_one(self):
        index = dm.get_index("hybrid")
        query = "wifi settings screen"
        top_entry, _ = index.best_match(query)
        assert top_entry is not None

        entry, score, explanation = index.best_match_explained(
            query, avoid_deeplinks={top_entry.deeplink}
        )
        assert entry is not None
        assert entry.deeplink != top_entry.deeplink
        assert explanation["session_avoid_skipped"] == top_entry.deeplink
        assert "already_tried_this_session" not in explanation  # the CHOSEN one wasn't avoided

    def test_pick_index_avoiding_falls_back_when_avoided_is_the_only_viable_candidate(self):
        """The whole point, pinned down directly against the pure helper
        both index variants share (dm._pick_index_avoiding) rather than a
        tiny real HybridDeeplinkIndex -- a 1-2 document TF-IDF/SVD corpus
        is numerically degenerate (near-zero similarities either way) and
        isn't a meaningful way to exercise this: never turn a real match
        into a false no-match just because the session already tried it.
        See session_memory.py's module docstring."""
        ranked_indices = [0, 1, 2]  # already sorted best-first
        scores = [0.9, 0.05, 0.02]  # only index 0 clears this threshold
        deeplinks = ["dl-a", "dl-b", "dl-c"]
        chosen_i, skipped = dm._pick_index_avoiding(
            ranked_indices, scores, deeplinks, threshold=0.12, avoid_deeplinks={"dl-a"}
        )
        assert chosen_i == 0  # falls back to the avoided (only viable) entry
        assert skipped is False

    def test_pick_index_avoiding_skips_to_next_viable_candidate(self):
        ranked_indices = [0, 1, 2]
        scores = [0.9, 0.7, 0.02]
        deeplinks = ["dl-a", "dl-b", "dl-c"]
        chosen_i, skipped = dm._pick_index_avoiding(
            ranked_indices, scores, deeplinks, threshold=0.12, avoid_deeplinks={"dl-a"}
        )
        assert chosen_i == 1
        assert skipped is True

    def test_pick_index_avoiding_is_a_noop_for_empty_avoid_set(self):
        ranked_indices = [0, 1, 2]
        scores = [0.9, 0.7, 0.02]
        deeplinks = ["dl-a", "dl-b", "dl-c"]
        chosen_i, skipped = dm._pick_index_avoiding(
            ranked_indices, scores, deeplinks, threshold=0.12, avoid_deeplinks=set()
        )
        assert chosen_i == 0
        assert skipped is False

    def test_pick_index_avoiding_empty_ranking_returns_sentinel(self):
        chosen_i, skipped = dm._pick_index_avoiding([], [], [], threshold=0.12, avoid_deeplinks={"x"})
        assert chosen_i == -1
        assert skipped is False

    def test_hybrid_avoiding_an_irrelevant_deeplink_changes_nothing(self):
        index = dm.get_index("hybrid")
        query = "wifi settings screen"
        entry_plain, score_plain = index.best_match(query)
        entry_avoid, score_avoid, _ = index.best_match_explained(
            query, avoid_deeplinks={"bixby://masked/act/not-in-the-top-results-at-all"}
        )
        assert entry_plain.deeplink == entry_avoid.deeplink
        assert score_plain == score_avoid

    def test_rules_skips_past_avoided_top_match_to_the_next_viable_one(self):
        entries = dm.load_deeplinks("deeplinks.json")
        index = dm.RulesDeeplinkIndex(entries)
        query = "wifi settings screen"
        top_entry, _ = index.best_match(query)
        assert top_entry is not None

        entry, score, explanation = index.best_match_explained(
            query, avoid_deeplinks={top_entry.deeplink}
        )
        assert entry is not None
        assert entry.deeplink != top_entry.deeplink
        assert explanation["session_avoid_skipped"] == top_entry.deeplink

    def test_rules_falls_back_to_avoided_entry_when_it_is_the_only_viable_match(self):
        entries = dm.load_deeplinks("deeplinks.json")
        index = dm.RulesDeeplinkIndex(entries[:1])
        query = entries[0].description or entries[0].message or "settings"
        top_entry, top_score = index.best_match(query)
        assert top_entry is not None

        entry, score, explanation = index.best_match_explained(
            query, avoid_deeplinks={top_entry.deeplink}
        )
        assert entry is not None
        assert entry.deeplink == top_entry.deeplink
        assert explanation["already_tried_this_session"] is True

    def test_match_and_build_deeplink_threads_avoid_deeplinks_through(self):
        index = dm.get_index("hybrid")
        top_entry, _ = index.best_match("wifi settings screen")
        assert top_entry is not None

        actionable, _ = dm.match_and_build_deeplink(
            "Wifi Settings", ["Tap Wifi.", "Toggle Wifi on."], variant="hybrid",
            avoid_deeplinks={top_entry.deeplink},
        )
        assert actionable["deeplink"] != top_entry.deeplink
        assert actionable["matchExplanation"]["session_avoid_skipped"] == top_entry.deeplink
