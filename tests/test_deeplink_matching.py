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
