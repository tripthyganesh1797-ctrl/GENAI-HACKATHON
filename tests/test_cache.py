"""cache.py — semantic (keyword-overlap) cache tests."""
import cache


def test_miss_on_empty_store():
    assert cache.get_cached("battery drains fast") is None


def test_exact_query_hits():
    cache.set_cached("battery drains fast", {"response": {"contexts": []}, "meta": {}})
    hit = cache.get_cached("battery drains fast")
    assert hit is not None
    assert hit["response"] == {"contexts": []}


def test_similar_paraphrase_hits_above_threshold():
    cache.set_cached("phone battery drains very fast after update", {"meta": {"fallback": None}})
    # Shares most content words -- should clear the 0.6 Jaccard threshold.
    hit = cache.get_cached("phone battery drains very fast update")
    assert hit is not None


def test_unrelated_query_misses():
    cache.set_cached("camera lags when opening", {"meta": {}})
    assert cache.get_cached("wifi keeps disconnecting at night") is None


def test_cached_at_timestamp_is_added():
    cache.set_cached("screen flickers", {"meta": {}})
    hit = cache.get_cached("screen flickers")
    assert "_cached_at" in hit
