"""request_log.py — per-request logging + /stats aggregation."""
import request_log as rl


def test_empty_log_returns_zeroed_stats():
    stats = rl.compute_stats()
    assert stats["total_requests"] == 0
    assert stats["avg_latency_ms"] == 0
    assert stats["requests_by_domain"] == {}
    assert stats["trending_issues"] == {}


def test_append_and_aggregate():
    rl.append_log({"domain_guess": "Battery", "cache_hit": False, "latency_ms": 10.0,
                    "total_tokens": 100, "cost_usd": 0.001, "fallback": None,
                    "used_offline_fallback": False})
    rl.append_log({"domain_guess": "Battery", "cache_hit": True, "latency_ms": 2.0,
                    "total_tokens": 0, "cost_usd": 0.0, "fallback": None,
                    "used_offline_fallback": False})
    rl.append_log({"domain_guess": "Display", "cache_hit": False, "latency_ms": 5.0,
                    "total_tokens": 0, "cost_usd": 0.0, "fallback": "no_match",
                    "used_offline_fallback": True})

    stats = rl.compute_stats()
    assert stats["total_requests"] == 3
    assert stats["avg_latency_ms"] == round((10.0 + 2.0 + 5.0) / 3, 1)
    assert stats["cache_hit_rate"] == round(1 / 3, 3)
    assert stats["no_match_rate"] == round(1 / 3, 3)
    assert stats["total_tokens"] == 100
    assert stats["requests_by_domain"] == {"Battery": 2, "Display": 1}


def test_corrupted_line_is_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "LOG_FILE", str(tmp_path / "log.jsonl"))
    with open(rl.LOG_FILE, "w") as f:
        f.write('{"domain_guess": "Battery", "latency_ms": 1.0}\n')
        f.write("not valid json at all\n")
        f.write('{"domain_guess": "Camera", "latency_ms": 3.0}\n')

    logs = rl.read_all_logs()
    assert len(logs) == 2  # the corrupted line was skipped, not fatal


class TestTrendingIssues:
    """Task 37: /stats surfaces a finer-grained 'top recurring issues'
    breakdown (pipeline.py's issue_guess), not just the 5-bucket domain
    count -- these tests exercise compute_trending_issues() directly,
    independent of what specific phrases pipeline.py's keyword matcher
    produces (that logic has its own coverage in test_pipeline.py)."""

    def test_ranks_by_frequency_descending(self):
        logs = (
            [{"issue_guess": "Battery draining fast"}] * 3
            + [{"issue_guess": "Screen flickering"}] * 5
            + [{"issue_guess": "Wi-Fi connectivity"}] * 1
        )
        trending = rl.compute_trending_issues(logs)
        assert list(trending.items()) == [
            ("Screen flickering", 5),
            ("Battery draining fast", 3),
            ("Wi-Fi connectivity", 1),
        ]

    def test_respects_top_n_cap(self):
        logs = [{"issue_guess": f"issue-{i}"} for i in range(20) for _ in range(1)]
        trending = rl.compute_trending_issues(logs, top_n=3)
        assert len(trending) == 3

    def test_falls_back_to_domain_guess_for_pre_task37_log_lines(self):
        """A log line written before issue_guess existed must still be
        counted -- toward its domain_guess -- rather than silently
        vanishing from the trending-issues totals."""
        logs = [
            {"domain_guess": "Battery"},  # old-format line, no issue_guess
            {"domain_guess": "Battery"},
            {"issue_guess": "Camera app crashing", "domain_guess": "Camera"},
        ]
        trending = rl.compute_trending_issues(logs)
        assert trending == {"Battery": 2, "Camera app crashing": 1}

    def test_missing_both_fields_buckets_as_unknown(self):
        logs = [{}, {}]
        trending = rl.compute_trending_issues(logs)
        assert trending == {"Unknown": 2}

    def test_empty_logs_returns_empty_dict(self):
        assert rl.compute_trending_issues([]) == {}

    def test_compute_stats_includes_trending_issues(self):
        rl.append_log({"domain_guess": "Battery", "issue_guess": "Battery draining fast",
                        "cache_hit": False, "latency_ms": 1.0, "total_tokens": 0,
                        "cost_usd": 0.0, "fallback": None, "used_offline_fallback": False})
        rl.append_log({"domain_guess": "Battery", "issue_guess": "Battery draining fast",
                        "cache_hit": False, "latency_ms": 1.0, "total_tokens": 0,
                        "cost_usd": 0.0, "fallback": None, "used_offline_fallback": False})
        stats = rl.compute_stats()
        assert stats["trending_issues"] == {"Battery draining fast": 2}
