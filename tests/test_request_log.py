"""request_log.py — per-request logging + /stats aggregation."""
import request_log as rl


def test_empty_log_returns_zeroed_stats():
    stats = rl.compute_stats()
    assert stats["total_requests"] == 0
    assert stats["avg_latency_ms"] == 0
    assert stats["requests_by_domain"] == {}


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
