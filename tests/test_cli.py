"""cli.py — the terminal client. Exercises both the in-process pipeline path
(no network) and the --api-base HTTP path (urllib.request.urlopen mocked,
so this suite never needs a real running server)."""
import io
import json
import urllib.error

import pytest

import cli


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_load_batch_items_accepts_sample_queries_real_shape(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([
        {"complaint": "battery drains fast", "siis_response": "drain fix text"},
        {"complaint": "no siis here"},
    ]))
    items = cli._load_batch_items(str(path))
    assert items == [
        {"query": "battery drains fast", "siis_response": "drain fix text"},
        {"query": "no siis here", "siis_response": ""},
    ]


def test_load_batch_items_accepts_items_wrapper_shape(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"items": [{"query": "screen flickers"}]}))
    items = cli._load_batch_items(str(path))
    assert items == [{"query": "screen flickers", "siis_response": ""}]


def test_load_batch_items_skips_entries_with_no_query(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([{"siis_response": "orphaned, no query/complaint key"}]))
    assert cli._load_batch_items(str(path)) == []


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def test_parser_requires_a_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_routes_each_subcommand_to_its_function():
    parser = cli.build_parser()
    assert parser.parse_args(["query", "x"]).func is cli.cmd_query
    assert parser.parse_args(["stream", "x"]).func is cli.cmd_stream
    assert parser.parse_args(["batch", "f.json"]).func is cli.cmd_batch
    assert parser.parse_args(["health"]).func is cli.cmd_health
    assert parser.parse_args(["stats"]).func is cli.cmd_stats


def test_api_base_is_a_top_level_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://localhost:9000", "health"])
    assert args.api_base == "http://localhost:9000"


# ---------------------------------------------------------------------------
# In-process query/batch (real pipeline, real official data -- no network)
# ---------------------------------------------------------------------------

def test_cmd_query_in_process_prints_formatted_result(capsys, real_samples):
    sample = next(s for s in real_samples if s.get("siis_response"))
    parser = cli.build_parser()
    args = parser.parse_args(["query", sample["complaint"], "--siis-response", sample["siis_response"]])
    cli.cmd_query(args)
    out = capsys.readouterr().out
    assert "latency" in out
    assert sample["complaint"] in out


def test_cmd_query_reads_siis_response_from_file(capsys, real_samples, tmp_path):
    sample = next(s for s in real_samples if s.get("siis_response"))
    siis_file = tmp_path / "siis.txt"
    siis_file.write_text(sample["siis_response"])
    parser = cli.build_parser()
    args = parser.parse_args(["query", sample["complaint"], "--siis-file", str(siis_file)])
    cli.cmd_query(args)
    out = capsys.readouterr().out
    # grounded via the file, not left as the ungrounded no-hallucination path
    assert "no match" not in out


def test_cmd_query_in_process_json_flag_prints_valid_json(capsys, real_samples):
    sample = next(s for s in real_samples if s.get("siis_response"))
    parser = cli.build_parser()
    args = parser.parse_args(["query", sample["complaint"], "--siis-response", sample["siis_response"], "--json"])
    cli.cmd_query(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "response" in data and "meta" in data


def test_cmd_query_verbose_shows_match_explanation(capsys, real_samples):
    sample = next(s for s in real_samples if s.get("siis_response"))
    parser = cli.build_parser()
    args = parser.parse_args(["query", sample["complaint"], "--siis-response", sample["siis_response"], "--verbose"])
    cli.cmd_query(args)
    out = capsys.readouterr().out
    assert "why:" in out


def test_cmd_query_no_match_prints_clear_message(capsys):
    parser = cli.build_parser()
    # No siis_response at all -- offline path's no-hallucination guarantee
    # means this always yields no_match, regardless of complaint text.
    args = parser.parse_args(["query", "some complaint with nothing to ground it"])
    cli.cmd_query(args)
    out = capsys.readouterr().out
    assert "no match" in out


def test_cmd_batch_in_process_writes_results_and_prints_summary(capsys, real_samples, tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(real_samples[:3]))
    out_path = tmp_path / "out.json"

    parser = cli.build_parser()
    args = parser.parse_args(["batch", str(src), "--out", str(out_path)])
    cli.cmd_batch(args)

    printed = capsys.readouterr().out
    assert "succeeded (in-process)" in printed
    assert out_path.exists()
    results = json.loads(out_path.read_text())
    assert len(results) == 3
    assert all("ok" in r for r in results)


def test_cmd_batch_reports_empty_file_and_exits(tmp_path, capsys):
    src = tmp_path / "empty.json"
    src.write_text(json.dumps([]))
    parser = cli.build_parser()
    args = parser.parse_args(["batch", str(src)])
    with pytest.raises(SystemExit):
        cli.cmd_batch(args)


# ---------------------------------------------------------------------------
# --api-base HTTP path (urllib.request.urlopen mocked -- no real server)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._body = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_post_json_success(monkeypatch):
    def fake_urlopen(req, timeout=None):
        assert req.get_method() == "POST"
        return _FakeResponse({"hello": "world"}, status=200)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    status, data = cli._http_post_json("http://fake/x", {"a": 1})
    assert status == 200
    assert data == {"hello": "world"}


def test_http_post_json_http_error_returns_parsed_body(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://fake/x", 429,
            "Too Many Requests", hdrs=None,
            fp=io.BytesIO(json.dumps({"error": {"code": "rate_limited"}}).encode()),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    status, data = cli._http_post_json("http://fake/x", {"a": 1})
    assert status == 429
    assert data["error"]["code"] == "rate_limited"


def test_cmd_health_via_api_base(monkeypatch, capsys):
    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"status": "ok"}, status=200)

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://fake", "health"])
    cli.cmd_health(args)
    out = capsys.readouterr().out
    assert json.loads(out) == {"status": "ok"}


def test_cmd_query_via_api_base_sends_post_and_prints_result(monkeypatch, capsys):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _FakeResponse({
            "query": "x", "response": {"contexts": []},
            "meta": {"latency_ms": 1.0, "model": "offline-rule-based",
                      "cache_hit": False, "cost_usd": 0.0, "total_tokens": 0},
        })

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://fake", "query", "battery drains fast"])
    cli.cmd_query(args)

    assert captured["url"] == "http://fake/v1/troubleshoot"
    assert captured["body"]["query"] == "battery drains fast"
    out = capsys.readouterr().out
    assert "no match" in out


def test_cmd_query_via_api_base_error_status_exits(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://fake/v1/troubleshoot", 500, "err", hdrs=None,
            fp=io.BytesIO(json.dumps({"error": {"message": "boom"}}).encode()),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://fake", "query", "x"])
    with pytest.raises(SystemExit):
        cli.cmd_query(args)


def test_cmd_batch_via_api_base_caps_at_20_items(monkeypatch, tmp_path, capsys):
    src = tmp_path / "big.json"
    src.write_text(json.dumps([{"query": f"q{i}"} for i in range(25)]))
    captured = {}

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        captured["items"] = body["items"]
        results = [{"ok": True, "result": {"response": {"contexts": []}}} for _ in body["items"]]
        return _FakeResponse({
            "request_id": "rid", "count": len(results),
            "succeeded": len(results), "results": results,
        })

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://fake", "batch", str(src)])
    cli.cmd_batch(args)

    assert len(captured["items"]) == 20
    out = capsys.readouterr().out
    assert "exceeds the batch endpoint's 20-item cap" in out
    assert "20/20 succeeded" in out


def test_stream_refuses_api_base(monkeypatch):
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://fake", "stream", "x"])
    with pytest.raises(SystemExit):
        cli.cmd_stream(args)


def test_http_post_json_url_error_exits(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit):
        cli._http_post_json("http://unreachable", {"a": 1})


def test_http_get_json_url_error_exits(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit):
        cli._http_get_json("http://unreachable")


# ---------------------------------------------------------------------------
# _print_match_explanation -- both matcher shapes plus the rejected/unknown
# fallback branch
# ---------------------------------------------------------------------------

def test_print_match_explanation_hybrid_shape(capsys):
    cli._print_match_explanation({
        "matcher": "hybrid_bm25_dense", "bm25_component": 1.0, "dense_component": 0.5,
        "feedback_adjustment": 0.0, "final_score": 0.75, "matched_keywords": ["wifi"],
    })
    out = capsys.readouterr().out
    assert "bm25=1.0" in out and "keywords: wifi" in out


def test_print_match_explanation_rules_shape(capsys):
    cli._print_match_explanation({
        "matcher": "rules_fuzzy", "fuzzy_score": 80.0, "feedback_adjustment_pts": 0.0,
        "final_score_pts": 80.0, "matched_keywords": [],
    })
    out = capsys.readouterr().out
    assert "fuzzy=80.0" in out


def test_print_match_explanation_unknown_matcher_falls_back_to_reason(capsys):
    # No "matcher" key at all (or an unrecognized one) -- the empty_index /
    # no_entry_scored_above_zero rejection shape from deeplink_matching.py.
    cli._print_match_explanation({"reason": "empty_index"})
    out = capsys.readouterr().out
    assert "empty_index" in out


def test_print_match_explanation_none_is_a_noop(capsys):
    cli._print_match_explanation(None)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# cmd_stream in-process (no --api-base)
# ---------------------------------------------------------------------------

def test_cmd_stream_in_process_prints_every_stage(capsys, real_samples):
    sample = next(s for s in real_samples if s.get("siis_response"))
    parser = cli.build_parser()
    args = parser.parse_args(["stream", sample["complaint"], "--siis-response", sample["siis_response"], "--verbose"])
    cli.cmd_stream(args)
    out = capsys.readouterr().out
    assert "streaming:" in out
    assert "enrich" in out and "extract" in out and "deeplink_match" in out
    assert "complete" in out


# ---------------------------------------------------------------------------
# Remaining API-mode branches: batch error, health error, stats
# ---------------------------------------------------------------------------

def test_cmd_batch_via_api_base_error_status_exits(monkeypatch, tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"query": "x"}]))

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://fake/v1/troubleshoot/batch", 429, "err", hdrs=None,
            fp=io.BytesIO(json.dumps({"error": {"code": "rate_limited"}}).encode()),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://fake", "batch", str(src)])
    with pytest.raises(SystemExit):
        cli.cmd_batch(args)


def test_cmd_batch_in_process_reports_a_failing_item_without_aborting(monkeypatch, tmp_path, capsys):
    import pipeline as pipeline_module
    real_run_pipeline = pipeline_module.run_pipeline

    def flaky(query, siis_response=""):
        if query == "boom":
            raise RuntimeError("simulated crash")
        return real_run_pipeline(query, siis_response)

    monkeypatch.setattr(pipeline_module, "run_pipeline", flaky)

    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"query": "boom"}, {"query": "ok query"}]))
    out_path = tmp_path / "out.json"
    parser = cli.build_parser()
    args = parser.parse_args(["batch", str(src), "--out", str(out_path)])
    cli.cmd_batch(args)

    out = capsys.readouterr().out
    assert "error: simulated crash" in out
    results = json.loads(out_path.read_text())
    assert results[0]["ok"] is False
    assert results[1]["ok"] is True


def test_cmd_health_error_status_exits(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise urllib.error.HTTPError(
            "http://fake/health", 503, "err", hdrs=None,
            fp=io.BytesIO(json.dumps({"error": {"message": "down"}}).encode()),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://fake", "health"])
    with pytest.raises(SystemExit):
        cli.cmd_health(args)


def test_cmd_stats_success_and_error(monkeypatch, capsys):
    def fake_ok(url, timeout=None):
        return _FakeResponse({"total_requests": 5})

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_ok)
    parser = cli.build_parser()
    args = parser.parse_args(["--api-base", "http://fake", "stats"])
    cli.cmd_stats(args)
    assert json.loads(capsys.readouterr().out) == {"total_requests": 5}

    def fake_error(url, timeout=None):
        raise urllib.error.HTTPError(
            "http://fake/stats", 500, "err", hdrs=None,
            fp=io.BytesIO(json.dumps({"error": {"message": "boom"}}).encode()),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_error)
    with pytest.raises(SystemExit):
        cli.cmd_stats(args)


# ---------------------------------------------------------------------------
# main() entrypoint dispatch
# ---------------------------------------------------------------------------

def test_main_dispatches_to_cmd_query(capsys):
    cli.main(["query", "some complaint with nothing to ground it"])
    out = capsys.readouterr().out
    assert "no match" in out


def test_http_post_json_non_json_error_body_falls_back_to_raw_text(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "http://fake/x", 502, "Bad Gateway", hdrs=None,
            fp=io.BytesIO(b"<html>502 Bad Gateway</html>"),
        )

    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    status, data = cli._http_post_json("http://fake/x", {"a": 1})
    assert status == 502
    assert "502 Bad Gateway" in data["error"]["message"]


def test_cmd_stream_in_process_prints_error_stage(monkeypatch, capsys):
    import pipeline as pipeline_module

    def boom(raw_complaint):
        raise RuntimeError("simulated stage0 crash")

    monkeypatch.setattr(pipeline_module, "stage0_enrich", boom)
    parser = cli.build_parser()
    args = parser.parse_args(["stream", "anything"])
    cli.cmd_stream(args)
    out = capsys.readouterr().out
    assert "error: simulated stage0 crash" in out
