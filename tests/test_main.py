"""main.py — the FastAPI service, exercised through TestClient (no real
network socket, no separate uvicorn process needed)."""
import json

import pytest
from fastapi.testclient import TestClient

import middleware
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers the @app.on_event("startup") index warm-up
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "deeplink_variant" in body
    assert "llm_available" in body


def test_troubleshoot_post(client, real_samples):
    sample = next(s for s in real_samples if s.get("siis_response"))
    res = client.post("/v1/troubleshoot", json={
        "query": sample["complaint"], "siis_response": sample["siis_response"],
    })
    assert res.status_code == 200
    body = res.json()
    assert "response" in body and "meta" in body
    assert "X-Request-ID" in res.headers
    assert body["meta"]["request_id"] == res.headers["X-Request-ID"]


def test_request_id_is_echoed_back_if_client_supplies_one(client):
    res = client.get("/health", headers={"X-Request-ID": "my-custom-trace-id"})
    assert res.headers["X-Request-ID"] == "my-custom-trace-id"


def test_unknown_route_returns_structured_404(client):
    res = client.get("/v1/does-not-exist")
    assert res.status_code == 404
    body = res.json()
    assert body["error"]["code"] == "404"
    assert "request_id" in body["error"]


def test_validation_error_is_structured(client):
    res = client.post("/v1/troubleshoot", json={"siis_response": "missing the required query field"})
    assert res.status_code == 422
    body = res.json()
    assert body["error"]["code"] == "422"
    assert "details" in body["error"]


def test_troubleshoot_stream_emits_sse_events(real_samples):
    """StreamingResponse doesn't play well with the shared module-scoped
    client fixture's connection pooling in some httpx versions, so this
    uses its own short-lived client."""
    sample = next(s for s in real_samples if s.get("siis_response"))
    with TestClient(app) as c:
        with c.stream(
            "GET", "/v1/troubleshoot/stream",
            params={"query": sample["complaint"], "siis_response": sample["siis_response"]},
        ) as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            raw = "".join(res.iter_text())

    events = [line for line in raw.split("\n\n") if line.strip().startswith("data:")]
    assert len(events) >= 5
    assert events[0].startswith("data: ") and '"stage": "start"' in events[0]
    assert '"stage": "complete"' in events[-1]


class TestBatchTroubleshoot:
    def test_runs_multiple_items_and_reports_counts(self, client, real_samples):
        grounded = [s for s in real_samples if s.get("siis_response")][:2]
        res = client.post("/v1/troubleshoot/batch", json={
            "items": [{"query": s["complaint"], "siis_response": s["siis_response"]} for s in grounded],
        })
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == len(grounded)
        assert body["succeeded"] == len(grounded)
        assert len(body["results"]) == len(grounded)
        for r in body["results"]:
            assert r["ok"] is True
            assert "response" in r["result"] and "meta" in r["result"]
        assert "X-Request-ID" in res.headers
        assert body["request_id"] == res.headers["X-Request-ID"]

    def test_one_bad_item_does_not_fail_the_whole_batch(self, client, real_samples, monkeypatch):
        import main
        real_run_pipeline = main.run_pipeline
        calls = {"n": 0}

        def flaky(query, siis_response="", device=None, session_id=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated per-item crash")
            return real_run_pipeline(query, siis_response, device=device, session_id=session_id)

        monkeypatch.setattr(main, "run_pipeline", flaky)
        res = client.post("/v1/troubleshoot/batch", json={
            "items": [{"query": "a"}, {"query": "b"}, {"query": "c"}],
        })
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 3
        assert body["succeeded"] == 2
        assert body["results"][0]["ok"] is True
        assert body["results"][1]["ok"] is False
        assert "simulated per-item crash" in body["results"][1]["error"]
        assert body["results"][2]["ok"] is True

    def test_rejects_empty_or_oversized_batches(self, client):
        res_empty = client.post("/v1/troubleshoot/batch", json={"items": []})
        assert res_empty.status_code == 422

        res_too_big = client.post("/v1/troubleshoot/batch", json={
            "items": [{"query": f"q{i}"} for i in range(21)],
        })
        assert res_too_big.status_code == 422

    def test_batch_has_its_own_tighter_rate_limit(self, monkeypatch):
        monkeypatch.setattr(middleware.batch_limiter, "max_requests", 1)
        with TestClient(app) as c:
            first = c.post("/v1/troubleshoot/batch", json={"items": [{"query": "x"}]})
            assert first.status_code == 200
            second = c.post("/v1/troubleshoot/batch", json={"items": [{"query": "x"}]})
            assert second.status_code == 429
            # the single-item endpoint has its own separate, more generous limit
            unaffected = c.post("/v1/troubleshoot", json={"query": "x"})
            assert unaffected.status_code == 200


def test_feedback_records_and_rejects_placeholder(client):
    res = client.post("/v1/feedback", json={
        "deeplink": "bixby://masked/act/doesnotmatter",
        "action_name": "Wifi Settings",
        "helpful": True,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "recorded"
    assert body["request_id"] == res.headers["X-Request-ID"]

    res2 = client.post("/v1/feedback", json={
        "deeplink": "bixby://dummy_positive",
        "action_name": "X",
        "helpful": False,
    })
    assert res2.status_code == 400
    err = res2.json()["error"]
    assert err["code"] == "400"
    assert "placeholder" in err["message"]
    assert err["request_id"] == res2.headers["X-Request-ID"]


class TestRateLimiting:
    def test_returns_429_with_retry_after_once_limit_exceeded(self, monkeypatch):
        monkeypatch.setattr(middleware.troubleshoot_limiter, "max_requests", 2)
        with TestClient(app) as c:
            for _ in range(2):
                res = c.get("/v1/troubleshoot/stream", params={"query": "battery drains fast"})
                assert res.status_code == 200
            blocked = c.get("/v1/troubleshoot/stream", params={"query": "battery drains fast"})
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
        body = blocked.json()
        assert body["error"]["code"] == "rate_limited"
        assert "request_id" in body["error"]

    def test_health_is_exempt_from_rate_limiting(self, monkeypatch):
        monkeypatch.setattr(middleware.troubleshoot_limiter, "max_requests", 1)
        with TestClient(app) as c:
            c.get("/v1/troubleshoot/stream", params={"query": "x"})  # uses up the one allowed slot
            for _ in range(5):
                res = c.get("/health")
                assert res.status_code == 200

    def test_different_endpoints_have_independent_limits(self, monkeypatch):
        monkeypatch.setattr(middleware.troubleshoot_limiter, "max_requests", 1)
        with TestClient(app) as c:
            c.get("/v1/troubleshoot/stream", params={"query": "x"})
            blocked = c.get("/v1/troubleshoot/stream", params={"query": "x"})
            assert blocked.status_code == 429
            # /v1/feedback has its own separate limiter -- not affected
            res = c.post("/v1/feedback", json={
                "deeplink": "bixby://masked/act/independent-limit-check",
                "action_name": "X", "helpful": True,
            })
            assert res.status_code == 200


def test_unhandled_exception_becomes_structured_500(monkeypatch):
    import main
    def boom(*a, **k):
        raise RuntimeError("simulated pipeline crash")
    monkeypatch.setattr(main, "run_pipeline", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        res = c.post("/v1/troubleshoot", json={"query": "anything"})
    assert res.status_code == 500
    body = res.json()
    assert body["error"]["code"] == "internal_error"
    assert "simulated pipeline crash" in body["error"]["message"]
    assert "request_id" in body["error"]


def test_stats_includes_feedback_section(client):
    res = client.get("/stats")
    assert res.status_code == 200
    body = res.json()
    assert "feedback" in body
    assert "total_feedback_events" in body["feedback"]


class TestDeviceContext:
    """Task 35: POST /v1/troubleshoot(/batch) and GET /v1/troubleshoot/stream
    all accept optional device-state signals and thread them into
    run_pipeline()/run_pipeline_streaming(). The actual reordering/notes
    logic is covered by tests/test_device_signals.py and
    tests/test_pipeline.py -- these only check the wiring at the HTTP
    boundary: payload -> dict passed to the pipeline call."""

    def test_troubleshoot_without_device_passes_none(self, monkeypatch, client):
        import main
        captured = {}

        def fake_run_pipeline(query, siis_response="", device=None, session_id=None):
            captured["device"] = device
            return {"query": query, "query_variations": [], "response": {"contexts": []},
                    "meta": {"latency_ms": 1.0, "cache_hit": False, "model": "x",
                              "cost_usd": 0.0, "device_context_notes": []}}

        monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
        res = client.post("/v1/troubleshoot", json={"query": "anything"})
        assert res.status_code == 200
        assert captured["device"] is None

    def test_troubleshoot_with_device_is_forwarded_as_a_plain_dict(self, monkeypatch, client):
        import main
        captured = {}

        def fake_run_pipeline(query, siis_response="", device=None, session_id=None):
            captured["device"] = device
            return {"query": query, "query_variations": [], "response": {"contexts": []},
                    "meta": {"latency_ms": 1.0, "cache_hit": False, "model": "x",
                              "cost_usd": 0.0, "device_context_notes": []}}

        monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
        res = client.post("/v1/troubleshoot", json={
            "query": "anything", "device": {"battery_pct": 6, "storage_free_pct": 3},
        })
        assert res.status_code == 200
        assert captured["device"] == {"battery_pct": 6.0, "storage_free_pct": 3.0}

    def test_troubleshoot_end_to_end_surfaces_device_context_notes(self, client, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        res = client.post("/v1/troubleshoot", json={
            "query": sample["complaint"], "siis_response": sample["siis_response"],
            "device": {"battery_pct": 4},
        })
        assert res.status_code == 200
        assert len(res.json()["meta"]["device_context_notes"]) == 1

    def test_batch_forwards_per_item_device_context(self, monkeypatch, client):
        import main
        captured = []

        def fake_run_pipeline(query, siis_response="", device=None, session_id=None):
            captured.append(device)
            return {"query": query, "query_variations": [], "response": {"contexts": []},
                    "meta": {"latency_ms": 1.0, "cache_hit": False, "model": "x",
                              "cost_usd": 0.0, "device_context_notes": []}}

        monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
        res = client.post("/v1/troubleshoot/batch", json={"items": [
            {"query": "a", "device": {"battery_pct": 5}},
            {"query": "b"},
        ]})
        assert res.status_code == 200
        assert captured[0] == {"battery_pct": 5.0}
        assert captured[1] is None

    @staticmethod
    def _parse_sse_events(raw: str) -> list:
        lines = [line for line in raw.split("\n\n") if line.strip().startswith("data:")]
        return [json.loads(line[len("data: "):]) for line in lines]

    def test_stream_accepts_flattened_device_query_params(self, real_samples):
        """See test_troubleshoot_stream_emits_sse_events above for why this
        uses its own short-lived TestClient rather than the shared module-
        scoped `client` fixture."""
        sample = next(s for s in real_samples if s.get("siis_response"))
        with TestClient(app) as c:
            with c.stream(
                "GET", "/v1/troubleshoot/stream",
                params={"query": sample["complaint"], "siis_response": sample["siis_response"],
                        "battery_pct": 4},
            ) as res:
                raw = "".join(res.iter_text())
        events = self._parse_sse_events(raw)
        assert events[-1]["stage"] == "complete"
        assert len(events[-1]["data"]["meta"]["device_context_notes"]) == 1

    def test_stream_with_no_device_params_has_empty_notes(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        with TestClient(app) as c:
            with c.stream(
                "GET", "/v1/troubleshoot/stream",
                params={"query": sample["complaint"], "siis_response": sample["siis_response"]},
            ) as res:
                raw = "".join(res.iter_text())
        events = self._parse_sse_events(raw)
        assert events[-1]["data"]["meta"]["device_context_notes"] == []


class TestSessionMemory:
    """Task 36: session_id wiring at the HTTP boundary -- POST
    /v1/troubleshoot(/batch), GET /v1/troubleshoot/stream, and POST
    /v1/feedback all thread it through to run_pipeline()/record_feedback().
    The actual avoidance/cache-bypass mechanics are covered end-to-end in
    tests/test_pipeline.py::TestSessionMemory; these only pin down that
    each route passes the field along correctly."""

    def test_troubleshoot_without_session_id_passes_none(self, monkeypatch, client):
        import main
        captured = {}

        def fake_run_pipeline(query, siis_response="", device=None, session_id=None):
            captured["session_id"] = session_id
            return {"query": query, "query_variations": [], "response": {"contexts": []},
                    "meta": {"latency_ms": 1.0, "cache_hit": False, "model": "x",
                              "cost_usd": 0.0, "device_context_notes": [], "session_notes": []}}

        monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
        res = client.post("/v1/troubleshoot", json={"query": "anything"})
        assert res.status_code == 200
        assert captured["session_id"] is None

    def test_troubleshoot_forwards_session_id(self, monkeypatch, client):
        import main
        captured = {}

        def fake_run_pipeline(query, siis_response="", device=None, session_id=None):
            captured["session_id"] = session_id
            return {"query": query, "query_variations": [], "response": {"contexts": []},
                    "meta": {"latency_ms": 1.0, "cache_hit": False, "model": "x",
                              "cost_usd": 0.0, "device_context_notes": [], "session_notes": []}}

        monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
        res = client.post("/v1/troubleshoot", json={"query": "anything", "session_id": "sess-xyz"})
        assert res.status_code == 200
        assert captured["session_id"] == "sess-xyz"

    def test_batch_forwards_per_item_session_id(self, monkeypatch, client):
        import main
        captured = []

        def fake_run_pipeline(query, siis_response="", device=None, session_id=None):
            captured.append(session_id)
            return {"query": query, "query_variations": [], "response": {"contexts": []},
                    "meta": {"latency_ms": 1.0, "cache_hit": False, "model": "x",
                              "cost_usd": 0.0, "device_context_notes": [], "session_notes": []}}

        monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
        res = client.post("/v1/troubleshoot/batch", json={"items": [
            {"query": "a", "session_id": "sess-a"},
            {"query": "b"},
        ]})
        assert res.status_code == 200
        assert captured[0] == "sess-a"
        assert captured[1] is None

    def test_stream_forwards_session_id_query_param(self, monkeypatch):
        import main
        captured = {}

        def fake_run_pipeline_streaming(query, siis_response="", device=None, session_id=None):
            captured["session_id"] = session_id
            yield {"stage": "complete", "status": "done", "data": {
                "query": query, "query_variations": [], "response": {"contexts": []},
                "meta": {"latency_ms": 1.0, "cache_hit": False, "model": "x", "cost_usd": 0.0,
                          "device_context_notes": [], "session_notes": []},
            }}

        monkeypatch.setattr(main, "run_pipeline_streaming", fake_run_pipeline_streaming)
        with TestClient(app) as c:
            with c.stream(
                "GET", "/v1/troubleshoot/stream",
                params={"query": "anything", "session_id": "sess-stream"},
            ) as res:
                "".join(res.iter_text())
        assert captured["session_id"] == "sess-stream"

    def test_feedback_forwards_session_id_to_record_feedback(self, monkeypatch, client):
        import main
        captured = {}

        def fake_record_feedback(deeplink, action_name, helpful, query="", comment="", session_id=""):
            captured["session_id"] = session_id
            return {"helpful": 0, "unhelpful": 1, "action_names": [], "adjustment": 0.0}

        monkeypatch.setattr(main.feedback_module, "record_feedback", fake_record_feedback)
        res = client.post("/v1/feedback", json={
            "deeplink": "bixby://masked/act/sess-fb", "action_name": "X",
            "helpful": False, "session_id": "sess-feedback-1",
        })
        assert res.status_code == 200
        assert captured["session_id"] == "sess-feedback-1"

    def test_feedback_without_session_id_passes_empty_string(self, monkeypatch, client):
        import main
        captured = {}

        def fake_record_feedback(deeplink, action_name, helpful, query="", comment="", session_id=""):
            captured["session_id"] = session_id
            return {"helpful": 0, "unhelpful": 1, "action_names": [], "adjustment": 0.0}

        monkeypatch.setattr(main.feedback_module, "record_feedback", fake_record_feedback)
        res = client.post("/v1/feedback", json={
            "deeplink": "bixby://masked/act/sess-fb2", "action_name": "X", "helpful": False,
        })
        assert res.status_code == 200
        assert captured["session_id"] == ""

    def test_troubleshoot_end_to_end_avoids_previously_unhelpful_action(self, client, real_samples):
        """Full stack, no monkeypatching: feedback -> session_memory ->
        pipeline -> deeplink_matching, all through real HTTP calls."""
        sample = next(s for s in real_samples if s.get("siis_response"))
        session_id = "e2e-session-1"

        first = client.post("/v1/troubleshoot", json={
            "query": sample["complaint"], "siis_response": sample["siis_response"],
        })
        deeplink = first.json()["response"]["contexts"][0]["actions"][0]["stepGroups"][0]["actionableDeeplink"]["deeplink"]

        fb = client.post("/v1/feedback", json={
            "deeplink": deeplink, "action_name": "X", "helpful": False, "session_id": session_id,
        })
        assert fb.status_code == 200

        second = client.post("/v1/troubleshoot", json={
            "query": sample["complaint"], "siis_response": sample["siis_response"],
            "session_id": session_id,
        })
        body = second.json()
        assert body["meta"]["cache_hit"] is False
        assert len(body["meta"]["session_notes"]) >= 1


class TestReportEndpoint:
    """Task 38: POST /v1/report packages an already-computed troubleshoot
    result into a shareable Markdown/HTML report -- see report.py."""

    def test_markdown_report_from_a_real_troubleshoot_result(self, client, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        first = client.post("/v1/troubleshoot", json={
            "query": sample["complaint"], "siis_response": sample["siis_response"],
        })
        assert first.status_code == 200

        res = client.post("/v1/report", json={"result": first.json(), "format": "markdown"})
        assert res.status_code == 200
        body = res.json()
        assert body["format"] == "markdown"
        assert body["content"].startswith("# Diagnostic Report")
        assert sample["complaint"] in body["content"]
        assert "request_id" in body
        assert body["request_id"] == res.headers["X-Request-ID"]

    def test_html_report_from_a_real_troubleshoot_result(self, client, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        first = client.post("/v1/troubleshoot", json={
            "query": sample["complaint"], "siis_response": sample["siis_response"],
        })
        res = client.post("/v1/report", json={"result": first.json(), "format": "html"})
        assert res.status_code == 200
        body = res.json()
        assert body["format"] == "html"
        assert body["content"].startswith("<!DOCTYPE html>")

    def test_format_defaults_to_markdown_when_omitted(self, client, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        first = client.post("/v1/troubleshoot", json={
            "query": sample["complaint"], "siis_response": sample["siis_response"],
        })
        res = client.post("/v1/report", json={"result": first.json()})
        assert res.status_code == 200
        assert res.json()["format"] == "markdown"

    def test_unsupported_format_is_a_structured_400(self, client, real_samples):
        sample = real_samples[0]
        first = client.post("/v1/troubleshoot", json={"query": sample["complaint"]})
        res = client.post("/v1/report", json={"result": first.json(), "format": "pdf"})
        assert res.status_code == 400
        body = res.json()
        assert "request_id" in body["error"]

    def test_no_match_result_still_produces_a_report(self, client, real_samples):
        sample = real_samples[0]
        first = client.post("/v1/troubleshoot", json={"query": sample["complaint"]})  # no siis_response
        assert first.json()["response"]["contexts"] == []
        res = client.post("/v1/report", json={"result": first.json()})
        assert res.status_code == 200
        assert "No matching troubleshooting plan" in res.json()["content"]

    def test_missing_result_field_is_a_validation_error(self, client):
        res = client.post("/v1/report", json={"format": "markdown"})
        assert res.status_code == 422

    def test_report_generation_does_not_write_a_request_log_entry(self, client, real_samples):
        """report.py is a pure formatter -- calling /v1/report must not
        itself append to request_log.jsonl (only real /v1/troubleshoot
        calls should count toward /stats)."""
        import request_log
        sample = real_samples[0]
        first = client.post("/v1/troubleshoot", json={"query": sample["complaint"]})
        count_before = len(request_log.read_all_logs())
        client.post("/v1/report", json={"result": first.json()})
        count_after = len(request_log.read_all_logs())
        assert count_after == count_before
