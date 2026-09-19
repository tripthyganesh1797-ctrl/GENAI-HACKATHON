"""main.py — the FastAPI service, exercised through TestClient (no real
network socket, no separate uvicorn process needed)."""
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

        def flaky(query, siis_response=""):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated per-item crash")
            return real_run_pipeline(query, siis_response)

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
