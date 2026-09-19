"""main.py — the FastAPI service, exercised through TestClient (no real
network socket, no separate uvicorn process needed)."""
import pytest
from fastapi.testclient import TestClient

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


def test_feedback_records_and_rejects_placeholder(client):
    res = client.post("/v1/feedback", json={
        "deeplink": "bixby://masked/act/doesnotmatter",
        "action_name": "Wifi Settings",
        "helpful": True,
    })
    assert res.status_code == 200
    assert res.json()["status"] == "recorded"

    res2 = client.post("/v1/feedback", json={
        "deeplink": "bixby://dummy_positive",
        "action_name": "X",
        "helpful": False,
    })
    assert res2.status_code == 400


def test_stats_includes_feedback_section(client):
    res = client.get("/stats")
    assert res.status_code == 200
    body = res.json()
    assert "feedback" in body
    assert "total_feedback_events" in body["feedback"]
