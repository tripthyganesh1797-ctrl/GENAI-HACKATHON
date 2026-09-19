"""
main.py — FastAPI service. Run with:
    uvicorn main:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/v1/troubleshoot \\
      -H "Content-Type: application/json" \\
      -d '{"query": "screen flickers and battery dies fast"}'
"""

import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from schema import TroubleshootRequest

from pipeline import run_pipeline, run_pipeline_streaming, llm_available
from request_log import compute_stats
from deeplink_matching import get_index

app = FastAPI(title="Smart Guided Troubleshooting Engine", version="0.1.0")

_startup_info = {}


@app.on_event("startup")
def _warm_up():
    """Builds the BM25/embedding index once at process startup instead of
    on a user's first request -- the index build (and, if
    sentence-transformers + network are available, the embedding model
    load) is the one genuinely slow step in this whole service, and it
    should never be on the hot path of a real request."""
    index = get_index("hybrid")
    _startup_info["deeplink_variant"] = type(index).__name__
    _startup_info["dense_kind"] = getattr(index, "dense_kind", "n/a (rules-based)")
    _startup_info["llm_available"] = llm_available()

# Allow the demo webpage (opened as a local file or on a different port) to
# call this API. Fine for a hackathon demo; would be scoped down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", **_startup_info}


@app.post("/v1/troubleshoot")
def troubleshoot(request: TroubleshootRequest):
    try:
        result = run_pipeline(request.query, request.siis_response or "")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/troubleshoot/stream")
def troubleshoot_stream(
    query: str = Query(..., description="Raw user complaint"),
    siis_response: str = Query("", description="Optional reference troubleshooting text"),
):
    """Server-Sent Events version of /v1/troubleshoot -- same pipeline, same
    final payload, but emits one event per stage (enrich -> cache check ->
    extract -> validate -> deeplink match -> complete) so a UI can show the
    engine actually working instead of a blank loading spinner. Uses GET +
    query params (not POST) because the browser EventSource API only
    supports GET.

    Try it:
        curl -N "http://localhost:8000/v1/troubleshoot/stream?query=battery+drains+fast"
    """

    def event_source():
        for event in run_pipeline_streaming(query, siis_response):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering so events stream live
        },
    )


@app.get("/stats")
def stats():
    """Live aggregate metrics across every request this server has processed:
    avg/P95 latency, cache hit rate, no-match rate, total tokens/cost, and a
    breakdown of requests by domain. Real numbers for your metrics report,
    computed from actual usage rather than a handful of manual test runs."""
    return compute_stats()
