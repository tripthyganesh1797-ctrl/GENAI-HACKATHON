"""
main.py — FastAPI service. Run with:
    uvicorn main:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/v1/troubleshoot \\
      -H "Content-Type: application/json" \\
      -d '{"query": "screen flickers and battery dies fast"}'
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from schema import TroubleshootRequest, FeedbackRequest, BatchTroubleshootRequest, ReportRequest

from pipeline import run_pipeline, run_pipeline_streaming, llm_available
from request_log import compute_stats
from deeplink_matching import get_index, DUMMY_POSITIVE_DEEPLINK
import feedback as feedback_module
from report import generate_report, SUPPORTED_FORMATS
from middleware import (
    RequestIDMiddleware, RateLimitMiddleware, error_body,
    troubleshoot_limiter, feedback_limiter, batch_limiter,
)

_startup_info = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Builds the BM25/embedding index once at process startup instead of
    on a user's first request -- the index build (and, if
    sentence-transformers + network are available, the embedding model
    load) is the one genuinely slow step in this whole service, and it
    should never be on the hot path of a real request."""
    index = get_index("hybrid")
    _startup_info["deeplink_variant"] = type(index).__name__
    _startup_info["dense_kind"] = getattr(index, "dense_kind", "n/a (rules-based)")
    _startup_info["llm_available"] = llm_available()
    yield


app = FastAPI(title="Smart Guided Troubleshooting Engine", version="0.1.0", lifespan=lifespan)

# Middleware order matters: Starlette runs the LAST-added middleware
# OUTERMOST, so a request passes CORS -> request-ID -> rate-limit -> route.
# That means a CORS preflight (OPTIONS) short-circuits before it can ever
# be counted against a client's rate limit, and every response -- success,
# 4xx, 429, or 500 -- already has an X-Request-ID by the time rate limiting
# or the route handler runs.
app.add_middleware(
    RateLimitMiddleware,
    limiter_by_prefix={
        "/v1/troubleshoot": troubleshoot_limiter,  # covers both POST and the /stream GET
        "/v1/troubleshoot/batch": batch_limiter,   # longest-prefix-wins over the line above
        "/v1/feedback": feedback_limiter,
    },
)
app.add_middleware(RequestIDMiddleware)
# Allow the demo webpage (opened as a local file or on a different port) to
# call this API. Fine for a hackathon demo; would be scoped down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Every error the service returns -- ours (raised as fastapi.HTTPException,
    a subclass of Starlette's) or Starlette's own routing-level errors (404s,
    405 method-not-allowed, etc., raised as the base class directly) -- comes
    back in the same {"error": {...}} shape with a request_id, so a client
    (or a judge poking at the API) never has to handle two different error
    formats. Registered on the base class specifically because FastAPI's
    HTTPException is a *subclass*: a handler keyed to the subclass would
    never catch the base-class 404s Starlette's router raises directly."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code=str(exc.status_code), message=str(exc.detail), request_id=request_id),
        headers=dict(exc.headers or {}),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", None)
    body = error_body(code="422", message="Request validation failed", request_id=request_id)
    body["error"]["details"] = exc.errors()
    return JSONResponse(status_code=422, content=body)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last resort: an uncaught exception anywhere in a route handler
    becomes a structured 500 instead of a bare traceback or FastAPI's
    default plaintext response -- the request_id here is what a judge (or
    a teammate) would quote back to us to find the matching stack trace
    in server logs."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content=error_body(code="internal_error", message=str(exc), request_id=request_id),
    )


@app.get("/health")
def health():
    return {"status": "ok", **_startup_info}


@app.post("/v1/troubleshoot")
def troubleshoot(payload: TroubleshootRequest, request: Request):
    # No try/except here -- an uncaught exception falls through to the
    # unhandled_exception_handler above, which returns the same structured
    # {"error": {...}} shape every other failure mode in this service uses.
    device = payload.device.model_dump(exclude_none=True) if payload.device else None
    result = run_pipeline(payload.query, payload.siis_response or "", device=device,
                           session_id=payload.session_id)
    result.setdefault("meta", {})["request_id"] = request.state.request_id
    return result


@app.post("/v1/troubleshoot/batch")
def troubleshoot_batch(payload: BatchTroubleshootRequest, request: Request):
    """Runs up to 20 complaints through the same run_pipeline() a single
    /v1/troubleshoot call uses, in one HTTP round trip -- e.g. a device
    health-check screen probing several known symptoms at once. One bad
    item (a pipeline exception on that item specifically) is reported
    inline as {"ok": false, "error": ...} at its own index rather than
    failing the whole batch; every other item's real result still comes
    back. Route registered *before* the bare "/v1/troubleshoot" POST route
    in this file doesn't matter for FastAPI's routing (it matches on the
    full path, not prefix-order) -- only middleware.py's rate limiting
    needed the explicit longest-prefix-wins fix for that ordering concern.
    """
    results = []
    for item in payload.items:
        try:
            device = item.device.model_dump(exclude_none=True) if item.device else None
            result = run_pipeline(item.query, item.siis_response or "", device=device,
                                   session_id=item.session_id)
            results.append({"ok": True, "result": result})
        except Exception as e:
            results.append({"ok": False, "error": str(e), "query": item.query})
    return {
        "request_id": request.state.request_id,
        "count": len(results),
        "succeeded": sum(1 for r in results if r["ok"]),
        "results": results,
    }


@app.get("/v1/troubleshoot/stream")
def troubleshoot_stream(
    query: str = Query(..., description="Raw user complaint"),
    siis_response: str = Query("", description="Optional reference troubleshooting text"),
    battery_pct: float = Query(None, ge=0, le=100, description="Task 35: optional device context"),
    storage_free_pct: float = Query(None, ge=0, le=100),
    os_version: str = Query(None),
    uptime_hours: float = Query(None, ge=0),
    last_restart_hours_ago: float = Query(None, ge=0),
    session_id: str = Query(None, description="Task 36: optional session-scoped avoidance"),
):
    """Server-Sent Events version of /v1/troubleshoot -- same pipeline, same
    final payload, but emits one event per stage (enrich -> cache check ->
    extract -> validate -> deeplink match -> complete) so a UI can show the
    engine actually working instead of a blank loading spinner. Uses GET +
    query params (not POST) because the browser EventSource API only
    supports GET -- which is also why device context (normally one nested
    JSON object, schema.DeviceContext) is flattened into individual query
    params here instead.

    Try it:
        curl -N "http://localhost:8000/v1/troubleshoot/stream?query=battery+drains+fast"
        curl -N "http://localhost:8000/v1/troubleshoot/stream?query=battery+drains+fast&battery_pct=6"
    """
    device_raw = {
        "battery_pct": battery_pct,
        "storage_free_pct": storage_free_pct,
        "os_version": os_version,
        "uptime_hours": uptime_hours,
        "last_restart_hours_ago": last_restart_hours_ago,
    }
    device = {k: v for k, v in device_raw.items() if v is not None} or None

    def event_source():
        for event in run_pipeline_streaming(query, siis_response, device=device, session_id=session_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering so events stream live
        },
    )


@app.post("/v1/feedback")
def submit_feedback(payload: FeedbackRequest, request: Request):
    """Human-in-the-loop signal on a specific deeplink match. Every event is
    logged (feedback_log.jsonl) and folded into a running per-deeplink
    aggregate that deeplink_matching.py consults on every future search --
    so the very next /v1/troubleshoot call for a similar query can already
    reflect it. See feedback.py for the bounded, explainable adjustment
    formula (a single click can't flip a match; a consistent pattern can)."""
    if payload.deeplink == DUMMY_POSITIVE_DEEPLINK:
        raise HTTPException(
            status_code=400,
            detail="Feedback on the placeholder deeplink isn't meaningful -- "
                   "that action had no real catalog match to begin with.",
        )
    result = feedback_module.record_feedback(
        deeplink=payload.deeplink,
        action_name=payload.action_name,
        helpful=payload.helpful,
        query=payload.query or "",
        comment=payload.comment or "",
        session_id=payload.session_id or "",
    )
    return {"status": "recorded", "request_id": request.state.request_id, **result}


@app.post("/v1/report")
def report(payload: ReportRequest, request: Request):
    """Task 38: packages an already-computed /v1/troubleshoot(/stream)
    result into a compact, shareable report -- Markdown (default) or a
    self-contained HTML page -- so a user can paste a fix into a support
    ticket or forward it to someone without re-explaining the diagnosis.
    See report.py for the rendering logic; this route never re-runs the
    pipeline, so it's cheap and side-effect-free (no cache write, no
    request-log entry)."""
    if payload.format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format {payload.format!r} -- expected one of {SUPPORTED_FORMATS}",
        )
    content = generate_report(payload.result, fmt=payload.format)
    return {
        "request_id": request.state.request_id,
        "format": payload.format,
        "content": content,
    }


@app.get("/stats")
def stats():
    """Live aggregate metrics across every request this server has processed:
    avg/P95 latency, cache hit rate, no-match rate, total tokens/cost, a
    breakdown of requests by domain, and (Task 37) a "trending issues" top-N
    breakdown of the actual recurring symptoms (e.g. "Battery draining
    fast") rather than just the coarse domain bucket. Real numbers for your
    metrics report, computed from actual usage rather than a handful of
    manual test runs."""
    return {**compute_stats(), "feedback": feedback_module.feedback_summary()}
