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
from schema import TroubleshootRequest, FeedbackRequest

from pipeline import run_pipeline, run_pipeline_streaming, llm_available
from request_log import compute_stats
from deeplink_matching import get_index, DUMMY_POSITIVE_DEEPLINK
import feedback as feedback_module
from middleware import (
    RequestIDMiddleware, RateLimitMiddleware, error_body,
    troubleshoot_limiter, feedback_limiter,
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
    result = run_pipeline(payload.query, payload.siis_response or "")
    result.setdefault("meta", {})["request_id"] = request.state.request_id
    return result


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
    )
    return {"status": "recorded", "request_id": request.state.request_id, **result}


@app.get("/stats")
def stats():
    """Live aggregate metrics across every request this server has processed:
    avg/P95 latency, cache hit rate, no-match rate, total tokens/cost, and a
    breakdown of requests by domain. Real numbers for your metrics report,
    computed from actual usage rather than a handful of manual test runs."""
    return {**compute_stats(), "feedback": feedback_module.feedback_summary()}
