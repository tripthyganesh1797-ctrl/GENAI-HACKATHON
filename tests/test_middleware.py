"""middleware.py — the rate limiter's own logic in isolation (fast, no
HTTP layer needed), plus its wiring through main.py in test_main.py."""
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from middleware import RateLimiter, RateLimitMiddleware


def test_allows_up_to_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60.0)
    for _ in range(3):
        allowed, _ = limiter.check("client-a")
        assert allowed


def test_rejects_over_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60.0)
    for _ in range(3):
        limiter.check("client-a")
    allowed, retry_after = limiter.check("client-a")
    assert not allowed
    assert retry_after > 0


def test_keys_are_independent():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    allowed_a, _ = limiter.check("client-a")
    allowed_b, _ = limiter.check("client-b")
    assert allowed_a and allowed_b


def test_window_expiry_allows_requests_again():
    limiter = RateLimiter(max_requests=1, window_seconds=0.05)
    allowed1, _ = limiter.check("client-a")
    assert allowed1
    allowed2, _ = limiter.check("client-a")
    assert not allowed2
    time.sleep(0.08)
    allowed3, _ = limiter.check("client-a")
    assert allowed3


def test_reset_clears_all_state():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    limiter.check("client-a")
    allowed_before_reset, _ = limiter.check("client-a")
    assert not allowed_before_reset
    limiter.reset()
    allowed_after_reset, _ = limiter.check("client-a")
    assert allowed_after_reset


def test_longest_matching_prefix_wins_over_a_broader_one():
    """Regression test for the Task 30 fix: a route registered under a more
    specific prefix (like "/v1/troubleshoot/batch") must use its own
    limiter even though a broader prefix ("/v1/troubleshoot") also matches
    and happens to be inserted into the dict first."""
    broad = RateLimiter(max_requests=100, window_seconds=60.0)
    specific = RateLimiter(max_requests=1, window_seconds=60.0)

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limiter_by_prefix={
        "/v1/troubleshoot": broad,          # inserted first, and a prefix of the path below
        "/v1/troubleshoot/batch": specific,  # more specific; must win
    })

    @app.get("/v1/troubleshoot/batch")
    def batch_route():
        return {"ok": True}

    with TestClient(app) as c:
        first = c.get("/v1/troubleshoot/batch")
        assert first.status_code == 200
        second = c.get("/v1/troubleshoot/batch")
        assert second.status_code == 429  # the tight "specific" limiter, not the generous "broad" one
        assert broad._hits == {}  # broad limiter was never touched
