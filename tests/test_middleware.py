"""middleware.py — the rate limiter's own logic in isolation (fast, no
HTTP layer needed), plus its wiring through main.py in test_main.py."""
import time

from middleware import RateLimiter


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
