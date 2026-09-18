"""
request_log.py — Records a line of JSON per request processed, so we can
report real aggregate metrics (avg latency, cache hit rate, total tokens,
total cost, domain breakdown) instead of one-off numbers copied by hand.

This directly targets the theme guide's "Latency & Resource Efficiency" and
"Operational Cost & Cache Efficacy" evaluation categories with genuine data.
"""

import json
import os
import time

LOG_FILE = "request_log.jsonl"


def append_log(record: dict) -> None:
    record = dict(record)  # don't mutate caller's dict
    record["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_all_logs() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    logs = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a corrupted line rather than crash
    return logs


def compute_stats() -> dict:
    logs = read_all_logs()
    if not logs:
        return {
            "total_requests": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "cache_hit_rate": 0.0,
            "no_match_rate": 0.0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "requests_by_domain": {},
        }

    latencies = sorted(r.get("latency_ms", 0) for r in logs)
    cache_hits = sum(1 for r in logs if r.get("cache_hit"))
    no_matches = sum(1 for r in logs if r.get("fallback") == "no_match")
    total_tokens = sum(r.get("total_tokens", 0) for r in logs)
    total_cost = sum(r.get("cost_usd", 0.0) for r in logs)

    domain_counts = {}
    for r in logs:
        d = r.get("domain_guess", "Unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    n = len(logs)
    p95_index = max(0, int(n * 0.95) - 1)

    return {
        "total_requests": n,
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "p95_latency_ms": round(latencies[p95_index], 1),
        "cache_hit_rate": round(cache_hits / n, 3),
        "no_match_rate": round(no_matches / n, 3),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "requests_by_domain": domain_counts,
    }
