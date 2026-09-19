#!/usr/bin/env python3
"""
cli.py — Terminal client for the Smart Guided Troubleshooting Engine.

Talks to pipeline.py directly, in-process, by default -- zero network, zero
server, works even with no LLM_API_KEY set (falls onto the same offline
path the API server uses). Pass --api-base to hit a *running* uvicorn
server instead (useful for exercising the real HTTP contract: rate
limiting, request IDs, structured errors -- see middleware.py).

Deliberately dependency-free beyond the stdlib (urllib, not requests/httpx)
for the --api-base HTTP calls, so this script still runs in a minimal
install that skips the "dev / test only" section of requirements.txt.

Examples:
    # Single query, in-process (no server needed)
    python cli.py query "battery drains fast" --siis-file samples/battery.txt

    # Same, but against a live server (exercises the real API contract)
    python cli.py --api-base http://localhost:8000 query "battery drains fast"

    # Live stage-by-stage progress, same events index.html's SSE view shows
    python cli.py stream "screen flickers and touch is laggy"

    # Many complaints from a JSON file -- accepts either
    # sample_queries_real.json's shape or {"items": [...]}
    python cli.py batch sample_queries_real.json --out results.json

    # Batch against a live server instead (uses POST /v1/troubleshoot/batch)
    python cli.py --api-base http://localhost:8000 batch queries.json

    python cli.py --api-base http://localhost:8000 health
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DUMMY_POSITIVE_DEEPLINK = "bixby://dummy_positive"

# ---------------------------------------------------------------------------
# Minimal ANSI color helpers -- no colorama dependency, auto-disabled when
# stdout isn't a real terminal (piping to a file, CI logs, etc.).
# ---------------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def cyan(t): return _c("36", t)
def green(t): return _c("32", t)
def red(t): return _c("31", t)
def yellow(t): return _c("33", t)
def dim(t): return _c("2", t)
def bold(t): return _c("1", t)


# ---------------------------------------------------------------------------
# Stdlib-only HTTP helpers for --api-base mode
# ---------------------------------------------------------------------------

def _http_post_json(url: str, payload: dict, timeout: float = 60.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"error": {"message": body.decode(errors="replace")}}
    except urllib.error.URLError as e:
        print(red(f"Could not reach {url}: {e.reason}"))
        sys.exit(1)


def _http_get_json(url: str, timeout: float = 15.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except urllib.error.URLError as e:
        print(red(f"Could not reach {url}: {e.reason}"))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Rendering a pipeline result to the terminal
# ---------------------------------------------------------------------------

def _print_match_explanation(exp: dict | None, indent: str = "      "):
    if not exp:
        return
    if exp.get("matcher") == "hybrid_bm25_dense":
        parts = [
            f"bm25={exp.get('bm25_component')}",
            f"dense={exp.get('dense_component')}",
            f"feedback={exp.get('feedback_adjustment')}",
            f"final={exp.get('final_score')}",
        ]
    elif exp.get("matcher") == "rules_fuzzy":
        parts = [
            f"fuzzy={exp.get('fuzzy_score')}",
            f"feedback_pts={exp.get('feedback_adjustment_pts')}",
            f"final_pts={exp.get('final_score_pts')}",
        ]
    else:
        parts = [exp.get("reason", "")]
    kws = exp.get("matched_keywords") or []
    line = f"{indent}{dim('why: ' + ', '.join(parts))}"
    if kws:
        line += dim(f" | keywords: {', '.join(kws)}")
    print(line)


def print_result(data: dict, verbose: bool = False):
    meta = data.get("meta", {})
    contexts = data.get("response", {}).get("contexts", [])

    print(bold(f"\nquery: {data.get('query', '')}"))
    print(dim(
        f"latency {meta.get('latency_ms', '?')}ms  ·  model {meta.get('model', '?')}  ·  "
        f"cache {'hit' if meta.get('cache_hit') else 'miss'}  ·  "
        f"cost ${meta.get('cost_usd', 0):.4f}  ·  "
        f"tokens {meta.get('total_tokens', 0)}"
    ))

    if not contexts:
        print(yellow("\n  no match in the diagnostic catalog\n"))
        return

    for goal in contexts:
        confidence_note = dim(f"(confidence {goal.get('score', 0):.2f})")
        print(f"\n{cyan(bold(goal.get('title', '')))}  {confidence_note}")
        print(dim(f"  {goal.get('goal', '')}"))

        for action in goal.get("actions", []):
            category = action.get("category", "manual")
            tag = {
                "auto": green("[auto]"),
                "critical": red("[critical]"),
            }.get(category, yellow("[manual]"))
            print(f"\n  {tag} {bold(action.get('actionName', ''))}")
            print(f"      {action.get('description', '')}")

            for sg in action.get("stepGroups", []):
                for step in sg.get("steps", []):
                    print(f"      · {step}")
                dl = sg.get("actionableDeeplink")
                if dl:
                    is_real = dl.get("deeplink") and dl["deeplink"] != DUMMY_POSITIVE_DEEPLINK
                    arrow = green("-> ") if is_real else dim("-> (placeholder) ")
                    print(f"      {arrow}{dl.get('deeplink', '')}")
                    if verbose:
                        _print_match_explanation(dl.get("matchExplanation"))
    print()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_query(args):
    siis_response = args.siis_response or ""
    if args.siis_file:
        with open(args.siis_file) as f:
            siis_response = f.read()

    if args.api_base:
        status, data = _http_post_json(
            f"{args.api_base}/v1/troubleshoot",
            {"query": args.query, "siis_response": siis_response},
        )
        if status != 200:
            print(red(f"API error {status}: {json.dumps(data)}"))
            sys.exit(1)
    else:
        from pipeline import run_pipeline
        data = run_pipeline(args.query, siis_response)

    if args.json:
        print(json.dumps(data, indent=2))
        return
    print_result(data, verbose=args.verbose)


def cmd_stream(args):
    if args.api_base:
        print(red("stream is in-process only (no server round trip) -- drop --api-base"))
        print(dim("(index.html's own live view talks to GET /v1/troubleshoot/stream directly over SSE instead)"))
        sys.exit(1)

    from pipeline import run_pipeline_streaming

    print(bold(f"streaming: {args.query}\n"))
    t0 = time.time()
    for event in run_pipeline_streaming(args.query, args.siis_response or ""):
        stage, status = event.get("stage"), event.get("status")
        elapsed_ms = (time.time() - t0) * 1000
        prefix = dim(f"[{elapsed_ms:7.1f}ms]")
        if stage == "complete":
            print(f"{prefix} {green('complete')}")
            print_result(event["data"], verbose=args.verbose)
        elif stage == "error":
            print(f"{prefix} {red('error: ' + event['data']['message'])}")
        else:
            detail = ""
            if event.get("data"):
                detail = dim(" " + json.dumps({k: v for k, v in event["data"].items()
                                                if k not in ("contexts",)}))
            print(f"{prefix} {stage:16s} {status:8s}{detail}")


def _load_batch_items(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    items = []
    for it in raw:
        query = it.get("query") or it.get("complaint")
        if not query:
            continue
        items.append({"query": query, "siis_response": it.get("siis_response") or ""})
    return items


def cmd_batch(args):
    items = _load_batch_items(args.file)
    if not items:
        print(red(f"no usable {{query, siis_response}} items found in {args.file}"))
        sys.exit(1)

    if args.api_base:
        if len(items) > 20:
            print(yellow(
                f"{len(items)} items exceeds the batch endpoint's 20-item cap "
                f"(see schema.BatchTroubleshootRequest) -- sending only the first 20."
            ))
            items = items[:20]
        status, data = _http_post_json(f"{args.api_base}/v1/troubleshoot/batch", {"items": items})
        if status != 200:
            print(red(f"API error {status}: {json.dumps(data)}"))
            sys.exit(1)
        results = data["results"]
        print(dim(f"batch: {data['succeeded']}/{data['count']} succeeded "
                   f"(request_id {data['request_id']})\n"))
    else:
        from pipeline import run_pipeline
        results = []
        for it in items:
            try:
                r = run_pipeline(it["query"], it["siis_response"])
                results.append({"ok": True, "result": r})
            except Exception as e:
                results.append({"ok": False, "error": str(e), "query": it["query"]})
        succeeded = sum(1 for r in results if r["ok"])
        print(dim(f"batch: {succeeded}/{len(results)} succeeded (in-process)\n"))

    for it, r in zip(items, results):
        label = it["query"][:64] + ("…" if len(it["query"]) > 64 else "")
        if r["ok"]:
            n = len(r["result"].get("response", {}).get("contexts", []))
            status_s = green(f"{n} goal(s)") if n else yellow("no match")
        else:
            status_s = red(f"error: {r.get('error', '?')}")
        print(f"  {label:<66s} {status_s}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(dim(f"\nwrote {args.out}"))


def cmd_health(args):
    base = args.api_base or "http://localhost:8000"
    status, data = _http_get_json(f"{base}/health")
    if status != 200:
        print(red(f"API error {status}: {json.dumps(data)}"))
        sys.exit(1)
    print(json.dumps(data, indent=2))


def cmd_stats(args):
    base = args.api_base or "http://localhost:8000"
    status, data = _http_get_json(f"{base}/stats")
    if status != 200:
        print(red(f"API error {status}: {json.dumps(data)}"))
        sys.exit(1)
    print(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Terminal client for the Smart Guided Troubleshooting Engine "
                     "(in-process by default -- no server required).",
    )
    parser.add_argument(
        "--api-base", default=None,
        help="Call a running API server instead of the in-process pipeline "
             "(e.g. http://localhost:8000). Must come before the subcommand.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Run a single complaint through the pipeline")
    q.add_argument("query", help="The raw complaint text")
    q.add_argument("--siis-response", default=None, help="Reference troubleshooting text to ground the plan in")
    q.add_argument("--siis-file", default=None, help="Read --siis-response from a file instead")
    q.add_argument("--json", action="store_true", help="Print the raw JSON response instead of formatted text")
    q.add_argument("--verbose", "-v", action="store_true", help="Also show each match's score breakdown (see deeplink_matching.py)")
    q.set_defaults(func=cmd_query)

    s = sub.add_parser("stream", help="Run a complaint with live stage-by-stage output (in-process only)")
    s.add_argument("query", help="The raw complaint text")
    s.add_argument("--siis-response", default=None)
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(func=cmd_stream)

    b = sub.add_parser("batch", help="Run many complaints from a JSON file (sample_queries_real.json shape, or {\"items\": [...]})")
    b.add_argument("file", help="Path to a JSON file of complaints")
    b.add_argument("--out", default=None, help="Write full per-item results to this JSON file")
    b.set_defaults(func=cmd_batch)

    h = sub.add_parser("health", help="Check API server health (requires --api-base)")
    h.set_defaults(func=cmd_health)

    st = sub.add_parser("stats", help="Fetch live API server metrics (requires --api-base)")
    st.set_defaults(func=cmd_stats)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
