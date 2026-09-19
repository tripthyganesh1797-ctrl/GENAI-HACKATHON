"""
eval/generate_results_jsonl.py — produces results.jsonl, the offline
results file the Theme 2 FAQ (Q17, Q2, gate G3) asks for alongside the
live API: one JSON object per line, each with the exact shape

    {"query": ..., "query_variations": [...], "response": {...ContextDeeplinkResponse...}}

covering the official query set (gate G3 wants >= 95% coverage; this
covers all 20/20). run_pipeline() already returns exactly this shape
(plus a "meta" block, which the FAQ's own worked example in
Theme2_Problem_Statement.pdf Appendix B also includes) -- this script
does no separate extraction logic, it just runs the real pipeline
against the real official queries and serializes what comes back, so
results.jsonl can never drift from what the live API actually returns
for the same input.

Run with: python eval/generate_results_jsonl.py
Works with zero setup (no LLM_API_KEY needed -- runs on the offline
fallback path exactly like the rest of this repo's eval tooling).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import CACHE_FILE
from pipeline import run_pipeline

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)

with open(os.path.join(PARENT_DIR, "sample_queries_real.json")) as f:
    QUERY_SET = json.load(f)

OUT_PATH = os.path.join(PARENT_DIR, "results.jsonl")


def main():
    # Force every query to run cold (not served from a stale cache entry
    # left over from a previous eval/demo run in this same checkout) so
    # results.jsonl always reflects a genuine fresh pipeline execution,
    # not whatever happened to already be cached.
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    lines = []
    matched = 0
    for case in QUERY_SET:
        result = run_pipeline(case["complaint"], case.get("siis_response", ""))
        if result["response"]["contexts"]:
            matched += 1
        lines.append(json.dumps({
            "query": result["query"],
            "query_variations": result["query_variations"],
            "response": result["response"],
            "meta": result["meta"],
        }))

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    covered = len(QUERY_SET)
    print(f"Wrote {covered} lines to {OUT_PATH}")
    print(f"Coverage: {covered}/{len(QUERY_SET)} official queries ({100*covered/len(QUERY_SET):.0f}%)")
    print(f"Non-empty (matched) responses: {matched}/{covered}")


if __name__ == "__main__":
    main()
