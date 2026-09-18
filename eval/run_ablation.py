"""
run_ablation.py — Runs all 3 deeplink-matching variants against the labeled
ground truth and reports real accuracy, latency, and cost per variant, in
the exact table format the theme guide's metrics.md template asks for.

Run with: python eval/run_ablation.py
"""

import json
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchers import match_rules_based, match_hybrid, match_llm
from deeplink_matching import load_deeplinks
from pipeline import llm_available
import llm_client

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)

CATALOG = load_deeplinks(os.path.join(PARENT_DIR, "deeplinks.json"))

with open(os.path.join(THIS_DIR, "deeplink_ground_truth.json")) as f:
    GROUND_TRUTH = json.load(f)


def run_variant(name: str, match_fn) -> dict:
    correct = 0
    latencies = []
    total_tokens = 0
    total_cost = 0.0

    for case in GROUND_TRUTH:
        start = time.time()
        predicted_id = match_fn(case["query"], CATALOG)
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

        if "Full-LLM" in name:
            total_tokens += llm_client.LAST_USAGE["total_tokens"]
            total_cost += llm_client.estimate_cost_usd(
                llm_client.LAST_USAGE["prompt_tokens"],
                llm_client.LAST_USAGE["completion_tokens"],
            )

        if predicted_id == case["expected_id"]:
            correct += 1

    n = len(GROUND_TRUTH)
    return {
        "variant": name,
        "step_accuracy": round(correct / n, 3),
        "latency_p95_ms": round(sorted(latencies)[int(n * 0.95) - 1] if n > 1 else latencies[0], 2),
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "cost_per_query": round(total_cost / n, 6),
        "correct": correct,
        "total": n,
    }


def main():
    print(f"Running ablation study on {len(GROUND_TRUTH)} labeled test cases...\n")

    variants = [("Variant B: Hybrid BM25 + Dense", match_hybrid),
                ("Variant A: Pure Rules-Based", match_rules_based)]
    if llm_available():
        variants.insert(0, ("Baseline: Full-LLM Mapping", match_llm))
    else:
        print("(Skipping Baseline: Full-LLM Mapping -- no LLM_API_KEY configured)\n")

    results = [run_variant(name, fn) for name, fn in variants]

    # Print as the exact markdown table shape used in Appendix C
    print("## 5. Architectural Ablation Analysis\n")
    print("| Architecture Variant | Step Accuracy | Latency (P95) | Cost / Query | Key Observations |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        accuracy_pct = f"{r['step_accuracy']*100:.1f}%"
        obs = f"{r['correct']}/{r['total']} correct matches"
        print(f"| {r['variant']} | {accuracy_pct} | {r['latency_p95_ms']}ms | ${r['cost_per_query']} | {obs} |")

    print("\n(Raw data below for your own analysis)")
    print(json.dumps(results, indent=2))

    # Also save to a file so it's easy to paste into the PPT / metrics.md
    out_path = os.path.join(THIS_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
