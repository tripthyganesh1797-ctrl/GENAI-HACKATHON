# Smart Guided Troubleshooting Engine

Samsung PRISM GenAI Hackathon 3rd Edition — Theme 2

Transforms a vague Galaxy device complaint ("screen flickers and battery dies
fast") into a validated, deeplink-mapped, one-tap troubleshooting plan.
Runs against the **real official Theme 2 dataset** (578 masked deeplinks,
20 official SIIS support queries) and ships with **two interchangeable
execution paths**:

| Path | Quality | Cost | Requires |
|---|---|---|---|
| **LLM path** (primary) | Best | Real $/token cost (or free on Groq) | `LLM_API_KEY` |
| **Offline fallback** (`offline_fallback.py`) | Good, fully deterministic | $0.00, always | Nothing |

The service picks the LLM path automatically when a key is configured, and
**falls back to the offline path automatically** — no crash, no manual
switch — whenever `LLM_API_KEY` isn't set, or an LLM call/JSON-parse fails
after retries. `meta.used_offline_fallback` in every response says which
path actually ran.

**Contents:** [Architecture](#architecture) · [Setup](#setup) · [Run the pipeline directly](#run-the-pipeline-directly-no-server-needed-fastest-way-to-test) · [Run the API server](#run-the-api-server) · [Run the tests](#run-the-tests) · [Run with Docker](#run-with-docker) · [Try the demo UI](#try-the-demo-ui) · [Project structure](#project-structure) · [Production-readiness notes](#production-readiness-notes) · [Submission checklist](#submission-checklist-per-hackathon_guidelinespdf) · [Known limitations](#known-limitations)

## Architecture

Every request flows through the same four stages regardless of which
execution path runs Stage 0/1 — Stage 2 (deeplink matching) and every
validator run identically either way, so the LLM path and the offline
fallback can never disagree about the data *contract*, only about how
good the extracted troubleshooting content is.

```mermaid
flowchart TD
    U["User complaint\n(text, voice, or any language)"] --> API

    subgraph API["FastAPI (main.py)"]
        direction TB
        MW["middleware.py: request ID -> rate limit -> CORS"]
        EP1["POST /v1/troubleshoot"]
        EP2["GET /v1/troubleshoot/stream (SSE)"]
        EP3["POST /v1/feedback"]
        EP4["GET /stats · /health (rate-limit exempt)"]
        MW --> EP1 & EP2 & EP3 & EP4
    end

    EP1 --> S0
    EP2 --> S0

    S0{"Stage 0: enrich\nLLM_API_KEY set?"}
    S0 -- "yes, LLM reachable" --> L0["LLM: normalise + translate\n+ 8-10 paraphrases\n(prompts.py / llm_client.py)"]
    S0 -- "no key, or call/JSON-parse fails" --> O0["offline_fallback.py\nrule-based enrich, $0, English-only"]

    L0 --> CACHE
    O0 --> CACHE
    CACHE{"Semantic cache hit?\n(cache.py, Jaccard overlap)"}
    CACHE -- "yes" --> RESP
    CACHE -- "no" --> S1

    S1{"Stage 1: extract"}
    S1 -- "LLM" --> L1["LLM: Goal/Action/StepGroup JSON\n+ multi-issue splitting"]
    S1 -- "offline" --> O1["offline_fallback.py\nparse SIIS text, group steps,\nsplit multi-issue complaints"]

    L1 --> VAL
    O1 --> VAL
    VAL["validators.py\nword counts · goal syntax · URL scrub\ncritical-safety override · re-sort"]

    VAL --> S2
    S2["Stage 2: deeplink matching\n(deeplink_matching.py — always code, never the LLM)"]
    S2 --> HYB["HybridDeeplinkIndex\nBM25 + dense (sentence-transformers\nor offline TF-IDF/SVD)"]
    S2 --> RUL["RulesDeeplinkIndex\nfuzzy keyword match (rapidfuzz)"]
    HYB & RUL --> FB[("feedback.py\nper-deeplink score adjustment")]

    HYB --> RESP
    RUL --> RESP
    RESP["Validated response\n(contexts + meta)"] --> LOG[("request_log.jsonl\n→ /stats")]

    EP3 --> FBWRITE["feedback.record_feedback()"] --> FB
```

**Why this shape:** Stage 2 and the validators are shared, not
duplicated, between the two Stage-0/1 paths — the ablation study and the
"no-hallucination" guarantee describe the code that actually ships, not a
separate demo-only code path. The feedback loop closes over the deeplink
matchers specifically (not the LLM/offline choice), because that's the
stage a human can meaningfully correct without needing to re-prompt an
LLM: "this deeplink was wrong" is a fact about the catalog match, not
about how the complaint was understood.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional — only needed for the LLM path
```

The service works with **zero setup beyond `pip install`** — without a
`.env`/`LLM_API_KEY` it just runs entirely on the offline fallback path.

## Run the pipeline directly (no server needed, fastest way to test)

```bash
python pipeline.py
```

Runs the first 3 complaints from `sample_queries_real.json` (the real
official queries, each paired with its real SIIS reference text) through
the full pipeline and prints the JSON output.

## Run the API server

```bash
uvicorn main:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/v1/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"query": "screen flickers and the battery dies fast"}'

curl http://localhost:8000/health
curl http://localhost:8000/stats

# Live stage-by-stage progress (Server-Sent Events) -- what index.html's UI consumes:
curl -N "http://localhost:8000/v1/troubleshoot/stream?query=screen+flickers+and+battery+dies+fast"

# Thumbs up/down on a specific matched deeplink (drives adaptive re-ranking, see feedback.py):
curl -X POST http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"deeplink": "bixby://masked/act/...", "action_name": "Battery Settings", "helpful": true}'
```

## Run the tests

```bash
pip install -r requirements.txt   # includes pytest / httpx (dev-only, see bottom of the file)
pytest                             # 98 tests, ~92% line coverage, runs in ~15s, no LLM key needed
pytest --cov=. --cov-report=term-missing   # optional, needs pytest-cov (already in requirements.txt)
```

Covers the data contract (`test_validators.py`, `test_schema.py`), the
zero-API-key offline path end-to-end against the real 20 official queries
(`test_offline_fallback.py`), retrieval against the real 578-entry catalog
plus the feedback-driven re-ranking (`test_deeplink_matching.py`,
`test_feedback.py`), the cache and stats logging (`test_cache.py`,
`test_request_log.py`), the full orchestration including a byte-for-byte
parity check between the streaming and non-streaming pipelines
(`test_pipeline.py`), and the live FastAPI service itself via `TestClient`
(`test_main.py`) — including the SSE endpoint. All isolated from your real
`cache_store.json` / `request_log.jsonl` / `feedback_*` files (see
`tests/conftest.py`).

## Run with Docker

```bash
docker build -t troubleshoot-engine .
docker run -p 8000:8000 troubleshoot-engine                       # offline mode
docker run -p 8000:8000 -e LLM_PROVIDER=groq -e LLM_API_KEY=xxx troubleshoot-engine  # LLM mode
```

## Try the demo UI

Open `index.html` directly in a browser (Chrome/Edge recommended — it
includes a **microphone button for voice input** via the Web Speech API,
and a language selector for both speech recognition and query language).
Point it at a running `uvicorn` server (`API_BASE` at the top of the
`<script>` block, defaults to `http://localhost:8000`). `demo.html` is a
simpler, earlier version of the same UI kept for reference.

Click **telemetry** in the top-right to open the **analytics dashboard**:
live stat tiles, a requests-by-domain bar chart, and a helpful/unhelpful
feedback breakdown — all real numbers from `/stats`, with hover tooltips
and a screen-reader-friendly table view, not mock data.

## Project structure

| File | Purpose |
|---|---|
| `schema.py` | Pydantic models — exact data contract from the theme guide |
| `validators.py` | Programmatic checks (word counts, URL scrubbing, category ordering + safety-net re-sort) |
| `prompts.py` | LLM prompts (Stage 0 enrichment — now explicitly multi-language, Stage 1 extraction) |
| `llm_client.py` | Swappable LLM API wrapper (Anthropic / OpenAI / Groq) |
| `offline_fallback.py` | **Deterministic, zero-API-key replacement for Stages 0-1** — rule-based query enrichment + SIIS text parsing, $0 cost, no network |
| `deeplink_matching.py` | Shared Stage 2 deeplink retrieval (BM25 + dense hybrid, or pure fuzzy-rules fallback), used by both execution paths and the ablation study |
| `pipeline.py` | Orchestrates: complaint → enrich → extract → validate → deeplink match → cache, choosing LLM vs offline path per-request |
| `cache.py` | Fast-path semantic cache (keyword overlap) |
| `main.py` | FastAPI app: `POST /v1/troubleshoot`, `GET /health`, `GET /stats` — warms the deeplink index at startup |
| `index.html` | Polished demo UI, with voice input + language selector |
| `demo.html` | Simpler legacy demo UI |
| `deeplinks.json` | **Real official catalog**: 578 masked deeplinks across the full Settings surface |
| `siis_responses.json` | **Real official data**: 20 official SIIS support queries + their raw reference text |
| `sample_queries_real.json` | The 20 official queries adapted into this project's `{domain, complaint, siis_response}` shape — what `pipeline.py` and `eval/eval_harness.py` actually run against |
| `sample_queries.json` | Original placeholder queries (Battery/Camera/Display/Performance), kept for quick manual smoke tests only |
| `deeplinks_placeholder_backup.json`, `deeplinks_expanded_placeholder_backup.json` | Original placeholder catalogs, kept for reference |
| `official_theme2_data/` | Untouched copy of the official team-kit dataset (deeplinks.json, siis_responses.json, sample_output.json, schema_reference.py) |
| `eval/eval_harness.py` | Runs the full pipeline against the 20 real queries, produces `eval/metrics.md` |
| `eval/run_ablation.py` + `eval/matchers.py` | 3-variant deeplink-matching ablation (Full-LLM / Hybrid BM25+dense / Pure rules) against `eval/deeplink_ground_truth.json` (real catalog entries) |
| `request_log.py` | Per-request JSONL log, powers `/stats` |
| `feedback.py` | Human-in-the-loop feedback (`POST /v1/feedback`) + the bounded per-deeplink score adjustment that `deeplink_matching.py` consults on every search |
| `middleware.py` | Request IDs, structured `{"error": {...}}` bodies, and per-route in-memory rate limiting |
| `tests/` | pytest suite, ~92% line coverage, zero LLM key required — see "Run the tests" above |

## Production-readiness notes

Every response — success or error, including 429s — carries an
`X-Request-ID` header (echoed back if the caller supplies one, so a
client-side trace ID survives the round trip). Every error the service
returns, whatever raised it (a validation failure, an unknown route, a
rejected feedback submission, an uncaught exception), comes back in the
same shape:

```json
{"error": {"code": "400", "message": "...", "request_id": "..."}}
```

`/v1/troubleshoot` (and its `/stream` variant) and `/v1/feedback` are each
rate-limited per client IP (30 req/min and 60 req/min respectively — tuned
for interactive demo/judge traffic, not a load test); `/health` is exempt
so uptime checks never trip it. The limiter is in-process and in-memory,
which is the right call for this submission's actual deployment shape (a
single `uvicorn` process — see the Dockerfile, no orchestration implied
anywhere else in this repo) and is genuinely enforced, not just logged;
it just doesn't coordinate across multiple replicas, which a real
multi-instance deployment would need a shared store (e.g. Redis) for
instead. See `middleware.py` for all of this.

## Submission checklist (per Hackathon_Guidelines.pdf)

- [x] Working prototype code, real official data wired in
- [x] README with reproducible setup + Docker
- [ ] Fill in `AI_Disclosure_Form_TEMPLATE.docx` as a team before submitting
- [ ] Demo video (max 5 min)
- [ ] Presentation file, named `CollegeName_TeamName_Submission_ppt`
- [ ] Push to a **public or shared GitHub repo**
- [ ] Tag the final commit `PRISM_GENAI_HACKATHON_Y2026` — the tagged commit is what gets judged
- [ ] Submit everything via the official Google Form by **25 Sep 2026, 11:59 PM**

## Known limitations

See `eval/metrics.md` section 6 ("Known Edge Cases & System Limitations")
for the full, honest list, generated from real measured runs — including
why the offline fallback is English-only, why this sandbox's ablation
numbers favor the rules-based matcher (HuggingFace Hub is unreachable
here, so the hybrid matcher runs on TF-IDF/SVD rather than real
sentence-transformer embeddings — re-run `eval/run_ablation.py` on a
machine with normal internet access to get the true comparison), and one
official query whose SIIS reference text is a genuine grounding mismatch
(floating "Assistive menu" circle, paired with Multi-Window/Edge-panel
docs) — the **offline fallback's keyword-overlap relevance gate is fooled
by it** (measured, not assumed: see metrics.md for why a stricter
threshold isn't a clean fix), while the **LLM path correctly rejects it**
via prompt rule #7's relevance judgment. This is the clearest concrete
example of why the LLM path is the primary path and the rule-based
fallback is a $0/always-available degrade, not a drop-in replacement.
