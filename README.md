# Smart Guided Troubleshooting Engine

**GenAI - Google Drive:** [[https://drive.google.com/drive/u/0/folders/1nGRBihNH4eYMUosMETzAHrRJh_vWTfOj]]


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

Beyond the core troubleshooting pipeline, the service also ships:

- **Confidence-gated escalation** (`escalation.py`) — a Goal the engine
  itself judged uncertain still returns its full plan, plus an honest
  caveat and a real backup action (`goal.escalation`).
- **Structured device-signal input** (`device_signals.py`) — optional
  battery/storage/uptime context reorders same-category actions toward
  what's actually relevant and adds a short advisory note; a no-op when
  omitted.
- **Session-scoped avoidance** (`session_memory.py`) — mark a deeplink
  unhelpful with a `session_id` and later requests in that same session
  steer away from re-suggesting it, without touching the global
  feedback-driven re-ranking.
- **Trending issues on `/stats`** (`request_log.py`) — a finer-grained
  "top recurring symptoms" breakdown (e.g. "Battery draining fast"), not
  just a 5-bucket domain count.
- **Shareable diagnostic reports** (`report.py`, `POST /v1/report`) —
  packages an already-computed result into a compact Markdown or
  self-contained HTML report a user can paste into a support ticket or
  forward to someone else.
- **Voice output** — `index.html`'s results panel can read the current
  plan aloud via the Web Speech API, alongside the existing voice *input*.
- **Physical-hazard safety short-circuit** (`safety.py`) — a swollen/
  bulging battery, smoke, sparking, or a burning/chemical smell is a real
  fire/burn risk, not a normal troubleshooting scenario. A narrow,
  deterministic keyword check runs before anything else (before Stage 0,
  before the cache) on both execution paths and, when it fires, replaces
  the plan entirely with a single "stop using, don't charge, contact
  Samsung Support" instruction — never software steps for this class of
  complaint. `meta.safety_alert` / `meta.safety_reason` flag it for any
  API/CLI/UI consumer; `index.html` renders an unmissable red banner.
- **Clarifying-question detection** (`clarify.py`) — a complaint like "my
  phone isn't working" or "it's broken" gives the engine almost no signal
  to work with. Rather than silently handing back a low-confidence guess,
  `meta.needs_clarification` / `meta.clarifying_question` /
  `meta.clarifying_topic_options` flag it, and `index.html` shows an
  amber "could you tell us more?" banner with one-click topic chips
  (Battery, Screen, Camera, Wi-Fi/Bluetooth, ...) that append the topic
  and re-run instantly. Purely additive — unlike the safety short-circuit
  above, the normal plan is still computed and returned in full; this
  never suppresses `response.contexts`, which is why it can never put
  gate G3's official-query coverage at risk (verified: none of the 20
  official queries trip it).

### Research-paper-inspired extensions

Four more features, each adapting an idea from a specific paper to this
project's existing dual-path (LLM/offline) architecture rather than a
literal port — see each module's docstring for the full honest mapping
between the paper's method and what's actually implemented here.

- **CLAM-inspired ambiguity confidence** (`ambiguity.py`, from *"CLAM:
  Selective Clarification for Ambiguous Questions with Generative Language
  Models"*) — a second, LLM-based opinion layered on top of `clarify.py`'s
  zero-cost keyword heuristic. It can only ever RAISE the heuristic's
  "needs clarification" verdict, never lower it, and it's a complete no-op
  with no LLM key configured — so every guarantee `clarify.py` already had
  (including never risking gate G3's coverage) stays fully intact.
  `meta.ambiguity`.
- **Guided-Retry recovery** (`recovery.py`, from *"When the Database
  Fails: Prompting LLM Dialogue Agents for Safe Recovery in Task-Oriented
  Dialogue"*) — when Stage 1's LLM call actually fails (not a legitimate
  empty result — a real error), one structured retry with an explicit
  recovery instruction is attempted before falling through to the offline
  path, instead of a silent, "naive" drop straight to offline with zero
  visibility into what happened. `meta.recovery`.
- **SIA-inspired interactive investigator** (`investigator.py`, `POST
  /v1/investigate/start` + `/answer`, from *"LLM-as-an-Investigator:
  Evidence-First Reasoning for Robust Interactive Problem Diagnosis"*) —
  an opt-in short Q&A that maintains a probability distribution over 2-3
  candidate root causes (reusing `related_issues.py`'s curated data),
  asks one targeted discriminative question at a time, updates the
  distribution from the answer, and stops once a hypothesis clears 90%
  confidence or 3 questions have been asked — then runs the *real*
  pipeline against the winning hypothesis rather than inventing its own
  plan. Fully offline-capable (a deterministic question sequence + a
  keyword-bucket probability update) with an LLM-assisted upgrade when a
  key is configured.
- **DiagGPT-inspired session topic stack** (`topic_manager.py`, from
  *"DiagGPT: An LLM-based and Multi-agent Dialogue System with Automatic
  Topic Management for Flexible Task-Oriented Dialogue"*) — tracks a
  per-session stack of symptom "topics" (stay on the current one, open a
  new one, or jump back to one mentioned earlier), using a deterministic
  bucket match instead of DiagGPT's own per-turn LLM agent. `meta.topic_stack`,
  shown in `index.html` as a small "this session:" chip trail.

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
        EP5["POST /v1/troubleshoot/batch\n(1-20 items, own rate limit)"]
        MW --> EP1 & EP2 & EP3 & EP4 & EP5
    end

    EP1 --> S0
    EP2 --> S0
    EP5 -. "runs the full pipeline\nonce per item" .-> S0

    S0{"Stage 0: enrich\nLLM_API_KEY set?"}
    S0 -- "yes, LLM reachable" --> L0["LLM: normalise + translate\n+ 8-10 paraphrases\n(prompts.py / llm_client.py)"]
    S0 -- "no key, or call/JSON-parse fails" --> O0["offline_fallback.py\nrule-based enrich, $0, + Hinglish phrasebook"]

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
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional — only needed for the LLM path
```

The service works with **zero setup beyond `pip install`** — without a
`.env`/`LLM_API_KEY` it just runs entirely on the offline fallback path.

**If you edit `.env` after the server is already running, you must fully
restart it (`Ctrl+C`, then re-run `uvicorn main:app --reload`).**
`--reload` only watches `.py` source files for changes, not `.env` —
`llm_client.py` reads `LLM_API_KEY`/`LLM_MODEL`/`LLM_PROVIDER` once at
import time, so a running process keeps using whatever `.env` said when
it first started until you restart it, even though the file on disk now
says something else. Also make sure `.env` actually sets `LLM_API_KEY`
(not `OPENAI_API_KEY` or `GROQ_API_KEY`) — that's the exact variable
name this project's code reads; setting a differently-named variable
can silently "work" (the `openai` Python SDK falls back to
`OPENAI_API_KEY` on its own if `LLM_API_KEY`/`api_key` comes through as
`None`) but leaves `LLM_MODEL` pointed at whatever stale default was
loaded at startup, which is a confusing failure mode to debug.

This includes a real person typing their **own** complaint with nothing
else supplied (no reference text to paste in) — e.g. `"my battery is
getting drained quickly"` on the offline path grounds itself in
`builtin_knowledge.py`, a small hand-written, explicitly-labeled
non-official Android/Samsung troubleshooting reference, then runs
through the exact same relevance-gating and Stage 2 deeplink-matching
code as every other query. `meta.used_builtin_reference` on the
response says whether a given plan came from that built-in text or a
real caller-supplied `siis_response`, so it's never ambiguous which one
grounded the answer. An out-of-scope complaint ("how do I cook pasta")
still correctly comes back `no_match` — see
`tests/test_builtin_knowledge.py`.

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

# Batch: up to 20 complaints in one round trip (e.g. a device health-check
# screen probing several known symptoms at once). One item failing doesn't
# fail the others -- each result is reported individually as {"ok": ...}.
curl -X POST http://localhost:8000/v1/troubleshoot/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"query": "battery drains fast"}, {"query": "screen flickers"}]}'

# Optional device context + session_id -- reorders relevant actions and adds
# an advisory note; session_id lets a later request in the same session
# avoid re-suggesting a deeplink marked unhelpful via POST /v1/feedback:
curl -X POST http://localhost:8000/v1/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"query": "battery drains fast", "device": {"battery_pct": 8}, "session_id": "demo-1"}'

# Package an already-computed result into a shareable Markdown/HTML report
# (never re-runs the pipeline -- see report.py):
curl -X POST http://localhost:8000/v1/report \
  -H "Content-Type: application/json" \
  -d '{"result": <the exact body /v1/troubleshoot returned>, "format": "markdown"}'

# Goal-level "did this fix it?" Yes/No (point 5, see resolution.py) -- a
# "No" reuses the same per-deeplink feedback/session-avoidance machinery
# POST /v1/feedback already drives:
curl -X POST http://localhost:8000/v1/resolution \
  -H "Content-Type: application/json" \
  -d '{"goal_title": "Battery fast drain", "resolved": false,
       "deeplinks": [{"deeplink": "bixby://masked/act/...", "action_name": "Battery Settings"}]}'

# Attach a photo of the problem (point 1, see image_analysis.py) -- folded
# into the complaint text as a factual description before Stage 0 runs.
# Degrades honestly (meta.image_analysis.reason) with no LLM key or a
# non-vision-capable model configured -- this codebase's own coded default
# (Groq's openai/gpt-oss-20b) is text-only, so set LLM_MODEL to a
# vision-capable one (see llm_client.py's _VISION_CAPABLE_MODELS) to see
# meta.image_analysis.analyzed=true:
curl -X POST http://localhost:8000/v1/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"query": "my screen looks weird", "image_data_url": "data:image/jpeg;base64,<...>"}'

# SIA-inspired interactive diagnosis (see investigator.py) -- instead of
# committing to the engine's first guess, ask a short targeted Q&A to
# narrow down WHICH of several candidate root causes is actually correct,
# then run the normal pipeline against the winning one:
curl -X POST http://localhost:8000/v1/investigate/start \
  -H "Content-Type: application/json" \
  -d '{"complaint": "my battery drains really fast", "session_id": "demo-1"}'
# -> {"investigation_id": "...", "status": "in_progress", "question": "...", "hypotheses": [...]}
curl -X POST http://localhost:8000/v1/investigate/answer \
  -H "Content-Type: application/json" \
  -d '{"investigation_id": "<from above>", "answer": "it happens every single time, consistently"}'
# -> repeat until "status": "resolved", which includes a real final_result
#    (the exact same shape POST /v1/troubleshoot returns)
```

## Run the tests

```bash
pip install -r requirements.txt   # includes pytest / httpx (dev-only, see bottom of the file)
pytest                             # 536 tests, ~97% line coverage, runs in ~13s, no LLM key needed
pytest --cov=. --cov-report=term-missing   # optional, needs pytest-cov (already in requirements.txt)
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

## Terminal CLI

No browser, no server, no LLM key needed -- `cli.py` calls `pipeline.py`
directly and prints formatted results (drop `--api-base` to run against a
live server instead, which exercises the real HTTP contract: rate
limiting, request IDs, structured errors):

```bash
python cli.py query "battery drains fast and camera lags on open"
python cli.py query "screen flickers" --siis-file samples/screen.txt --verbose   # --verbose shows the matchExplanation breakdown
python cli.py stream "screen flickers and touch is laggy"                       # live stage-by-stage, same events the SSE endpoint emits
python cli.py batch sample_queries_real.json --out results.json                 # runs all 20 official queries, writes full results
python cli.py --api-base http://localhost:8000 batch queries.json               # same, but against a live server (uses POST /v1/troubleshoot/batch)
python cli.py --api-base http://localhost:8000 health
python cli.py --api-base http://localhost:8000 stats
python cli.py feedback bixby://masked/act/... "Wifi Settings" --unhelpful --session-id demo-1  # then re-run `query` with the same --session-id to see it steer away
python cli.py report "camera app keeps crashing" --format html -o report.html                  # package a result as a shareable report
```

Zero third-party dependencies for the `--api-base` HTTP calls (stdlib
`urllib`, not `requests`/`httpx`) -- the CLI works even in a minimal
install that skips the "dev / test only" section of `requirements.txt`.

## Run with Docker

```bash
docker build -t troubleshoot-engine .
docker run -p 8000:8000 troubleshoot-engine                       # offline mode
docker run -p 8000:8000 -e LLM_PROVIDER=groq -e LLM_API_KEY=xxx troubleshoot-engine  # LLM mode
```

## Try the demo UI

Open `index.html` directly in a browser (Chrome/Edge recommended — it
includes a **microphone button for voice input** via the Web Speech API,
and a language selector for both speech recognition and query language).
Point it at a running `uvicorn` server (`API_BASE` at the top of the
`<script>` block, defaults to `http://localhost:8000`). `demo.html` is a
simpler, earlier version of the same UI kept for reference.

A result's panel also offers an optional **device state** disclosure
(battery/storage/uptime, reorders relevant steps), a **📷 photo attach**
button (client-side downscale to a 1024px JPEG before it's sent — see
`image_analysis.py`), **📋 copy report / ⬇ download report** buttons
(packages the result via `POST /v1/report`), and, on a browser that
supports it, a **🔊 read steps aloud** control (Web Speech API
`SpeechSynthesis`, entirely client-side). A low-confidence match also
shows an amber **escalation banner** with a backup action, plus a
**"might it be one of these instead?"** chip list of related possibilities
(`related_issues.py`) and, under every goal, a **"did this fix it?"**
Yes/No prompt (`resolution.py`) — a "No" surfaces those same alternative
chips right there. A collapsible **"where did this answer come from?"**
disclosure (`answer_source.py`) tells you whether a plan is official
Samsung reference text, generic built-in knowledge, or an AI's general
knowledge. 👍/👎 feedback on a matched deeplink is session-scoped — mark
one unhelpful (directly, or via a "did this fix it?" No) and the *next*
query in that same browser tab steers away from suggesting it again.
A **"this session:" topic chip trail** (`topic_manager.py`) quietly
appears once you've asked about more than one thing in the same tab, and
a small note appears on any answer where the AI model failed once and a
guided retry recovered it (`recovery.py`) — both silent otherwise. A
**🔍 "not sure this is right? run a guided diagnosis"** button
(`investigator.py`) opens an inline short Q&A that narrows down which of
2-3 candidate causes is actually correct before showing you the real plan
for that specific cause.

Click **telemetry** in the top-right to open the **analytics dashboard**:
live stat tiles, a requests-by-domain bar chart, a **trending issues**
breakdown (the actual recurring symptoms, not just the domain bucket), and
a helpful/unhelpful feedback breakdown — all real numbers from `/stats`,
with hover tooltips and a screen-reader-friendly table view, not mock data.

## Project structure

| File | Purpose |
|---|---|
| `schema.py` | Pydantic models — exact data contract from the theme guide |
| `validators.py` | Programmatic checks (word counts, URL scrubbing, category ordering + safety-net re-sort) |
| `prompts.py` | LLM prompts (Stage 0 enrichment — now explicitly multi-language, Stage 1 extraction) |
| `llm_client.py` | Swappable LLM API wrapper (Anthropic / OpenAI / Groq) — plus `call_llm_vision()`/`is_vision_capable()` for point 1's image analysis, checked against the model actually configured rather than assumed |
| `offline_fallback.py` | **Deterministic, zero-API-key replacement for Stages 0-1** — rule-based query enrichment + SIIS text parsing, $0 cost, no network |
| `deeplink_matching.py` | Shared Stage 2 deeplink retrieval (BM25 + dense hybrid, or pure fuzzy-rules fallback), used by both execution paths and the ablation study; also owns the session-avoidance ranking helper (see `session_memory.py`) |
| `escalation.py` | Shared confidence-gated escalation recommendation — attaches an honest caveat + real backup action to a low-confidence Goal without ever replacing the plan |
| `device_signals.py` | Optional device-state context (battery/storage/uptime) — reorders same-category actions toward what's relevant, adds an advisory note; a no-op when omitted |
| `session_memory.py` | Session-scoped avoidance: a deeplink marked unhelpful (with a `session_id`) is steered away from in that session's later requests — distinct from `feedback.py`'s global re-ranking |
| `safety.py` | Physical-hazard short-circuit (swollen battery, smoke, fire, sparks, chemical smell, burns) — narrow keyword-based detection that replaces the plan with a single "stop, don't charge, contact Samsung Support" Goal, on both execution paths, before Stage 0 or the cache ever runs |
| `clarify.py` | Vague-complaint detection ("it's broken", "not working") — additive-only `meta.needs_clarification`/`clarifying_question` flag that never suppresses `response.contexts`; verified to never trip on any of the 20 official queries |
| `answer_source.py` | Classifies WHERE a returned plan's content actually came from (official Samsung reference, generic built-in knowledge, AI general knowledge, or the safety rule) — `meta.answer_source`, surfaced in the UI as a small collapsible "where did this come from?" disclosure so a user can tell official Samsung guidance from an AI's best guess |
| `related_issues.py` | When a plan's own confidence was borderline enough to already carry an `escalation` object, suggests 2-3 other plausible root causes for the same symptom (a hand-curated map keyed on `pipeline.py`'s existing symptom buckets) — `meta.related_possibilities`, empty whenever the engine is confident |
| `resolution.py` | The goal-level "did this fix it?" Yes/No signal (`POST /v1/resolution`) — distinct from `feedback.py`'s per-deeplink thumbs up/down; fans a "No" out into that SAME per-deeplink scoring/session-avoidance machinery rather than duplicating it |
| `image_analysis.py` | Point 1: turns an optionally-attached photo into a short factual description (never a diagnosis) that `pipeline.py` folds into the complaint text before Stage 0 — degrades honestly (`meta.image_analysis.reason`) when no key is configured, the model isn't vision-capable, or the call fails |
| `ambiguity.py` | CLAM-inspired second opinion on `clarify.py`'s heuristic — an LLM-based confidence score that can only RAISE `meta.needs_clarification`, never lower it; a complete no-op with no LLM key |
| `recovery.py` | Guided-Retry-inspired ("When the Database Fails") structured recovery when Stage 1's LLM call actually fails — one retry with an explicit recovery instruction before falling through to the offline path — `meta.recovery` |
| `investigator.py` | SIA-inspired interactive diagnosis (`POST /v1/investigate/start`/`/answer`) — a probability vector over candidate root causes (reusing `related_issues.py`'s curated data), narrowed down via targeted questions, then handed to the real `run_pipeline()` |
| `topic_manager.py` | DiagGPT-inspired per-session topic stack (stay / create / jump), deterministic bucket matching instead of DiagGPT's own per-turn LLM agent — `meta.topic_stack` |
| `report.py` | Packages an already-computed result into a shareable Markdown/HTML report (`POST /v1/report`) — a pure formatter, never re-runs the pipeline |
| `pipeline.py` | Orchestrates: (optional image →) complaint → enrich → extract → validate → deeplink match → cache, choosing LLM vs offline path per-request |
| `cache.py` | Fast-path semantic cache (keyword overlap) |
| `main.py` | FastAPI app: `POST /v1/troubleshoot(/batch)`, `GET /v1/troubleshoot/stream` (SSE), `POST /v1/feedback`, `POST /v1/resolution`, `POST /v1/report`, `POST /v1/investigate/start`, `POST /v1/investigate/answer`, `GET /v1/investigate/{id}`, `GET /health`, `GET /stats` — warms the deeplink index at startup |
| `index.html` | Polished demo UI — voice input + output, language selector, device-state panel, photo attach with client-side downscale, "where did this come from?" disclosure, related-possibilities chips, a "did this fix it?" resolution prompt, session topic-stack chips, a guided-diagnosis (investigator) widget, shareable reports, live analytics dashboard with trending issues |
| `demo.html` | Simpler legacy demo UI |
| `deeplinks.json` | **Real official catalog**: 578 masked deeplinks across the full Settings surface |
| `siis_responses.json` | **Real official data**: 20 official SIIS support queries + their raw reference text |
| `sample_queries_real.json` | The 20 official queries adapted into this project's `{domain, complaint, siis_response}` shape — what `pipeline.py` and `eval/eval_harness.py` actually run against |
| `sample_queries.json` | Original placeholder queries (Battery/Camera/Display/Performance), kept for quick manual smoke tests only |
| `deeplinks_placeholder_backup.json`, `deeplinks_expanded_placeholder_backup.json` | Original placeholder catalogs, kept for reference |
| `official_theme2_data/` | Untouched copy of the official team-kit dataset (deeplinks.json, siis_responses.json, sample_output.json, schema_reference.py) |
| `eval/eval_harness.py` | Runs the full pipeline against the 20 real queries, produces `eval/metrics.md` |
| `eval/run_ablation.py` + `eval/matchers.py` | 3-variant deeplink-matching ablation (Full-LLM / Hybrid BM25+dense / Pure rules) against `eval/deeplink_ground_truth.json` (real catalog entries) |
| `eval/generate_results_jsonl.py` → `results.jsonl` | The offline results file the Theme 2 FAQ asks for alongside the live API: one JSON line per official query (`{"query", "query_variations", "response", "meta"}`), covering all 20/20. Each line is a direct serialization of `run_pipeline()`'s real output — never separately-derived — so it can't drift from what the live API actually returns for the same input. Re-run after any change that could affect the 20 official queries' output. |
| `request_log.py` | Per-request JSONL log, powers `/stats` |
| `feedback.py` | Human-in-the-loop feedback (`POST /v1/feedback`) + the bounded per-deeplink score adjustment that `deeplink_matching.py` consults on every search |
| `middleware.py` | Request IDs, structured `{"error": {...}}` bodies, and per-route in-memory rate limiting |
| `cli.py` | Terminal client (`query` / `stream` / `batch` / `feedback` / `report` / `health` / `stats`) -- runs the pipeline in-process by default, or against a live server with `--api-base` |
| `tests/` | pytest suite, ~97% line coverage, zero LLM key required — see "Run the tests" above |

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

Every `actionableDeeplink` in a response (real match or the placeholder)
also carries a `matchExplanation` object — the score breakdown behind
*why* that specific screen was picked, not just that it was:

```json
"matchExplanation": {
  "matcher": "hybrid_bm25_dense",
  "dense_kind": "tfidf-svd-128 (offline fallback; sentence-transformers unavailable)",
  "alpha": 0.5,
  "bm25_component": 1.0,
  "dense_component": 0.8466,
  "combined_before_feedback": 0.9233,
  "feedback_adjustment": 0.0,
  "final_score": 0.9233,
  "threshold": 0.12,
  "matched_keywords": ["data", "fi", "mobile", "stable", "wi"]
}
```

The rules-variant matcher (`RulesDeeplinkIndex`) reports the analogous
`fuzzy_score` / `feedback_adjustment_pts` / `final_score_pts` shape
instead. This is genuine introspection into the actual ranking that ran
— not a post-hoc guess — so it's exactly as trustworthy as the match
itself, and it's the same code path `deeplink_matching.best_match()`
uses internally (`best_match_explained()` is a superset, verified in
`tests/test_deeplink_matching.py::TestMatchExplainability` to never
disagree with `best_match()` on which entry wins). `index.html` renders
it as a collapsed "why this match?" disclosure under each resolved
deeplink, with the overlapping keywords as chips.

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
the offline fallback's narrow Hindi/Hinglish phrasebook (helps the
common code-mixed case, isn't general multi-language support -- the LLM
path is still the robust option for that), why this sandbox's ablation
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

A few more, specific to this batch's 4 research-inspired extensions:

- **`ambiguity.py`'s CLAM-style confidence is a self-reported score, not a
  literal log-probability.** The paper's own method reads the model's
  log-prob on a "True"/"False" token; none of this project's provider SDKs
  (Anthropic/OpenAI/Groq chat-completions) expose per-token log-probs
  without switching to a completions-style API this codebase doesn't use
  elsewhere, so a structured few-shot self-report is used instead —
  documented as exactly that in the module docstring, not dressed up as
  the paper's literal method. Fully inert with no LLM key configured.
- **`investigator.py`'s offline mode has a fixed, 3-question generic
  question pool** (`_GENERIC_DISCRIMINATIVE_QUESTIONS`) rather than
  genuinely tailored per-symptom questions — an LLM key upgrades this to
  real per-turn generated questions. The offline probability update is a
  simple keyword-bucket heuristic (software/hardware/external), not a
  learned or LLM-driven Bayesian update — see the module docstring for the
  full honest mapping to the paper's method.
- **`investigator.py`'s session state is in-process only**, same
  documented tradeoff as `middleware.py`'s rate limiter — an investigation
  in progress is lost on a server restart. Fine for a short, few-round-trip
  interaction in a hackathon-scale demo; a real deployment would move this
  to a shared store (Redis, same as the rate limiter's own noted upgrade
  path).
- **`recovery.py`'s Guided-Retry only covers Stage 1's LLM call**, not
  Stage 0's enrichment call — this project has no live backend database to
  inject the paper's specific empty-result/wrong-domain fault types into,
  so the scope is narrowed to the one place an actual LLM-call failure can
  happen and be usefully retried; see the module's own scope note.
- **`topic_manager.py`'s topic identity is `pipeline.py`'s existing
  ~17-bucket keyword vocabulary**, not a genuine semantic topic model — two
  complaints that are really about different things but share a bucket
  (or vice versa) can be mis-tracked. Same tradeoff this codebase already
  makes everywhere else it uses that bucket vocabulary (`/stats`'s
  trending issues, `related_issues.py`).




in this readme at the beginning add this drive link and mention it as "drive link as video and ppt link" GenAI - Google Drive 
don't change anything else
