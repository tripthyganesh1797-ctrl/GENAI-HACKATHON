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
```

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
machine with normal internet access to get the true comparison), and the
one official query/SIIS-text pairing that's a genuine grounding mismatch
by design (correctly triggers `no_match` rather than hallucinating steps).
