# Smart Guided Troubleshooting Engine

Samsung PRISM GenAI Hackathon 3rd Edition — Theme 2

Transforms a vague Galaxy device complaint ("screen flickers and battery dies
fast") into a validated, deeplink-mapped, one-tap troubleshooting plan.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your real LLM_API_KEY
```

## Run the pipeline directly (no server needed, fastest way to test)

```bash
python pipeline.py
```

This runs the first 3 complaints from `sample_queries.json` through the full
pipeline and prints the JSON output — good for quickly checking prompt changes.

## Run the API server

```bash
uvicorn main:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/v1/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"query": "screen flickers and battery dies fast"}'

curl http://localhost:8000/health
```

## Project structure

| File | Purpose |
|---|---|
| `schema.py` | Pydantic models — exact data contract from the theme guide |
| `validators.py` | Programmatic checks (word counts, URL scrubbing, category ordering) |
| `prompts.py` | All LLM prompts (Stage 0 enrichment, Stage 1 extraction) |
| `llm_client.py` | Swappable LLM API wrapper (Anthropic / OpenAI) |
| `pipeline.py` | Orchestrates: complaint → enrich → extract → validate → deeplink match → cache |
| `cache.py` | Fast-path semantic-ish cache (keyword overlap; upgradeable to embeddings) |
| `main.py` | FastAPI app: `POST /v1/troubleshoot`, `GET /health` |
| `deeplinks.json` | Sample catalog of ~15 masked deeplinks across Battery/Display/Camera/Performance — **expand this** |
| `sample_queries.json` | Test complaints across all 4 domains + edge cases (multi-issue, typo, no-match) |

## Known limitations (fill in as you find more)

- Deeplink matching uses simple string-similarity, not embeddings — upgrade
  `cache.py`/`pipeline.py` `_match_score()` with `sentence-transformers` if time allows.
- Cost tracking (`cost_usd`) is not yet computed from real token usage.
- `deeplinks.json` currently has ~15 sample entries; expand toward realistic
  coverage of all 4 domains before final submission.
