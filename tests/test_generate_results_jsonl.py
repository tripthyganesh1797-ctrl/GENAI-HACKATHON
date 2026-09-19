"""tests/test_generate_results_jsonl.py -- covers eval/generate_results_jsonl.py,
which produces the results.jsonl offline results file the Theme 2 FAQ (Q17,
Q2, gate G3) asks for alongside the live API. Runs the real script against
the real 20 official queries (no mocking -- this IS the submission
artifact), so this doubles as an end-to-end regression test that the file
stays well-formed as the rest of the pipeline changes."""
import importlib
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "eval"))


def _run_generator():
    import generate_results_jsonl as gen
    importlib.reload(gen)  # pick up any monkeypatched QUERY_SET/paths cleanly
    gen.main()
    return gen.OUT_PATH


class TestGenerateResultsJsonl:
    def test_writes_one_line_per_official_query(self):
        out_path = _run_generator()
        with open(out_path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        with open(os.path.join(REPO_ROOT, "sample_queries_real.json")) as f:
            query_set = json.load(f)
        assert len(lines) == len(query_set) == 20

    def test_every_line_is_valid_json_with_the_required_keys(self):
        out_path = _run_generator()
        with open(out_path) as f:
            for line in f.read().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)  # raises if not valid JSON
                assert set(["query", "query_variations", "response"]) <= set(obj.keys())
                assert isinstance(obj["query"], str) and obj["query"]
                assert isinstance(obj["query_variations"], list)
                assert "contexts" in obj["response"]

    def test_query_variations_are_within_the_8_to_10_spec_range(self):
        out_path = _run_generator()
        with open(out_path) as f:
            for line in f.read().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                assert 8 <= len(obj["query_variations"]) <= 10

    def test_all_20_official_queries_produce_a_non_empty_response(self):
        """The 20 official queries all come paired with real grounding
        text, so none of them should legitimately come back no_match."""
        out_path = _run_generator()
        with open(out_path) as f:
            for line in f.read().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                assert obj["response"]["contexts"], obj["query"]

    def test_output_matches_what_run_pipeline_actually_returns(self):
        """results.jsonl must never drift from live API behaviour -- each
        line is a direct serialization of run_pipeline()'s own output,
        not separately-derived logic."""
        import pipeline
        with open(os.path.join(REPO_ROOT, "sample_queries_real.json")) as f:
            case = json.load(f)[0]
        live = pipeline.run_pipeline(case["complaint"], case.get("siis_response", ""))

        out_path = _run_generator()
        with open(out_path) as f:
            first = json.loads(f.readline())
        assert first["query"] == live["query"] == case["complaint"]
        assert first["response"]["contexts"][0]["title"] == live["response"]["contexts"][0]["title"]
