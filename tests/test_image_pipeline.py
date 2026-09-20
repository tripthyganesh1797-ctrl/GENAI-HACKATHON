"""tests/test_image_pipeline.py -- pipeline.py's wiring of point 1 (picture
upload): image_data_url must be a complete no-op when absent, must fold a
successfully-analyzed image's description into the complaint text BEFORE
Stage 0 (so it can influence safety/clarify/technical_query, not just be
decoration), must degrade honestly when analysis fails, and must never
leak into response["query"] (which should always echo exactly what the
user typed)."""
import pipeline

FAKE_IMAGE_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


class TestNoImageIsANoOp:
    def test_meta_shape_present_but_not_provided(self):
        result = pipeline.run_pipeline("my battery is draining too fast", "")
        assert result["meta"]["image_analysis"] == {
            "provided": False, "analyzed": False, "description": None, "reason": None,
        }

    def test_safety_short_circuit_also_carries_the_field(self):
        result = pipeline.run_pipeline("my phone is smoking", "")
        assert result["meta"]["image_analysis"]["provided"] is False

    def test_streaming_never_accepts_images_but_still_has_the_field(self):
        events = list(pipeline.run_pipeline_streaming("my battery is draining too fast", ""))
        final = events[-1]["data"]
        assert final["meta"]["image_analysis"] == {
            "provided": False, "analyzed": False, "description": None, "reason": None,
        }


class TestImageFoldedIntoComplaint:
    def test_analyzed_description_feeds_stage0_and_technical_query_changes(self, monkeypatch):
        """Stage 0 must see the image-informed text, not just the raw
        complaint -- proven here by a fake stage0_enrich that echoes back
        whatever text it actually received."""
        captured = {}

        def fake_describe_image(url):
            return {"provided": True, "analyzed": True,
                    "description": "a cracked screen with visible spiderweb pattern", "reason": None}

        def fake_stage0(text):
            captured["text"] = text
            return {"technical_query": text, "query_variations": []}, True

        monkeypatch.setattr(pipeline, "describe_image", fake_describe_image)
        monkeypatch.setattr(pipeline, "stage0_enrich", fake_stage0)

        result = pipeline.run_pipeline("my screen looks weird", "", image_data_url=FAKE_IMAGE_DATA_URL)

        assert "cracked screen" in captured["text"]
        assert "my screen looks weird" in captured["text"]
        # response["query"] must still be exactly what the user typed --
        # the image description is an internal signal, not user-facing text.
        assert result["query"] == "my screen looks weird"
        assert result["meta"]["image_analysis"]["analyzed"] is True

    def test_image_can_trigger_safety_even_when_text_alone_would_not(self, monkeypatch):
        """A photo showing something hazardous (smoke, a swollen battery)
        must be able to trip safety.py's hazard check even if the typed
        text is innocuous on its own -- proves the image is folded in
        BEFORE the hazard check, not after."""
        def fake_describe_image(url):
            return {"provided": True, "analyzed": True,
                    "description": "visible smoke coming from the back of the device", "reason": None}

        monkeypatch.setattr(pipeline, "describe_image", fake_describe_image)

        result = pipeline.run_pipeline("it's acting weird", "", image_data_url=FAKE_IMAGE_DATA_URL)
        assert result["meta"]["safety_alert"] is True
        assert result["meta"]["image_analysis"]["analyzed"] is True
        # response["query"] is still the original innocuous text.
        assert result["query"] == "it's acting weird"


class TestImageAnalysisFailureDegradesGracefully:
    def test_failed_analysis_does_not_change_behavior(self, monkeypatch):
        """When image analysis fails (no key, wrong model, provider error),
        the request must proceed exactly as if no image had been sent --
        never crash, never silently drop the rest of the request."""
        def fake_describe_image(url):
            return {"provided": True, "analyzed": False, "description": None,
                    "reason": "no LLM API key configured -- image was skipped, text-only analysis still ran"}

        monkeypatch.setattr(pipeline, "describe_image", fake_describe_image)

        result = pipeline.run_pipeline("my battery is draining too fast", "", image_data_url=FAKE_IMAGE_DATA_URL)
        assert result["meta"]["image_analysis"]["provided"] is True
        assert result["meta"]["image_analysis"]["analyzed"] is False
        assert result["response"]["contexts"]  # plan still comes back


class TestCacheHitRecomputesImageAnalysisFresh:
    def test_second_request_with_different_image_gets_its_own_result(self, monkeypatch):
        """image_analysis is request-specific (like needs_clarification) --
        a cache hit must reflect THIS request's own image result, never a
        stale value baked in by whichever request first wrote the cache
        entry. stage0_enrich is pinned to a fixed technical_query here
        (same isolation technique used elsewhere in this suite) so the
        test controls cache-hit/miss directly rather than depending on
        exactly how image text happens to affect normalize_query()."""
        calls = {"n": 0}

        def fake_describe_image(url):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"provided": False, "analyzed": False, "description": None, "reason": None}
            return {"provided": True, "analyzed": True, "description": "some new visible detail", "reason": None}

        def fake_stage0(text):
            return {"technical_query": "pinned wifi technical query", "query_variations": []}, True

        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Wifi Troubleshooting",
                "title": "Wifi issue",
                "score": 0.9,
                "actions": [],
            }
            return {"contexts": [goal]}, True  # a real match, so it actually gets cached

        monkeypatch.setattr(pipeline, "describe_image", fake_describe_image)
        monkeypatch.setattr(pipeline, "stage0_enrich", fake_stage0)
        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)

        first = pipeline.run_pipeline("wifi keeps dropping randomly, cache image test", "")
        assert first["meta"]["cache_hit"] is False
        assert first["meta"]["image_analysis"]["provided"] is False

        second = pipeline.run_pipeline("wifi keeps dropping randomly, cache image test", "",
                                        image_data_url=FAKE_IMAGE_DATA_URL)
        assert second["meta"]["cache_hit"] is True
        assert second["meta"]["image_analysis"]["provided"] is True
        assert second["meta"]["image_analysis"]["description"] == "some new visible detail"
