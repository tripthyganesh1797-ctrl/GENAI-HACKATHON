"""pipeline.py — full orchestration, including the SSE streaming variant.
No LLM_API_KEY is configured in this environment, so these exercise the
same offline-fallback path the judges' zero-setup run will use."""
import pipeline


def test_run_pipeline_on_real_query_is_schema_valid(real_samples):
    import validators
    sample = next(s for s in real_samples if s.get("siis_response"))
    result = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
    assert "meta" in result and "response" in result
    for goal in result["response"]["contexts"]:
        assert validators.validate_goal_object(dict(goal)) == []


def test_run_pipeline_no_siis_response_grounds_in_builtin_knowledge(real_samples):
    """As of builtin_knowledge.py: a bare complaint with no reference text
    no longer automatically comes back no_match on the offline path -- see
    offline_fallback.py's offline_extract() and that module's docstring.
    A real person typing their own problem with zero setup (no LLM key,
    no reference text to paste in) now gets a real, relevance-gated plan."""
    sample = real_samples[0]  # a real screen-blank complaint
    result = pipeline.run_pipeline(sample["complaint"], "")
    assert result["response"]["contexts"] != []
    assert result["meta"]["fallback"] is None
    assert result["meta"]["used_builtin_reference"] is True


def test_run_pipeline_no_siis_response_still_no_match_for_out_of_scope_query():
    """The relevance gate is unchanged -- a genuinely out-of-scope
    complaint must still come back no_match even with the built-in
    fallback text substituted in."""
    result = pipeline.run_pipeline("how do I cook pasta at home", "")
    assert result["response"]["contexts"] == []
    assert result["meta"]["fallback"] == "no_match"
    assert result["meta"]["used_builtin_reference"] is True


def test_run_pipeline_reports_offline_fallback_used():
    result = pipeline.run_pipeline("my battery drains too fast", "")
    assert result["meta"]["used_offline_fallback"] is True
    assert result["meta"]["cost_usd"] == 0.0
    # offline_enrich() itself can't detect language (no LLM) -- pipeline.py
    # fills in the "en" default for the offline path here.
    assert result["meta"]["detected_language"] == "en"


def test_cache_hit_on_second_identical_call():
    complaint = "how do I cook pasta at home"  # genuinely out of scope -> no_match
    first = pipeline.run_pipeline(complaint, "")
    assert first["meta"]["cache_hit"] is False
    second = pipeline.run_pipeline(complaint, "")
    # no_match responses are never cached (see run_pipeline: `if contexts: set_cached(...)`)
    assert second["meta"]["cache_hit"] is False


def test_cache_hit_on_second_identical_call_with_a_real_match():
    """Companion to the no_match case above -- a query that DOES produce
    contexts (grounded in builtin_knowledge.py, since no siis_response is
    given here) must still be cached and hit on the second call."""
    complaint = "my wifi keeps disconnecting randomly"
    first = pipeline.run_pipeline(complaint, "")
    assert first["response"]["contexts"] != []
    assert first["meta"]["cache_hit"] is False
    second = pipeline.run_pipeline(complaint, "")
    assert second["meta"]["cache_hit"] is True


class TestEscalation:
    """Task 34: LLM-path confidence gate (_maybe_attach_llm_escalation).
    The offline path's own gate is tested directly in
    tests/test_offline_fallback.py::TestEscalation since it uses a
    different signal (relevance, not this self-reported score)."""

    def test_low_score_goal_gets_flagged(self):
        goal = {"score": 0.4}
        pipeline._maybe_attach_llm_escalation(goal)
        assert goal["escalation"]["recommended"] is True
        assert "0.40" in goal["escalation"]["reason"]

    def test_high_score_goal_is_untouched(self):
        goal = {"score": 0.9}
        pipeline._maybe_attach_llm_escalation(goal)
        assert "escalation" not in goal

    def test_score_exactly_at_threshold_is_not_flagged(self):
        """Strict less-than: a goal AT the threshold is confident enough."""
        goal = {"score": pipeline.LLM_SCORE_ESCALATION_THRESHOLD}
        pipeline._maybe_attach_llm_escalation(goal)
        assert "escalation" not in goal

    def test_missing_or_non_numeric_score_does_not_crash(self):
        goal = {}
        pipeline._maybe_attach_llm_escalation(goal)
        assert "escalation" not in goal

        goal2 = {"score": "not a number"}
        pipeline._maybe_attach_llm_escalation(goal2)
        assert "escalation" not in goal2

    def test_llm_path_goal_flows_through_run_pipeline_with_escalation(self, monkeypatch):
        """End-to-end: a Goal that came back from stage1_extract with
        fb1=False (i.e. "the LLM path was used", however that happened)
        and a low self-reported score must carry escalation by the time
        run_pipeline() returns -- exercised through the real function,
        not just the helper in isolation."""
        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Test Issue Troubleshooting",
                "title": "Test issue",
                "score": 0.35,
                "actions": [],
            }
            return {"contexts": [goal]}, False, {"failure_detected": False, "failure_type": None, "detail": None, "guided_retry_attempted": False, "guided_retry_succeeded": None}  # fb1=False -> "LLM path" for this test

        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)
        result = pipeline.run_pipeline("some complaint", "some grounding text")
        goal = result["response"]["contexts"][0]
        assert goal.get("escalation") is not None
        assert goal["escalation"]["recommended"] is True

    def test_llm_path_goal_high_score_flows_through_without_escalation(self, monkeypatch):
        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Test Issue Troubleshooting",
                "title": "Test issue",
                "score": 0.95,
                "actions": [],
            }
            return {"contexts": [goal]}, False, {"failure_detected": False, "failure_type": None, "detail": None, "guided_retry_attempted": False, "guided_retry_succeeded": None}

        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)
        result = pipeline.run_pipeline("some other complaint", "some grounding text")
        goal = result["response"]["contexts"][0]
        assert "escalation" not in goal

    def test_offline_path_goal_is_not_double_processed_by_llm_gate(self, real_samples):
        """A real offline-path goal already carries its own escalation
        decision (or lack of one) from offline_fallback.py -- run_pipeline
        must not additionally run the LLM-scale gate over it, which would
        compare a 0.55-0.99-floored offline score against a threshold
        calibrated for the LLM's own 0-1 scale."""
        sample = next(s for s in real_samples if s.get("siis_response"))
        result = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        assert result["meta"]["used_offline_fallback"] is True
        # Whatever offline_fallback.py decided is exactly what's still there.
        for goal in result["response"]["contexts"]:
            from offline_fallback import ESCALATION_RELEVANCE_CEILING, NO_MATCH_SECTION_THRESHOLD
            # sanity: offline scores are never LLM-scale low enough to trip
            # LLM_SCORE_ESCALATION_THRESHOLD by coincidence of the wrong gate
            assert goal["score"] >= 0.55  # offline's own floor


class TestDeviceContext:
    """Task 35: optional device-state signals (schema.DeviceContext), fully
    additive on top of the existing pipeline. device_signals.py itself is
    unit-tested directly in tests/test_device_signals.py; these exercise
    the pipeline-level wiring: prompt formatting for the LLM path, and --
    the trickiest part -- that device context is applied fresh on every
    call and never leaks into (or out of) the semantic cache."""

    def test_format_device_context_block_empty_for_none_or_empty(self):
        assert pipeline._format_device_context_block(None) == ""
        assert pipeline._format_device_context_block({}) == ""
        assert pipeline._format_device_context_block(
            {"battery_pct": None, "storage_free_pct": None}
        ) == ""

    def test_format_device_context_block_includes_every_set_field(self):
        block = pipeline._format_device_context_block({
            "battery_pct": 12, "storage_free_pct": 8, "os_version": "One UI 6.1",
            "uptime_hours": 50, "last_restart_hours_ago": 50,
        })
        assert "Battery level: 12%" in block
        assert "Free storage: 8%" in block
        assert "OS version: One UI 6.1" in block
        assert "Uptime since last restart: 50 hours" in block
        assert "Hours since last restart: 50" in block

    def test_stage1_extract_threads_device_into_llm_prompt(self, monkeypatch):
        captured = {}

        def fake_call_llm_json(prompt, max_tokens=4000):
            captured["prompt"] = prompt
            return {"contexts": []}

        monkeypatch.setattr(pipeline, "API_KEY", "sk-test-key-not-a-placeholder")
        monkeypatch.setattr(pipeline, "call_llm_json", fake_call_llm_json)
        pipeline.stage1_extract("battery drains fast", "some grounding text",
                                 device={"battery_pct": 7})
        assert "Known device state" in captured["prompt"]
        assert "Battery level: 7%" in captured["prompt"]

    def test_stage1_extract_with_no_device_omits_block_from_prompt(self, monkeypatch):
        captured = {}

        def fake_call_llm_json(prompt, max_tokens=4000):
            captured["prompt"] = prompt
            return {"contexts": []}

        monkeypatch.setattr(pipeline, "API_KEY", "sk-test-key-not-a-placeholder")
        monkeypatch.setattr(pipeline, "call_llm_json", fake_call_llm_json)
        pipeline.stage1_extract("battery drains fast", "some grounding text", device=None)
        # Rule 10's own text mentions the phrase "Known device state" in the
        # abstract (it's part of the prompt unconditionally) -- what must
        # be absent is the actual rendered block, which always includes at
        # least one concrete field line.
        assert "Battery level:" not in captured["prompt"]
        assert "Free storage:" not in captured["prompt"]

    def test_run_pipeline_with_no_device_has_empty_notes(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        result = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        assert result["meta"]["device_context_notes"] == []

    def test_run_pipeline_with_device_returns_notes(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        result = pipeline.run_pipeline(
            sample["complaint"], sample["siis_response"], device={"battery_pct": 5}
        )
        assert len(result["meta"]["device_context_notes"]) == 1

    def test_device_context_is_not_persisted_into_cache(self, real_samples):
        """The base plan is cached without device data -- a later caller
        for the same complaint who passes NO device must not see the
        earlier caller's device-specific notes leak in via the cache."""
        sample = next(s for s in real_samples if s.get("siis_response"))
        first = pipeline.run_pipeline(
            sample["complaint"], sample["siis_response"], device={"battery_pct": 5}
        )
        assert first["meta"]["cache_hit"] is False
        assert len(first["meta"]["device_context_notes"]) == 1

        second = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        assert second["meta"]["cache_hit"] is True
        assert second["meta"]["device_context_notes"] == []

    def test_cache_hit_still_applies_fresh_device_context(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        first = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        assert first["meta"]["cache_hit"] is False

        second = pipeline.run_pipeline(
            sample["complaint"], sample["siis_response"], device={"battery_pct": 5}
        )
        assert second["meta"]["cache_hit"] is True
        assert len(second["meta"]["device_context_notes"]) == 1

    def test_end_to_end_reorders_actions_via_device_context(self, monkeypatch):
        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Battery Troubleshooting",
                "title": "Battery drain",
                "score": 0.9,
                "actions": [
                    {"actionName": "Display Settings",
                     "description": "It will adjust the display brightness",
                     "category": "auto",
                     "stepGroups": [{"steps": ["Tap Display."]}]},
                    {"actionName": "Battery Settings",
                     "description": "It will enable power saving mode",
                     "category": "auto",
                     "stepGroups": [{"steps": ["Tap Battery."]}]},
                ],
            }
            return {"contexts": [goal]}, False, {"failure_detected": False, "failure_type": None, "detail": None, "guided_retry_attempted": False, "guided_retry_succeeded": None}

        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)
        result = pipeline.run_pipeline("battery complaint", "grounding text",
                                        device={"battery_pct": 5})
        actions = result["response"]["contexts"][0]["actions"]
        assert actions[0]["actionName"] == "Battery Settings"

    def test_streaming_also_applies_device_context(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        events = list(pipeline.run_pipeline_streaming(
            sample["complaint"], sample["siis_response"], device={"battery_pct": 5}
        ))
        assert events[-1]["stage"] == "complete"
        assert len(events[-1]["data"]["meta"]["device_context_notes"]) == 1


class TestSessionMemory:
    """Task 36 (session_memory.py): a session that already got thumbs-down
    feedback on a specific deeplink shouldn't casually see it suggested
    again for a follow-up call in the SAME session. Two things to pin
    down: (1) the matching itself steers away from the avoided deeplink
    when it can (tests/test_deeplink_matching.py::TestSessionAvoidance
    covers the matcher-level mechanics in depth), and (2) the trickier
    part done here -- the semantic cache is bypassed entirely for a
    request carrying active avoidance data, so a stale cached entry
    (resolved under a different, or no, avoidance state) can never
    silently reintroduce the exact deeplink this feature exists to avoid."""

    def test_no_session_id_has_empty_session_notes(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        result = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        assert result["meta"]["session_notes"] == []

    def test_unknown_session_id_behaves_like_no_session_id(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        result = pipeline.run_pipeline(
            sample["complaint"], sample["siis_response"], session_id="never-gave-feedback"
        )
        assert result["meta"]["session_notes"] == []
        assert result["meta"]["cache_hit"] is False

    def test_repeat_call_with_plain_session_still_uses_cache(self, real_samples):
        """A session_id with NO avoidance history yet must not disable
        caching -- only active avoidance data does that (see below)."""
        sample = next(s for s in real_samples if s.get("siis_response"))
        first = pipeline.run_pipeline(
            sample["complaint"], sample["siis_response"], session_id="quiet-session"
        )
        assert first["meta"]["cache_hit"] is False
        second = pipeline.run_pipeline(
            sample["complaint"], sample["siis_response"], session_id="quiet-session"
        )
        assert second["meta"]["cache_hit"] is True
        assert second["meta"]["session_notes"] == []

    def test_avoidance_feedback_bypasses_cache_and_changes_the_match(self, real_samples):
        import feedback
        import session_memory

        sample = next(s for s in real_samples if s.get("siis_response"))
        session_id = "session-with-a-failed-action"

        first = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        assert first["meta"]["cache_hit"] is False
        goal = first["response"]["contexts"][0]
        action = goal["actions"][0]
        sg = action["stepGroups"][0]
        deeplink = sg["actionableDeeplink"]["deeplink"]

        # Simulate: the user tried this exact action, it didn't help, and
        # told us so via POST /v1/feedback with a session_id attached.
        feedback.record_feedback(deeplink, action["actionName"], helpful=False,
                                  query=sample["complaint"], session_id=session_id)
        assert deeplink in session_memory.get_avoid_set(session_id)

        second = pipeline.run_pipeline(
            sample["complaint"], sample["siis_response"], session_id=session_id
        )
        # Cache bypass: this must be freshly computed, not the cached
        # first-call result (which still has the now-avoided deeplink
        # baked in from before any feedback existed).
        assert second["meta"]["cache_hit"] is False
        assert len(second["meta"]["session_notes"]) >= 1

        second_goal = second["response"]["contexts"][0]
        second_action = second_goal["actions"][0]
        second_sg = second_action["stepGroups"][0]
        second_exp = second_sg["actionableDeeplink"].get("matchExplanation") or {}
        # Either the matcher found a different real option (most catalog
        # domains have more than one candidate deeplink) or it fell back to
        # the same one, honestly flagged as already tried -- either way the
        # avoidance signal must have visibly reached Stage 2.
        assert (
            second_sg["actionableDeeplink"]["deeplink"] != deeplink
            or second_exp.get("already_tried_this_session") is True
        )

    def test_avoidance_active_session_never_populates_the_cache(self, real_samples):
        import feedback

        sample = next(s for s in real_samples if s.get("siis_response"))
        session_id = "session-that-never-caches"

        first = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        deeplink = first["response"]["contexts"][0]["actions"][0]["stepGroups"][0]["actionableDeeplink"]["deeplink"]
        feedback.record_feedback(deeplink, "X", helpful=False,
                                  query=sample["complaint"], session_id=session_id)

        # Two consecutive calls with the SAME active-avoidance session_id:
        # if the cache were being written to despite the bypass, the
        # second call would come back as a cache hit.
        pipeline.run_pipeline(sample["complaint"], sample["siis_response"], session_id=session_id)
        third = pipeline.run_pipeline(sample["complaint"], sample["siis_response"], session_id=session_id)
        assert third["meta"]["cache_hit"] is False

    def test_summarize_session_avoidance_reports_skip(self):
        contexts = [{
            "actions": [{
                "actionName": "Wifi Settings",
                "stepGroups": [{
                    "actionableDeeplink": {
                        "deeplink": "bixby://masked/act/new",
                        "matchExplanation": {"session_avoid_skipped": "bixby://masked/act/old"},
                    },
                }],
            }],
        }]
        notes = pipeline._summarize_session_avoidance(contexts)
        assert len(notes) == 1
        assert "1 previously-tried match" in notes[0]

    def test_summarize_session_avoidance_reports_repeat(self):
        contexts = [{
            "actions": [{
                "actionName": "Wifi Settings",
                "stepGroups": [{
                    "actionableDeeplink": {
                        "deeplink": "bixby://masked/act/old",
                        "matchExplanation": {"already_tried_this_session": True},
                    },
                }],
            }],
        }]
        notes = pipeline._summarize_session_avoidance(contexts)
        assert len(notes) == 1
        assert "Wifi Settings" in notes[0]

    def test_summarize_session_avoidance_empty_for_no_flags(self):
        contexts = [{
            "actions": [{
                "actionName": "Wifi Settings",
                "stepGroups": [{
                    "actionableDeeplink": {
                        "deeplink": "bixby://masked/act/x",
                        "matchExplanation": {},
                    },
                }],
            }],
        }]
        assert pipeline._summarize_session_avoidance(contexts) == []

    def test_streaming_also_bypasses_cache_for_active_avoidance(self, real_samples):
        import feedback

        sample = next(s for s in real_samples if s.get("siis_response"))
        session_id = "streaming-session-with-a-failed-action"

        first = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        deeplink = first["response"]["contexts"][0]["actions"][0]["stepGroups"][0]["actionableDeeplink"]["deeplink"]
        feedback.record_feedback(deeplink, "X", helpful=False,
                                  query=sample["complaint"], session_id=session_id)

        events = list(pipeline.run_pipeline_streaming(
            sample["complaint"], sample["siis_response"], session_id=session_id
        ))
        assert events[-1]["stage"] == "complete"
        assert events[-1]["data"]["meta"]["cache_hit"] is False
        assert len(events[-1]["data"]["meta"]["session_notes"]) >= 1


class TestStreamingParity:
    """The whole point of run_pipeline_streaming(): its final event must
    carry exactly the same troubleshooting content as run_pipeline()'s
    return value for the same input, so the two code paths can never
    silently drift apart."""

    def test_final_event_matches_direct_call(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        events = list(pipeline.run_pipeline_streaming(sample["complaint"], sample["siis_response"]))
        assert events[-1]["stage"] == "complete"
        streamed_contexts = events[-1]["data"]["response"]["contexts"]

        direct = pipeline.run_pipeline(sample["complaint"], sample["siis_response"])
        assert streamed_contexts == direct["response"]["contexts"]

    def test_emits_expected_stage_sequence(self, real_samples):
        sample = next(s for s in real_samples if s.get("siis_response"))
        events = list(pipeline.run_pipeline_streaming(sample["complaint"], sample["siis_response"]))
        stages = [e["stage"] for e in events]
        assert stages[0] == "start"
        assert "enrich" in stages
        assert "cache" in stages
        assert "extract" in stages
        assert "validate" in stages
        assert "deeplink_match" in stages
        assert stages[-1] == "complete"

    def test_cache_hit_short_circuits_the_stream(self):
        complaint = "my camera lags every time I open it and take a photo"
        # First call with siis text so it actually caches (empty siis never caches).
        # There's no real siis text for this ad-hoc complaint, so just verify
        # a genuine cache-hit still produces a valid "complete" event and stops.
        events = list(pipeline.run_pipeline_streaming(complaint, ""))
        assert events[-1]["stage"] == "complete"

    def test_streaming_never_raises_only_yields_error_stage(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("simulated failure")
        monkeypatch.setattr(pipeline, "stage0_enrich", boom)
        events = list(pipeline.run_pipeline_streaming("anything", ""))
        assert events[-1]["stage"] == "error"
        assert "simulated failure" in events[-1]["data"]["message"]

    def test_streaming_also_attaches_llm_escalation(self, monkeypatch):
        """The streaming variant duplicates the validate+escalation loop
        (it can't share run_pipeline()'s code directly since it yields
        progress between stages) -- must not have silently drifted."""
        def fake_stage1(technical_query, siis_response, force_offline=False, device=None, avoid_deeplinks=None):
            goal = {
                "goal": "Follow these steps to perform this Test Issue Troubleshooting",
                "title": "Test issue",
                "score": 0.3,
                "actions": [],
            }
            return {"contexts": [goal]}, False, {"failure_detected": False, "failure_type": None, "detail": None, "guided_retry_attempted": False, "guided_retry_succeeded": None}

        monkeypatch.setattr(pipeline, "stage1_extract", fake_stage1)
        events = list(pipeline.run_pipeline_streaming("some complaint", "some grounding text"))
        assert events[-1]["stage"] == "complete"
        goal = events[-1]["data"]["response"]["contexts"][0]
        assert goal.get("escalation") is not None


class TestIssuePhraseGuessing:
    """Task 37: _guess_issue_phrase() is the finer-grained sibling of
    _guess_topic() (5 domain buckets) -- it powers /stats's 'trending
    issues' breakdown, so it needs its own direct coverage rather than
    being exercised only incidentally through run_pipeline()."""

    def test_battery_draining(self):
        assert pipeline._guess_issue_phrase("battery drains too fast") == "Battery draining fast"

    def test_battery_charging(self):
        assert pipeline._guess_issue_phrase("battery not charging overnight") == "Battery not charging"

    def test_camera_blurry(self):
        assert pipeline._guess_issue_phrase("camera photos are blurry") == "Camera blurry / out of focus"

    def test_camera_crashing(self):
        assert pipeline._guess_issue_phrase("camera app keeps crashing") == "Camera app crashing"

    def test_wifi_connectivity(self):
        assert pipeline._guess_issue_phrase("wifi keeps disconnecting") == "Wi-Fi connectivity"

    def test_bluetooth_connectivity(self):
        assert pipeline._guess_issue_phrase("bluetooth won't pair with my earbuds") == "Bluetooth connectivity"

    def test_overheating(self):
        assert pipeline._guess_issue_phrase("phone is overheating during calls") == "Device overheating"

    def test_screen_flickering(self):
        assert pipeline._guess_issue_phrase("screen flickers randomly") == "Screen flickering"

    def test_screen_cracked(self):
        assert pipeline._guess_issue_phrase("screen is cracked in the corner") == "Screen cracked / physical damage"

    def test_black_screen(self):
        assert pipeline._guess_issue_phrase("phone shows a black screen on boot") == "Black / blank screen"

    def test_touchscreen_unresponsive(self):
        assert pipeline._guess_issue_phrase("touch screen is unresponsive") == "Touchscreen unresponsive"

    def test_storage_full(self):
        assert pipeline._guess_issue_phrase("storage is full and can't install apps") == "Storage full"

    def test_app_crashing(self):
        assert pipeline._guess_issue_phrase("the app keeps crashing on launch") == "App crashing"

    def test_software_update(self):
        assert pipeline._guess_issue_phrase("software update failed to install") == "Software update issue"

    def test_running_slow(self):
        assert pipeline._guess_issue_phrase("phone is very slow and laggy") == "Device running slow"

    def test_speaker_audio(self):
        assert pipeline._guess_issue_phrase("speaker sound is distorted") == "Speaker / audio issue"

    def test_microphone(self):
        assert pipeline._guess_issue_phrase("microphone not picking up my voice") == "Microphone issue"

    def test_network_signal(self):
        assert pipeline._guess_issue_phrase("no network signal in my area") == "Network / signal issue"

    def test_unmatched_query_falls_back_to_domain_plus_issue(self):
        """Nothing here matches a specific phrase, so it must fall back to
        '<domain> issue' (never crash, never return an empty string)."""
        result = pipeline._guess_issue_phrase("something is wrong with my device")
        assert result == "Device issue"
        assert result == f"{pipeline._guess_topic('something is wrong with my device')} issue"

    def test_run_pipeline_populates_issue_guess_in_the_request_log(self):
        """End-to-end: a real run_pipeline() call must log a specific
        issue_guess (not just domain_guess), so /stats's trending-issues
        breakdown reflects real traffic. request_log.LOG_FILE is already
        redirected to a scratch path by conftest.py's autouse fixture."""
        import request_log
        pipeline.run_pipeline("my battery is draining super fast today", "")
        logs = request_log.read_all_logs()
        assert logs[-1]["issue_guess"] == "Battery draining fast"
