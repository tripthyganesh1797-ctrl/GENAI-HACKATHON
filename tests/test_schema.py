"""schema.py — the Pydantic data contract (theme guide Appendix A)."""
import pytest
from pydantic import ValidationError

from schema import (
    Action, ActionCategory, BatchTroubleshootRequest, DeviceContext, EscalationAction,
    EscalationRecommendation, FeedbackRequest, Goal, Meta, StepGroup, TroubleshootRequest,
)


def test_troubleshoot_request_requires_query():
    with pytest.raises(ValidationError):
        TroubleshootRequest()
    req = TroubleshootRequest(query="battery drains fast")
    assert req.siis_response is None


def test_goal_object_builds_from_valid_dict():
    goal = Goal(
        goal="Follow these steps to perform this Battery Troubleshooting",
        title="Battery drain",
        score=0.9,
        actions=[
            Action(
                actionName="Battery Settings",
                description="It will let you enable power saving mode",
                category=ActionCategory.auto,
                stepGroups=[StepGroup(steps=["Tap Battery."])],
            )
        ],
    )
    assert goal.actions[0].category == ActionCategory.auto


def test_action_category_rejects_invalid_enum_value():
    with pytest.raises(ValidationError):
        Action(
            actionName="X",
            description="It will do something helpful here",
            category="destructive",  # not a valid ActionCategory
            stepGroups=[StepGroup(steps=["Tap X."])],
        )


def test_feedback_request_requires_deeplink_action_helpful():
    with pytest.raises(ValidationError):
        FeedbackRequest(action_name="X", helpful=True)  # missing deeplink

    fb = FeedbackRequest(deeplink="bixby://masked/act/x", action_name="Wifi Settings", helpful=False)
    assert fb.query is None
    assert fb.comment is None


def test_goal_accepts_optional_escalation():
    goal = Goal(
        goal="Follow these steps to perform this Battery Troubleshooting",
        title="Battery drain",
        score=0.4,
        actions=[],
        escalation=EscalationRecommendation(
            recommended=True,
            reason="low confidence",
            action=EscalationAction(
                deeplink="bixby://masked/act/5930a08d3d",
                message="Run Full Device Diagnostic",
                description="Runs a full device diagnostic.",
            ),
        ),
    )
    assert goal.escalation.recommended is True
    assert goal.escalation.action.deeplink.startswith("bixby://")

    goal_without = Goal(
        goal="Follow these steps to perform this Battery Troubleshooting",
        title="Battery drain", score=0.9, actions=[],
    )
    assert goal_without.escalation is None


def test_troubleshoot_request_device_is_optional_and_defaults_to_none():
    req = TroubleshootRequest(query="battery drains fast")
    assert req.device is None


def test_troubleshoot_request_accepts_partial_device_context():
    req = TroubleshootRequest(query="battery drains fast", device={"battery_pct": 6})
    assert req.device.battery_pct == 6
    assert req.device.storage_free_pct is None


def test_device_context_rejects_out_of_range_percentages():
    with pytest.raises(ValidationError):
        DeviceContext(battery_pct=150)
    with pytest.raises(ValidationError):
        DeviceContext(storage_free_pct=-5)


def test_device_context_rejects_negative_uptime():
    with pytest.raises(ValidationError):
        DeviceContext(uptime_hours=-1)


def test_meta_device_context_notes_defaults_to_empty_list():
    meta = Meta(latency_ms=1.0, cache_hit=False, model="x", cost_usd=0.0)
    assert meta.device_context_notes == []


def test_batch_request_enforces_one_to_twenty_items():
    with pytest.raises(ValidationError):
        BatchTroubleshootRequest(items=[])
    with pytest.raises(ValidationError):
        BatchTroubleshootRequest(items=[{"query": f"q{i}"} for i in range(21)])

    req = BatchTroubleshootRequest(items=[{"query": "battery drains fast"}])
    assert len(req.items) == 1
    assert req.items[0].siis_response is None
    assert req.items[0].device is None


def test_batch_item_accepts_device_context():
    req = BatchTroubleshootRequest(items=[
        {"query": "battery drains fast", "device": {"battery_pct": 5}},
    ])
    assert req.items[0].device.battery_pct == 5
