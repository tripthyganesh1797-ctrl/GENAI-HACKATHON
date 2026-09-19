"""schema.py — the Pydantic data contract (theme guide Appendix A)."""
import pytest
from pydantic import ValidationError

from schema import (
    Action, ActionCategory, BatchTroubleshootRequest, FeedbackRequest, Goal,
    StepGroup, TroubleshootRequest,
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


def test_batch_request_enforces_one_to_twenty_items():
    with pytest.raises(ValidationError):
        BatchTroubleshootRequest(items=[])
    with pytest.raises(ValidationError):
        BatchTroubleshootRequest(items=[{"query": f"q{i}"} for i in range(21)])

    req = BatchTroubleshootRequest(items=[{"query": "battery drains fast"}])
    assert len(req.items) == 1
    assert req.items[0].siis_response is None
