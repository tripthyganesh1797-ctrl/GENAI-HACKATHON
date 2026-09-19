"""device_signals.py — the confidence-context-free companion to
escalation.py: reorders same-category actions and appends honest advisory
notes from optional device-state signals (Task 35). See the module
docstring for what this deliberately does NOT do (invent actions, rewrite
step text, or use os_version for anything offline)."""
import device_signals as ds


def _goal(actions):
    return {"title": "Test goal", "actions": actions}


def _action(name, category="auto", description="It will do the thing needed"):
    return {"actionName": name, "description": description, "category": category}


def test_no_device_is_a_noop():
    contexts = [_goal([_action("Battery Settings")])]
    before = [dict(a) for a in contexts[0]["actions"]]
    notes = ds.apply_device_context(contexts, None)
    assert notes == []
    assert contexts[0]["actions"] == before


def test_empty_device_dict_is_a_noop():
    contexts = [_goal([_action("Battery Settings")])]
    notes = ds.apply_device_context(contexts, {})
    assert notes == []


def test_device_with_all_none_fields_is_a_noop():
    contexts = [_goal([_action("Battery Settings")])]
    device = {"battery_pct": None, "storage_free_pct": None, "os_version": None,
              "uptime_hours": None, "last_restart_hours_ago": None}
    notes = ds.apply_device_context(contexts, device)
    assert notes == []


def test_low_battery_adds_note_and_boosts_matching_action():
    contexts = [_goal([
        _action("Display Settings"),
        _action("Battery Settings", description="It will enable power saving mode"),
    ])]
    notes = ds.apply_device_context(contexts, {"battery_pct": 5})
    assert len(notes) == 1
    assert "5%" in notes[0]
    assert contexts[0]["actions"][0]["actionName"] == "Battery Settings"


def test_battery_above_threshold_does_not_trigger():
    contexts = [_goal([
        _action("Display Settings"),
        _action("Battery Settings"),
    ])]
    before = [a["actionName"] for a in contexts[0]["actions"]]
    notes = ds.apply_device_context(contexts, {"battery_pct": 50})
    assert notes == []
    assert [a["actionName"] for a in contexts[0]["actions"]] == before


def test_battery_exactly_at_threshold_triggers():
    """Threshold semantics are <=, matching escalation.py's documented
    style of stating the boundary explicitly rather than leaving it
    ambiguous which side of the line "critical" falls on."""
    contexts = [_goal([_action("Battery Settings")])]
    notes = ds.apply_device_context(contexts, {"battery_pct": ds.CRITICAL_BATTERY_PCT})
    assert len(notes) == 1


def test_low_storage_adds_note_and_boosts_matching_action():
    contexts = [_goal([
        _action("Display Settings"),
        _action("Clear Cache", category="manual", description="It will free up device storage space"),
    ])]
    notes = ds.apply_device_context(contexts, {"storage_free_pct": 3})
    assert len(notes) == 1
    assert "3%" in notes[0]
    # Clear Cache is the only "manual" action -- nothing to reorder it past,
    # but it must not have been moved out of its category.
    assert contexts[0]["actions"][1]["category"] == "manual"


def test_stale_uptime_adds_note_without_reorder():
    contexts = [_goal([
        _action("Display Settings"),
        _action("Restart Device", category="critical"),
    ])]
    before = [a["actionName"] for a in contexts[0]["actions"]]
    notes = ds.apply_device_context(contexts, {"uptime_hours": 96})
    assert len(notes) == 1
    assert "96" in notes[0]
    assert [a["actionName"] for a in contexts[0]["actions"]] == before  # no reorder


def test_stale_uptime_suppressed_when_recently_restarted():
    """A device can have high uptime_hours (e.g. it just hasn't been asked
    in a while) but a recent last_restart_hours_ago -- e.g. inconsistent
    client instrumentation -- so the note requires BOTH signals to agree
    the device is actually stale."""
    contexts = [_goal([_action("Display Settings")])]
    notes = ds.apply_device_context(
        contexts, {"uptime_hours": 96, "last_restart_hours_ago": 2}
    )
    assert notes == []


def test_multiple_signals_combine_into_multiple_notes():
    contexts = [_goal([
        _action("Battery Settings"),
        _action("Clear Cache", category="manual"),
        _action("Restart Device", category="critical"),
    ])]
    notes = ds.apply_device_context(
        contexts,
        {"battery_pct": 5, "storage_free_pct": 4, "uptime_hours": 80},
    )
    assert len(notes) == 3


def test_category_ordering_invariant_survives_boosted_reorder():
    """No matter what gets boosted, auto < manual < critical must still
    hold after apply_device_context -- this is the same invariant
    validators.py's validate_category_ordering() checks, just verified
    here from the device_signals.py side of the seam."""
    contexts = [_goal([
        _action("Restart Device", category="critical", description="It will restart the device fully"),
        _action("Battery Settings", category="auto"),
        _action("Clean Charging Port", category="manual"),
    ])]
    ds.apply_device_context(contexts, {"battery_pct": 5})
    cats = [a["category"] for a in contexts[0]["actions"]]
    rank = {"auto": 0, "manual": 1, "critical": 2}
    assert [rank[c] for c in cats] == sorted(rank[c] for c in cats)


def test_single_action_goal_is_not_touched():
    contexts = [_goal([_action("Display Settings")])]
    before = dict(contexts[0]["actions"][0])
    ds.apply_device_context(contexts, {"battery_pct": 5})
    assert contexts[0]["actions"][0] == before


def test_no_matching_action_still_returns_note_but_no_reorder():
    contexts = [_goal([_action("Display Settings"), _action("Wifi Settings")])]
    before = [a["actionName"] for a in contexts[0]["actions"]]
    notes = ds.apply_device_context(contexts, {"battery_pct": 5})
    assert len(notes) == 1  # still informative even if this particular goal has nothing to reorder
    assert [a["actionName"] for a in contexts[0]["actions"]] == before


def test_multiple_goals_each_get_reordered_independently():
    contexts = [
        _goal([_action("Display Settings"), _action("Battery Settings")]),
        _goal([_action("Camera Settings"), _action("Battery Optimization", description="It will optimize battery usage now")]),
    ]
    ds.apply_device_context(contexts, {"battery_pct": 5})
    assert contexts[0]["actions"][0]["actionName"] == "Battery Settings"
    assert contexts[1]["actions"][0]["actionName"] == "Battery Optimization"
