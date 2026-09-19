"""escalation.py — the shared confidence-gated escalation recommendation,
plus the honest documentation of what it does and does NOT catch (see
module docstring in escalation.py)."""
import json

import escalation as esc


def test_build_escalation_recommendation_shape():
    rec = esc.build_escalation_recommendation("some reason")
    assert rec["recommended"] is True
    assert rec["reason"] == "some reason"
    assert rec["action"]["deeplink"].startswith("bixby://")
    assert rec["action"]["message"]
    assert rec["action"]["description"]


def test_escalation_action_is_a_real_catalog_entry():
    """The recommended action must be a REAL deeplink from the official
    578-entry catalog, not an invented "contact support" URI that doesn't
    exist anywhere in the actual data -- see the module docstring's
    rationale for why "View Device Details" (Samsung Members full device
    diagnostic) was chosen specifically."""
    import deeplink_matching as dm
    entries = dm.load_deeplinks("deeplinks.json")
    real_deeplinks = {e.deeplink for e in entries}
    rec = esc.build_escalation_recommendation("x")
    assert rec["action"]["deeplink"] in real_deeplinks


def test_recommendation_is_a_fresh_dict_each_call():
    """Regression guard: build_escalation_recommendation() must not hand
    back a shared mutable reference to the module-level action template --
    otherwise one caller's dict mutation could corrupt every other Goal's
    escalation object silently."""
    rec1 = esc.build_escalation_recommendation("a")
    rec2 = esc.build_escalation_recommendation("b")
    rec1["action"]["message"] = "mutated"
    assert rec2["action"]["message"] != "mutated"


def test_recommendation_is_json_serializable():
    rec = esc.build_escalation_recommendation("x")
    json.dumps(rec)  # must not raise
