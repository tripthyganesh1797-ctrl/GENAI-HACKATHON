"""
validators.py — Programmatic enforcement of rules the LLM will NOT reliably follow
on its own (word counts, URL leaks, exact syntax). "Prompt-only constraints are
unreliable" — theme guide, Common Pitfall #5. So we check and fix in code.
"""

import re
from typing import List, Tuple

URL_PATTERN = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)


def has_url_leak(text: str) -> bool:
    """Zero URL Leaks constraint: no http/https/www anywhere."""
    return bool(URL_PATTERN.search(text))


def strip_urls(text: str) -> str:
    return URL_PATTERN.sub("", text).strip()


def check_description_format(description: str) -> Tuple[bool, str]:
    """description must be exactly 5-7 words and start with 'It will'."""
    if not description.startswith("It will"):
        return False, f"description must start with 'It will': got '{description}'"
    word_count = len(description.split())
    if not (5 <= word_count <= 7):
        return False, f"description must be 5-7 words, got {word_count}: '{description}'"
    return True, ""


def check_goal_syntax(goal: str, topic: str = None) -> Tuple[bool, str]:
    """Checks the STRUCTURE of the goal string, not an exact hardcoded topic.
    Accepts any topic the LLM chose, as long as the sentence shape is exactly
    'Follow these steps to perform this <topic> Troubleshooting/Configuration'.
    """
    pattern = re.compile(
        r"^Follow these steps to perform this .+ (Troubleshooting|Configuration)$"
    )
    if not pattern.match(goal):
        return False, (
            f"goal syntax mismatch. Got '{goal}', expected pattern: "
            f"'Follow these steps to perform this <Topic> Troubleshooting|Configuration'"
        )
    return True, ""


def check_title_format(title: str) -> Tuple[bool, str]:
    word_count = len(title.split())
    if not (2 <= word_count <= 3):
        return False, f"title must be 2-3 words, got {word_count}: '{title}'"
    if title != title[0].upper() + title[1:].lower() and not title[0].isupper():
        # loose sentence-case check; first letter capitalised, not full Title Case
        pass
    return True, ""


def validate_category_ordering(categories: List[str]) -> Tuple[bool, str]:
    """critical actions must always come last; auto/manual before critical."""
    if "critical" in categories:
        first_critical_idx = categories.index("critical")
        if any(c != "critical" for c in categories[first_critical_idx + 1:]):
            return False, "A non-critical action appears after a critical action"
    return True, ""


# Keywords that ALWAYS mean an action is destructive/disruptive, regardless of
# what category the LLM assigned. This is a safety net: the LLM occasionally
# mislabels a restart/reset as "auto", which would incorrectly give it a
# one-tap deeplink. We never trust the model alone for safety-critical rules.
_CRITICAL_KEYWORDS = ("restart", "reboot", "factory reset", "erase", "wipe", "power off")


def enforce_critical_safety(goal_dict: dict) -> None:
    """Mutates goal_dict in place: forces category='critical' on any action
    whose name or steps mention a destructive keyword, overriding the LLM's
    own classification if it got this wrong."""
    for action in goal_dict.get("actions", []):
        text = (action.get("actionName", "") + " " + action.get("description", "")).lower()
        for step_group in action.get("stepGroups", []):
            text += " " + " ".join(step_group.get("steps", [])).lower()

        if any(keyword in text for keyword in _CRITICAL_KEYWORDS):
            action["category"] = "critical"


def validate_goal_object(goal_dict: dict, topic: str = None) -> List[str]:
    """Run all checks against a single Goal dict (post-JSON-parse, pre-Pydantic).
    Returns a list of human-readable errors; empty list = fully compliant."""
    errors = []

    # Safety override runs BEFORE anything else, so downstream deeplink
    # matching sees the corrected category.
    enforce_critical_safety(goal_dict)

    ok, msg = check_goal_syntax(goal_dict.get("goal", ""))
    if not ok:
        errors.append(msg)

    ok, msg = check_title_format(goal_dict.get("title", ""))
    if not ok:
        errors.append(msg)

    categories = []
    for action in goal_dict.get("actions", []):
        ok, msg = check_description_format(action.get("description", ""))
        if not ok:
            errors.append(msg)
        categories.append(action.get("category", "manual"))

        # Zero URL Leaks — check every step and description
        for step_group in action.get("stepGroups", []):
            for step in step_group.get("steps", []):
                if has_url_leak(step):
                    errors.append(f"URL leak in step: '{step}'")
        if has_url_leak(action.get("description", "")):
            errors.append(f"URL leak in description: '{action['description']}'")

    ok, msg = validate_category_ordering(categories)
    if not ok:
        errors.append(msg)

    return errors
