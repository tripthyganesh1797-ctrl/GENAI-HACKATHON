"""
report.py — Task 38: packages an already-computed troubleshooting result
(the exact dict shape run_pipeline()/run_pipeline_streaming() return) into
a compact, human-shareable report -- Markdown by default, a self-contained
HTML page as the alternative -- so a user can paste a fix into a support
ticket, a forum post, or a message to a technician/friend without
re-explaining the whole diagnosis from scratch.

Deliberately takes an already-computed result rather than re-running the
pipeline: the caller (the demo UI, the CLI, a judge poking at the API)
already paid for that computation once via /v1/troubleshoot(/stream), and
formatting it is a pure, free function of data it already has. This also
means report generation can never itself trigger an LLM call, a cache
write, or a request-log entry -- it's just a renderer.

Two-layer design, same pattern as deeplink_matching.py's shared ranking
helper: build_report_sections() extracts a plain, renderer-independent
structure from the pipeline response (never mutates the input, never
invents a field that isn't already somewhere in schema.py's Goal/Action/
StepGroup or meta.*), and render_markdown()/render_html() turn that
structure into text. Adding a third format later means adding one more
render_* function against the same sections dict, not touching the
extraction logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape as _h

from escalation import LLM_SCORE_ESCALATION_THRESHOLD

SUPPORTED_FORMATS = ("markdown", "html")

# Confidence label bands. The Low/Medium boundary deliberately reuses
# escalation.py's own LLM_SCORE_ESCALATION_THRESHOLD (0.6) rather than a
# separate invented number, so a Goal the engine itself flagged for
# escalation is consistently the same Goal a reader sees labeled "Low"
# here -- one confidence story, not two slightly different ones.
_HIGH_CONFIDENCE_THRESHOLD = 0.8


def _confidence_label(score: float) -> str:
    if score >= _HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if score >= LLM_SCORE_ESCALATION_THRESHOLD:
        return "Medium"
    return "Low"


def build_report_sections(pipeline_response: dict) -> dict:
    """Pulls a compact, renderer-agnostic structure out of a run_pipeline()
    -shaped response dict. Never mutates `pipeline_response`."""
    pipeline_response = pipeline_response or {}
    meta = pipeline_response.get("meta") or {}
    contexts = (pipeline_response.get("response") or {}).get("contexts") or []

    goals = []
    for goal in contexts:
        actions = []
        for action in goal.get("actions", []) or []:
            steps: list[str] = []
            deeplink = None
            deeplink_message = None
            for sg in action.get("stepGroups", []) or []:
                steps.extend(sg.get("steps", []) or [])
                if deeplink is None:
                    dl = sg.get("actionableDeeplink")
                    if dl and dl.get("deeplink"):
                        deeplink = dl.get("deeplink")
                        deeplink_message = dl.get("message") or dl.get("description")
            actions.append({
                "name": action.get("actionName", ""),
                "description": action.get("description", ""),
                "category": action.get("category") or "manual",
                "steps": steps,
                "deeplink": deeplink,
                "deeplink_message": deeplink_message,
            })

        score = goal.get("score", 0.0) or 0.0
        escalation = goal.get("escalation")
        goals.append({
            "title": goal.get("title", ""),
            "goal": goal.get("goal", ""),
            "score": score,
            "confidence_label": _confidence_label(score),
            "actions": actions,
            "escalation": {
                "reason": escalation.get("reason", ""),
                "message": escalation.get("action", {}).get("message", ""),
                "deeplink": escalation.get("action", {}).get("deeplink", ""),
            } if escalation and escalation.get("recommended") else None,
        })

    return {
        "query": pipeline_response.get("query", ""),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "goals": goals,
        "no_match": not goals,
        "used_offline_fallback": bool(meta.get("used_offline_fallback")),
        "device_context_notes": meta.get("device_context_notes") or [],
        "session_notes": meta.get("session_notes") or [],
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_markdown(sections: dict) -> str:
    lines = ["# Diagnostic Report", ""]
    lines.append(f"**Complaint:** {sections['query']}")
    lines.append(f"**Generated:** {sections['generated_at']}")
    lines.append("")

    if sections["no_match"]:
        lines.append("_No matching troubleshooting plan was found in the catalog for this complaint._")
    else:
        for i, goal in enumerate(sections["goals"], start=1):
            lines.append(f"## {i}. {goal['title']} — Confidence: {goal['confidence_label']} "
                         f"({goal['score']:.2f})")
            lines.append(f"_{goal['goal']}_")
            lines.append("")
            for j, action in enumerate(goal["actions"], start=1):
                lines.append(f"### {i}.{j} {action['name']} `[{action['category']}]`")
                lines.append(action["description"])
                for step in action["steps"]:
                    lines.append(f"- {step}")
                if action["deeplink"]:
                    label = action["deeplink_message"] or "Open in Samsung Members"
                    lines.append(f"- **→ {label}:** `{action['deeplink']}`")
                lines.append("")
            if goal["escalation"]:
                esc = goal["escalation"]
                lines.append(f"> ⚠ **If these steps don't resolve it:** {esc['reason']} "
                             f"— try **{esc['message']}** (`{esc['deeplink']}`)")
                lines.append("")

    notes = sections["device_context_notes"] + sections["session_notes"]
    if notes:
        lines.append("---")
        lines.append("**Notes:**")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("---")
    lines.append("_Generated by the Samsung PRISM Smart Guided Troubleshooting Engine. "
                 "Manual/critical steps should be reviewed carefully; when in doubt, use the "
                 "official Samsung Members app or a certified service center._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML — self-contained, no external assets, safe to open directly or
# email as an attachment. Every user-supplied string is escaped.
# ---------------------------------------------------------------------------

_HTML_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 720px;
       margin: 2rem auto; padding: 0 1.25rem; color: #1a1f2b; line-height: 1.5; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
.meta { color: #5b6472; font-size: 0.9rem; margin-bottom: 1.5rem; }
.goal { border: 1px solid #dfe3ea; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.25rem; }
.goal h2 { font-size: 1.15rem; margin: 0 0 0.15rem; }
.goal .desc { color: #5b6472; font-style: italic; margin-bottom: 0.75rem; }
.confidence { display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 0.1rem 0.5rem;
              border-radius: 999px; margin-left: 0.5rem; vertical-align: middle; }
.confidence.High { background: #d7f5e9; color: #0a7a4e; }
.confidence.Medium { background: #fff2d6; color: #915a00; }
.confidence.Low { background: #fde2e1; color: #a3231d; }
.action { margin: 0.9rem 0; padding-top: 0.75rem; border-top: 1px dashed #e3e6ec; }
.action h3 { font-size: 1rem; margin: 0 0 0.2rem; }
.tag { font-size: 0.7rem; text-transform: uppercase; color: #5b6472; }
.action .desc { color: #333; margin-bottom: 0.4rem; }
.action ul { margin: 0.3rem 0; padding-left: 1.3rem; }
.deeplink { font-family: monospace; font-size: 0.85rem; background: #f3f5f8; padding: 0.15rem 0.4rem;
            border-radius: 4px; }
.escalation { background: #fff8e8; border: 1px solid #f0dca0; border-radius: 8px;
              padding: 0.6rem 0.9rem; margin-top: 0.5rem; font-size: 0.9rem; }
.notes { background: #f3f5f8; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1rem; }
.notes ul { margin: 0.25rem 0 0; padding-left: 1.2rem; }
.footer { color: #8890a3; font-size: 0.78rem; margin-top: 2rem; border-top: 1px solid #e3e6ec;
          padding-top: 0.75rem; }
""".strip()


def render_html(sections: dict) -> str:
    body = [f"<h1>Diagnostic Report</h1>",
            f'<div class="meta"><strong>Complaint:</strong> {_h(sections["query"])}<br>'
            f'<strong>Generated:</strong> {_h(sections["generated_at"])}</div>']

    if sections["no_match"]:
        body.append("<p><em>No matching troubleshooting plan was found in the catalog "
                    "for this complaint.</em></p>")
    else:
        for i, goal in enumerate(sections["goals"], start=1):
            body.append('<div class="goal">')
            body.append(f'<h2>{i}. {_h(goal["title"])}'
                        f'<span class="confidence {goal["confidence_label"]}">'
                        f'{goal["confidence_label"]} · {goal["score"]:.2f}</span></h2>')
            body.append(f'<div class="desc">{_h(goal["goal"])}</div>')
            for j, action in enumerate(goal["actions"], start=1):
                body.append('<div class="action">')
                body.append(f'<h3>{i}.{j} {_h(action["name"])} '
                            f'<span class="tag">[{_h(action["category"])}]</span></h3>')
                body.append(f'<div class="desc">{_h(action["description"])}</div>')
                if action["steps"]:
                    body.append("<ul>" + "".join(f"<li>{_h(s)}</li>" for s in action["steps"]) + "</ul>")
                if action["deeplink"]:
                    label = action["deeplink_message"] or "Open in Samsung Members"
                    body.append(f'<div>→ {_h(label)}: '
                                f'<span class="deeplink">{_h(action["deeplink"])}</span></div>')
                body.append("</div>")
            if goal["escalation"]:
                esc = goal["escalation"]
                body.append(f'<div class="escalation">⚠ <strong>If these steps don\'t resolve it:</strong> '
                            f'{_h(esc["reason"])} — try <strong>{_h(esc["message"])}</strong> '
                            f'(<span class="deeplink">{_h(esc["deeplink"])}</span>)</div>')
            body.append("</div>")

    notes = sections["device_context_notes"] + sections["session_notes"]
    if notes:
        body.append('<div class="notes"><strong>Notes:</strong><ul>' +
                    "".join(f"<li>{_h(n)}</li>" for n in notes) + "</ul></div>")

    body.append('<div class="footer">Generated by the Samsung PRISM Smart Guided Troubleshooting '
                "Engine. Manual/critical steps should be reviewed carefully; when in doubt, use the "
                "official Samsung Members app or a certified service center.</div>")

    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>Diagnostic Report — {_h(sections['query'][:60])}</title>\n"
        f"<style>{_HTML_STYLE}</style>\n</head>\n<body>\n" + "\n".join(body) + "\n</body>\n</html>\n"
    )


def generate_report(pipeline_response: dict, fmt: str = "markdown") -> str:
    """Single entry point used by main.py's /v1/report route and cli.py's
    `report` subcommand. Raises ValueError on an unsupported format so
    callers get a clear 400 instead of a confusing fallback."""
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported report format {fmt!r} -- expected one of {SUPPORTED_FORMATS}")
    sections = build_report_sections(pipeline_response)
    return render_markdown(sections) if fmt == "markdown" else render_html(sections)
