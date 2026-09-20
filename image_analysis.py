"""
image_analysis.py — point 1: let a user attach a photo of the problem (a
cracked screen, a swollen battery, an error dialog, a charging cable) so
"the problem will be very clear and get accurate results" (the user's own
framing for this feature).

This module's only job is turning that photo into a short, honest, FACTUAL
description of what's visibly there -- never a diagnosis, never a fix
suggestion -- which pipeline.py then folds into the complaint text BEFORE
Stage 0 runs. That's a deliberate architectural choice: this codebase's LLM
prompts (prompts.py) are plain-text templates, so folding a description in
as text keeps every existing guarantee -- offline fallback, caching by
technical_query, safety.py's keyword-based hazard detection, clarify.py's
vague-complaint detection, deterministic Stage 2 deeplink matching --
working exactly as before with zero special-casing anywhere else in the
pipeline. An attached image becomes "a few more words in the complaint",
architecturally nothing more. It also means a photo that shows something
alarming (visible smoke, a swollen/bulging battery) naturally flows into
safety.py's existing hazard check for free, since that check just looks at
the (now image-informed) complaint text -- no separate image-specific
safety path was needed.

Degrades honestly, same philosophy as safety.py/clarify.py/answer_source.py:
  - No image supplied: complete no-op ({"provided": False, ...}) -- the
    other 3 features in this batch, and everything before them, are
    unaffected whether or not a caller ever sends an image.
  - Image supplied but no LLM key configured, or the configured
    provider/model isn't vision-capable (llm_client.is_vision_capable()):
    the image is silently skipped and text-only troubleshooting still
    runs on the original complaint -- meta.image_analysis says exactly
    why, rather than pretending the picture was looked at.
  - Image supplied, vision-capable, but the call itself fails (corrupted
    upload, network hiccup, provider error): caught here and degraded the
    same way -- one bad photo never crashes the whole request.
"""
from __future__ import annotations

import llm_client

_DESCRIBE_IMAGE_PROMPT = (
    "You are looking at a photo a user attached to a phone troubleshooting "
    "request. In 1-2 short sentences, describe ONLY what is factually "
    "visible in the image that's relevant to a hardware or software "
    "problem (for example: a cracked or discolored screen, a swollen or "
    "bulging battery, an error message or dialog box, a charging cable or "
    "port, visible smoke, burn marks, or liquid damage). Do not diagnose "
    "the cause and do not suggest a fix -- just state plainly what you "
    "see. If nothing problem-relevant is visible, say so plainly instead "
    "of guessing."
)

# The complete no-op shape, reused wherever no image was ever supplied (no
# request-specific reason to compute) -- shared here so pipeline.py's
# streaming path (which doesn't accept images at all, see its own comment)
# and the non-streaming path's "no image" case always agree on the exact
# same shape for meta.image_analysis.
NO_IMAGE_ANALYSIS = {"provided": False, "analyzed": False, "description": None, "reason": None}


def describe_image(image_data_url: str | None) -> dict:
    """Returns a dict always shaped as:
        {"provided": bool, "analyzed": bool, "description": str|None, "reason": str|None}
    Never raises -- every failure mode degrades to analyzed=False with an
    honest `reason`, so pipeline.py can fold this straight into
    meta.image_analysis without needing its own try/except around this
    call."""
    if not image_data_url:
        return dict(NO_IMAGE_ANALYSIS)

    if not llm_client.api_key_configured():
        return {
            "provided": True, "analyzed": False, "description": None,
            "reason": "no LLM API key configured -- image was skipped, text-only analysis still ran",
        }

    if not llm_client.is_vision_capable():
        return {
            "provided": True, "analyzed": False, "description": None,
            "reason": f"configured model ({llm_client.MODEL}) doesn't support image input -- "
                      f"image was skipped, text-only analysis still ran",
        }

    try:
        description = llm_client.call_llm_vision(_DESCRIBE_IMAGE_PROMPT, image_data_url, max_tokens=150)
        description = (description or "").strip()
        if not description:
            return {
                "provided": True, "analyzed": False, "description": None,
                "reason": "vision model returned an empty description -- image was skipped, "
                          "text-only analysis still ran",
            }
        return {"provided": True, "analyzed": True, "description": description, "reason": None}
    except Exception as e:
        return {
            "provided": True, "analyzed": False, "description": None,
            "reason": f"image analysis failed ({e}) -- image was skipped, text-only analysis still ran",
        }
