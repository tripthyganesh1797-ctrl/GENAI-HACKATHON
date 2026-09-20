"""
llm_client.py — Single place that calls whichever LLM API you choose.
Swap the internals here without touching pipeline.py.

Set LLM_API_KEY and LLM_PROVIDER in your .env file.
Supported providers out of the box: "anthropic", "openai".
Add more by extending call_llm().
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq")
API_KEY = os.getenv("LLM_API_KEY")
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "groq": "openai/gpt-oss-20b",
}
MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(PROVIDER, "openai/gpt-oss-20b"))

# Which models actually accept an image in their chat/messages API --
# checked explicitly (image_analysis.py's is_vision_capable()) rather than
# assumed, so a text-only model (this codebase's own coded default, Groq's
# openai/gpt-oss-20b) degrades honestly when a photo is attached instead of
# erroring deep inside a provider SDK call. Extend this set as new
# vision-capable models get added.
_VISION_CAPABLE_MODELS = {
    # Anthropic: every current Claude model accepts images.
    "claude-sonnet-4-6",
    # OpenAI: the gpt-4o / gpt-4.1 / o-series multimodal family.
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o1", "o3", "o4-mini",
    # Groq: Llama 4's natively multimodal models -- confirmed available on
    # Groq's free developer tier at the time this feature was built.
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
}

# Real $ per 1M tokens (input, output) for cost tracking. Groq's tier used
# here is free, so cost is genuinely $0 — but token counts are still tracked
# honestly from the API's real usage data, not faked.
_COST_PER_1M_TOKENS = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.0, 0.0),  # free on Groq's developer tier
}

# Running totals for the current process — pipeline.py reads this after each
# stage to report real token usage per request, instead of a hardcoded 0.
LAST_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    LAST_USAGE["prompt_tokens"] = prompt_tokens
    LAST_USAGE["completion_tokens"] = completion_tokens
    LAST_USAGE["total_tokens"] = prompt_tokens + completion_tokens


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = _COST_PER_1M_TOKENS.get(MODEL, (0.0, 0.0))
    return round((prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate, 6)


def api_key_configured() -> bool:
    """Same "is there really a usable key" check pipeline.py's own
    llm_available() does, exposed here too so image_analysis.py (which
    must not import pipeline.py -- pipeline.py imports IT) can make the
    same honest degrade decision without duplicating the placeholder-value
    list."""
    return API_KEY not in (None, "", "your_key_here")


def is_vision_capable() -> bool:
    """Whether the CURRENTLY CONFIGURED model (not just the provider) is
    known to accept an image input. See _VISION_CAPABLE_MODELS above."""
    return MODEL in _VISION_CAPABLE_MODELS


def _parse_data_url(data_url: str) -> tuple[str, str]:
    """Splits a `data:<mime>;base64,<data>` URL into (mime, base64_data).
    Raises ValueError on anything else -- call_llm_vision()'s only caller
    (image_analysis.py) always catches this, so a malformed/truncated
    upload degrades to an honest "couldn't analyze" rather than a 500."""
    if not data_url.startswith("data:") or ";base64," not in data_url:
        raise ValueError("expected a base64 data URL (data:<mime>;base64,<data>)")
    header, b64data = data_url.split(";base64,", 1)
    if not b64data:
        raise ValueError("data URL has no base64 payload")
    return header[len("data:"):] or "image/jpeg", b64data


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite instructions not to."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def call_llm(prompt: str, max_tokens: int = 1000) -> str:
    """Sends a prompt, returns the raw text response."""
    if PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=API_KEY)
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        _record_usage(response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text

    elif PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        _record_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content

    elif PROVIDER == "groq":
        # Groq has a free API tier (no card required) and is OpenAI-SDK
        # compatible — we just point the OpenAI client at Groq's base_url.
        # NOTE: gpt-oss reasoning models can burn the entire token budget on
        # invisible "thinking" and return empty content if reasoning_effort
        # isn't capped. "low" keeps enough tokens free for the real answer.
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"reasoning_effort": "low"},
        )
        content = response.choices[0].message.content
        if response.usage:
            _record_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        if not content:
            # Surface the full response so you can see exactly what came back
            # instead of a bare empty string that's hard to debug.
            raise ValueError(
                f"Groq returned empty content. Full response object:\n{response}"
            )
        return content

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {PROVIDER}")


def call_llm_vision(prompt: str, image_data_url: str, max_tokens: int = 300) -> str:
    """Same shape as call_llm() above but attaches one image alongside the
    text prompt. Only ever called after is_vision_capable() has already
    said yes -- see image_analysis.py, this function's sole caller, which
    never reaches here without that check and always catches whatever this
    raises (a bad/corrupted image, a provider error) rather than letting
    it crash the request."""
    if PROVIDER == "anthropic":
        import anthropic
        media_type, b64data = _parse_data_url(image_data_url)
        client = anthropic.Anthropic(api_key=API_KEY)
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64data}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        _record_usage(response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text

    elif PROVIDER in ("openai", "groq"):
        # Both are the OpenAI content-array format, which accepts a data
        # URL directly in image_url.url -- no base64/media-type splitting
        # needed on this branch, unlike Anthropic's source-object format.
        from openai import OpenAI
        client_kwargs = {"api_key": API_KEY}
        if PROVIDER == "groq":
            client_kwargs["base_url"] = "https://api.groq.com/openai/v1"
        client = OpenAI(**client_kwargs)
        extra_kwargs = {"extra_body": {"reasoning_effort": "low"}} if PROVIDER == "groq" else {}
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }],
            **extra_kwargs,
        )
        content = response.choices[0].message.content
        if response.usage:
            _record_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        if not content:
            raise ValueError(
                f"{PROVIDER} returned empty content for a vision request. Full response object:\n{response}"
            )
        return content

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER for vision: {PROVIDER}")


def call_llm_json(prompt: str, max_tokens: int = 1000, retries: int = 2) -> dict:
    """Calls the LLM and parses the response as JSON. Retries automatically on
    truncated/malformed JSON (a known intermittent quirk with free-tier
    models), since one bad call shouldn't fail the whole request. Raises
    ValueError with the raw text only after all retries are exhausted."""
    last_error = None
    for attempt in range(retries + 1):
        raw = call_llm(prompt, max_tokens=max_tokens)
        cleaned = _strip_code_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = ValueError(
                f"LLM did not return valid JSON (attempt {attempt + 1}/{retries + 1}).\n"
                f"Error: {e}\nRaw output:\n{raw}"
            )
    raise last_error
