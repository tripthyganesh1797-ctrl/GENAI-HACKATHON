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
