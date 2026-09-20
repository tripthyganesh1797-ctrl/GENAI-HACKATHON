"""tests/test_image_analysis.py -- point 1 (picture upload). Covers
llm_client.py's vision helpers (capability check, data-URL parsing, the
vision call itself against a fake SDK client) and image_analysis.py's
describe_image(), which must NEVER raise and must always degrade honestly
-- no key, no vision-capable model, a corrupted upload, or a failed call
should all fall back to text-only troubleshooting rather than a 500."""
import pytest

import image_analysis
import llm_client
from image_analysis import NO_IMAGE_ANALYSIS, describe_image

# A syntactically-valid (if trivial) base64 PNG data URL for tests that
# only need to get PAST the data-URL parsing step, not decode a real image.
FAKE_IMAGE_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


class TestIsVisionCapable:
    def test_known_vision_model_is_capable(self, monkeypatch):
        monkeypatch.setattr(llm_client, "MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        assert llm_client.is_vision_capable() is True

    def test_default_groq_model_is_not_capable(self, monkeypatch):
        monkeypatch.setattr(llm_client, "MODEL", "openai/gpt-oss-20b")
        assert llm_client.is_vision_capable() is False

    def test_unknown_model_is_not_capable(self, monkeypatch):
        monkeypatch.setattr(llm_client, "MODEL", "some-made-up-model")
        assert llm_client.is_vision_capable() is False


class TestApiKeyConfigured:
    def test_none_is_not_configured(self, monkeypatch):
        monkeypatch.setattr(llm_client, "API_KEY", None)
        assert llm_client.api_key_configured() is False

    def test_placeholder_is_not_configured(self, monkeypatch):
        monkeypatch.setattr(llm_client, "API_KEY", "your_key_here")
        assert llm_client.api_key_configured() is False

    def test_real_looking_key_is_configured(self, monkeypatch):
        monkeypatch.setattr(llm_client, "API_KEY", "gsk_realkeylookingvalue")
        assert llm_client.api_key_configured() is True


class TestParseDataUrl:
    def test_parses_mime_and_base64(self):
        mime, data = llm_client._parse_data_url("data:image/jpeg;base64,QUJD")
        assert mime == "image/jpeg"
        assert data == "QUJD"

    def test_rejects_non_data_url(self):
        with pytest.raises(ValueError):
            llm_client._parse_data_url("https://example.com/photo.jpg")

    def test_rejects_missing_base64_marker(self):
        with pytest.raises(ValueError):
            llm_client._parse_data_url("data:image/png,notbase64")

    def test_rejects_empty_payload(self):
        with pytest.raises(ValueError):
            llm_client._parse_data_url("data:image/png;base64,")


class TestCallLlmVision:
    def test_anthropic_provider_sends_image_block_and_records_usage(self, monkeypatch):
        monkeypatch.setattr(llm_client, "PROVIDER", "anthropic")
        monkeypatch.setattr(llm_client, "MODEL", "claude-sonnet-4-6")
        monkeypatch.setattr(llm_client, "API_KEY", "fake-key")

        captured = {}

        class FakeUsage:
            input_tokens = 120
            output_tokens = 20

        class FakeContentBlock:
            text = "A cracked screen with a visible spiderweb crack pattern."

        class FakeResponse:
            usage = FakeUsage()
            content = [FakeContentBlock()]

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeAnthropicClient:
            def __init__(self, api_key):
                captured["api_key"] = api_key
                self.messages = FakeMessages()

        import sys
        import types
        fake_anthropic_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

        result = llm_client.call_llm_vision("describe this", FAKE_IMAGE_DATA_URL)
        assert result == "A cracked screen with a visible spiderweb crack pattern."
        assert captured["api_key"] == "fake-key"
        content_blocks = captured["messages"][0]["content"]
        assert content_blocks[0]["type"] == "image"
        assert content_blocks[0]["source"]["media_type"] == "image/png"
        assert content_blocks[1] == {"type": "text", "text": "describe this"}
        assert llm_client.LAST_USAGE["prompt_tokens"] == 120

    def test_groq_provider_sends_image_url_content(self, monkeypatch):
        monkeypatch.setattr(llm_client, "PROVIDER", "groq")
        monkeypatch.setattr(llm_client, "MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        monkeypatch.setattr(llm_client, "API_KEY", "fake-key")

        captured = {}

        class FakeMessage:
            content = "A swollen battery visibly bulging the back cover."

        class FakeChoice:
            message = FakeMessage()

        class FakeUsage:
            prompt_tokens = 200
            completion_tokens = 15

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeOpenAIClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.chat = FakeChat()

        import sys
        import types
        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAIClient)
        monkeypatch.setitem(sys.modules, "openai", fake_openai_module)

        result = llm_client.call_llm_vision("describe this", FAKE_IMAGE_DATA_URL)
        assert result == "A swollen battery visibly bulging the back cover."
        assert captured["client_kwargs"]["base_url"] == "https://api.groq.com/openai/v1"
        content_blocks = captured["messages"][0]["content"]
        assert content_blocks[1] == {"type": "image_url", "image_url": {"url": FAKE_IMAGE_DATA_URL}}

    def test_unsupported_provider_raises(self, monkeypatch):
        monkeypatch.setattr(llm_client, "PROVIDER", "made-up-provider")
        with pytest.raises(ValueError):
            llm_client.call_llm_vision("x", FAKE_IMAGE_DATA_URL)


class TestDescribeImage:
    def test_no_image_is_a_complete_noop(self):
        assert describe_image(None) == NO_IMAGE_ANALYSIS
        assert describe_image("") == NO_IMAGE_ANALYSIS

    def test_no_api_key_degrades_honestly(self, monkeypatch):
        monkeypatch.setattr(llm_client, "API_KEY", None)
        result = describe_image(FAKE_IMAGE_DATA_URL)
        assert result["provided"] is True
        assert result["analyzed"] is False
        assert result["description"] is None
        assert "no LLM API key" in result["reason"]

    def test_non_vision_model_degrades_honestly(self, monkeypatch):
        monkeypatch.setattr(llm_client, "API_KEY", "fake-key")
        monkeypatch.setattr(llm_client, "MODEL", "openai/gpt-oss-20b")
        result = describe_image(FAKE_IMAGE_DATA_URL)
        assert result["provided"] is True
        assert result["analyzed"] is False
        assert "doesn't support image input" in result["reason"]

    def test_successful_call_returns_description(self, monkeypatch):
        monkeypatch.setattr(llm_client, "API_KEY", "fake-key")
        monkeypatch.setattr(llm_client, "MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        monkeypatch.setattr(llm_client, "call_llm_vision",
                             lambda prompt, url, max_tokens=150: "  A cracked screen.  ")
        result = describe_image(FAKE_IMAGE_DATA_URL)
        assert result == {
            "provided": True, "analyzed": True,
            "description": "A cracked screen.", "reason": None,
        }

    def test_empty_description_degrades_honestly(self, monkeypatch):
        monkeypatch.setattr(llm_client, "API_KEY", "fake-key")
        monkeypatch.setattr(llm_client, "MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        monkeypatch.setattr(llm_client, "call_llm_vision", lambda prompt, url, max_tokens=150: "   ")
        result = describe_image(FAKE_IMAGE_DATA_URL)
        assert result["analyzed"] is False
        assert "empty description" in result["reason"]

    def test_call_failure_degrades_honestly_never_raises(self, monkeypatch):
        monkeypatch.setattr(llm_client, "API_KEY", "fake-key")
        monkeypatch.setattr(llm_client, "MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

        def boom(prompt, url, max_tokens=150):
            raise RuntimeError("provider is down")

        monkeypatch.setattr(llm_client, "call_llm_vision", boom)
        result = describe_image(FAKE_IMAGE_DATA_URL)  # must not raise
        assert result["provided"] is True
        assert result["analyzed"] is False
        assert "provider is down" in result["reason"]
