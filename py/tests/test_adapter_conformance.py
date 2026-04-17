"""Adapter-level conformance tests: frontier metadata shape, provider values,
usage structure, session integration, and advice injection.

Uses mocks to avoid requiring real API keys or running services."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fit.adapters import AnthropicAdapter, OllamaAdapter, OpenAIAdapter
from fit.adapters.base import Adapter
from fit.advisor import Advisor
from fit.reward import DimensionScorer
from fit.session import Session, SessionConfig
from fit.types import Advice

REQUIRED_FRONTIER_KEYS = {"model", "provider", "output", "usage"}
REQUIRED_USAGE_KEYS = {"prompt_tokens", "completion_tokens", "total_tokens"}


def _make_advice() -> Advice:
    return Advice(
        domain="test",
        steering_text="Be concise.",
        confidence=0.9,
    )


def _mock_anthropic_response() -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text="mock anthropic output")]
    resp.model = "claude-sonnet-4-6"
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


def _mock_openai_response() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="mock openai output"))]
    resp.model = "gpt-5"
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return resp


def _mock_ollama_response() -> dict[str, Any]:
    return {
        "message": {"content": "mock ollama output"},
        "model": "llama3",
        "prompt_eval_count": 80,
        "eval_count": 40,
    }


def _make_anthropic() -> tuple[AnthropicAdapter, MagicMock]:
    """Anthropic adapter with injected mock client."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response()
    return AnthropicAdapter(api_key="test-key", client=mock_client), mock_client


def _make_openai() -> tuple[OpenAIAdapter, MagicMock]:
    """OpenAI adapter with injected mock client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response()
    return OpenAIAdapter(api_key="test-key", client=mock_client), mock_client


def _call_ollama(
    adapter: OllamaAdapter,
    prompt: str,
    advice: Advice,
) -> tuple[str, dict[str, Any]]:
    """Call ollama adapter with httpx.post patched."""
    with patch("fit.adapters.ollama.httpx.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_ollama_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        return adapter.call(prompt, advice)


def _call_provider(
    provider: str,
    prompt: str = "hello",
    advice: Advice | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call any provider's adapter with full mocking. Returns (output, meta)."""
    advice = advice or _make_advice()
    if provider == "anthropic":
        adapter, _ = _make_anthropic()
        return adapter.call(prompt, advice)
    elif provider == "openai":
        adapter, _ = _make_openai()
        return adapter.call(prompt, advice)
    else:
        return _call_ollama(OllamaAdapter(), prompt, advice)


ADAPTER_PROVIDERS = [
    ("anthropic", AnthropicAdapter),
    ("openai", OpenAIAdapter),
    ("ollama", OllamaAdapter),
]


# ---------------------------------------------------------------------------
# 1. Adapter metadata shape
# ---------------------------------------------------------------------------

class TestAdapterMetadataShape:
    """Each adapter's call() must return metadata with all required fields."""

    @pytest.fixture(params=ADAPTER_PROVIDERS, ids=["anthropic", "openai", "ollama"])
    def adapter_result(self, request: pytest.FixtureRequest) -> tuple[str, dict[str, Any]]:
        provider, _cls = request.param
        return _call_provider(provider)

    def test_metadata_has_required_keys(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        missing = REQUIRED_FRONTIER_KEYS - set(meta.keys())
        assert not missing, f"adapter missing keys: {missing}"

    def test_model_is_string(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        assert isinstance(meta["model"], str)

    def test_provider_is_string(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        assert isinstance(meta["provider"], str)

    def test_output_is_string(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        assert isinstance(meta["output"], str)

    def test_usage_is_dict(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        assert isinstance(meta["usage"], dict)

    def test_call_returns_tuple(self, adapter_result: tuple) -> None:
        assert isinstance(adapter_result, tuple)
        assert len(adapter_result) == 2
        assert isinstance(adapter_result[0], str)
        assert isinstance(adapter_result[1], dict)


# ---------------------------------------------------------------------------
# 2. Provider field values
# ---------------------------------------------------------------------------

class TestProviderFieldValues:

    def test_anthropic_provider(self) -> None:
        _, meta = _call_provider("anthropic", "q")
        assert meta["provider"] == "anthropic"

    def test_openai_provider(self) -> None:
        _, meta = _call_provider("openai", "q")
        assert meta["provider"] == "openai"

    def test_ollama_provider(self) -> None:
        _, meta = _call_provider("ollama", "q")
        assert meta["provider"] == "ollama"


# ---------------------------------------------------------------------------
# 3. Usage structure
# ---------------------------------------------------------------------------

class TestUsageStructure:

    @pytest.fixture(params=ADAPTER_PROVIDERS, ids=["anthropic", "openai", "ollama"])
    def adapter_result(self, request: pytest.FixtureRequest) -> tuple[str, dict[str, Any]]:
        provider, _cls = request.param
        return _call_provider(provider)

    def test_usage_has_exactly_three_keys(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        assert set(meta["usage"].keys()) == REQUIRED_USAGE_KEYS

    def test_prompt_tokens_is_int(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        assert isinstance(meta["usage"]["prompt_tokens"], int)

    def test_completion_tokens_is_int(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        assert isinstance(meta["usage"]["completion_tokens"], int)

    def test_total_tokens_is_int(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        assert isinstance(meta["usage"]["total_tokens"], int)

    def test_usage_values_non_negative(self, adapter_result: tuple) -> None:
        _, meta = adapter_result
        usage = meta["usage"]
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0
        assert usage["total_tokens"] >= 0


# ---------------------------------------------------------------------------
# 4. Session frontier metadata
# ---------------------------------------------------------------------------

def _stub_advisor() -> Advisor:
    advisor = MagicMock(spec=Advisor)
    advisor.generate_advice.return_value = Advice(
        domain="test-domain",
        steering_text="Be precise.",
        confidence=0.8,
    )
    return advisor


class _MockAdapter(Adapter):
    """Adapter that returns spec-compliant metadata without network calls."""

    def __init__(self, provider: str) -> None:
        self._provider = provider

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        return f"mock-{self._provider}-output", {
            "model": f"mock-{self._provider}-model",
            "provider": self._provider,
            "output": f"mock-{self._provider}-output",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class TestSessionFrontierMetadata:

    @pytest.fixture(params=["anthropic", "openai", "ollama"])
    def session_result(self, request: pytest.FixtureRequest) -> tuple[str, Any, Any]:
        adapter = _MockAdapter(request.param)
        scorer = DimensionScorer("accuracy")
        session = Session(
            advisor=_stub_advisor(),
            adapter=adapter,
            scorer=scorer,
            config=SessionConfig(),
        )
        return session.run("test prompt", {"key": "val"})

    def test_frontier_has_required_keys(self, session_result: tuple) -> None:
        _, _, trace = session_result
        missing = REQUIRED_FRONTIER_KEYS - set(trace.frontier.keys())
        assert not missing, f"trace.frontier missing keys: {missing}"

    def test_frontier_model_is_string(self, session_result: tuple) -> None:
        _, _, trace = session_result
        assert isinstance(trace.frontier["model"], str)

    def test_frontier_provider_is_string(self, session_result: tuple) -> None:
        _, _, trace = session_result
        assert isinstance(trace.frontier["provider"], str)

    def test_frontier_output_is_string(self, session_result: tuple) -> None:
        _, _, trace = session_result
        assert isinstance(trace.frontier["output"], str)

    def test_frontier_usage_has_required_keys(self, session_result: tuple) -> None:
        _, _, trace = session_result
        usage = trace.frontier["usage"]
        missing = REQUIRED_USAGE_KEYS - set(usage.keys())
        assert not missing, f"frontier.usage missing keys: {missing}"

    def test_frontier_usage_values_are_int(self, session_result: tuple) -> None:
        _, _, trace = session_result
        usage = trace.frontier["usage"]
        for key in REQUIRED_USAGE_KEYS:
            assert isinstance(usage[key], int), f"{key} is not int"


# ---------------------------------------------------------------------------
# 5. Advice injection (system prompt prefix)
# ---------------------------------------------------------------------------

class TestAdviceInjection:

    def test_anthropic_passes_system_prompt(self) -> None:
        adapter, mock_client = _make_anthropic()
        advice = Advice(domain="tax", steering_text="Cite IRS publications.", confidence=0.9)
        adapter.call("q", advice)
        call_kwargs = mock_client.messages.create.call_args
        assert "[Advisor Guidance]" in call_kwargs.kwargs["system"]
        assert "Cite IRS publications." in call_kwargs.kwargs["system"]

    def test_openai_passes_system_prompt(self) -> None:
        adapter, mock_client = _make_openai()
        advice = Advice(domain="tax", steering_text="Cite IRS publications.", confidence=0.9)
        adapter.call("q", advice)
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "[Advisor Guidance]" in system_msg["content"]
        assert "Cite IRS publications." in system_msg["content"]

    def test_ollama_passes_system_prompt(self) -> None:
        adapter = OllamaAdapter()
        advice = Advice(domain="tax", steering_text="Cite IRS publications.", confidence=0.9)
        with patch("fit.adapters.ollama.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_ollama_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp
            adapter.call("q", advice)
        messages = mock_post.call_args.kwargs["json"]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "[Advisor Guidance]" in system_msg["content"]
        assert "Cite IRS publications." in system_msg["content"]

    def test_steering_text_propagated(self) -> None:
        text = "Always cite sources."
        advice = Advice(domain="test", steering_text=text, confidence=0.5)
        adapter, mock_client = _make_openai()
        adapter.call("direct", advice)
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert text in system_msg["content"]
