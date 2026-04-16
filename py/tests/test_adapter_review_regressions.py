"""Regression tests for issues found in PR code review.

These tests validate fixes for:
1. OpenAI content=None not guarded
2. Anthropic multi-block content truncated
3. Ollama request payload consistency across code paths
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fit.adapters import AnthropicAdapter, OllamaAdapter, OpenAIAdapter
from fit.types import Advice


def _make_advice() -> Advice:
    return Advice(domain="test", steering_text="Be concise.", confidence=0.9)


# ---------------------------------------------------------------------------
# 1. OpenAI content=None guard
# ---------------------------------------------------------------------------

class TestOpenAINullContentRegression:
    """OpenAI SDK allows choices[0].message.content to be None.
    The adapter must handle this — output must always be a string."""

    def test_content_none_returns_string(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=None))
        ]
        mock_response.model = "gpt-5"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIAdapter(api_key="test-key", client=mock_client)
        output, meta = adapter.call("q", _make_advice())

        assert isinstance(output, str), (
            f"output must be str, got {type(output).__name__}"
        )
        assert isinstance(meta["output"], str), (
            f"meta['output'] must be str, got {type(meta['output']).__name__}"
        )


# ---------------------------------------------------------------------------
# 2. Anthropic multi-block content
# ---------------------------------------------------------------------------

class TestAnthropicMultiBlockRegression:
    """Anthropic responses can contain multiple content blocks.
    The adapter must return the full text, not just the first block."""

    def test_multi_block_content_not_truncated(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="First paragraph. "),
            MagicMock(text="Second paragraph. "),
            MagicMock(text="Third paragraph."),
        ]
        mock_response.model = "claude-sonnet-4-6"
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=30)
        mock_client.messages.create.return_value = mock_response

        adapter = AnthropicAdapter(api_key="test-key", client=mock_client)
        output, _ = adapter.call("q", _make_advice())

        assert "First paragraph." in output
        assert "Second paragraph." in output
        assert "Third paragraph." in output


# ---------------------------------------------------------------------------
# 3. Ollama payload consistency
# ---------------------------------------------------------------------------

class TestOllamaPayloadConsistency:
    """Injected http_client and default httpx.post must produce
    identical request payloads."""

    def _mock_ollama_response(self) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "test"},
            "model": "llama3",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_payload_consistency(self) -> None:
        adapter = OllamaAdapter()

        # Capture payload via injected http_client
        mock_http = MagicMock()
        mock_http.post.return_value = self._mock_ollama_response()
        injected = OllamaAdapter(http_client=mock_http)
        injected.call("q", _make_advice())
        injected_call = mock_http.post.call_args

        # Capture payload via patched httpx.post
        with patch("fit.adapters.ollama.httpx.post") as mock_post:
            mock_post.return_value = self._mock_ollama_response()
            adapter.call("q", _make_advice())
            patched_call = mock_post.call_args

        # Compare payloads
        assert injected_call.args[0] == patched_call.args[0], (
            "URL mismatch between code paths"
        )
        assert injected_call.kwargs["json"] == patched_call.kwargs["json"], (
            "JSON payload mismatch between code paths"
        )
        assert injected_call.kwargs["timeout"] == patched_call.kwargs["timeout"], (
            "Timeout mismatch between code paths"
        )
