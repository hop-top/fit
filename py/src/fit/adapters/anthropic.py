from __future__ import annotations

import os
from typing import Any

import httpx

from ..errors import (
    ADAPTER_AUTH,
    ADAPTER_MODEL,
    ADAPTER_RATE,
    ADAPTER_TIMEOUT,
    FitError,
)
from ..types import Advice
from .base import Adapter

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore[assignment]


class AnthropicAdapter(Adapter):
    """Anthropic Claude adapter."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if _anthropic is None and client is None:
            raise ImportError(
                "anthropic package is required: pip install fit[adapters]"
            )
        self._model = model
        self._api_key = api_key
        self._client = client

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        system_prompt = f"[Advisor Guidance]\n{advice.steering_text}"

        if self._client is not None:
            client = self._client
        else:
            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise FitError(
                    ADAPTER_AUTH,
                    "Anthropic API key required: "
                    "pass api_key or set ANTHROPIC_API_KEY",
                    fix="export ANTHROPIC_API_KEY=... or pass api_key=",
                )
            client = _anthropic.Anthropic(api_key=api_key)

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise _map_anthropic_error(exc) from exc

        output = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        metadata = {
            "model": response.model,
            "provider": "anthropic",
            "output": output,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            },
        }
        return output, metadata


def _map_anthropic_error(exc: Exception) -> FitError:
    """Convert Anthropic SDK / transport exceptions to FitError."""
    cls_name = type(exc).__name__

    if _anthropic is not None:
        if isinstance(exc, _anthropic.AuthenticationError):
            return FitError(
                ADAPTER_AUTH, str(exc),
                cause=cls_name,
                fix="Check ANTHROPIC_API_KEY is valid",
            )
        if isinstance(exc, _anthropic.RateLimitError):
            return FitError(
                ADAPTER_RATE, str(exc),
                cause=cls_name,
                fix="Back off and retry",
                retryable=True,
            )
        if isinstance(exc, _anthropic.NotFoundError):
            return FitError(
                ADAPTER_MODEL, str(exc),
                cause=cls_name,
                fix="Check model name is valid",
            )

    if isinstance(exc, httpx.TimeoutException):
        return FitError(
            ADAPTER_TIMEOUT, str(exc),
            cause=cls_name,
            fix="Increase timeout or retry",
            retryable=True,
        )

    # Unknown SDK error — still wrap with generic code
    return FitError(
        ADAPTER_MODEL, str(exc),
        cause=cls_name,
    )
