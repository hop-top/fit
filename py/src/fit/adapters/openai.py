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
    import openai as _openai
except ImportError:
    _openai = None  # type: ignore[assignment]


class OpenAIAdapter(Adapter):
    """OpenAI adapter."""

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if _openai is None and client is None:
            raise ImportError(
                "openai package is required: pip install fit[adapters]"
            )
        self._model = model
        self._api_key = api_key
        self._client = client

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        system_prompt = f"[Advisor Guidance]\n{advice.steering_text}"

        if self._client is not None:
            client = self._client
        else:
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise FitError(
                    ADAPTER_AUTH,
                    "OpenAI API key required: "
                    "pass api_key or set OPENAI_API_KEY",
                    fix="export OPENAI_API_KEY=... or pass api_key=",
                )
            client = _openai.OpenAI(api_key=api_key)

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise _map_openai_error(exc) from exc

        output = response.choices[0].message.content or ""
        usage = response.usage
        metadata = {
            "model": response.model,
            "provider": "openai",
            "output": output,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            },
        }
        return output, metadata


def _map_openai_error(exc: Exception) -> FitError:
    """Convert OpenAI SDK / transport exceptions to FitError."""
    cls_name = type(exc).__name__

    if _openai is not None:
        if isinstance(exc, _openai.AuthenticationError):
            return FitError(
                ADAPTER_AUTH, str(exc),
                cause=cls_name,
                fix="Check OPENAI_API_KEY is valid",
            )
        if isinstance(exc, _openai.RateLimitError):
            return FitError(
                ADAPTER_RATE, str(exc),
                cause=cls_name,
                fix="Back off and retry",
                retryable=True,
            )
        if isinstance(exc, _openai.NotFoundError):
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

    return FitError(
        ADAPTER_MODEL, str(exc),
        cause=cls_name,
    )
