from __future__ import annotations

import os
from typing import Any

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
                raise ValueError(
                    "Anthropic API key required: "
                    "pass api_key or set ANTHROPIC_API_KEY"
                )
            client = _anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

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
