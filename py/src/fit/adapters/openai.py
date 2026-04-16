from __future__ import annotations

import os
from typing import Any

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
                raise ValueError(
                    "OpenAI API key required: "
                    "pass api_key or set OPENAI_API_KEY"
                )
            client = _openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        output = response.choices[0].message.content
        metadata = {
            "model": response.model,
            "provider": "openai",
            "output": output,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        return output, metadata
