from __future__ import annotations

from typing import Any

from ..types import Advice
from .base import Adapter


class OpenAIAdapter(Adapter):
    """OpenAI adapter."""

    def __init__(self, model: str = "gpt-5", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        system = f"[Advisor Guidance]\n{advice.steering_text}"
        return "(openai stub)", {
            "model": self._model,
            "provider": "openai",
            "output": "(openai stub)",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
