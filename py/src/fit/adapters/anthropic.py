from __future__ import annotations

from typing import Any

from ..types import Advice
from .base import Adapter


class AnthropicAdapter(Adapter):
    """Anthropic Claude adapter."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        # Stub: real implementation uses anthropic SDK
        system = f"[Advisor Guidance]\n{advice.steering_text}"
        return "(anthropic stub)", {
            "model": self._model,
            "provider": "anthropic",
            "output": "(anthropic stub)",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
