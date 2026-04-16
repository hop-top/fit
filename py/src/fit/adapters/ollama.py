from __future__ import annotations

from typing import Any

from ..types import Advice
from .base import Adapter


class OllamaAdapter(Adapter):
    """Ollama local adapter."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base_url = base_url

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        system = f"[Advisor Guidance]\n{advice.steering_text}"
        return "(ollama stub)", {
            "model": self._model,
            "provider": "ollama",
            "output": "(ollama stub)",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
