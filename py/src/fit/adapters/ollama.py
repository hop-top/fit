from __future__ import annotations

from typing import Any

import httpx

from ..types import Advice
from .base import Adapter


class OllamaAdapter(Adapter):
    """Ollama local adapter."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        http_client: Any | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        system_prompt = f"[Advisor Guidance]\n{advice.steering_text}"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        url = f"{self._base_url}/api/chat"
        client = self._http_client or httpx
        resp = client.post(url, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()

        output = data["message"]["content"]
        metadata = {
            "model": data.get("model", self._model),
            "provider": "ollama",
            "output": output,
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0)
                + data.get("eval_count", 0),
            },
        }
        return output, metadata
