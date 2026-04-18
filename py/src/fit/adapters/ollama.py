from __future__ import annotations

from typing import Any

import httpx

from ..errors import (
    ADAPTER_AUTH,
    ADAPTER_RATE,
    ADAPTER_TIMEOUT,
    FitError,
)
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
        try:
            resp = client.post(url, json=payload, timeout=60.0)
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise FitError(
                ADAPTER_TIMEOUT, str(exc),
                cause=type(exc).__name__,
                fix="Increase timeout or check Ollama is running",
                retryable=True,
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise _map_ollama_http_error(exc) from exc
        except Exception as exc:
            raise FitError(
                ADAPTER_TIMEOUT, str(exc),
                cause=type(exc).__name__,
                fix="Check Ollama is running at "
                f"{self._base_url}",
            ) from exc

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


def _map_ollama_http_error(exc: httpx.HTTPStatusError) -> FitError:
    """Map Ollama HTTP status errors to FitError."""
    status = exc.response.status_code
    cls_name = type(exc).__name__

    if status == 401:
        return FitError(
            ADAPTER_AUTH, str(exc),
            cause=cls_name,
            fix="Check Ollama authentication config",
        )
    if status == 429:
        return FitError(
            ADAPTER_RATE, str(exc),
            cause=cls_name,
            fix="Back off and retry",
            retryable=True,
        )

    return FitError(
        ADAPTER_TIMEOUT, str(exc),
        cause=cls_name,
        fix="Check Ollama server status",
    )
