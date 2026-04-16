from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import Advice


class Advisor(ABC):
    """Loads a trained advisor and generates per-instance advice."""

    @abstractmethod
    def generate_advice(self, context: dict[str, Any]) -> Advice:
        """Generate steering advice for the given context."""

    @abstractmethod
    def model_id(self) -> str:
        """Return advisor model identifier."""


class RemoteAdvisor(Advisor):
    """Advisor backed by a remote inference endpoint."""

    def __init__(self, endpoint: str, timeout_ms: int = 5000) -> None:
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms

    def generate_advice(self, context: dict[str, Any]) -> Advice:
        import httpx
        import json

        resp = httpx.post(
            f"{self._endpoint}/advise",
            json=context,
            timeout=self._timeout_ms / 1000,
        )
        resp.raise_for_status()
        data = resp.json()
        return Advice(
            domain=data["domain"],
            steering_text=data["steering_text"],
            confidence=data["confidence"],
            constraints=data.get("constraints", []),
            metadata=data.get("metadata", {}),
        )

    def model_id(self) -> str:
        return f"remote:{self._endpoint}"

    @classmethod
    def from_endpoint(cls, url: str) -> RemoteAdvisor:
        return cls(endpoint=url)
