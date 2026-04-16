from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import Advice


class Adapter(ABC):
    """Frontier LLM adapter interface."""

    @abstractmethod
    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        """Call frontier LLM with advice injected. Returns (output, metadata)."""
