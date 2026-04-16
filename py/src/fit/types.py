from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Advice:
    """Advisor output (advice-format-v1)."""

    domain: str
    steering_text: str
    confidence: float
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"


@dataclass(frozen=True)
class Reward:
    """Reward scoring result (reward-schema-v1)."""

    score: float | None
    breakdown: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trace:
    """Session trace record (trace-format-v1)."""

    id: str
    session_id: str
    timestamp: str
    input: dict[str, Any]
    advice: Advice
    frontier: dict[str, Any]
    reward: Reward
    metadata: dict[str, Any] = field(default_factory=dict)
