from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from .types import Reward


class RewardScorer(ABC):
    """Scores frontier LLM output."""

    @abstractmethod
    def score(self, output: str, context: dict[str, Any]) -> Reward:
        """Score the frontier output given context."""


class CompositeScorer(RewardScorer):
    """Weighted combination of multiple scorers."""

    def __init__(
        self,
        scorers: Sequence[RewardScorer],
        weights: Sequence[float] | None = None,
    ) -> None:
        self._scorers = list(scorers)
        self._weights = (
            list(weights) if weights else [1.0 / len(scorers)] * len(scorers)
        )

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        rewards = [s.score(output, context) for s in self._scorers]
        total_weight = sum(self._weights)
        combined = sum(r.score * w for r, w in zip(rewards, self._weights))
        return Reward(
            score=combined / total_weight if total_weight else 0.0,
            breakdown=rewards[0].breakdown if rewards else {},
            metadata={"scorers": len(rewards)},
        )

    @classmethod
    def composite(cls, names: list[str]) -> CompositeScorer:
        """Convenience: create from dimension names."""
        scorers = [DimensionScorer(n) for n in names]
        return cls(scorers)


class DimensionScorer(RewardScorer):
    """Stub scorer for a single dimension (returns neutral scores)."""

    def __init__(self, dimension: str) -> None:
        self._dimension = dimension

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        return Reward(
            score=0.5,
            breakdown={self._dimension: 0.5},
        )
