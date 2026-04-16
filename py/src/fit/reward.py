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
            list(weights) if weights
            else [] if not scorers
            else [1.0 / len(scorers)] * len(scorers)
        )
        if weights is not None and len(self._weights) != len(self._scorers):
            raise ValueError(
                f"weights/scorers length mismatch: "
                f"{len(self._scorers)} scorers but "
                f"{len(self._weights)} weights"
            )

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        rewards = [s.score(output, context) for s in self._scorers]
        # Merge breakdowns from all scorers
        merged_breakdown: dict[str, float] = {}
        for r in rewards:
            merged_breakdown.update(r.breakdown)
        # If any child score is None, propagate None (failure semantics per reward-schema-v1)
        if any(r.score is None for r in rewards):
            return Reward(
                score=None,
                breakdown=merged_breakdown,
                metadata={"scorers": len(rewards), "error": "child_score_is_null"},
            )
        total_weight = sum(self._weights)
        combined = sum(r.score * w for r, w in zip(rewards, self._weights))
        return Reward(
            score=combined / total_weight if total_weight else 0.0,
            breakdown=merged_breakdown,
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
