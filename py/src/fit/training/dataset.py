"""Convert TraceRecords into (context, advice, reward) tuples for GRPO training."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .tracer import TraceRecord


@dataclass(frozen=True)
class TrainingExample:
    """A single training example for the advisor model."""

    context: str  # formatted context (prompt + metadata)
    advice: str  # steering_text from advisor
    reward: float  # normalized reward score
    session_id: str = ""  # for episode grouping
    metadata: dict[str, Any] = field(default_factory=dict)


class FitDataset:
    """Simple list-based dataset of TrainingExamples."""

    def __init__(self, examples: list[TrainingExample]) -> None:
        self._examples = list(examples)

    def split(
        self,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[FitDataset, FitDataset]:
        """Split into (train, val) datasets. Deterministic with seed."""
        if not 0.0 <= val_ratio <= 1.0:
            raise ValueError(
                f"val_ratio must be between 0.0 and 1.0, got {val_ratio}"
            )
        rng = random.Random(seed)
        indices = list(range(len(self._examples)))
        rng.shuffle(indices)
        val_count = int(len(indices) * val_ratio)
        if val_count == 0 and val_ratio > 0 and len(indices) > 0:
            val_count = 1
        if val_count >= len(indices) and len(indices) > 0:
            val_count = len(indices) - 1
        val_idx = set(indices[:val_count])

        train = [e for i, e in enumerate(self._examples) if i not in val_idx]
        val = [e for i, e in enumerate(self._examples) if i in val_idx]
        return FitDataset(train), FitDataset(val)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self._examples[idx]

    def __iter__(self):
        return iter(self._examples)

    @property
    def examples(self) -> list[TrainingExample]:
        return list(self._examples)

    def reward_stats(self) -> dict[str, float]:
        """Compute reward statistics over the dataset."""
        if not self._examples:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
        rewards = [e.reward for e in self._examples]
        n = len(rewards)
        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / n
        return {
            "mean": mean,
            "std": variance**0.5,
            "min": min(rewards),
            "max": max(rewards),
            "count": float(n),
        }


class DatasetBuilder:
    """Build a FitDataset from TraceRecords."""

    def __init__(self, records: list[TraceRecord]) -> None:
        self._records = list(records)

    def build(
        self,
        normalize_rewards: bool = True,
        group_by_session: bool = True,
    ) -> FitDataset:
        """Build dataset from loaded trace records.

        Args:
            normalize_rewards: Min-max normalize reward scores to [0, 1].
            group_by_session: Sort examples by session for episode grouping.
        """
        examples = self._records_to_examples(self._records)

        if group_by_session:
            examples = self._group_episodes(examples)

        if normalize_rewards:
            examples = self._normalize_rewards(examples)

        return FitDataset(examples)

    def _records_to_examples(
        self, records: list[TraceRecord]
    ) -> list[TrainingExample]:
        """Convert TraceRecords to TrainingExamples."""
        examples: list[TrainingExample] = []
        for rec in records:
            if rec.reward_score is None:
                continue  # skip traces with null rewards

            context = self._format_context(rec)
            examples.append(
                TrainingExample(
                    context=context,
                    advice=rec.advice_text,
                    reward=rec.reward_score,
                    session_id=rec.session_id,
                    metadata={
                        "domain": rec.advice_domain,
                        "confidence": rec.advice_confidence,
                        "frontier_model": rec.frontier_model,
                        "breakdown": rec.reward_breakdown,
                    },
                )
            )
        return examples

    def _format_context(self, rec: TraceRecord) -> str:
        """Format a TraceRecord into a context string for training."""
        parts = [f"Domain: {rec.advice_domain}"]
        if rec.prompt:
            parts.append(f"Prompt: {rec.prompt}")
        if rec.context:
            ctx_parts = [f"{k}={v}" for k, v in rec.context.items()]
            parts.append(f"Context: {', '.join(ctx_parts)}")
        if rec.frontier_output:
            parts.append(f"Output: {rec.frontier_output}")
        return "\n".join(parts)

    def _normalize_rewards(
        self, examples: list[TrainingExample]
    ) -> list[TrainingExample]:
        """Min-max normalize reward scores to [0, 1]."""
        if not examples:
            return examples
        rewards = [e.reward for e in examples]
        r_min, r_max = min(rewards), max(rewards)
        span = r_max - r_min
        if span == 0:
            # All same reward — normalize to 0.5
            return [
                TrainingExample(
                    context=e.context,
                    advice=e.advice,
                    reward=0.5,
                    session_id=e.session_id,
                    metadata=e.metadata,
                )
                for e in examples
            ]
        return [
            TrainingExample(
                context=e.context,
                advice=e.advice,
                reward=(e.reward - r_min) / span,
                session_id=e.session_id,
                metadata=e.metadata,
            )
            for e in examples
        ]

    def _group_episodes(
        self, examples: list[TrainingExample]
    ) -> list[TrainingExample]:
        """Sort examples by session_id for multi-turn episode grouping."""
        return sorted(examples, key=lambda e: e.session_id)
