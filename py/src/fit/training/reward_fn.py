"""Composable reward functions for GRPO training.

Each implements the RewardFn protocol. Heavy deps (adapters, LLMs) are
lazily imported so core fit works without them.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


class RewardFn(ABC):
    """Protocol for reward functions used during training."""

    @abstractmethod
    def __call__(self, context: str, advice: str, output: str) -> float:
        """Compute reward score for the given (context, advice, output) triple."""


class ExactMatchReward(RewardFn):
    """Binary reward: 1.0 if output contains expected substring, else 0.0."""

    def __init__(self, expected: str, case_sensitive: bool = False) -> None:
        self._expected = expected
        self._case_sensitive = case_sensitive

    def __call__(self, context: str, advice: str, output: str) -> float:
        target = self._expected if self._case_sensitive else self._expected.lower()
        text = output if self._case_sensitive else output.lower()
        return 1.0 if target in text else 0.0


class RubricJudgeReward(RewardFn):
    """Keyword-based rubric scoring.

    Each rubric entry is a (pattern, weight) pair. Score is the weighted
    fraction of matched patterns.
    """

    def __init__(
        self,
        rubrics: list[tuple[str, float]],
        case_sensitive: bool = False,
    ) -> None:
        self._rubrics = rubrics
        self._flags = 0 if case_sensitive else re.IGNORECASE

    def __call__(self, context: str, advice: str, output: str) -> float:
        if not self._rubrics:
            return 0.0
        total_weight = sum(w for _, w in self._rubrics)
        if total_weight == 0:
            return 0.0
        matched_weight = 0.0
        for pattern, weight in self._rubrics:
            if re.search(pattern, output, self._flags):
                matched_weight += weight
        return matched_weight / total_weight


class LLMJudgeReward(RewardFn):
    """LLM-as-judge reward. Uses a frontier model to score output quality.

    Lazy-imports adapters so this is only pulled in when actually used.
    """

    def __init__(
        self,
        prompt_template: str | None = None,
        model: str = "claude-sonnet-4-6",
        adapter_config: dict[str, Any] | None = None,
    ) -> None:
        self._prompt_template = prompt_template or _DEFAULT_JUDGE_TEMPLATE
        self._model = model
        self._adapter_config = adapter_config or {}

    def __call__(self, context: str, advice: str, output: str) -> float:
        prompt = self._prompt_template.format(
            context=context, advice=advice, output=output
        )
        # Lazy import — adapters are optional deps
        try:
            from ..adapters.anthropic import AnthropicAdapter
            from ..types import Advice

            adapter = AnthropicAdapter(model=self._model, **self._adapter_config)
            judge_advice = Advice(
                domain="judge",
                steering_text="Score the output.",
                confidence=1.0,
            )
            result = adapter.call(prompt, judge_advice)
            score_text = result[0] if isinstance(result, tuple) else str(result)
            return _parse_score(score_text)
        except (ImportError, ValueError):
            # Fallback: return neutral score if adapter unavailable
            # or API key missing
            return 0.5


class UserSignalReward(RewardFn):
    """Reward from pre-computed scores keyed by output text hash.

    On each call, computes a stable SHA-256 hash of ``output``,
    truncates it to 16 hex characters, and looks up the corresponding
    score in ``scores``. Returns ``default`` when no matching hash
    key is present.
    """

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        default: float = 0.5,
    ) -> None:
        self._scores = scores or {}
        self._default = default

    def __call__(self, context: str, advice: str, output: str) -> float:
        # Look up score by SHA-256 hash of output text
        import hashlib

        key = hashlib.sha256(output.encode()).hexdigest()[:16]
        return self._scores.get(key, self._default)

    def add_score(self, key: str, score: float) -> None:
        self._scores[key] = score


class CompositeReward(RewardFn):
    """Weighted combination of multiple reward functions."""

    def __init__(self, fns: list[tuple[RewardFn, float]]) -> None:
        if not fns:
            raise ValueError("CompositeReward requires at least one (fn, weight) pair")
        total = sum(w for _, w in fns)
        if total <= 0:
            raise ValueError("Total weight must be positive")
        self._fns = fns
        self._total_weight = total

    def __call__(self, context: str, advice: str, output: str) -> float:
        weighted_sum = sum(fn(context, advice, output) * w for fn, w in self._fns)
        return weighted_sum / self._total_weight


def _parse_score(text: str) -> float:
    """Extract a numeric score from LLM judge output."""
    # Try prefixed fraction first: "Score: 4/5", "Rating: 3/10"
    match = re.search(
        r"(?:score|rating)[:\s]+(\d+\.?\d*)\s*/\s*(\d+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if match:
        denom = float(match.group(2))
        return min(float(match.group(1)) / denom, 1.0) if denom > 0 else 0.5

    # Try prefixed decimal/integer: "Score: 0.8", "Rating: 8"
    match = re.search(r"(?:score|rating)[:\s]+(\d+\.?\d*)", text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        raw = val if val <= 1.0 else val / 10.0
        return min(max(raw, 0.0), 1.0)

    # Try bare fraction: "4/5"
    match = re.search(r"(\d+\.?\d*)\s*/\s*(\d+)", text)
    if match:
        denom = float(match.group(2))
        return min(float(match.group(1)) / denom, 1.0) if denom > 0 else 0.5

    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        val = float(match.group(1))
        raw = val if val <= 1.0 else val / 10.0
        return min(max(raw, 0.0), 1.0)

    return 0.5


_DEFAULT_JUDGE_TEMPLATE = (
    "Rate the quality of the following output on a scale of 0.0 to 1.0.\n\n"
    "Context: {context}\n"
    "Advice given: {advice}\n"
    "Output: {output}\n\n"
    "Score:"
)
