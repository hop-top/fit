"""Regression tests for session context shape bug.

Bug: generate_advice() was called with {"prompt": prompt, **ctx}
which flattened the context dict into the top level. The session
protocol spec requires {"prompt": prompt, "context": ctx}.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fit.advisor import Advisor
from fit.reward import RewardScorer
from fit.session import Session
from fit.types import Advice, Reward


class CaptureAdvisor(Advisor):
    """Advisor that captures the context dict it receives."""

    def __init__(self) -> None:
        self.captured_context: dict[str, Any] | None = None

    def generate_advice(self, context: dict[str, Any]) -> Advice:
        self.captured_context = context
        return Advice(domain="test", steering_text="test advice", confidence=0.9)

    def model_id(self) -> str:
        return "capture-advisor"


class FixedScorer(RewardScorer):
    """Scorer that returns a fixed reward."""

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        return Reward(score=0.8, breakdown={"accuracy": 0.8})


class FakeAdapter:
    """Minimal adapter that echoes the prompt."""

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        return prompt, {"provider": "fake"}


def test_advisor_receives_nested_context_shape():
    """Regression: advisor must receive {"prompt": ..., "context": {...}}.

    Before fix: advisor received {"prompt": ..., **ctx} which flattened
    context keys into the top-level dict. For example, with context
    {"user_id": "abc"}, the advisor would get {"prompt": "...", "user_id": "abc"}
    instead of {"prompt": "...", "context": {"user_id": "abc"}}.
    """
    advisor = CaptureAdvisor()
    adapter = FakeAdapter()
    scorer = FixedScorer()
    session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

    ctx = {"user_id": "abc", "locale": "en-US"}
    session.run("hello", context=ctx)

    assert advisor.captured_context is not None
    # Top-level must have exactly two keys: "prompt" and "context"
    assert set(advisor.captured_context.keys()) == {"prompt", "context"}, (
        f"advisor received keys {set(advisor.captured_context.keys())}, "
        f"expected {{'prompt', 'context'}}"
    )
    # "context" must be a dict with the original context entries
    assert advisor.captured_context["prompt"] == "hello"
    assert advisor.captured_context["context"] == ctx


def test_advisor_no_context_collision():
    """Regression: context keys must not leak to top level.

    Before fix: if context had a key like "prompt", the **ctx spread
    would overwrite the "prompt" value. This test ensures the nested
    shape prevents such collisions.
    """
    advisor = CaptureAdvisor()
    adapter = FakeAdapter()
    scorer = FixedScorer()
    session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

    # context with a key that would collide if flattened
    ctx = {"prompt": "evil override", "domain": "test-domain"}
    session.run("real prompt", context=ctx)

    assert advisor.captured_context is not None
    # "prompt" at top level must be the session prompt, not the context one
    assert advisor.captured_context["prompt"] == "real prompt"
    # The collision key should live inside "context"
    assert advisor.captured_context["context"]["prompt"] == "evil override"
