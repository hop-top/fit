"""Regression tests for session and trace bugs.

Includes:
- Session context shape bug (flattened context dict)
- Trace serialization dropping advice.version
- Adapter failure producing no trace (PR#11)
- NaN→None for failure rewards (PR#15)
- Trace serialization dropping reward.metadata (PR#15)
"""

from __future__ import annotations

from typing import Any

from fit.advisor import Advisor
from fit.reward import RewardScorer
from fit.session import Session
from fit.trace import _trace_to_dict
from fit.types import Advice, Reward, Trace


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


def test_trace_to_dict_includes_advice_version():
    """Regression: _trace_to_dict() dropped advice.version from output.

    Before fix: the advice dict in serialized traces was missing the
    "version" key even though Advice.version was set (default "1.0").
    """
    advice = Advice(
        domain="test",
        steering_text="steer",
        confidence=0.9,
        version="1.0",
    )
    reward = Reward(score=0.95, breakdown={"accuracy": 1.0})
    trace = Trace(
        id="test-id",
        session_id="sess-1",
        timestamp="2026-04-15T10:00:00Z",
        input={"prompt": "hello", "context": {}},
        advice=advice,
        frontier={"model": "stub"},
        reward=reward,
    )

    d = _trace_to_dict(trace)

    assert "version" in d["advice"], (
        "advice dict missing 'version' key"
    )
    assert d["advice"]["version"] == "1.0"


class ErrorAdapter:
    """Adapter whose call() always raises RuntimeError."""

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        raise RuntimeError("adapter failed")


def test_adapter_failure_produces_partial_trace():
    """Regression: adapter failure must produce a partial trace.

    Before fix: session.run() propagated the exception with no trace.
    Spec requires frontier failures to still produce a trace with
    partial fields (null reward score, frontier error info).

    PR#15: changed from NaN to None per reward-schema-v1.
    """
    advisor = CaptureAdvisor()
    adapter = ErrorAdapter()
    scorer = FixedScorer()
    session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

    output, reward, trace = session.run("test")

    # Output should be empty string, not an exception
    assert isinstance(output, str)

    # Trace must be present
    assert trace is not None

    # Reward score must be None per reward-schema-v1 (not NaN)
    assert reward.score is None

    # Frontier must contain the error
    assert "error" in trace.frontier
    assert "adapter failed" in trace.frontier["error"]


class ErrorScorer:
    """Scorer whose score() always raises RuntimeError."""

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        raise RuntimeError("scorer exploded")


def test_scorer_failure_produces_null_reward():
    """Regression: scorer failure must produce null score (not NaN).

    PR#15: reward-schema-v1 specifies score: null on failure.
    """
    advisor = CaptureAdvisor()
    adapter = FakeAdapter()
    scorer = ErrorScorer()
    session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

    output, reward, trace = session.run("test")

    # Adapter succeeded, so output is non-empty
    assert output == "test"

    # Reward score must be None per reward-schema-v1
    assert reward.score is None


def test_trace_to_dict_includes_reward_metadata():
    """Regression: _trace_to_dict() dropped reward.metadata.

    PR#15: reward object in serialized trace must include metadata
    alongside score and breakdown.
    """
    advice = Advice(
        domain="test",
        steering_text="steer",
        confidence=0.9,
    )
    reward = Reward(
        score=0.95,
        breakdown={"accuracy": 1.0},
        metadata={"model_version": "2.1", "latency_ms": 120},
    )
    trace = Trace(
        id="test-id",
        session_id="sess-1",
        timestamp="2026-04-15T10:00:00Z",
        input={"prompt": "hello", "context": {}},
        advice=advice,
        frontier={"model": "stub"},
        reward=reward,
    )

    d = _trace_to_dict(trace)

    assert "metadata" in d["reward"], (
        "reward dict missing 'metadata' key"
    )
    assert d["reward"]["metadata"] == {"model_version": "2.1", "latency_ms": 120}


def test_trace_null_score_yaml_roundtrip():
    """Regression: trace with null score must round-trip through YAML.

    PR#15: null score must be parseable after YAML serialization.
    """
    import tempfile

    from fit.trace import TraceWriter, TraceReader

    advice = Advice(
        domain="test",
        steering_text="steer",
        confidence=0.9,
    )
    reward = Reward(score=None, breakdown={})
    trace = Trace(
        id="null-score-id",
        session_id="sess-null",
        timestamp="2026-04-16T10:00:00Z",
        input={"prompt": "test", "context": {}},
        advice=advice,
        frontier={"error": "adapter failed"},
        reward=reward,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TraceWriter(tmpdir)
        writer.write(trace, step=1)

        reader = TraceReader(tmpdir)
        loaded = reader.read("sess-null", step=1)

        # score must deserialize as None (not NaN string)
        assert loaded["reward"]["score"] is None
        assert loaded["reward"]["breakdown"] == {}
