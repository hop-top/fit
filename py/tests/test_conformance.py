"""Conformance tests: load spec/fixtures/ and verify round-trip parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from fit.types import Advice, Reward, Trace

FIXTURES = Path(__file__).resolve().parent.parent.parent / "spec" / "fixtures"


def _load_yaml(name: str) -> dict[str, Any]:
    return yaml.safe_load((FIXTURES / name).read_text())


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


class TestAdviceConformance:
    def test_parse_yaml(self) -> None:
        data = _load_yaml("advice-v1.yaml")
        a = Advice(
            domain=data["domain"],
            steering_text=data["steering_text"],
            confidence=data["confidence"],
            constraints=data.get("constraints", []),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
        )
        assert a.domain == "tax-compliance"
        assert a.confidence == pytest.approx(0.87)
        assert len(a.constraints) == 3
        assert a.version == "1.0"
        assert "model" in a.metadata

    def test_parse_json(self) -> None:
        data = _load_json("advice-v1.json")
        a = Advice(
            domain=data["domain"],
            steering_text=data["steering_text"],
            confidence=data["confidence"],
            constraints=data.get("constraints", []),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
        )
        assert a.domain == "tax-compliance"
        assert a.confidence == pytest.approx(0.87)

    def test_yaml_json_equivalence(self) -> None:
        yml = _load_yaml("advice-v1.yaml")
        jsn = _load_json("advice-v1.json")
        assert yml["domain"] == jsn["domain"]
        assert yml["confidence"] == jsn["confidence"]
        assert yml["constraints"] == jsn["constraints"]

    def test_round_trip_defaults(self) -> None:
        a = Advice(domain="x", steering_text="y", confidence=0.5)
        assert a.constraints == []
        assert a.metadata == {}
        assert a.version == "1.0"

    def test_confidence_range(self) -> None:
        for val in [0.0, 0.5, 1.0]:
            a = Advice(domain="x", steering_text="y", confidence=val)
            assert 0.0 <= a.confidence <= 1.0


class TestRewardConformance:
    def test_parse_json(self) -> None:
        data = _load_json("reward-v1.json")
        r = Reward(
            score=data["score"],
            breakdown=data["breakdown"],
            metadata=data.get("metadata", {}),
        )
        assert r.score == pytest.approx(0.62)
        assert r.breakdown["accuracy"] == pytest.approx(0.7)
        assert r.breakdown["safety"] == pytest.approx(1.0)
        assert r.metadata["scorer"] == "rubric-judge-v2"

    def test_score_range(self) -> None:
        data = _load_json("reward-v1.json")
        assert 0.0 <= data["score"] <= 1.0
        for v in data["breakdown"].values():
            assert 0.0 <= v <= 1.0

    def test_round_trip_defaults(self) -> None:
        r = Reward(score=0.5, breakdown={"accuracy": 0.5})
        assert r.metadata == {}


class TestTraceConformance:
    def test_parse_yaml(self) -> None:
        data = _load_yaml("trace-v1.yaml")
        a = Advice(
            domain=data["advice"]["domain"],
            steering_text=data["advice"]["steering_text"],
            confidence=data["advice"]["confidence"],
            constraints=data["advice"].get("constraints", []),
            metadata=data["advice"].get("metadata", {}),
        )
        r = Reward(
            score=data["reward"]["score"],
            breakdown=data["reward"]["breakdown"],
        )
        t = Trace(
            id=data["id"],
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            input=data["input"],
            advice=a,
            frontier=data["frontier"],
            reward=r,
            metadata=data.get("metadata", {}),
        )
        assert t.id == "550e8400-e29b-41d4-a716-446655440000"
        assert t.session_id == "sess_abc123"
        assert t.advice.domain == "tax-compliance"
        assert t.frontier["provider"] == "anthropic"
        assert t.reward.score == pytest.approx(0.95)

    def test_required_fields_present(self) -> None:
        data = _load_yaml("trace-v1.yaml")
        for key in ("id", "session_id", "timestamp", "input", "advice", "frontier", "reward"):
            assert key in data, f"missing required field: {key}"

    def test_reward_determinism(self) -> None:
        data = _load_yaml("trace-v1.yaml")
        score1 = data["reward"]["score"]
        data2 = _load_yaml("trace-v1.yaml")
        score2 = data2["reward"]["score"]
        assert score1 == score2


class TestSessionMultiConformance:
    def test_multi_turn_parse(self) -> None:
        data = _load_yaml("session-multi.yaml")
        assert data["mode"] == "multi-turn"
        assert data["max_steps"] == 3
        assert len(data["steps"]) == 3

    def test_multi_turn_step_order(self) -> None:
        data = _load_yaml("session-multi.yaml")
        timestamps = [s["timestamp"] for s in data["steps"]]
        assert timestamps == sorted(timestamps)

    def test_multi_turn_session_id_consistent(self) -> None:
        data = _load_yaml("session-multi.yaml")
        sid = data["session_id"]
        for step in data["steps"]:
            assert step["session_id"] == sid

    def test_multi_turn_reward_progression(self) -> None:
        data = _load_yaml("session-multi.yaml")
        scores = [s["reward"]["score"] for s in data["steps"]]
        assert scores[0] < scores[-1]
