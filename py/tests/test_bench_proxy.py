from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from fit.types import Advice


def _make_advice(text: str = "steer left") -> Advice:
    return Advice(
        domain="test",
        steering_text=text,
        confidence=0.9,
    )


def _make_messages(
    system: str = "You are helpful.",
    user: str = "Hello",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _make_upstream_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi!"}}
        ],
    }


def _sys_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def advisor() -> MagicMock:
    mock = MagicMock()
    mock.generate_advice.return_value = _make_advice()
    return mock


@pytest.fixture
def forwarder() -> MagicMock:
    mock = MagicMock()
    mock.return_value = _make_upstream_response()
    return mock


# ── Mode selection ──────────────────────────────────────────────


class TestModeSelection:
    def test_plan_mode(self, advisor: MagicMock, forwarder: MagicMock) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="plan", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)
        assert proc.config.mode == "plan"

    def test_session_mode(self, advisor: MagicMock, forwarder: MagicMock) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="session", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)
        assert proc.config.mode == "session"

    def test_oneshot_mode(self, advisor: MagicMock, forwarder: MagicMock) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)
        assert proc.config.mode == "oneshot"

    def test_invalid_mode_raises(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        with pytest.raises(ValueError, match="mode"):
            cfg = ProxyConfig(mode="bogus", upstream="openai")
            RequestProcessor(cfg, advisor, forwarder)


# ── System prompt injection ─────────────────────────────────────


class TestInjection:
    def test_advice_prepended_to_system(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)
        msgs = _make_messages()

        proc.process({"model": "gpt-5", "messages": msgs})

        call_args = forwarder.call_args
        sent_msgs = call_args[0][0]["messages"]
        sys_msg = next(m for m in sent_msgs if m["role"] == "system")
        assert sys_msg["content"].startswith("[Advisor Guidance]")
        assert "steer left" in sys_msg["content"]
        assert "You are helpful." in sys_msg["content"]

    def test_no_system_message_creates_one(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)
        msgs = [{"role": "user", "content": "Hello"}]

        proc.process({"model": "gpt-5", "messages": msgs})

        sent_msgs = forwarder.call_args[0][0]["messages"]
        sys_msg = sent_msgs[0]
        assert sys_msg["role"] == "system"
        assert "steer left" in sys_msg["content"]


# ── Plan mode ───────────────────────────────────────────────────


class TestPlanMode:
    def test_first_request_gets_advice(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="plan", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages()})
        assert advisor.generate_advice.call_count == 1

    def test_second_request_passthrough(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="plan", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages()})
        proc.process({"model": "gpt-5", "messages": _make_messages()})

        assert advisor.generate_advice.call_count == 1
        # second call forwards messages unmodified
        second_msgs = forwarder.call_args_list[1][0][0]["messages"]
        sys_content = next(
            m["content"] for m in second_msgs if m["role"] == "system"
        )
        assert "[Advisor Guidance]" not in sys_content

    def test_different_system_prompt_gets_advice(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="plan", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages("sys A")})
        proc.process({"model": "gpt-5", "messages": _make_messages("sys B")})

        assert advisor.generate_advice.call_count == 2


# ── Session mode ────────────────────────────────────────────────


class TestSessionMode:
    def test_per_turn_advice(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="session", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages()})
        proc.process({"model": "gpt-5", "messages": _make_messages()})

        assert advisor.generate_advice.call_count == 2

    def test_independent_state_per_system_prefix(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="session", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages("sys A")})
        proc.process({"model": "gpt-5", "messages": _make_messages("sys B")})

        # context passed to advisor should differ by system prefix
        ctx_a = advisor.generate_advice.call_args_list[0][0][0]
        ctx_b = advisor.generate_advice.call_args_list[1][0][0]
        assert ctx_a["session_key"] != ctx_b["session_key"]

    def test_same_prefix_same_session(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="session", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages("sys A")})
        proc.process({"model": "gpt-5", "messages": _make_messages("sys A")})

        ctx_1 = advisor.generate_advice.call_args_list[0][0][0]
        ctx_2 = advisor.generate_advice.call_args_list[1][0][0]
        assert ctx_1["session_key"] == ctx_2["session_key"]
        assert ctx_2["turn"] == 2


# ── Oneshot mode ────────────────────────────────────────────────


class TestOneshotMode:
    def test_every_request_gets_advice(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages()})
        proc.process({"model": "gpt-5", "messages": _make_messages()})
        proc.process({"model": "gpt-5", "messages": _make_messages()})

        assert advisor.generate_advice.call_count == 3

    def test_retry_detection_increments_count(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages()})
        proc.process({"model": "gpt-5", "messages": _make_messages()})

        ctx_1 = advisor.generate_advice.call_args_list[0][0][0]
        ctx_2 = advisor.generate_advice.call_args_list[1][0][0]
        assert ctx_1["request_num"] == 1
        assert ctx_2["request_num"] == 2

    def test_different_prefix_resets_count(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "gpt-5", "messages": _make_messages("sys A")})
        proc.process({"model": "gpt-5", "messages": _make_messages("sys B")})

        ctx_b = advisor.generate_advice.call_args_list[1][0][0]
        assert ctx_b["request_num"] == 1


# ── Response passthrough ────────────────────────────────────────


class TestResponsePassthrough:
    def test_upstream_response_returned_unmodified(
        self, advisor: MagicMock, forwarder: MagicMock
    ) -> None:
        from fit.bench.proxy import ProxyConfig, RequestProcessor

        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        result = proc.process(
            {"model": "gpt-5", "messages": _make_messages()}
        )
        assert result == _make_upstream_response()
