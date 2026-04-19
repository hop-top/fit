from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from fit.bench.proxy import ProxyConfig, RequestProcessor
from fit.types import Advice


# ── Helpers ────────────────────────────────────────────────────


def _advice(text: str = "steer left") -> Advice:
    return Advice(domain="test", steering_text=text, confidence=0.9)


def _msgs(
    system: str | None = "You are helpful.",
    user: str = "Hello",
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if system is not None:
        out.append({"role": "system", "content": system})
    out.append({"role": "user", "content": user})
    return out


def _upstream() -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi!"},
            }
        ],
    }


def _sys_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def advisor() -> MagicMock:
    m = MagicMock()
    m.generate_advice.return_value = _advice()
    return m


@pytest.fixture
def forwarder() -> MagicMock:
    m = MagicMock()
    m.return_value = _upstream()
    return m


# ── 1. End-to-end flow ────────────────────────────────────────


class TestEndToEnd:
    """Full request → advisor → injection → forward → response."""

    def test_full_pipeline(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        req = {"model": "gpt-5", "messages": _msgs()}
        result = proc.process(req)

        # advisor called once
        assert advisor.generate_advice.call_count == 1
        ctx = advisor.generate_advice.call_args[0][0]
        assert ctx["system_prompt"] == "You are helpful."

        # forwarder received modified messages
        fwd_req = forwarder.call_args[0][0]
        sys_msg = next(
            m for m in fwd_req["messages"] if m["role"] == "system"
        )
        assert sys_msg["content"].startswith("[Advisor Guidance]")
        assert "steer left" in sys_msg["content"]
        assert "You are helpful." in sys_msg["content"]

        # model param preserved
        assert fwd_req["model"] == "gpt-5"

        # upstream response returned
        assert result == _upstream()

    def test_original_request_not_mutated(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        original_msgs = _msgs()
        original_sys = original_msgs[0]["content"]
        req = {"model": "gpt-5", "messages": original_msgs}

        proc.process(req)

        # original message list content preserved (deep copy in process)
        assert original_msgs[0]["content"] == original_sys

    def test_extra_request_fields_preserved(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        req = {
            "model": "gpt-5",
            "messages": _msgs(),
            "temperature": 0.7,
            "max_tokens": 100,
        }
        proc.process(req)

        fwd = forwarder.call_args[0][0]
        assert fwd["temperature"] == 0.7
        assert fwd["max_tokens"] == 100


# ── 2. Mode transitions ──────────────────────────────────────


class TestModeTransitions:
    """Plan vs session state isolation; cross-mode independence."""

    def test_plan_state_persists(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """First request advised, subsequent passthrough for same sys."""
        cfg = ProxyConfig(mode="plan", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        req = {"model": "gpt-5", "messages": _msgs()}
        proc.process(req)
        proc.process(req)
        proc.process(req)

        assert advisor.generate_advice.call_count == 1
        # last forwarded messages should NOT have guidance
        last = forwarder.call_args_list[-1][0][0]["messages"]
        sys_c = next(m["content"] for m in last if m["role"] == "system")
        assert "[Advisor Guidance]" not in sys_c

    def test_session_independent_per_key(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Different system prompts = different sessions."""
        cfg = ProxyConfig(mode="session", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "m", "messages": _msgs("sys A")})
        proc.process({"model": "m", "messages": _msgs("sys B")})
        proc.process({"model": "m", "messages": _msgs("sys A")})

        # sys A turn 2, sys B turn 1
        ctx_a2 = advisor.generate_advice.call_args_list[2][0][0]
        ctx_b1 = advisor.generate_advice.call_args_list[1][0][0]
        assert ctx_a2["turn"] == 2
        assert ctx_b1["turn"] == 1

    def test_plan_and_session_state_not_shared(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Two processors in different modes have independent state."""
        plan_cfg = ProxyConfig(mode="plan", upstream="openai")
        sess_cfg = ProxyConfig(mode="session", upstream="openai")
        plan_proc = RequestProcessor(plan_cfg, advisor, forwarder)
        sess_proc = RequestProcessor(sess_cfg, advisor, forwarder)

        req = {"model": "m", "messages": _msgs()}

        plan_proc.process(req)
        plan_proc.process(req)  # passthrough (plan seen)
        sess_proc.process(req)
        sess_proc.process(req)  # still advised (session always advises)

        # plan: 1 advice, session: 2 advice = 3 total
        assert advisor.generate_advice.call_count == 3

    def test_plan_new_system_resets_advisor(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Plan mode advises again when system prompt changes."""
        cfg = ProxyConfig(mode="plan", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "m", "messages": _msgs("A")})
        proc.process({"model": "m", "messages": _msgs("A")})  # skip
        proc.process({"model": "m", "messages": _msgs("B")})  # new
        proc.process({"model": "m", "messages": _msgs("B")})  # skip
        proc.process({"model": "m", "messages": _msgs("A")})  # skip

        assert advisor.generate_advice.call_count == 2


# ── 3. Edge cases ─────────────────────────────────────────────


class TestEdgeCases:
    def test_no_system_message_injects_new(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        req = {"model": "m", "messages": _msgs(system=None)}
        proc.process(req)

        fwd = forwarder.call_args[0][0]["messages"]
        assert fwd[0]["role"] == "system"
        assert "steer left" in fwd[0]["content"]
        # user message still present after system
        assert fwd[1]["role"] == "user"

    def test_empty_messages_array(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Empty messages — proxy should still process (no system)."""
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        req = {"model": "m", "messages": []}
        proc.process(req)

        fwd = forwarder.call_args[0][0]["messages"]
        assert len(fwd) == 1
        assert fwd[0]["role"] == "system"

    def test_empty_steering_text(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Advisor returns empty steering — still injects wrapper."""
        advisor.generate_advice.return_value = _advice(text="")
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "m", "messages": _msgs()})

        fwd = forwarder.call_args[0][0]["messages"]
        sys_c = next(m["content"] for m in fwd if m["role"] == "system")
        # still has wrapper even with empty text
        assert "[Advisor Guidance]" in sys_c
        assert "You are helpful." in sys_c

    def test_forwarder_raises_exception(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Forwarder exception propagates (no swallowing)."""
        forwarder.side_effect = ConnectionError("upstream down")
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        with pytest.raises(ConnectionError, match="upstream down"):
            proc.process({"model": "m", "messages": _msgs()})

        # advisor was still called before forwarder blew up
        assert advisor.generate_advice.call_count == 1

    def test_very_long_system_message_passed_through(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Long system messages not truncated."""
        long_sys = "x" * 100_000
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "m", "messages": _msgs(system=long_sys)})

        fwd = forwarder.call_args[0][0]["messages"]
        sys_c = next(m["content"] for m in fwd if m["role"] == "system")
        assert long_sys in sys_c
        assert len(sys_c) > 100_000

    def test_advisor_context_includes_empty_string_for_no_system(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """When no system message, advisor context uses empty string."""
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "m", "messages": _msgs(system=None)})

        ctx = advisor.generate_advice.call_args[0][0]
        assert ctx["system_prompt"] == ""

    def test_multiple_system_messages_only_first_injected(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Only first system message gets injection."""
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        msgs = [
            {"role": "system", "content": "First system"},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Second system"},
        ]
        proc.process({"model": "m", "messages": msgs})

        fwd = forwarder.call_args[0][0]["messages"]
        first_sys = fwd[0]
        second_sys = fwd[2]
        assert "[Advisor Guidance]" in first_sys["content"]
        assert "[Advisor Guidance]" not in second_sys["content"]
        assert second_sys["content"] == "Second system"

    def test_system_hash_stability(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Same system content always produces same hash/session key."""
        cfg = ProxyConfig(mode="session", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "m", "messages": _msgs("same")})
        proc.process({"model": "m", "messages": _msgs("same")})

        k1 = advisor.generate_advice.call_args_list[0][0][0]["session_key"]
        k2 = advisor.generate_advice.call_args_list[1][0][0]["session_key"]
        assert k1 == k2 == _sys_hash("same")

    def test_request_missing_messages_key(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Missing 'messages' key raises KeyError."""
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        with pytest.raises(KeyError):
            proc.process({"model": "m"})


# ── 4. Concurrency safety ────────────────────────────────────


class TestConcurrencySafety:
    """Sequential rapid access — state correctness under interleaving."""

    def test_interleaved_sessions_no_interference(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Alternating session keys maintain correct turn counts."""
        cfg = ProxyConfig(mode="session", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        keys = ["A", "B", "A", "B", "A"]
        for k in keys:
            proc.process({"model": "m", "messages": _msgs(k)})

        # A: turns 1, 2, 3; B: turns 1, 2
        calls = advisor.generate_advice.call_args_list
        a_turns = [
            c[0][0]["turn"] for c in calls
            if c[0][0]["session_key"] == _sys_hash("A")
        ]
        b_turns = [
            c[0][0]["turn"] for c in calls
            if c[0][0]["session_key"] == _sys_hash("B")
        ]
        assert a_turns == [1, 2, 3]
        assert b_turns == [1, 2]

    def test_rapid_plan_mode_same_key(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Many rapid plan-mode requests for same key — only one advice."""
        cfg = ProxyConfig(mode="plan", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        for _ in range(50):
            proc.process({"model": "m", "messages": _msgs()})

        assert advisor.generate_advice.call_count == 1

    def test_oneshot_counts_across_keys(
        self, advisor: MagicMock, forwarder: MagicMock,
    ) -> None:
        """Oneshot request_num tracked per key, not globally."""
        cfg = ProxyConfig(mode="oneshot", upstream="openai")
        proc = RequestProcessor(cfg, advisor, forwarder)

        proc.process({"model": "m", "messages": _msgs("X")})
        proc.process({"model": "m", "messages": _msgs("X")})
        proc.process({"model": "m", "messages": _msgs("Y")})
        proc.process({"model": "m", "messages": _msgs("X")})

        calls = advisor.generate_advice.call_args_list
        x_nums = [
            c[0][0]["request_num"] for c in calls
            if c[0][0]["session_key"] == _sys_hash("X")
        ]
        y_nums = [
            c[0][0]["request_num"] for c in calls
            if c[0][0]["session_key"] == _sys_hash("Y")
        ]
        assert x_nums == [1, 2, 3]
        assert y_nums == [1]
