from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from ..advisor import Advisor

_VALID_MODES = {"plan", "session", "oneshot"}


@dataclass
class ProxyConfig:
    mode: str
    upstream: str
    host: str = "0.0.0.0"
    port: int = 8090

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {self.mode!r}"
            )


Forwarder = Callable[[dict[str, Any]], dict[str, Any]]


def _system_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _extract_system(
    messages: list[dict[str, str]],
) -> tuple[str | None, int | None]:
    """Return (system_content, index) or (None, None)."""
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            return m["content"], i
    return None, None


class RequestProcessor:
    """Core logic for intercepting, advising, and forwarding requests."""

    def __init__(
        self,
        config: ProxyConfig,
        advisor: Advisor,
        forwarder: Forwarder,
    ) -> None:
        self.config = config
        self._advisor = advisor
        self._forwarder = forwarder

        # plan mode: set of system hashes already advised
        self._plan_seen: set[str] = set()

        # session mode: {session_key: turn_count}
        self._session_turns: dict[str, int] = {}

        # oneshot mode: {system_hash: request_count}
        self._oneshot_counts: dict[str, int] = {}

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = [dict(m) for m in request["messages"]]
        sys_content, sys_idx = _extract_system(messages)
        sys_key = _system_hash(sys_content or "")

        handler = getattr(self, f"_handle_{self.config.mode}")
        messages = handler(messages, sys_content, sys_idx, sys_key)

        forwarded = dict(request)
        forwarded["messages"] = messages
        return self._forwarder(forwarded)

    # ── plan ────────────────────────────────────────────────────

    def _handle_plan(
        self,
        messages: list[dict[str, str]],
        sys_content: str | None,
        sys_idx: int | None,
        sys_key: str,
    ) -> list[dict[str, str]]:
        if sys_key in self._plan_seen:
            return messages
        self._plan_seen.add(sys_key)
        return self._inject(messages, sys_content, sys_idx, sys_key)

    # ── session ─────────────────────────────────────────────────

    def _handle_session(
        self,
        messages: list[dict[str, str]],
        sys_content: str | None,
        sys_idx: int | None,
        sys_key: str,
    ) -> list[dict[str, str]]:
        turn = self._session_turns.get(sys_key, 0) + 1
        self._session_turns[sys_key] = turn
        return self._inject(
            messages, sys_content, sys_idx, sys_key, turn=turn,
        )

    # ── oneshot ─────────────────────────────────────────────────

    def _handle_oneshot(
        self,
        messages: list[dict[str, str]],
        sys_content: str | None,
        sys_idx: int | None,
        sys_key: str,
    ) -> list[dict[str, str]]:
        count = self._oneshot_counts.get(sys_key, 0) + 1
        self._oneshot_counts[sys_key] = count
        return self._inject(
            messages, sys_content, sys_idx, sys_key,
            request_num=count,
        )

    # ── shared injection ────────────────────────────────────────

    def _inject(
        self,
        messages: list[dict[str, str]],
        sys_content: str | None,
        sys_idx: int | None,
        sys_key: str,
        turn: int = 1,
        request_num: int = 1,
    ) -> list[dict[str, str]]:
        ctx: dict[str, Any] = {
            "system_prompt": sys_content or "",
            "session_key": sys_key,
            "turn": turn,
            "request_num": request_num,
        }
        advice = self._advisor.generate_advice(ctx)
        steering = f"[Advisor Guidance]\n{advice.steering_text}"

        if sys_idx is not None:
            original = sys_content or ""
            messages[sys_idx] = {
                "role": "system",
                "content": f"{steering}\n\n{original}",
            }
        else:
            messages.insert(0, {"role": "system", "content": steering})

        return messages
