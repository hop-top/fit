"""In-memory pub/sub with MQTT-style topic matching.

Wildcards (dot-separated segments):
  ``*`` — matches exactly one segment
  ``#`` — matches zero or more trailing segments
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

# -- fit-specific event topics ------------------------------------------------

TOPIC_TRACE_CREATED = "fit.trace.created"
TOPIC_TRACE_BATCH = "fit.trace.batch"
TOPIC_ADVISOR_UPDATED = "fit.advisor.updated"


# -- core types ---------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """Immutable bus event."""

    topic: str
    source: str
    timestamp: datetime
    payload: Any


def create_event(topic: str, source: str, payload: Any) -> Event:
    """Build an ``Event`` stamped with the current UTC time."""
    return Event(
        topic=topic,
        source=source,
        timestamp=datetime.now(UTC),
        payload=payload,
    )


# -- topic matching -----------------------------------------------------------

Handler = Callable[[Event], None]
Unsubscribe = Callable[[], None]


def match_topic(topic: str, pattern: str) -> bool:
    """MQTT-style pattern match against a dot-separated topic."""
    t_parts = topic.split(".")
    p_parts = pattern.split(".")

    ti = 0
    pi = 0

    while pi < len(p_parts):
        if p_parts[pi] == "#":
            return pi == len(p_parts) - 1
        if ti >= len(t_parts):
            return False
        if p_parts[pi] != "*" and p_parts[pi] != t_parts[ti]:
            return False
        ti += 1
        pi += 1

    return ti == len(t_parts)


# -- adapter protocol ---------------------------------------------------------


@dataclass
class _Subscription:
    id: int
    pattern: str
    handler: Handler


@runtime_checkable
class BusAdapter(Protocol):
    """Pluggable transport layer for the bus."""

    def publish(self, event: Event) -> None: ...
    def subscribe(self, pattern: str, handler: Handler) -> Unsubscribe: ...
    def close(self) -> None: ...


class MemoryAdapter:
    """Default in-process adapter."""

    def __init__(self) -> None:
        self._subs: list[_Subscription] = []
        self._next_id: int = 0
        self._closed: bool = False

    def publish(self, event: Event) -> None:
        if self._closed:
            raise RuntimeError("bus: publish after close")
        for s in [s for s in self._subs if match_topic(event.topic, s.pattern)]:
            s.handler(event)

    def subscribe(self, pattern: str, handler: Handler) -> Unsubscribe:
        sub_id = self._next_id
        self._next_id += 1
        self._subs.append(_Subscription(id=sub_id, pattern=pattern, handler=handler))

        def _unsub() -> None:
            self._subs = [s for s in self._subs if s.id != sub_id]

        return _unsub

    def close(self) -> None:
        self._closed = True
        self._subs = []


# -- bus facade ---------------------------------------------------------------


class Bus:
    """Pub/sub bus backed by a pluggable ``BusAdapter``."""

    def __init__(self, adapter: BusAdapter | None = None) -> None:
        self._adapter: BusAdapter = adapter or MemoryAdapter()

    def publish(self, event: Event) -> None:
        self._adapter.publish(event)

    def subscribe(self, pattern: str, handler: Handler) -> Unsubscribe:
        return self._adapter.subscribe(pattern, handler)

    def close(self) -> None:
        self._adapter.close()


def create_bus(adapter: BusAdapter | None = None) -> Bus:
    """Create a new ``Bus`` (defaults to ``MemoryAdapter``)."""
    return Bus(adapter=adapter)
