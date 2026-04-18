from __future__ import annotations

from fit.bus import (
    Bus,
    Event,
    MemoryAdapter,
    TOPIC_TRACE_CREATED,
    TOPIC_TRACE_BATCH,
    TOPIC_ADVISOR_UPDATED,
    create_bus,
    create_event,
    match_topic,
)


# -- topic matching -----------------------------------------------------------


class TestMatchTopic:
    def test_exact(self) -> None:
        assert match_topic("fit.trace.created", "fit.trace.created")

    def test_exact_no_match(self) -> None:
        assert not match_topic("fit.trace.created", "fit.trace.batch")

    def test_single_wildcard(self) -> None:
        assert match_topic("fit.trace.created", "fit.*.created")

    def test_single_wildcard_no_match(self) -> None:
        assert not match_topic("fit.trace.created", "fit.*.batch")

    def test_multi_level_wildcard(self) -> None:
        assert match_topic("fit.trace.created", "fit.#")

    def test_multi_level_wildcard_root(self) -> None:
        assert match_topic("fit.trace.created", "#")

    def test_multi_level_wildcard_no_trailing(self) -> None:
        # '#' only valid as last segment
        assert not match_topic("fit.trace.created", "#.created")

    def test_shorter_pattern(self) -> None:
        assert not match_topic("fit.trace.created", "fit.trace")

    def test_longer_pattern(self) -> None:
        assert not match_topic("fit.trace", "fit.trace.created")


# -- create_event -------------------------------------------------------------


class TestCreateEvent:
    def test_fields(self) -> None:
        evt = create_event("fit.trace.created", "test", {"id": "t1"})
        assert evt.topic == "fit.trace.created"
        assert evt.source == "test"
        assert evt.payload == {"id": "t1"}
        assert evt.timestamp is not None


# -- publish / subscribe ------------------------------------------------------


class TestBusPubSub:
    def test_publish_delivers(self) -> None:
        bus = create_bus()
        received: list[Event] = []
        bus.subscribe("fit.trace.created", received.append)
        evt = create_event(TOPIC_TRACE_CREATED, "session", {"id": "t1"})
        bus.publish(evt)
        assert len(received) == 1
        assert received[0] is evt

    def test_wildcard_subscribe(self) -> None:
        bus = create_bus()
        received: list[Event] = []
        bus.subscribe("fit.#", received.append)

        bus.publish(create_event(TOPIC_TRACE_CREATED, "s", {}))
        bus.publish(create_event(TOPIC_TRACE_BATCH, "s", {}))
        bus.publish(create_event(TOPIC_ADVISOR_UPDATED, "s", {}))
        assert len(received) == 3

    def test_no_match_no_delivery(self) -> None:
        bus = create_bus()
        received: list[Event] = []
        bus.subscribe("other.topic", received.append)
        bus.publish(create_event(TOPIC_TRACE_CREATED, "s", {}))
        assert len(received) == 0

    def test_unsubscribe(self) -> None:
        bus = create_bus()
        received: list[Event] = []
        unsub = bus.subscribe("fit.#", received.append)
        bus.publish(create_event(TOPIC_TRACE_CREATED, "s", {}))
        assert len(received) == 1

        unsub()
        bus.publish(create_event(TOPIC_TRACE_BATCH, "s", {}))
        assert len(received) == 1  # no new delivery

    def test_multiple_subscribers(self) -> None:
        bus = create_bus()
        a: list[Event] = []
        b: list[Event] = []
        bus.subscribe(TOPIC_TRACE_CREATED, a.append)
        bus.subscribe(TOPIC_TRACE_CREATED, b.append)
        bus.publish(create_event(TOPIC_TRACE_CREATED, "s", {}))
        assert len(a) == 1
        assert len(b) == 1


# -- close ---------------------------------------------------------------------


class TestBusClose:
    def test_publish_after_close_raises(self) -> None:
        adapter = MemoryAdapter()
        bus = Bus(adapter)
        bus.close()
        try:
            bus.publish(create_event(TOPIC_TRACE_CREATED, "s", {}))
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass

    def test_close_clears_subs(self) -> None:
        adapter = MemoryAdapter()
        bus = Bus(adapter)
        received: list[Event] = []
        bus.subscribe("fit.#", received.append)
        bus.close()
        assert adapter._subs == []


# -- adapter protocol ---------------------------------------------------------


class TestAdapterProtocol:
    def test_memory_adapter_satisfies_protocol(self) -> None:
        from fit.bus import BusAdapter
        assert isinstance(MemoryAdapter(), BusAdapter)
