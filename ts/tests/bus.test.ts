import { describe, it, expect } from "vitest";
import {
  createBus,
  createEvent,
  matchTopic,
  MemoryAdapter,
  TOPIC_TRACE_CREATED,
  TOPIC_TRACE_BATCH,
  TOPIC_ADVISOR_UPDATED,
  type BusEvent,
} from "../src/bus.js";

// -- topic matching -----------------------------------------------------------

describe("matchTopic", () => {
  it("exact match", () => {
    expect(matchTopic("fit.trace.created", "fit.trace.created")).toBe(true);
  });

  it("exact no match", () => {
    expect(matchTopic("fit.trace.created", "fit.trace.batch")).toBe(false);
  });

  it("single wildcard", () => {
    expect(matchTopic("fit.trace.created", "fit.*.created")).toBe(true);
  });

  it("single wildcard no match", () => {
    expect(matchTopic("fit.trace.created", "fit.*.batch")).toBe(false);
  });

  it("multi-level wildcard", () => {
    expect(matchTopic("fit.trace.created", "fit.#")).toBe(true);
  });

  it("multi-level wildcard root", () => {
    expect(matchTopic("fit.trace.created", "#")).toBe(true);
  });

  it("shorter pattern no match", () => {
    expect(matchTopic("fit.trace.created", "fit.trace")).toBe(false);
  });

  it("longer pattern no match", () => {
    expect(matchTopic("fit.trace", "fit.trace.created")).toBe(false);
  });
});

// -- createEvent --------------------------------------------------------------

describe("createEvent", () => {
  it("populates fields", () => {
    const evt = createEvent("fit.trace.created", "test", { id: "t1" });
    expect(evt.topic).toBe("fit.trace.created");
    expect(evt.source).toBe("test");
    expect(evt.payload).toEqual({ id: "t1" });
    expect(evt.timestamp).toBeInstanceOf(Date);
  });
});

// -- publish / subscribe ------------------------------------------------------

describe("bus pub/sub", () => {
  it("publish delivers to subscriber", () => {
    const bus = createBus();
    const received: BusEvent[] = [];
    bus.subscribe("fit.trace.created", (e) => received.push(e));

    const evt = createEvent(TOPIC_TRACE_CREATED, "session", { id: "t1" });
    bus.publish(evt);

    expect(received).toHaveLength(1);
    expect(received[0]).toBe(evt);
  });

  it("wildcard subscribe", () => {
    const bus = createBus();
    const received: BusEvent[] = [];
    bus.subscribe("fit.#", (e) => received.push(e));

    bus.publish(createEvent(TOPIC_TRACE_CREATED, "s", {}));
    bus.publish(createEvent(TOPIC_TRACE_BATCH, "s", {}));
    bus.publish(createEvent(TOPIC_ADVISOR_UPDATED, "s", {}));

    expect(received).toHaveLength(3);
  });

  it("no match no delivery", () => {
    const bus = createBus();
    const received: BusEvent[] = [];
    bus.subscribe("other.topic", (e) => received.push(e));
    bus.publish(createEvent(TOPIC_TRACE_CREATED, "s", {}));
    expect(received).toHaveLength(0);
  });

  it("unsubscribe stops delivery", () => {
    const bus = createBus();
    const received: BusEvent[] = [];
    const unsub = bus.subscribe("fit.#", (e) => received.push(e));

    bus.publish(createEvent(TOPIC_TRACE_CREATED, "s", {}));
    expect(received).toHaveLength(1);

    unsub();
    bus.publish(createEvent(TOPIC_TRACE_BATCH, "s", {}));
    expect(received).toHaveLength(1);
  });

  it("multiple subscribers", () => {
    const bus = createBus();
    const a: BusEvent[] = [];
    const b: BusEvent[] = [];
    bus.subscribe(TOPIC_TRACE_CREATED, (e) => a.push(e));
    bus.subscribe(TOPIC_TRACE_CREATED, (e) => b.push(e));

    bus.publish(createEvent(TOPIC_TRACE_CREATED, "s", {}));
    expect(a).toHaveLength(1);
    expect(b).toHaveLength(1);
  });
});

// -- close --------------------------------------------------------------------

describe("bus close", () => {
  it("publish after close throws", () => {
    const bus = createBus();
    bus.close();
    expect(() =>
      bus.publish(createEvent(TOPIC_TRACE_CREATED, "s", {})),
    ).toThrow("bus: publish after close");
  });

  it("close clears subscriptions", () => {
    const adapter = new MemoryAdapter();
    const bus = createBus(adapter);
    const received: BusEvent[] = [];
    bus.subscribe("fit.#", (e) => received.push(e));
    bus.close();
    // adapter is closed, no more subs
    expect(() =>
      adapter.publish(createEvent(TOPIC_TRACE_CREATED, "s", {})),
    ).toThrow();
  });
});
