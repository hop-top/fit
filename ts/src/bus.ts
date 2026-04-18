/**
 * In-memory pub/sub with MQTT-style topic matching.
 *
 * Wildcards (dot-separated segments):
 *   `*` — matches exactly one segment
 *   `#` — matches zero or more trailing segments
 */

// -- fit-specific event topics ------------------------------------------------

export const TOPIC_TRACE_CREATED = "fit.trace.created";
export const TOPIC_TRACE_BATCH = "fit.trace.batch";
export const TOPIC_ADVISOR_UPDATED = "fit.advisor.updated";

// -- core types ---------------------------------------------------------------

export interface BusEvent {
  topic: string;
  source: string;
  timestamp: Date;
  payload: unknown;
}

export type BusHandler = (event: BusEvent) => void | Promise<void>;
export type Unsubscribe = () => void;

export function createEvent(
  topic: string,
  source: string,
  payload: unknown,
): BusEvent {
  return { topic, source, timestamp: new Date(), payload };
}

// -- topic matching -----------------------------------------------------------

export function matchTopic(topic: string, pattern: string): boolean {
  const tParts = topic.split(".");
  const pParts = pattern.split(".");

  let ti = 0;
  let pi = 0;

  while (pi < pParts.length) {
    if (pParts[pi] === "#") {
      return pi === pParts.length - 1;
    }
    if (ti >= tParts.length) return false;
    if (pParts[pi] !== "*" && pParts[pi] !== tParts[ti]) return false;
    ti++;
    pi++;
  }

  return ti === tParts.length;
}

// -- adapter ------------------------------------------------------------------

interface Subscription {
  id: number;
  pattern: string;
  handler: BusHandler;
}

/** Pluggable transport layer for the bus. */
export interface BusAdapter {
  publish(event: BusEvent): void;
  subscribe(pattern: string, handler: BusHandler): Unsubscribe;
  close(): void;
}

/** Default in-process adapter. */
export class MemoryAdapter implements BusAdapter {
  private subs: Subscription[] = [];
  private nextId = 0;
  private closed = false;

  publish(event: BusEvent): void {
    if (this.closed) throw new Error("bus: publish after close");

    const matching = this.subs.filter((s) =>
      matchTopic(event.topic, s.pattern),
    );
    for (const s of matching) {
      const result = s.handler(event);
      if (result && typeof (result as Promise<void>).then === "function") {
        Promise.resolve(result).then(
          () => {},
          () => {},
        );
      }
    }
  }

  subscribe(pattern: string, handler: BusHandler): Unsubscribe {
    const id = this.nextId++;
    this.subs.push({ id, pattern, handler });
    return () => {
      this.subs = this.subs.filter((s) => s.id !== id);
    };
  }

  close(): void {
    this.closed = true;
    this.subs = [];
  }
}

// -- bus facade ---------------------------------------------------------------

export interface Bus {
  publish(event: BusEvent): void;
  subscribe(pattern: string, handler: BusHandler): Unsubscribe;
  close(): void;
}

/** Create a Bus backed by the given adapter (default: MemoryAdapter). */
export function createBus(adapter?: BusAdapter): Bus {
  const a = adapter ?? new MemoryAdapter();
  return {
    publish: (event) => a.publish(event),
    subscribe: (pattern, handler) => a.subscribe(pattern, handler),
    close: () => a.close(),
  };
}
