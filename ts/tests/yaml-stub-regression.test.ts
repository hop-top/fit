import { describe, it, expect } from "vitest";
import { dump, load } from "../src/yaml-stub.js";

// PR#12 regression: dump/load schema mismatch.
//
// Before fix: dump() used js-yaml DEFAULT_SCHEMA (emits YAML-specific
// tags like !!timestamp), but load() used JSON_SCHEMA which rejects them.
// After fix: both use JSON_SCHEMA, so round-trips are safe.

describe("yaml-stub dump/load round-trip", () => {
  it("round-trips a trace-like object with numbers, strings, nested objects", () => {
    const obj = {
      id: "trace-001",
      session_id: "sess-42",
      timestamp: "2025-04-10T12:00:00Z",
      step: 3,
      advice: {
        domain: "test",
        steering_text: "prefer concise answers",
        confidence: 0.85,
      },
      reward: {
        score: 0.92,
        breakdown: { accuracy: 0.9, relevance: 0.94 },
      },
    };

    const yaml = dump(obj);
    const result = load(yaml);

    expect(result).toEqual(obj);
  });

  it("handles null values (NaN→null boundary)", () => {
    const obj = { score: null, label: "unknown" };
    const yaml = dump(obj);
    const result = load(yaml);

    expect(result).toEqual(obj);
  });

  it("rejects Date objects on dump with JSON_SCHEMA (no silent tag emission)", () => {
    // Before the fix, DEFAULT_SCHEMA would serialize Date as !!timestamp,
    // which JSON_SCHEMA load() would reject. Now both use JSON_SCHEMA,
    // so dump() throws immediately — consistent, no silent data corruption.
    const obj = { ts: new Date("2025-04-10T12:00:00Z") };

    expect(() => dump(obj)).toThrow();
  });
});
