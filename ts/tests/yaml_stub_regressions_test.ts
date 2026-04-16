import { describe, it, expect } from "vitest";
import { dump, load } from "../src/yaml-stub.js";
import type { Trace } from "../src/types.js";

const traceObj: Trace = {
  id: "550e8400-e29b-41d4-a716-446655440000",
  session_id: "sess_abc123",
  timestamp: "2025-04-14T12:00:00Z",
  input: { prompt: "What is the VAT rate in Germany?", lang: "en" },
  advice: {
    domain: "tax-compliance",
    steering_text: "Ensure accurate VAT rates",
    confidence: 0.87,
    constraints: ["cite official source", "no speculation"],
    metadata: { model: "advisor-v2" },
    version: "1.0",
  },
  frontier: { provider: "anthropic", model: "claude-3-opus" },
  reward: { score: 0.95, breakdown: { accuracy: 0.9, safety: 1.0 } },
  metadata: { environment: "test" },
};

describe("yaml-stub dump()", () => {
  it("produces valid YAML, not JSON", () => {
    const out = dump({ key: "value", num: 42 });
    expect(out[0]).not.toBe("{");
    expect(out).toContain("key: value\n");
    expect(out).toContain("num: 42\n");
  });

  it("handles nested objects", () => {
    const out = dump({ a: { b: { c: 1 } } });
    expect(out).toContain("a:\n");
    expect(out).toContain("b:\n");
    expect(out).toContain("c: 1\n");
  });

  it("handles arrays", () => {
    const out = dump({ items: ["alpha", "beta", "gamma"] });
    expect(out).toContain("- alpha\n");
    expect(out).toContain("- beta\n");
    expect(out).toContain("- gamma\n");
  });
});

describe("yaml-stub load()", () => {
  it("parses YAML strings", () => {
    const result = load("name: test\ncount: 3\n") as Record<string, unknown>;
    expect(result.name).toBe("test");
    expect(result.count).toBe(3);
  });

  it("parses nested YAML", () => {
    const doc = "parent:\n  child: value\n";
    const result = load(doc) as Record<string, unknown>;
    const parent = result.parent as Record<string, unknown>;
    expect(parent.child).toBe("value");
  });
});

describe("yaml-stub round-trip", () => {
  it("round-trips a complex Trace object", () => {
    const yaml = dump(traceObj);
    const parsed = load(yaml) as Trace;
    expect(parsed.id).toBe(traceObj.id);
    expect(parsed.session_id).toBe(traceObj.session_id);
    expect(parsed.timestamp).toBe(traceObj.timestamp);
    expect(parsed.advice.domain).toBe(traceObj.advice.domain);
    expect(parsed.advice.confidence).toBeCloseTo(traceObj.advice.confidence);
    expect(parsed.advice.constraints).toEqual(traceObj.advice.constraints);
    expect(parsed.reward.score).toBeCloseTo(traceObj.reward.score);
    expect(parsed.reward.breakdown.accuracy).toBeCloseTo(
      traceObj.reward.breakdown.accuracy,
    );
    expect(parsed.frontier.provider).toBe(traceObj.frontier.provider);
  });
});

describe("yaml-stub cross-tool compatibility", () => {
  it("output is parseable by load()", () => {
    const obj = { foo: "bar", list: [1, 2, 3], nested: { a: true } };
    const out = dump(obj);
    // Verify no JSON artifacts
    expect(out).not.toMatch(/^\s*\{/);
    // Round-trip through the same parser
    const parsed = load(out);
    expect(parsed).toEqual(obj);
  });

  it("output uses standard YAML syntax for numbers", () => {
    const out = dump({ score: 0.95, count: 10 });
    expect(out).toMatch(/score: 0\.95/);
    expect(out).toMatch(/count: 10/);
  });

  it("output uses standard YAML syntax for booleans", () => {
    const out = dump({ active: true, deleted: false });
    expect(out).toMatch(/active: true/);
    expect(out).toMatch(/deleted: false/);
  });
});

// Regression: load() must use JSON_SCHEMA, not DEFAULT_SCHEMA.
//
// Before fix: js-yaml DEFAULT_SCHEMA auto-converts bare date strings
// (e.g. "2025-04-14") into JS Date objects. This is surprising and
// unsafe for callers expecting JSON-compatible types. After fix: load()
// uses JSON_SCHEMA which keeps date strings as plain strings.
describe("yaml-stub unsafe schema rejection", () => {
  it("keeps bare date strings as strings, not Date objects", () => {
    const doc = "ts: 2025-04-14\n";
    const result = load(doc) as Record<string, unknown>;
    // With JSON_SCHEMA: string. With DEFAULT_SCHEMA: Date object.
    expect(typeof result.ts).toBe("string");
    expect(result.ts).toBe("2025-04-14");
  });

  it("keeps binary tags out", () => {
    // JSON_SCHEMA rejects !!binary and other non-JSON YAML tags
    const doc = 'data: !!binary "aGVsbG8="\n';
    expect(() => load(doc)).toThrow();
  });

  // Note: JSON_SCHEMA does NOT reject merge keys (<<). They pass through
  // as a regular key. This is documented here for clarity; merge keys
  // are a DEFAULT_SCHEMA feature but JSON_SCHEMA does not error on them.
  it("documents merge key behavior under JSON_SCHEMA", () => {
    const doc = "default: &default\n  key: value\ncustom:\n  <<: *default\n  extra: data\n";
    const result = load(doc) as Record<string, unknown>;
    // Merge key passes through as a plain key (not resolved into merged object)
    const custom = result.custom as Record<string, unknown>;
    expect(custom).toHaveProperty("<<");
    expect(custom).toHaveProperty("extra", "data");
  });
});
