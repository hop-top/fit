import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Advice, Reward, Trace } from "../src/types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(__dirname, "..", "..", "spec", "fixtures");

function loadJSON(name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(FIXTURES, name), "utf-8"));
}

describe("Advice conformance", () => {
  it("parses JSON fixture into Advice type", () => {
    const data = loadJSON("advice-v1.json") as Record<string, unknown>;
    const a: Advice = {
      domain: data.domain as string,
      steering_text: data.steering_text as string,
      confidence: data.confidence as number,
      constraints: (data.constraints as string[]) ?? [],
      metadata: (data.metadata as Record<string, unknown>) ?? {},
      version: (data.version as string) ?? "1.0",
    };
    expect(a.domain).toBe("tax-compliance");
    expect(a.confidence).toBeCloseTo(0.87);
    expect(a.constraints).toHaveLength(3);
    expect(a.version).toBe("1.0");
    expect(a.metadata).toHaveProperty("model");
  });

  it("round-trips through JSON serialize/parse", () => {
    const data = loadJSON("advice-v1.json") as Record<string, unknown>;
    const a: Advice = {
      domain: data.domain as string,
      steering_text: data.steering_text as string,
      confidence: data.confidence as number,
      constraints: data.constraints as string[],
      metadata: data.metadata as Record<string, unknown>,
      version: data.version as string,
    };
    const serialized = JSON.stringify(a);
    const parsed = JSON.parse(serialized) as Advice;
    expect(parsed.domain).toBe(a.domain);
    expect(parsed.confidence).toBeCloseTo(a.confidence);
    expect(parsed.constraints).toEqual(a.constraints);
  });

  it("has confidence in [0, 1]", () => {
    const data = loadJSON("advice-v1.json") as Record<string, unknown>;
    const conf = data.confidence as number;
    expect(conf).toBeGreaterThanOrEqual(0);
    expect(conf).toBeLessThanOrEqual(1);
  });
});

describe("Reward conformance", () => {
  it("parses JSON fixture into Reward type", () => {
    const data = loadJSON("reward-v1.json") as Record<string, unknown>;
    const r: Reward = {
      score: data.score as number,
      breakdown: data.breakdown as Record<string, number>,
      metadata: data.metadata as Record<string, unknown>,
    };
    expect(r.score).toBeCloseTo(0.62);
    expect(r.breakdown.accuracy).toBeCloseTo(0.7);
    expect(r.breakdown.safety).toBeCloseTo(1.0);
    expect((r.metadata as Record<string, unknown>).scorer).toBe(
      "rubric-judge-v2",
    );
  });

  it("has all scores in [0, 1]", () => {
    const data = loadJSON("reward-v1.json") as Record<string, unknown>;
    expect(data.score).toBeGreaterThanOrEqual(0);
    expect(data.score).toBeLessThanOrEqual(1);
    const bd = data.breakdown as Record<string, number>;
    for (const v of Object.values(bd)) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("round-trips through JSON serialize/parse", () => {
    const data = loadJSON("reward-v1.json") as Record<string, unknown>;
    const r: Reward = {
      score: data.score as number,
      breakdown: data.breakdown as Record<string, number>,
    };
    const serialized = JSON.stringify(r);
    const parsed = JSON.parse(serialized) as Reward;
    expect(parsed.score).toBeCloseTo(r.score);
  });
});

describe("Trace conformance", () => {
  it("loads trace fixture and verifies required fields", () => {
    const data = loadJSON("trace-v1.json") as Record<string, unknown>;
    const adviceData = data.advice as Record<string, unknown>;
    const rewardData = data.reward as Record<string, unknown>;
    const breakdown = rewardData.breakdown as Record<string, number>;
    const t: Trace = {
      id: data.id as string,
      session_id: data.session_id as string,
      timestamp: data.timestamp as string,
      input: data.input as Record<string, unknown>,
      advice: {
        domain: adviceData.domain as string,
        steering_text: adviceData.steering_text as string,
        confidence: adviceData.confidence as number,
        constraints: adviceData.constraints as string[],
        metadata: adviceData.metadata as Record<string, unknown>,
      },
      frontier: data.frontier as Record<string, unknown>,
      reward: {
        score: rewardData.score as number,
        breakdown,
      },
      metadata: data.metadata as Record<string, unknown>,
    };
    expect(t.id).toBe("550e8400-e29b-41d4-a716-446655440000");
    expect(t.session_id).toBe("sess_abc123");
    expect(t.advice.domain).toBe("tax-compliance");
    expect(t.frontier.provider).toBe("anthropic");
    expect(t.reward.score).toBeCloseTo(0.95);
  });

  it("round-trips trace through JSON serialize/parse", () => {
    const data = loadJSON("trace-v1.json") as Record<string, unknown>;
    const adviceData = data.advice as Record<string, unknown>;
    const rewardData = data.reward as Record<string, unknown>;
    const t: Trace = {
      id: data.id as string,
      session_id: data.session_id as string,
      timestamp: data.timestamp as string,
      input: data.input as Record<string, unknown>,
      advice: {
        domain: adviceData.domain as string,
        steering_text: adviceData.steering_text as string,
        confidence: adviceData.confidence as number,
      },
      frontier: data.frontier as Record<string, unknown>,
      reward: {
        score: rewardData.score as number,
        breakdown: rewardData.breakdown as Record<string, number>,
      },
    };
    const serialized = JSON.stringify(t);
    const parsed = JSON.parse(serialized) as Trace;
    expect(parsed.id).toBe(t.id);
    expect(parsed.session_id).toBe(t.session_id);
    expect(parsed.advice.domain).toBe(t.advice.domain);
    expect(parsed.reward.score).toBeCloseTo(t.reward.score);
  });
});

describe("Multi-turn session conformance", () => {
  it("loads session-multi fixture and verifies structure", () => {
    const path = join(FIXTURES, "session-multi.json");
    expect(existsSync(path)).toBe(true);

    const data = JSON.parse(
      readFileSync(path, "utf-8"),
    ) as Record<string, unknown>;
    const steps = data.steps as Record<string, unknown>[];

    expect(steps).toHaveLength(3);
    expect(data.session_id).toBe("sess_def456");
    expect(data.mode).toBe("multi-turn");
    expect(data.max_steps).toBe(3);

    const first = steps[0] as Record<string, unknown>;
    const last = steps[2] as Record<string, unknown>;
    const firstReward = (first.reward as Record<string, unknown>)
      .score as number;
    const lastReward = (last.reward as Record<string, unknown>)
      .score as number;
    expect(firstReward).toBeLessThan(lastReward);

    steps.forEach((s) => {
      expect(s.session_id).toBe("sess_def456");
    });
  });
});
