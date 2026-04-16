import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Advice, Reward, Trace } from "../src/types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(__dirname, "..", "..", "spec", "fixtures");

describe("Trace conformance regression", () => {
  it("loads trace-v1.json fixture instead of hardcoded values", () => {
    const path = join(FIXTURES, "trace-v1.json");
    expect(existsSync(path)).toBe(true);

    const data = JSON.parse(
      readFileSync(path, "utf-8"),
    ) as Record<string, unknown>;
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
        constraints: adviceData.constraints as string[],
        metadata: adviceData.metadata as Record<string, unknown>,
      },
      frontier: data.frontier as Record<string, unknown>,
      reward: {
        score: rewardData.score as number,
        breakdown: rewardData.breakdown as Record<string, number>,
      },
      metadata: data.metadata as Record<string, unknown>,
    };

    // Assert against actual fixture values, not hardcoded ones
    expect(t.id).toBe("550e8400-e29b-41d4-a716-446655440000");
    expect(t.session_id).toBe("sess_abc123");
    expect(t.advice.domain).toBe("tax-compliance");
    expect(t.advice.steering_text).toContain("IRS publication");
    expect(t.frontier.provider).toBe("anthropic");
    expect(t.reward.score).toBeCloseTo(0.95);
  });
});

describe("Multi-turn session conformance regression", () => {
  it("loads session-multi.json and asserts 3 steps from fixture", () => {
    const path = join(FIXTURES, "session-multi.json");
    expect(existsSync(path)).toBe(true);

    const data = JSON.parse(
      readFileSync(path, "utf-8"),
    ) as Record<string, unknown>;
    const steps = data.steps as Record<string, unknown>[];

    // Fixture has 3 steps, not 2
    expect(steps).toHaveLength(3);
    expect(data.session_id).toBe("sess_def456");

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
