import { describe, it, expect } from "vitest";
import { Session, SessionState } from "../src/session.js";
import type {
  Advisor,
  RewardScorer,
  Advice,
  Reward,
  Trace,
} from "../src/types.js";

// Stub advisor that always succeeds.
const stubAdvisor: Advisor = {
  async generateAdvice() {
    return {
      domain: "test",
      steering_text: "steer",
      confidence: 0.5,
      version: "1.0",
      constraints: [],
      metadata: {},
    };
  },
  modelId() {
    return "stub-advisor";
  },
};

// Stub scorer that always succeeds.
const stubScorer: RewardScorer = {
  async score(): Promise<Reward> {
    return { score: 0.8, breakdown: { accuracy: 0.8 } };
  },
};

// Adapter whose call() always rejects.
const errorAdapter = {
  async call() {
    throw new Error("adapter failed");
  },
};

describe("PR#11: adapter failure produces partial trace", () => {
  it("returns partial result with null score when adapter throws", async () => {
    const session = new Session(
      stubAdvisor,
      errorAdapter as any,
      stubScorer,
      { mode: "one-shot" },
    );

    const result = await session.run("test");

    // Result must be returned, not thrown
    expect(result).toBeDefined();

    // Destructure the single-shot result
    const r = result as {
      output: string;
      reward: Reward;
      trace: Trace;
      state: SessionState;
    };

    // Trace must be present
    expect(r.trace).toBeDefined();
    expect(r.trace.frontier).toBeDefined();

    // Reward score must be null (spec: reward-schema-v1 allows null for failures)
    expect(r.reward.score).toBeNull();

    // Metadata must contain error indicator
    expect(r.reward.metadata).toBeDefined();
    expect(r.reward.metadata!.error).toBe("frontier_failure");

    // Frontier must contain error info
    expect(r.trace.frontier.error).toContain("adapter failed");
  });
});
