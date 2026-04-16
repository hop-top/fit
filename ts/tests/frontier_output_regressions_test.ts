import { describe, it, expect } from "vitest";
import { Session } from "../src/session.js";
import type {
  Advisor,
  RewardScorer,
  Advice,
  Reward,
} from "../src/types.js";

/**
 * PR#19 regression: spec/trace-format-v1.md requires frontier.output
 * in every trace. The session must inject adapter output into
 * frontierMeta even when the adapter omits it from its metadata.
 */

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

const stubScorer: RewardScorer = {
  async score(): Promise<Reward> {
    return { score: 0.8, breakdown: { accuracy: 0.8 } };
  },
};

describe("PR#19: frontier.output in trace", () => {
  it("injects output into frontier when adapter omits it from meta", async () => {
    const adapter = {
      async call() {
        return {
          output: "hello",
          meta: { model: "test" },
        };
      },
    };

    const session = new Session(
      stubAdvisor,
      adapter as any,
      stubScorer,
      { mode: "one-shot" },
    );

    const result = (await session.run("test")) as any;

    expect(result.trace.frontier.output).toBe("hello");
  });

  it("injects empty output into frontier on adapter failure", async () => {
    const adapter = {
      async call() {
        throw new Error("boom");
      },
    };

    const session = new Session(
      stubAdvisor,
      adapter as any,
      stubScorer,
      { mode: "one-shot" },
    );

    const result = (await session.run("test")) as any;

    // Even on failure, frontier must contain output (empty string).
    expect(result.trace.frontier.output).toBe("");
  });
});
