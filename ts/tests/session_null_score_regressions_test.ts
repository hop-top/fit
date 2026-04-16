import { describe, it, expect } from "vitest";
import { Session, SessionState } from "../src/session.js";
import type { Advisor } from "../src/advisor.js";
import type { RewardScorer } from "../src/reward.js";
import type { Advice, Reward, Adapter } from "../src/types.js";

// --- Stubs ---

class StubAdvisor implements Advisor {
  constructor(private advice: Advice) {}

  async generateAdvice(
    _context: Record<string, unknown>,
  ): Promise<Advice> {
    return this.advice;
  }

  modelId(): string {
    return "stub";
  }
}

class StubAdapter implements Adapter {
  constructor(private output: string) {}

  async call(
    _prompt: string,
    _advice: Advice,
  ): Promise<{ output: string; meta: Record<string, unknown> }> {
    return { output: this.output, meta: { provider: "stub" } };
  }
}

class ThrowingAdapter implements Adapter {
  async call(
    _prompt: string,
    _advice: Advice,
  ): Promise<{ output: string; meta: Record<string, unknown> }> {
    throw new Error("frontier down");
  }
}

class ConstantScorer implements RewardScorer {
  constructor(private fixed: number) {}

  async score(
    _output: string,
    _context: Record<string, unknown>,
  ): Promise<Reward> {
    return { score: this.fixed, breakdown: {} };
  }
}

class ThrowingScorer implements RewardScorer {
  async score(
    _output: string,
    _context: Record<string, unknown>,
  ): Promise<Reward> {
    throw new Error("scorer unavailable");
  }
}

class SequentialScorer implements RewardScorer {
  private scores: (number | null)[];
  private idx = 0;

  constructor(scores: (number | null)[]) {
    this.scores = scores;
  }

  async score(
    _output: string,
    _context: Record<string, unknown>,
  ): Promise<Reward> {
    const score = this.scores[this.idx] ?? 0;
    this.idx += 1;
    return { score, breakdown: { step: score ?? 0 } };
  }
}

// --- Tests ---

describe("Session null-score regressions (PR#13)", () => {
  const baseAdvice: Advice = {
    domain: "test",
    steering_text: "steer",
    confidence: 0.5,
    version: "1.0",
  };

  it("frontier failure produces score: null (not NaN)", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new ThrowingAdapter();
    const scorer = new ConstantScorer(0.5);

    const session = new Session(advisor, adapter, scorer, {
      mode: "one-shot",
    });

    const result = (await session.run("prompt")) as {
      reward: Reward;
    };

    expect(result.reward.score).toBeNull();
    // NaN would fail the above — null !== NaN
    expect(Number.isNaN(result.reward.score as number)).toBe(false);
  });

  it("scorer failure produces score: null (not NaN)", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("output");
    const scorer = new ThrowingScorer();

    const session = new Session(advisor, adapter, scorer, {
      mode: "one-shot",
    });

    const result = (await session.run("prompt")) as {
      reward: Reward;
    };

    expect(result.reward.score).toBeNull();
    expect(Number.isNaN(result.reward.score as number)).toBe(false);
  });

  it("frontier failure includes error metadata", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new ThrowingAdapter();
    const scorer = new ConstantScorer(0.5);

    const session = new Session(advisor, adapter, scorer, {
      mode: "one-shot",
    });

    const result = (await session.run("prompt")) as {
      reward: Reward;
    };

    expect(result.reward.metadata).toBeDefined();
    expect(result.reward.metadata!.error).toBe("frontier_failure");
  });

  it("scorer failure includes error metadata", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("output");
    const scorer = new ThrowingScorer();

    const session = new Session(advisor, adapter, scorer, {
      mode: "one-shot",
    });

    const result = (await session.run("prompt")) as {
      reward: Reward;
    };

    expect(result.reward.metadata).toBeDefined();
    expect(result.reward.metadata!.error).toBe("scorer_failure");
  });

  it("null-score reward does not trigger early termination", async () => {
    // Adapter throws on first call, succeeds on second.
    // With threshold 0.8, the null score should NOT stop the loop.
    let adapterCallCount = 0;
    const flakyAdapter: Adapter = {
      async call(
        _prompt: string,
        _advice: Advice,
      ): Promise<{ output: string; meta: Record<string, unknown> }> {
        adapterCallCount++;
        if (adapterCallCount === 1) {
          throw new Error("transient frontier error");
        }
        return { output: "recovered", meta: {} };
      },
    };

    const advisor = new StubAdvisor(baseAdvice);
    // First call: scorer not reached (frontier fails).
    // Second call: score 0.9 — should stop.
    const scorer = new SequentialScorer([0.9]);

    const session = new Session(advisor, flakyAdapter, scorer, {
      mode: "multi-turn",
      maxSteps: 5,
      rewardThreshold: 0.8,
    });

    const results = (await session.run("prompt")) as Array<{
      reward: Reward;
    }>;

    // Should have 2 steps: one with null score, one with 0.9
    expect(results).toHaveLength(2);
    expect(results[0].reward.score).toBeNull();
    expect(results[1].reward.score).toBe(0.9);
  });

  it("JSON.stringify works on result with null score", () => {
    const reward: Reward = {
      score: null,
      breakdown: {},
      metadata: { error: "frontier_failure" },
    };

    const serialized = JSON.stringify({ reward });

    // NaN would produce '{"reward":{"score":null}}' but only
    // because JSON.stringify converts NaN to null silently —
    // the point is null serializes cleanly without surprises.
    const parsed = JSON.parse(serialized);
    expect(parsed.reward.score).toBeNull();

    // Round-trip preserves null
    expect(parsed.reward.metadata.error).toBe("frontier_failure");
  });
});
