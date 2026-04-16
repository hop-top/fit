import { describe, it, expect } from "vitest";
import { CompositeScorer } from "../src/reward.js";
import type { Reward, RewardScorer } from "../src/types.js";

class StubScorer implements RewardScorer {
  constructor(private result: Reward) {}

  async score(
    _output: string,
    _context: Record<string, unknown>,
  ): Promise<Reward> {
    return this.result;
  }
}

describe("CompositeScorer regressions", () => {
  it("returns 0.0 score when scorers array is empty (NaN guard)", async () => {
    const scorer = new CompositeScorer([]);
    const result = await scorer.score("test", {});
    expect(Number.isNaN(result.score)).toBe(false);
    expect(result.score).toBe(0.0);
  });

  it("returns 0.0 score when all weights are 0", async () => {
    const stub = new StubScorer({
      score: 0.8,
      breakdown: { dim: 0.8 },
    });
    const scorer = new CompositeScorer([stub], [0]);
    const result = await scorer.score("test", {});
    expect(Number.isNaN(result.score)).toBe(false);
    expect(result.score).toBe(0.0);
  });

  it("throws when weights.length differs from scorers.length", () => {
    const stub = new StubScorer({
      score: 0.5,
      breakdown: {},
    });
    expect(() => new CompositeScorer([stub], [1, 2])).toThrow(
      /weights\.length/,
    );
  });

  it("merges breakdowns from all scorers, not just the first", async () => {
    const accuracy = new StubScorer({
      score: 0.9,
      breakdown: { accuracy: 0.9 },
    });
    const safety = new StubScorer({
      score: 0.7,
      breakdown: { safety: 0.7 },
    });
    const scorer = new CompositeScorer([accuracy, safety], [0.6, 0.4]);
    const result = await scorer.score("test", {});

    // Before fix: only first scorer's breakdown was kept
    expect(result.breakdown).toHaveProperty("accuracy");
    expect(result.breakdown).toHaveProperty("safety");
    expect(Object.keys(result.breakdown).length).toBe(2);
  });

  it("reports actual scorer count in metadata when totalWeight is 0", async () => {
    const stub = new StubScorer({
      score: 0.8,
      breakdown: {},
    });
    const scorer = new CompositeScorer([stub], [0]);
    const result = await scorer.score("test", {});

    // Before fix: metadata.scorers was 0 even though 1 scorer exists
    expect(result.metadata.scorers).toBe(1);
  });
});
