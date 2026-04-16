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
});
