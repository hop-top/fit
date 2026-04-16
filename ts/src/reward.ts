import type { Reward } from "./types.js";

export interface RewardScorer {
  score(output: string, context: Record<string, unknown>): Promise<Reward>;
}

export class CompositeScorer implements RewardScorer {
  private scorers: RewardScorer[];
  private weights: number[];

  constructor(scorers: RewardScorer[], weights?: number[]) {
    if (weights !== undefined && weights.length !== scorers.length) {
      throw new Error(
        `weights.length (${weights.length}) must equal scorers.length (${scorers.length})`,
      );
    }
    this.scorers = scorers;
    this.weights =
      weights ??
      (scorers.length === 0
        ? []
        : scorers.map(() => 1 / scorers.length));
  }

  async score(
    output: string,
    context: Record<string, unknown>,
  ): Promise<Reward> {
    const rewards = await Promise.all(
      this.scorers.map((s) => s.score(output, context)),
    );
    // If any scorer returns null score, propagate null for the composite.
    // Must check BEFORE totalWeight short-circuit to preserve failure semantics.
    if (rewards.some((r) => r.score === null)) {
      const mergedBreakdown: Record<string, number> = {};
      for (const r of rewards) {
        Object.assign(mergedBreakdown, r.breakdown);
      }
      return {
        score: null,
        breakdown: mergedBreakdown,
        metadata: { scorers: rewards.length, error: "child_score_is_null" },
      };
    }
    const totalWeight = this.weights.reduce((a, b) => a + b, 0);
    if (totalWeight === 0) {
      return {
        score: 0.0,
        breakdown: {},
        metadata: { scorers: rewards.length },
      };
    }
    const combined = rewards.reduce(
      (sum, r, i) => sum + (r.score as number) * this.weights[i],
      0,
    );
    const mergedBreakdown: Record<string, number> = {};
    for (const r of rewards) {
      Object.assign(mergedBreakdown, r.breakdown);
    }
    return {
      score: combined / totalWeight,
      breakdown: mergedBreakdown,
      metadata: { scorers: rewards.length },
    };
  }

  static composite(names: string[]): CompositeScorer {
    return new CompositeScorer(names.map((n) => new DimensionScorer(n)));
  }
}

class DimensionScorer implements RewardScorer {
  constructor(private dimension: string) {}

  async score(
    _output: string,
    _context: Record<string, unknown>,
  ): Promise<Reward> {
    return { score: 0.5, breakdown: { [this.dimension]: 0.5 } };
  }
}
