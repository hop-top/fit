import { randomUUID } from "node:crypto";
import type { Advice, Reward, Trace, Adapter } from "./types.js";
import type { Advisor } from "./advisor.js";
import type { RewardScorer } from "./reward.js";

export interface SessionConfig {
  mode: "one-shot" | "multi-turn";
  maxSteps: number;
  rewardThreshold: number;
}

export class Session {
  private config: SessionConfig;

  constructor(
    private advisor: Advisor,
    private adapter: Adapter,
    private scorer: RewardScorer,
    config?: Partial<SessionConfig>,
  ) {
    this.config = {
      mode: config?.mode ?? "one-shot",
      maxSteps: config?.maxSteps ?? 10,
      rewardThreshold: config?.rewardThreshold ?? 1.0,
    };
  }

  async run(
    prompt: string,
    context: Record<string, unknown> = {},
  ): Promise<{ output: string; reward: Reward; trace: Trace }> {
    const sessionId = randomUUID();
    const input = { prompt, ...context };

    let advice: Advice;
    try {
      advice = await this.advisor.generateAdvice(input);
    } catch {
      advice = { domain: "unknown", steering_text: "", confidence: 0 };
    }

    const { output, meta: frontierMeta } = await this.adapter.call(
      prompt,
      advice,
    );

    let reward: Reward;
    try {
      reward = await this.scorer(output, context);
    } catch {
      reward = { score: NaN, breakdown: {} };
    }

    const trace: Trace = {
      id: randomUUID(),
      session_id: sessionId,
      timestamp: new Date().toISOString(),
      input,
      advice,
      frontier: frontierMeta,
      reward,
    };

    return { output, reward, trace };
  }

  private async scorer(
    output: string,
    context: Record<string, unknown>,
  ): Promise<Reward> {
    return this.scorer.score(output, context);
  }
}
