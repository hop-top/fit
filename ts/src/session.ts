import { randomUUID } from "node:crypto";
import type { Advice, Reward, Trace, Adapter } from "./types.js";
import type { Advisor } from "./advisor.js";
import type { RewardScorer } from "./reward.js";

export interface SessionConfig {
  mode: "one-shot" | "multi-turn";
  maxSteps: number;
  rewardThreshold: number;
}

/** Session lifecycle state (mirrors spec/session-protocol.md state machine). */
export enum SessionState {
  Init = "init",
  Advise = "advise",
  Frontier = "frontier",
  Score = "score",
  Trace = "trace",
  Done = "done",
}

/** Result of a single advise->frontier->score->trace step. */
export interface SessionResult {
  output: string;
  reward: Reward;
  trace: Trace;
  state: SessionState;
}

/** Valid state transitions per session-protocol.md. */
const VALID_TRANSITIONS: Record<string, Set<string>> = {
  [SessionState.Init]: new Set([SessionState.Advise, SessionState.Done]),
  [SessionState.Advise]: new Set([SessionState.Frontier]),
  [SessionState.Frontier]: new Set([SessionState.Score]),
  [SessionState.Score]: new Set([SessionState.Trace]),
  [SessionState.Trace]: new Set([SessionState.Advise, SessionState.Done]),
};

export class Session {
  private config: SessionConfig;
  private state: SessionState = SessionState.Init;
  private step = 0;
  private traces: Trace[] = [];

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

  /** Current state machine state. */
  getState(): SessionState {
    return this.state;
  }

  /** All traces recorded so far. */
  getTraces(): Trace[] {
    return this.traces;
  }

  /** Current step count. */
  getStep(): number {
    return this.step;
  }

  private transition(to: SessionState): void {
    const allowed = VALID_TRANSITIONS[this.state];
    if (!allowed || !allowed.has(to)) {
      throw new Error(
        `invalid state transition: ${this.state} -> ${to}`,
      );
    }
    this.state = to;
  }

  /** Run a single advise->frontier->score->trace step with a given sessionId. */
  private async runStep(
    sessionId: string,
    prompt: string,
    context: Record<string, unknown>,
  ): Promise<SessionResult> {
    // First step: Init -> Advise. Subsequent steps: Trace -> Advise.
    this.transition(SessionState.Advise);

    const input = { prompt, context };

    let advice: Advice;
    try {
      advice = await this.advisor.generateAdvice(input);
    } catch {
      advice = {
        domain: "unknown",
        steering_text: "",
        confidence: 0,
        version: "1.0",
        constraints: [],
        metadata: {},
      };
    }

    this.transition(SessionState.Frontier);

    const { output, meta: frontierMeta } = await this.adapter.call(
      prompt,
      advice,
    );

    this.transition(SessionState.Score);

    let reward: Reward;
    try {
      reward = await this.scorer.score(output, context);
    } catch {
      reward = { score: NaN, breakdown: {} };
    }

    this.transition(SessionState.Trace);
    this.step += 1;

    const trace: Trace = {
      id: randomUUID(),
      session_id: sessionId,
      timestamp: new Date().toISOString(),
      input,
      advice,
      frontier: frontierMeta,
      reward,
    };

    this.traces.push(trace);

    return { output, reward, trace, state: this.state };
  }

  /**
   * Run a session. Behavior depends on config.mode:
   * - "one-shot": single step, returns one result.
   * - "multi-turn": loops up to maxSteps, stops early if
   *   reward.score >= rewardThreshold. Returns all step results.
   */
  async run(
    prompt: string,
    context: Record<string, unknown> = {},
  ): Promise<SessionResult | SessionResult[]> {
    const sessionId = randomUUID();
    this.step = 0;
    this.traces = [];
    this.state = SessionState.Init;

    if (this.config.mode === "multi-turn") {
      return this.runMultiTurn(sessionId, prompt, context);
    }

    return this.runOneShot(sessionId, prompt, context);
  }

  private async runOneShot(
    sessionId: string,
    prompt: string,
    context: Record<string, unknown>,
  ): Promise<SessionResult> {
    const result = await this.runStep(sessionId, prompt, context);
    this.transition(SessionState.Done);
    return { ...result, state: this.state };
  }

  private async runMultiTurn(
    sessionId: string,
    prompt: string,
    context: Record<string, unknown>,
  ): Promise<SessionResult[]> {
    // Short-circuit: 0-step session transitions Init -> Done.
    if (this.config.maxSteps === 0) {
      this.transition(SessionState.Done);
      return [];
    }

    const results: SessionResult[] = [];

    while (this.step < this.config.maxSteps) {
      const result = await this.runStep(sessionId, prompt, context);
      results.push(result);

      if (result.reward.score >= this.config.rewardThreshold) {
        break;
      }
    }

    this.transition(SessionState.Done);
    return results;
  }
}
