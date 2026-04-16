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

class ThrowingAdvisor implements Advisor {
  async generateAdvice(
    _context: Record<string, unknown>,
  ): Promise<Advice> {
    throw new Error("advisor unavailable");
  }

  modelId(): string {
    return "throwing";
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

class SequentialScorer implements RewardScorer {
  private scores: number[];
  private idx = 0;

  constructor(scores: number[]) {
    this.scores = scores;
  }

  async score(
    _output: string,
    _context: Record<string, unknown>,
  ): Promise<Reward> {
    const score = this.scores[this.idx] ?? 0;
    this.idx += 1;
    return { score, breakdown: { step: score } };
  }
}

class ConstantScorer implements RewardScorer {
  private fixed: number;

  constructor(score: number) {
    this.fixed = score;
  }

  async score(
    _output: string,
    _context: Record<string, unknown>,
  ): Promise<Reward> {
    return { score: this.fixed, breakdown: {} };
  }
}

// --- Tests ---

describe("Session regressions", () => {
  const baseAdvice: Advice = {
    domain: "test",
    steering_text: "steer",
    confidence: 0.5,
  };

  it("one-shot mode returns a single SessionResult", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("hello");
    const scorer = new ConstantScorer(0.8);

    const session = new Session(advisor, adapter, scorer, {
      mode: "one-shot",
    });

    const result = await session.run("prompt");

    // One-shot returns a single result, not an array
    expect(Array.isArray(result)).toBe(false);
    const r = result as { output: string; reward: Reward; trace: Record<string, unknown>; state: string };

    expect(r.output).toBe("hello");
    expect(r.reward.score).toBeCloseTo(0.8);
    expect(r.trace.session_id).toBeDefined();
    expect(r.state).toBe(SessionState.Done);
  });

  it("one-shot mode defaults when config omitted", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    const scorer = new ConstantScorer(0.5);

    // No config — defaults to one-shot
    const session = new Session(advisor, adapter, scorer);
    const result = await session.run("prompt");

    expect(Array.isArray(result)).toBe(false);
    expect((result as { output: string }).output).toBe("out");
  });

  it("multi-turn loops and stops on rewardThreshold", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    // Scores: 0.2, 0.5, 0.9 — threshold 0.8 stops at step 3
    const scorer = new SequentialScorer([0.2, 0.5, 0.9]);

    const session = new Session(advisor, adapter, scorer, {
      mode: "multi-turn",
      maxSteps: 10,
      rewardThreshold: 0.8,
    });

    const results = await session.run("prompt");

    expect(Array.isArray(results)).toBe(true);
    const steps = results as Array<{
      output: string;
      reward: Reward;
      trace: Record<string, unknown>;
      state: string;
    }>;

    // Should stop after 3 steps (0.9 >= 0.8)
    expect(steps).toHaveLength(3);
    expect(steps[2].reward.score).toBeGreaterThanOrEqual(0.8);

    // All traces share the same sessionId
    const sessionIds = steps.map((s) => s.trace.session_id);
    expect(new Set(sessionIds).size).toBe(1);

    // Final state is Done
    expect(session.getState()).toBe(SessionState.Done);
  });

  it("multi-turn respects maxSteps cap", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    // Scores never reach threshold
    const scorer = new ConstantScorer(0.1);

    const session = new Session(advisor, adapter, scorer, {
      mode: "multi-turn",
      maxSteps: 4,
      rewardThreshold: 0.9,
    });

    const results = await session.run("prompt");

    expect(Array.isArray(results)).toBe(true);
    const steps = results as Array<{ reward: Reward }>;

    // Should run exactly maxSteps times
    expect(steps).toHaveLength(4);
    steps.forEach((s) => {
      expect(s.reward.score).toBeCloseTo(0.1);
    });

    expect(session.getState()).toBe(SessionState.Done);
    expect(session.getStep()).toBe(4);
  });

  it("multi-turn shares sessionId across all steps", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    const scorer = new SequentialScorer([0.3, 0.6, 1.0]);

    const session = new Session(advisor, adapter, scorer, {
      mode: "multi-turn",
      maxSteps: 10,
      rewardThreshold: 1.0,
    });

    const results = await session.run("prompt");
    const steps = results as Array<{ trace: Record<string, unknown> }>;

    const ids = steps.map((s) => s.trace.session_id);
    expect(new Set(ids).size).toBe(1);
    expect(ids[0]).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });

  it("multi-turn collects all traces in Session.getTraces()", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    const scorer = new SequentialScorer([0.3, 0.7]);

    const session = new Session(advisor, adapter, scorer, {
      mode: "multi-turn",
      maxSteps: 10,
      rewardThreshold: 0.7,
    });

    await session.run("prompt");

    expect(session.getTraces()).toHaveLength(2);
    expect(session.getStep()).toBe(2);
  });

  it("state machine rejects invalid transitions", () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    const scorer = new ConstantScorer(0.5);

    const session = new Session(advisor, adapter, scorer);
    expect(session.getState()).toBe(SessionState.Init);

    // Jumping from Init to Score is invalid
    expect(() => session["transition"](SessionState.Score)).toThrow(
      /invalid state transition/,
    );
  });

  it("multi-turn follows Trace -> Advise (not Trace -> Init)", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    const scorer = new SequentialScorer([0.3, 0.9]);

    const session = new Session(advisor, adapter, scorer, {
      mode: "multi-turn",
      maxSteps: 5,
      rewardThreshold: 0.8,
    });

    // Run multi-turn — the internal state machine should never
    // reset to Init between steps. If it did, Trace -> Init would
    // be an invalid transition and throw.
    const results = await session.run("prompt");
    const steps = results as Array<{ reward: Reward }>;
    expect(steps).toHaveLength(2);
    expect(session.getState()).toBe(SessionState.Done);
  });

  it("one-shot transitions Trace -> Done via transition()", async () => {
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    const scorer = new ConstantScorer(0.5);

    const session = new Session(advisor, adapter, scorer, {
      mode: "one-shot",
    });

    const result = await session.run("prompt");
    expect((result as { state: string }).state).toBe(SessionState.Done);
    expect(session.getState()).toBe(SessionState.Done);
  });

  it("multi-turn maxSteps=0 returns empty array without throwing", async () => {
    // Regression: maxSteps=0 skipped the while loop then called
    // transition(Done) from Init state, which is not in VALID_TRANSITIONS.
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    const scorer = new ConstantScorer(0.5);

    const session = new Session(advisor, adapter, scorer, {
      mode: "multi-turn",
      maxSteps: 0,
      rewardThreshold: 1.0,
    });

    // Must not throw; returns empty results
    const results = await session.run("prompt");
    expect(Array.isArray(results)).toBe(true);
    expect(results).toHaveLength(0);
  });

  it("multi-turn maxSteps=0 transitions state to Done (not Init)", async () => {
    // Regression: PR#7 left session in Init for maxSteps=0, but all
    // other ports transition to Done. Session must end in Done state.
    const advisor = new StubAdvisor(baseAdvice);
    const adapter = new StubAdapter("out");
    const scorer = new ConstantScorer(0.5);

    const session = new Session(advisor, adapter, scorer, {
      mode: "multi-turn",
      maxSteps: 0,
      rewardThreshold: 1.0,
    });

    await session.run("prompt");

    expect(session.getState()).toBe(SessionState.Done);
  });

  it("fallback advice has all fields when advisor throws", async () => {
    // Regression: catch block omitted version, constraints, metadata
    const advisor = new ThrowingAdvisor();
    const adapter = new StubAdapter("out");
    const scorer = new ConstantScorer(0.5);

    const session = new Session(advisor, adapter, scorer, {
      mode: "one-shot",
    });

    const result = await session.run("prompt");
    const r = result as {
      trace: { advice: Advice };
    };

    const advice = r.trace.advice;
    expect(advice.domain).toBe("unknown");
    expect(advice.steering_text).toBe("");
    expect(advice.confidence).toBe(0);
    expect(advice.version).toBe("1.0");
    expect(advice.constraints).toEqual([]);
    expect(advice.metadata).toEqual({});
  });
});
