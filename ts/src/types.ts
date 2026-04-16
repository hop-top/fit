/** Advisor output (advice-format-v1) */
export interface Advice {
  domain: string;
  steering_text: string;
  confidence: number;
  constraints?: string[];
  metadata?: Record<string, unknown>;
  version: string;
}

/** Reward scoring result (reward-schema-v1) */
export interface Reward {
  score: number | null;
  breakdown: Record<string, number>;
  metadata?: Record<string, unknown>;
}

/** Session trace record (trace-format-v1) */
export interface Trace {
  id: string;
  session_id: string;
  timestamp: string;
  input: Record<string, unknown>;
  advice: Advice;
  frontier: Record<string, unknown>;
  reward: Reward;
  metadata?: Record<string, unknown>;
}

/** Frontier LLM adapter interface */
export interface Adapter {
  call(
    prompt: string,
    advice: Advice,
  ): Promise<{ output: string; meta: Record<string, unknown> }>;
}
