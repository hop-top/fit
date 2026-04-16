import type { Advice } from "./types.js";

export interface Advisor {
  generateAdvice(context: Record<string, unknown>): Promise<Advice>;
  modelId(): string;
}

export class RemoteAdvisor implements Advisor {
  constructor(
    private endpoint: string,
    private timeoutMs = 5000,
  ) {}

  async generateAdvice(context: Record<string, unknown>): Promise<Advice> {
    const resp = await fetch(`${this.endpoint}/advise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(context),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!resp.ok) throw new Error(`advisor error: ${resp.status}`);
    const data = await resp.json();
    return {
      domain: data.domain,
      steering_text: data.steering_text,
      confidence: data.confidence,
      constraints: data.constraints ?? [],
      metadata: data.metadata ?? {},
      version: data.version ?? "1.0",
    };
  }

  modelId(): string {
    return `remote:${this.endpoint}`;
  }

  static fromEndpoint(url: string): RemoteAdvisor {
    return new RemoteAdvisor(url);
  }
}
