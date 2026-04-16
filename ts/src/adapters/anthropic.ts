import type { Advice, Adapter } from "../types.js";

export class AnthropicAdapter implements Adapter {
  constructor(
    private model = "claude-sonnet-4-6",
    private apiKey?: string,
  ) {}

  async call(
    prompt: string,
    advice: Advice,
  ): Promise<{ output: string; meta: Record<string, unknown> }> {
    return {
      output: "(anthropic stub)",
      meta: {
        model: this.model,
        provider: "anthropic",
        output: "(anthropic stub)",
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      },
    };
  }
}
