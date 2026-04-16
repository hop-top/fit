import type { Advice, Adapter } from "../types.js";

export class OpenAIAdapter implements Adapter {
  constructor(
    private model = "gpt-5",
    private apiKey?: string,
  ) {}

  async call(
    prompt: string,
    advice: Advice,
  ): Promise<{ output: string; meta: Record<string, unknown> }> {
    return {
      output: "(openai stub)",
      meta: {
        model: this.model,
        provider: "openai",
        output: "(openai stub)",
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      },
    };
  }
}
