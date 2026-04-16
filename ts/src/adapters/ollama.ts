import type { Advice, Adapter } from "../types.js";

export class OllamaAdapter implements Adapter {
  constructor(
    private model = "llama3",
    private baseUrl = "http://localhost:11434",
  ) {}

  async call(
    prompt: string,
    advice: Advice,
  ): Promise<{ output: string; meta: Record<string, unknown> }> {
    return {
      output: "(ollama stub)",
      meta: {
        model: this.model,
        provider: "ollama",
        output: "(ollama stub)",
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      },
    };
  }
}
