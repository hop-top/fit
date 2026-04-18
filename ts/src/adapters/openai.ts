import type { Advice, Adapter } from "../types.js";

/**
 * OpenAI adapter stub — returns a fixed response without calling the API.
 *
 * Not yet wired to the OpenAI SDK. See the Python port for a working
 * implementation. This adapter satisfies the Adapter interface so the
 * session pipeline can be exercised end-to-end without network access.
 */
export class OpenAIAdapter implements Adapter {
  constructor(
    private model = "gpt-5",
    private apiKey?: string,
  ) {}

  async call(
    _prompt: string,
    _advice: Advice,
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
