import type { Advice, Adapter } from "../types.js";

/**
 * Anthropic adapter stub — returns a fixed response without calling the API.
 *
 * Not yet wired to the Anthropic SDK. See the Python port for a working
 * implementation. This adapter satisfies the Adapter interface so the
 * session pipeline can be exercised end-to-end without network access.
 */
export class AnthropicAdapter implements Adapter {
  constructor(
    private model = "claude-sonnet-4-6",
    private apiKey?: string,
  ) {}

  async call(
    _prompt: string,
    _advice: Advice,
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
