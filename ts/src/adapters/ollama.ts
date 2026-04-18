import type { Advice, Adapter } from "../types.js";

/**
 * Ollama adapter stub — returns a fixed response without calling the API.
 *
 * Not yet wired to the Ollama HTTP API. See the Python port for a working
 * implementation. This adapter satisfies the Adapter interface so the
 * session pipeline can be exercised end-to-end without network access.
 */
export class OllamaAdapter implements Adapter {
  constructor(
    private model = "llama3",
    private baseUrl = "http://localhost:11434",
  ) {}

  async call(
    _prompt: string,
    _advice: Advice,
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
