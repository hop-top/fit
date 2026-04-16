# TypeScript: Using fit with Node.js

## Installation

```bash
npm install @hop/fit
```

## Basic usage

```typescript
import { Session, RemoteAdvisor, CompositeScorer, SessionResult } from "@hop/fit";
import { AnthropicAdapter } from "@hop/fit/adapters";

const advisor = RemoteAdvisor.fromEndpoint("http://localhost:8080");
const adapter = new AnthropicAdapter("claude-sonnet-4-6", process.env.ANTHROPIC_API_KEY);
const scorer = CompositeScorer.composite(["accuracy", "relevance", "safety"]);

const session = new Session(advisor, adapter, scorer);

// run() returns SessionResult for one-shot (default mode)
// and SessionResult[] for multi-turn. Use Array.isArray to narrow:
const raw = await session.run(
  "What is the standard deduction?",
  { jurisdiction: "US", filing_status: "single" },
);
const result = Array.isArray(raw) ? raw[0] : raw;

console.log("Output:", result.output);
console.log("Reward:", result.reward.score);
```

## Adapter configuration

Three adapters included:

```typescript
import { AnthropicAdapter, OpenAIAdapter, OllamaAdapter } from "@hop/fit/adapters";

// Anthropic (Claude)
const anthropic = new AnthropicAdapter("claude-sonnet-4-6", apiKey);

// OpenAI (GPT)
const openai = new OpenAIAdapter("gpt-5", apiKey);

// Ollama (local)
const ollama = new OllamaAdapter("llama3");
```

Custom adapters implement the `Adapter` interface:

```typescript
interface Adapter {
  call(
    prompt: string,
    advice: Advice,
  ): Promise<{ output: string; meta: Record<string, unknown> }>;
}
```

The adapter is responsible for injecting advice into the frontier call
as hidden system context. End users never see advisor guidance.

## Custom reward functions

Implement the `RewardScorer` interface:

```typescript
import type { RewardScorer, Reward } from "@hop/fit";

class TaxAccuracyScorer implements RewardScorer {
  async score(
    output: string,
    context: Record<string, unknown>,
  ): Promise<Reward> {
    const accuracy = computeAccuracy(output, context);
    return {
      score: accuracy,
      breakdown: {
        accuracy,
        relevance: 0.9,
        safety: 1.0,
        efficiency: 0.8,
      },
    };
  }
}
```

Combine multiple scorers:

```typescript
const scorer = new CompositeScorer(
  [new TaxAccuracyScorer(), new SafetyScorer()],
  [0.7, 0.3], // weights
);
```

## Trace handling

```typescript
import { TraceWriter, TraceReader } from "@hop/fit";

const writer = new TraceWriter("./traces");
await writer.write(trace, 1); // step number

const reader = new TraceReader("./traces");
const sessions = await reader.listSessions();
const loaded = await reader.read("sess_abc123", 1);
```

Traces are xrr-compatible YAML cassettes stored as:
```
traces/{session_id}/step-001.yaml
```

## Multi-turn sessions

```typescript
const session = new Session(advisor, adapter, scorer, {
  mode: "multi-turn",
  maxSteps: 10,
  rewardThreshold: 0.95,
});
```

## Express integration

```typescript
import express from "express";

const app = express();
app.use(express.json());

app.post("/ask", async (req, res) => {
  const { prompt, context } = req.body;
  const result = await session.run(prompt, context);
  res.json({ output: result.output, reward: result.reward });
});

app.listen(3000);
```
