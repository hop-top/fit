# Trace Format v1

xrr-compatible YAML cassette format for session traces.

## Schema

```yaml
# Required
id: string              # Unique trace ID (UUID)
session_id: string      # Parent session ID
timestamp: string       # ISO 8601 UTC
input:
  prompt: string        # Original user/task input
  context:              # Arbitrary context k/v
    key: value

# Advice given
advice:
  domain: string
  steering_text: string
  confidence: float
  constraints: [string]
  metadata: {}

# Frontier model call
frontier:
  model: string         # e.g. "claude-sonnet-4-6", "gpt-5"
  provider: string      # "anthropic" | "openai" | "ollama"
  output: string        # Raw frontier response
  usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# Reward scoring
reward:
  score: float
  breakdown:
    accuracy: float
    relevance: float
    safety: float
    efficiency: float

# Optional
metadata:
  duration_ms: int
  advisor_model: string
  advisor_version: string
  trace_version: string  # "1.0"
```

## Example: One-shot trace

```yaml
id: "550e8400-e29b-41d4-a716-446655440000"
session_id: "sess_abc123"
timestamp: "2026-04-15T10:30:00Z"
input:
  prompt: "What is the standard deduction for 2025?"
  context:
    jurisdiction: "US"
    filing_status: "single"

advice:
  domain: "tax-compliance"
  steering_text: "Cite IRS publication numbers. Flag if rule changed in 2025."
  confidence: 0.91
  constraints: ["cite sources", "no speculation"]
  metadata:
    model: "advisor-tax-v2.3"

frontier:
  model: "claude-sonnet-4-6"
  provider: "anthropic"
  output: "For tax year 2025, the standard deduction..."
  usage:
    prompt_tokens: 342
    completion_tokens: 156
    total_tokens: 498

reward:
  score: 0.95
  breakdown:
    accuracy: 1.0
    relevance: 0.9
    safety: 1.0
    efficiency: 0.9

metadata:
  duration_ms: 1830
  advisor_model: "advisor-tax-v2.3"
  trace_version: "1.0"
```

## Example: Multi-turn agent trace (one step)

```yaml
id: "660e8400-e29b-41d4-a716-446655440001"
session_id: "sess_def456"
timestamp: "2026-04-15T10:31:15Z"
input:
  prompt: "Fix the off-by-one in src/parser.rs:142"
  context:
    repo: "hop-top/fit"
    issue: 42
    step: 3

advice:
  domain: "code-agent"
  steering_text: |
    Search for related usages of the parser function first.
    Apply minimal patch. Run cargo test after edit.
  confidence: 0.78
  constraints: ["no changes outside src/parser.rs"]
  metadata:
    model: "advisor-swe-v1.1"
    step: 3

frontier:
  model: "gpt-5"
  provider: "openai"
  output: "I'll search for usages of parse_header..."
  usage:
    prompt_tokens: 1205
    completion_tokens: 832
    total_tokens: 2037

reward:
  score: 0.7
  breakdown:
    accuracy: 0.8
    relevance: 0.9
    safety: 1.0
    efficiency: 0.1

metadata:
  duration_ms: 4500
  advisor_model: "advisor-swe-v1.1"
  trace_version: "1.0"
```

## File layout

Traces stored as YAML cassettes:

```
traces/
  {session_id}/
    step-001.yaml
    step-002.yaml
    ...
```

## xrr compatibility

- Each YAML file is an xrr cassette
- Replay: load cassette, verify advice → output → reward chain
- Conformance: all ports must produce identical traces for
  identical (input, advice, model, reward_fn)
