# Session Protocol v1

Language-agnostic session lifecycle for fit advisor steering.

## State machine

```
    ┌──────┐
    │ Init │──── max_steps=0 ────┐
    └──┬───┘                     │
       │                         │
       ▼                         │
   ┌────────┐     advice         │
   │ Advise │──────────────┐     │
   └────────┘              │     │
       ▲                   ▼     │
       │             ┌──────────┐│
       │             │ Frontier ││
       │             └────┬─────┘│
       │                  │      │
       │                  ▼      │
       │             ┌────────┐  │
       │             │ Score  │  │
       │             └────┬───┘  │
       │                  │      │
       │                  ▼      │
       │             ┌────────┐  │
       │             │ Trace  │  │
       │             └────┬───┘  │
       │                  │      │
       │    more turns?   │      │
       ├──────────────────┘      │
       │                         │
       ▼                         ▼
   ┌──────────────────────────────┐
   │            Done              │
   └──────────────────────────────┘
```

## States

| State | Description | Transitions |
|-------|-------------|-------------|
| Init | Session created, context loaded | → Advise or Done |
| Advise | Advisor generates per-instance advice | → Frontier |
| Frontier | Frontier LLM processes input + advice | → Score |
| Score | Reward scorer evaluates output | → Trace |
| Trace | Record trace (input, advice, output, reward) | → Advise or Done |
| Done | Session complete, final trace written | — terminal |

## One-shot session

```
1. Init: load context, select advisor + adapter + scorer
2. Advise: advisor.generate_advice(context)
3. Frontier: adapter.call(prompt, advice)
4. Score: scorer.score(output, context)
5. Trace: tracer.record(trace)
6. Done: return (output, reward, trace)
```

## Multi-turn session (agent loop)

```
1. Init: load context, max_steps=N
2. Loop for step in 1..N:
   a. Advise: advisor.generate_advice(context + observations)
   b. Frontier: adapter.call(prompt, advice)
   c. Observations from frontier response (tool calls, etc.)
   d. Score: scorer.score(output, context)
   e. Trace: tracer.record(trace)
   f. If done_condition_met: break
3. Done: return (output, final_reward, traces[])
```

## Session configuration

```yaml
session:
  mode: "one-shot" | "multi-turn"
  max_steps: 10            # multi-turn only
  advisor:
    model: string          # advisor model ID or endpoint URL
    timeout_ms: 5000
  frontier:
    provider: string       # "anthropic" | "openai" | "ollama"
    model: string
    timeout_ms: 30000
  reward:
    scorer: string         # scorer ID or composite config
    weights: {}            # optional per-dimension weights
  trace:
    output_dir: string     # cassette output directory
    format: "yaml"         # always yaml for v1
```

## Done conditions

A session transitions to Done when:
- One-shot: always after first complete cycle
- Multi-turn: any of:
  - `max_steps` reached
  - `reward.score >= threshold` (configurable)
  - Frontier output contains termination signal
  - Explicit cancel via context

## Error handling

- Advisor failure: use empty advice (no steering), continue
- Frontier failure: record error in trace, return partial result
- Scorer failure: set score to null, add error details to metadata
- Any failure: trace is still written (partial fields allowed)

## Advice injection

The adapter is responsible for injecting advice into the frontier
call. Recommended approach:

```
system_prompt = base_system_prompt + "\n\n" +
  "[Advisor Guidance]\n" + advice.steering_text
```

Advice is hidden from end-user output. Only visible in traces.
