# fit

Train small advisor models to steer black-box LLMs without fine-tuning.

Based on: *How to Train Your Advisor — Steering Black-Box LLMs with
ADVISOR MODELS*.

## What

fit provides a polyglot serving layer that:

1. Loads a trained advisor model (from fit-coach pipeline)
2. Generates per-instance steering advice for each request
3. Injects advice into frontier LLM calls as hidden context
4. Scores outputs via pluggable reward functions
5. Records traces for continuous advisor improvement

The frontier model is **never modified**.

## Quick start

```python
from fit.session import Session
from fit.advisor import RemoteAdvisor
from fit.reward import CompositeScorer
from fit.adapters import AnthropicAdapter

advisor = RemoteAdvisor.from_endpoint("http://localhost:8080")
adapter = AnthropicAdapter()
scorer = CompositeScorer.composite(["accuracy", "relevance", "safety"])
session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

output, reward, trace = session.run("What is the standard deduction?")
```

## Architecture

```
Input → Advisor → [hidden advice] → Frontier LLM → Output
                                               ↓
                                          Reward Scorer
                                               ↓
                                           Trace Store
                                               ↓
                                      Advisor Training (fit-coach)
```

## Monorepo map

| Subdir | Published as | Language |
|--------|-------------|----------|
| `go/` | `hop.top/fit` | Go module |
| `ts/` | `@hop/fit` | npm package |
| `py/` | `fit` | PyPI package |
| `rs/` | `fit` | Rust crate |
| `php/` | `hop/fit` | Composer package |
| `spec/` | — | Language-agnostic specs |
| `vs/` | — | Migration guides |
| `docs/` | — | Paper summaries, architecture |

## Building

```bash
task check        # lint + test all ports
task test:py      # python only
task test:go      # go only
task lint         # lint all
```

## Spec

All ports implement against shared specs in `spec/`:

- `advice-format-v1.md` — Advisor output format
- `reward-schema-v1.md` — Reward scoring schema
- `trace-format-v1.md` — xrr-compatible trace format
- `session-protocol.md` — Session lifecycle state machine

## License

MIT — see [LICENSE](LICENSE).
