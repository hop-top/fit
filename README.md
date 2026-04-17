# fit

Train small advisor models to steer black-box LLMs without
fine-tuning.

## What

fit provides two things:

1. **Polyglot serving layer** — loads a trained advisor, injects
   per-request steering advice into frontier LLM calls, scores
   outputs, and records traces.
2. **Training pipeline** (`fit.training`) — ingests traces, builds
   datasets, runs GRPO optimization, and exports deployable
   artifacts.

The frontier model is **never modified**.

## Quick start

### Inference

```python
from fit.session import Session
from fit.advisor import RemoteAdvisor
from fit.reward import CompositeScorer
from fit.adapters import AnthropicAdapter

advisor = RemoteAdvisor.from_endpoint("http://localhost:8080")
adapter = AnthropicAdapter()
scorer = CompositeScorer.composite(
    ["accuracy", "relevance", "safety"]
)
session = Session(
    advisor=advisor, adapter=adapter, scorer=scorer
)

output, reward, trace = session.run(
    "What is the standard deduction?"
)
```

### Training

```bash
# Dry-run (no torch required)
python -m examples.train_advisor \
  --traces spec/fixtures --dry-run

# Full training
python -m examples.train_advisor \
  --traces ./traces --base-model Qwen/Qwen2-0.5B \
  --epochs 3 --output ./advisor-output
```

### Serving a trained advisor

```bash
python -m examples.serve_advisor \
  --model-path ./advisor-output --port 8080
```

## Architecture

```
Input → Advisor → [hidden advice] → Frontier LLM → Output
                                               ↓
                                          Reward Scorer
                                               ↓
                                           Trace Store
                                               ↓
                                      Training Pipeline
                                               ↓
                                        Trained Advisor
```

### Training pipeline

```
Traces (JSONL/YAML/SQLite)
  → TraceIngester        (ingest + filter)
  → DatasetBuilder       (normalize + split)
  → GRPOTrainer          (TRL or simplified fallback)
  → ModelExporter         (safetensors/GGUF/ONNX)
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

### Python extras

```bash
pip install fit                  # core (inference)
pip install fit[adapters]        # + Anthropic/OpenAI
pip install fit[dev]             # + pytest, ruff, mypy
# Training deps (optional):
pip install torch transformers   # simplified GRPO
pip install trl                  # TRL-backed GRPO
pip install safetensors          # safetensors export
```

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

## Citation

```bibtex
@misc{asawa2025trainadvisor,
  title={How to Train Your Advisor: Steering Black-Box
         LLMs with Advisor Models},
  author={Asawa, Parth and Zhu, Alan and Zaharia, Matei
          and Dimakis, Alexandros G.
          and Gonzalez, Joseph E.},
  year={2025},
  eprint={2510.02453},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2510.02453},
  url={https://arxiv.org/abs/2510.02453}
}
```
