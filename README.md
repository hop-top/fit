# fit

[![Nightly](https://github.com/hop-top/fit/actions/workflows/nightly.yml/badge.svg)](https://github.com/hop-top/fit/actions/workflows/nightly.yml)

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

## Docker

### Run fit CLI

```bash
docker run --rm ghcr.io/hop-top/fit:latest serve \
  --config /data/config.yaml

docker run --rm ghcr.io/hop-top/fit:latest eval \
  --dataset /data/eval.jsonl
```

### Run fit-coach (training pipeline)

```bash
docker run --rm ghcr.io/hop-top/fit-coach:latest \
  train --config /data/coach.yaml

docker run --rm -p 8080:8080 ghcr.io/hop-top/fit-coach:latest \
  serve --port 8080
```

### Nightly images

Nightly builds from `main` are pushed daily at 04:00 UTC:

```bash
docker pull ghcr.io/hop-top/fit:nightly
docker pull ghcr.io/hop-top/fit-coach:nightly
```

### Local development with docker-compose

```bash
# Start all services
docker compose up -d

# Open a shell in the dev container (all 5 toolchains)
docker compose exec dev bash

# Run tests inside dev container
make check
```

### VS Code / Codespaces devcontainer

Open this repo in VS Code and select "Reopen in Container", or
launch a GitHub Codespace. The devcontainer installs Go, Node,
Python, Rust, PHP, and all project dependencies automatically.

## Building

```bash
make check        # lint + test all ports
make test\:py     # python only
make test\:go     # go only
make lint         # lint all
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
