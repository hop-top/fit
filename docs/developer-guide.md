# fit Developer Manual

Train small advisor models to steer black-box LLMs without fine-tuning.
App authors use fit to load trained advisors, inject per-request steering
advice into frontier LLM calls, score outputs via reward functions, and
record traces for continuous improvement.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Installation](#installation)
5. [Inference Guide](#inference-guide)
6. [Training Guide](#training-guide)
7. [Adapter Reference](#adapter-reference)
8. [Reward Functions](#reward-functions)
9. [Configuration](#configuration)
10. [Error Handling](#error-handling)
11. [Event Bus](#event-bus)
12. [Multi-Language Support](#multi-language-support)
13. [API Reference](#api-reference)

---

## Overview

fit is a production-ready framework for steering frontier LLMs with
trained advisor models. The architecture decouples advice generation
from frontier execution:

```
Input
  ↓
[Advisor] → generates per-request steering advice
  ↓
[Frontier LLM] → receives input + advice as hidden context
  ↓
[Output] → scored by pluggable reward functions
  ↓
[Trace Store] → accumulated for advisor retraining
```

### When to use fit

- You need consistent, domain-specific behavior from an LLM without
  fine-tuning the base model
- You want to steer responses toward specific criteria (compliance,
  tone, factuality, safety) without modifying the frontier model
- You need to collect feedback signals and improve steering over time
- You operate multiple advisor domains and want to swap them at runtime
- You want cross-language compatibility (Python, Go, TypeScript, Rust,
  PHP all supported)

### What fit does NOT do

- fine-tune the frontier model (advisors never touch base weights)
- replace evaluation or safety testing
- guarantee output correctness (advisors provide soft guidance, not
  guarantees)
- handle multi-turn conversations (current scope: single-prompt advice)

---

## Quick Start

### Minimal Inference Example

```python
from fit.session import Session
from fit.advisor import RemoteAdvisor
from fit.reward import CompositeScorer
from fit.adapters import AnthropicAdapter

# Load a remote advisor
advisor = RemoteAdvisor.from_endpoint("http://localhost:8080")

# Create reward scorer (uses placeholder scorers for demo)
scorer = CompositeScorer.composite(["accuracy", "relevance"])

# Set up adapter for Claude
adapter = AnthropicAdapter(model="claude-sonnet-4-6")

# Instantiate session
session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

# Run a single inference
output, reward, trace = session.run("What is the standard deduction?")

print(f"Output: {output}")
print(f"Reward score: {reward.score}")
print(f"Trace ID: {trace.id}")
```

### Minimal Training Example

```bash
# Dry-run: test pipeline without torch installed
python -m examples.train_advisor \
  --traces ./traces --dry-run

# Full training
python -m examples.train_advisor \
  --traces ./traces \
  --base-model Qwen/Qwen2-0.5B \
  --epochs 3 \
  --output ./advisor-output
```

### Serve the Trained Advisor

```bash
python -m examples.serve_advisor \
  --model-path ./advisor-output \
  --port 8080
```

---

## Core Concepts

### Advice

An `Advice` object represents per-request steering guidance from the
advisor model:

```python
from fit.types import Advice

advice = Advice(
    domain="tax-compliance",
    steering_text="Cite IRS publication numbers. Verify filing status.",
    confidence=0.85,
    constraints=[
        "never fabricate tax code sections",
        "cite specific publication numbers"
    ],
    metadata={"model": "advisor-v1.0", "version": "1.0"}
)
```

**Fields:**
- `domain` (str): Topic/category this advice applies to
- `steering_text` (str): Instruction injected into the frontier model's
  system prompt
- `confidence` (float): [0.0–1.0] model confidence in this advice
- `constraints` (list[str]): Hard constraints the output must respect
- `metadata` (dict): Arbitrary key-value data
- `version` (str): Spec version ("1.0")

### Session

A `Session` orchestrates the full inference cycle: advisor generation
→ frontier call → reward scoring → trace recording.

```python
from fit.session import Session, SessionConfig

config = SessionConfig(
    mode="one-shot",      # only mode currently supported
    max_steps=10,         # unused in one-shot
    reward_threshold=1.0  # unused in one-shot
)

session = Session(
    advisor=advisor,
    adapter=adapter,
    scorer=scorer,
    config=config
)

output, reward, trace = session.run(
    prompt="Explain capital gains tax.",
    context={"jurisdiction": "US", "filing_status": "married"}
)
```

**Session.run() returns** a tuple:
- `output` (str): Frontier model response
- `reward` (Reward): Scoring result
- `trace` (Trace): Complete inference record (for storage/training)

### Adapter

An `Adapter` calls a frontier LLM and injects advice into the system
prompt. fit provides adapters for Anthropic, OpenAI, and Ollama:

```python
from fit.adapters import AnthropicAdapter, OpenAIAdapter, OllamaAdapter

# Anthropic Claude
adapter = AnthropicAdapter(
    model="claude-sonnet-4-6",
    api_key="sk-ant-..."  # or ANTHROPIC_API_KEY env var
)

# OpenAI
adapter = OpenAIAdapter(
    model="gpt-5",
    api_key="sk-..."  # or OPENAI_API_KEY env var
)

# Ollama (local)
adapter = OllamaAdapter(
    model="llama3",
    base_url="http://localhost:11434"
)
```

Each adapter's `call(prompt, advice)` method returns
`(output_text, metadata_dict)`.

### Reward

A `Reward` object represents the quality assessment of a frontier
output:

```python
from fit.types import Reward

reward = Reward(
    score=0.92,  # [0.0–1.0] or None if scoring failed
    breakdown={
        "accuracy": 0.95,
        "relevance": 0.89,
        "safety": 1.0
    },
    metadata={"scorer": "composite", "latency_ms": 42}
)
```

**Null reward semantics:** If `score` is `None`, the output is
considered unscored (e.g., frontier model failed, scorer crashed).
Null rewards are still recorded in traces for debugging.

### Trace

A `Trace` is the complete record of one inference, used for training
and auditing:

```python
from fit.types import Trace

trace = Trace(
    id="trace-uuid",
    session_id="session-uuid",
    timestamp="2025-04-17T12:34:56Z",
    input={"prompt": "...", "context": {...}},
    advice=advice_obj,
    frontier={"model": "claude-...", "output": "...", ...},
    reward=reward_obj,
    metadata={}
)
```

Traces conform to `trace-format-v1.md` and can be written to disk in
YAML format via `TraceWriter`.

---

## Installation

### Core Package

```bash
# Core inference only (no adapter dependencies)
pip install fit
```

### With Adapters

```bash
# Anthropic + OpenAI adapters
pip install fit[adapters]
```

### With Training

```bash
# GRPO training + dataset builder
pip install fit[training]

# Optional: for TRL-backed GRPO (faster, more stable)
pip install trl

# Optional: for safetensors export
pip install safetensors

# Optional: for development
pip install fit[dev]
```

### Docker

```bash
# Run fit CLI
docker run --rm ghcr.io/hop-top/fit:latest serve --config config.yaml

# Run fit-coach (training pipeline)
docker run --rm ghcr.io/hop-top/fit-coach:latest \
  train --config coach.yaml

# Nightly images
docker pull ghcr.io/hop-top/fit:nightly
docker pull ghcr.io/hop-top/fit-coach:nightly
```

---

## Inference Guide

### Step 1: Choose an Adapter

Pick the frontier model you're steering:

```python
from fit.adapters import AnthropicAdapter

adapter = AnthropicAdapter(
    model="claude-sonnet-4-6",
    api_key=None  # uses ANTHROPIC_API_KEY env var by default
)
```

Alternatively, pass a pre-initialized client:

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")
adapter = AnthropicAdapter(client=client)
```

### Step 2: Load an Advisor

Either from a remote endpoint (after training + deployment):

```python
from fit.advisor import RemoteAdvisor

advisor = RemoteAdvisor.from_endpoint("http://localhost:8080")
```

Or implement the `Advisor` interface locally:

```python
from fit.advisor import Advisor
from fit.types import Advice

class CustomAdvisor(Advisor):
    def generate_advice(self, context: dict) -> Advice:
        # Your logic here
        return Advice(
            domain="custom",
            steering_text="...",
            confidence=0.8
        )
    
    def model_id(self) -> str:
        return "custom-advisor-v1"
```

### Step 3: Build a Reward Scorer

Use the built-in `CompositeScorer` for convenience:

```python
from fit.reward import CompositeScorer

scorer = CompositeScorer.composite([
    "accuracy",
    "relevance",
    "safety"
])
```

Or compose custom scorers:

```python
from fit.reward import DimensionScorer, CompositeScorer

scorers = [
    DimensionScorer("accuracy"),
    DimensionScorer("relevance")
]
weights = [0.7, 0.3]

scorer = CompositeScorer(scorers, weights)
```

For production, implement `RewardScorer`:

```python
from fit.reward import RewardScorer
from fit.types import Reward

class LLMJudgeScorer(RewardScorer):
    def score(self, output: str, context: dict) -> Reward:
        # Use an LLM as judge
        return Reward(
            score=...,
            breakdown={"quality": ...}
        )
```

### Step 4: Create a Session

```python
from fit.session import Session

session = Session(
    advisor=advisor,
    adapter=adapter,
    scorer=scorer
)
```

### Step 5: Run Inference

```python
output, reward, trace = session.run(
    prompt="What are capital gains taxes?",
    context={"jurisdiction": "US"}
)

print(f"Advice: {trace.advice.steering_text}")
print(f"Output: {output}")
print(f"Score: {reward.score}")
```

### Step 6: Store Traces

For training dataset collection, persist traces to disk:

```python
from fit.trace import TraceWriter

writer = TraceWriter("./traces")
path = writer.write(trace, step=1)
print(f"Trace saved to {path}")
```

Traces are stored as YAML cassettes in `./traces/{session_id}/step-XXX.yaml`.

### Error Handling

By default, `Session.run()` is fault-tolerant:
- If advisor fails, advice becomes empty (confidence=0.0)
- If frontier fails, output is empty, reward is null
- If scorer fails, reward is null

For stricter error handling, wrap in try-catch and inspect trace:

```python
try:
    output, reward, trace = session.run(prompt)
except Exception as e:
    logger.error(f"Session failed: {e}")
    return

if reward.score is None:
    logger.warning(f"Scoring failed: {reward.metadata}")

if trace.advice.confidence < 0.5:
    logger.info("Advisor low-confidence; treating output cautiously")
```

---

## Training Guide

### Overview

The training pipeline ingests production traces and optimizes a small
advisor model using GRPO (Generalized Reward Policy Optimization):

```
Traces (JSONL/YAML/SQLite)
  ↓
TraceIngester (load + normalize)
  ↓
DatasetBuilder (format → TrainingExample)
  ↓
GRPOTrainer (TRL or simplified torch)
  ↓
ModelExporter (safetensors/GGUF/ONNX)
  ↓
FileAdvisor (serve via HTTP)
```

### Collecting Traces

Enable trace collection in your app by storing session results:

```python
from fit.trace import TraceWriter

# In your production loop
session = Session(advisor, adapter, scorer)
output, reward, trace = session.run(prompt)

# Persist for later training
writer = TraceWriter("./production_traces")
writer.write(trace)

# Also record manually with feedback
trace.reward = Reward(score=user_feedback_score, breakdown={})
writer.write(trace)
```

### Ingesting Traces

The `TraceIngester` reads traces from disk:

```python
from fit.training.tracer import TraceIngester

ingester = TraceIngester()
ingester.load_batch(["./traces"])  # directory or file paths

# Optionally filter by domain
ingester = ingester.filter(domain="tax-compliance")

records = ingester.to_trace_records()
print(f"Loaded {len(records)} traces")
```

Supported formats:
- YAML (xrr-compatible): `session-id/step-001.yaml`
- JSONL: one trace per line
- SQLite: table with `data` column (JSON blob)

### Building Datasets

The `DatasetBuilder` normalizes traces into training examples:

```python
from fit.training.dataset import DatasetBuilder

builder = DatasetBuilder(records)
dataset = builder.build(
    normalize_rewards=True,    # min-max → [0, 1]
    group_by_session=True      # sort by session for episode grouping
)

# Check dataset stats
stats = dataset.reward_stats()
print(f"Mean reward: {stats['mean']:.3f}")
print(f"Dataset size: {len(dataset)}")

# Split for validation
train_ds, val_ds = dataset.split(val_ratio=0.2, seed=42)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
```

### Training with GRPO

```python
from fit.training.grpo import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    base_model="Qwen/Qwen2-0.5B",  # or any HF model
    learning_rate=1e-5,
    epochs=3,
    batch_size=8,
    output_dir="./advisor-output",
    use_trl=True  # try TRL first, fall back to simplified trainer
)

trainer = GRPOTrainer(config)
result = trainer.train(train_ds)

print(f"Training complete!")
print(f"Final loss: {result.final_loss:.4f}")
print(f"Model path: {result.model_path}")
```

**Requirements:**
- `torch` and `transformers` for training
- `trl` strongly recommended (faster, more stable)
- For CPU-only: set `torch.device('cpu')` (slower)

### Dry-Run Mode

Test the full pipeline without installing torch:

```bash
python -m examples.train_advisor \
  --traces ./traces \
  --dry-run
```

Dry-run skips the actual training step, useful for validating data
ingestion and dataset building.

### Exporting Models

The `ModelExporter` converts trained models to deployment formats:

```python
from fit.training.export import ModelExporter

exporter = ModelExporter("./advisor-output")

# Export to safetensors
exporter.to_safetensors("./advisor-safetensors")

# Export to GGUF (quantized, for inference)
exporter.to_gguf("./advisor-gguf")

# Generate model card (metadata)
card = exporter.generate_model_card(result)
print(card)
```

### Serving the Advisor

Once trained and exported, serve via HTTP:

```bash
python -m examples.serve_advisor \
  --model-path ./advisor-output \
  --port 8080 \
  --host 0.0.0.0
```

Endpoints:
- `POST /advise` — accepts JSON context, returns advice-format-v1
- `GET /health` — liveness check
- `GET /model` — model metadata

### Example: End-to-End Training Script

```python
"""Complete training workflow."""
from fit.training import (
    TraceIngester,
    DatasetBuilder,
    GRPOConfig,
    GRPOTrainer,
    ModelExporter
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Ingest
logger.info("Loading traces...")
ingester = TraceIngester()
ingester.load_batch(["./production_traces"])
records = ingester.to_trace_records()
logger.info(f"Loaded {len(records)} traces")

# 2. Build dataset
logger.info("Building dataset...")
builder = DatasetBuilder(records)
dataset = builder.build(normalize_rewards=True)
train_ds, val_ds = dataset.split(val_ratio=0.2)
logger.info(f"Dataset: train={len(train_ds)}, val={len(val_ds)}")

# 3. Train
logger.info("Training advisor...")
config = GRPOConfig(
    base_model="Qwen/Qwen2-0.5B",
    epochs=3,
    output_dir="./advisor-v2"
)
trainer = GRPOTrainer(config)
result = trainer.train(train_ds)
logger.info(f"Training complete. Loss: {result.final_loss:.4f}")

# 4. Export
logger.info("Exporting model...")
exporter = ModelExporter(result.model_path)
exporter.to_safetensors("./advisor-v2-safetensors")
logger.info("Export complete")
```

---

## Adapter Reference

### AnthropicAdapter

Calls Claude models via the Anthropic API.

```python
from fit.adapters import AnthropicAdapter

# Default: claude-sonnet-4-6, uses ANTHROPIC_API_KEY env var
adapter = AnthropicAdapter()

# Explicit config
adapter = AnthropicAdapter(
    model="claude-opus-4-1",
    api_key="sk-ant-..."
)

# With pre-initialized client
import anthropic
client = anthropic.Anthropic()
adapter = AnthropicAdapter(client=client)
```

**Advice injection:** Advice is prepended to system prompt as:
```
[Advisor Guidance]
{steering_text}
```

**Metadata returned:**
- `model` — actual model identifier from response
- `provider` — always "anthropic"
- `output` — the response text
- `usage` — token counts (prompt, completion, total)

### OpenAIAdapter

Calls GPT models via OpenAI API.

```python
from fit.adapters import OpenAIAdapter

# Default: gpt-5, uses OPENAI_API_KEY env var
adapter = OpenAIAdapter()

# Explicit config
adapter = OpenAIAdapter(
    model="gpt-4",
    api_key="sk-..."
)

# With pre-initialized client
import openai
client = openai.OpenAI()
adapter = OpenAIAdapter(client=client)
```

**Advice injection:** Advice is the system message (via
`messages[0]` with role "system").

### OllamaAdapter

Calls locally-running Ollama instance.

```python
from fit.adapters import OllamaAdapter

# Default: llama3 on localhost:11434
adapter = OllamaAdapter()

# Custom model + host
adapter = OllamaAdapter(
    model="mistral",
    base_url="http://192.168.1.100:11434"
)
```

**Advice injection:** Advice is the system message.

**Note:** Requires a running Ollama server. Start with:
```bash
ollama serve
ollama pull llama3
```

### Custom Adapters

Implement the `Adapter` interface:

```python
from fit.adapters.base import Adapter
from fit.types import Advice

class MyCustomAdapter(Adapter):
    def call(
        self, prompt: str, advice: Advice
    ) -> tuple[str, dict]:
        # 1. Inject advice into your model's input/context
        system_prompt = f"[Advisor]\n{advice.steering_text}"
        
        # 2. Call your model
        response = my_model.generate(
            prompt=prompt,
            system=system_prompt
        )
        
        # 3. Return (output, metadata)
        return response.text, {
            "model": response.model_id,
            "provider": "custom",
            "latency_ms": response.latency
        }
```

---

## Reward Functions

### Built-in Scorers

#### DimensionScorer

Placeholder scorer that returns neutral (0.5) scores. Useful for demo
and testing:

```python
from fit.reward import DimensionScorer

scorer = DimensionScorer("accuracy")
reward = scorer.score("Any output", {})
# Reward(score=0.5, breakdown={"accuracy": 0.5})
```

#### CompositeScorer

Weighted combination of multiple scorers:

```python
from fit.reward import CompositeScorer, DimensionScorer

scorers = [
    DimensionScorer("accuracy"),
    DimensionScorer("safety")
]
weights = [0.7, 0.3]

scorer = CompositeScorer(scorers, weights)
reward = scorer.score("output", context)

# Returns weighted average + merged breakdown
print(reward.score)          # 0.5 (0.7*0.5 + 0.3*0.5)
print(reward.breakdown)      # {"accuracy": 0.5, "safety": 0.5}
```

Convenience factory:

```python
scorer = CompositeScorer.composite(
    ["accuracy", "relevance", "safety"]
)
# Creates equal-weight DimensionScorers
```

### Training-Only Reward Functions

These are used during advisor training, not during inference.

#### ExactMatchReward

Binary reward: 1.0 if output contains substring, else 0.0.

```python
from fit.training.reward_fn import ExactMatchReward

scorer = ExactMatchReward(
    expected="IRS Publication 544",
    case_sensitive=False
)

reward = scorer("What's capital gains tax?", "...", 
                "See IRS Publication 544...")
# Returns 1.0
```

#### RubricJudgeReward

Keyword-based scoring with rubric entries (pattern, weight):

```python
from fit.training.reward_fn import RubricJudgeReward

scorer = RubricJudgeReward([
    (r"IRS Publication \d+", 0.4),
    (r"filing status", 0.3),
    (r"state-specific", 0.3)
])

reward = scorer("context", "advice", 
                "Per IRS Pub 544, filing status matters...")
# Rewards matched patterns: (0.4 + 0.3) / 1.0 = 0.7
```

#### LLMJudgeReward

Uses a frontier LLM as judge:

```python
from fit.training.reward_fn import LLMJudgeReward

scorer = LLMJudgeReward(
    prompt_template="""
    Context: {context}
    Advice: {advice}
    Output: {output}
    
    Score quality (0-10): 
    """,
    model="claude-sonnet-4-6"
)

reward = scorer(context, advice, output)
```

#### CompositeReward

Weighted combination for training:

```python
from fit.training.reward_fn import CompositeReward, ExactMatchReward

scorers = [
    ExactMatchReward("IRS Publication"),
    ExactMatchReward("filing status")
]
weights = [0.6, 0.4]

scorer = CompositeReward(scorers, weights)
reward = scorer(context, advice, output)
```

### Custom Reward Scorers

Inference (implement `RewardScorer`):

```python
from fit.reward import RewardScorer
from fit.types import Reward

class SemanticSimilarityScorer(RewardScorer):
    def score(self, output: str, context: dict) -> Reward:
        expected = context.get("expected_summary", "")
        
        # Your similarity logic
        similarity = self._compute_similarity(expected, output)
        
        return Reward(
            score=similarity,
            breakdown={"semantic_similarity": similarity}
        )
    
    def _compute_similarity(self, a: str, b: str) -> float:
        # e.g., use sentence-transformers
        pass
```

Training (implement `RewardFn`):

```python
from fit.training.reward_fn import RewardFn

class CustomRewardFn(RewardFn):
    def __call__(self, context: str, advice: str, 
                 output: str) -> float:
        # Compute scalar reward
        if "tax" in output.lower():
            return 0.8
        return 0.2
```

---

## Configuration

### SessionConfig

Controls session-level behavior:

```python
from fit.session import SessionConfig, Session

config = SessionConfig(
    mode="one-shot",           # Currently only mode
    max_steps=10,              # Unused in one-shot
    reward_threshold=1.0       # Unused in one-shot
)

session = Session(advisor, adapter, scorer, config=config)
```

### GRPOConfig

Controls training:

```python
from fit.training.grpo import GRPOConfig

config = GRPOConfig(
    base_model="Qwen/Qwen2-0.5B",
    learning_rate=1e-5,
    epochs=3,
    batch_size=8,
    max_seq_length=512,
    reward_shaping="linear",   # linear, exponential, clipped
    output_dir="./advisor-output",
    use_trl=True,              # try TRL first
    seed=42,
    beta=0.1,                  # KL penalty (simplified trainer)
    clip_range=0.2             # PPO clip (simplified trainer)
)
```

### Environment Variables

All adapters respect standard env vars:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export ADVISOR_MODEL_PATH=./advisor-output
```

### TraceIngestConfig

Controls trace loading:

```python
from fit.training.tracer import TraceIngestConfig, TraceIngester

config = TraceIngestConfig(
    yaml_glob="*.y*ml",
    required_keys=("input", "frontier"),
    sqlite_data_column="data",
    metadata_filters={}  # Optional filters
)

ingester = TraceIngester(config)
```

---

## Error Handling

> **Coming soon** (Next release)
>
> Structured error codes and recovery strategies. For now:
> - Errors are caught and logged within `Session.run()`
> - Check `reward.metadata` for scorer failures
> - Check `trace.metadata` for frontier failures
> - Inspect `trace.advice.confidence` for advisor uncertainty

Current fallback behavior:
- Advisor error → empty advice (confidence=0.0)
- Frontier error → empty output, null reward
- Scorer error → null reward with error metadata

Example defensive code:

```python
output, reward, trace = session.run(prompt)

if trace.advice.confidence == 0.0:
    logger.warning("Advisor failed; advice unavailable")

if reward.score is None:
    logger.warning(f"Scoring failed: {reward.metadata.get('error')}")

if "error" in trace.frontier:
    logger.error(f"Frontier failed: {trace.frontier['error']}")
```

---

## Event Bus

> **Coming soon** (Next release)
>
> Pub/sub event bus for trace creation, batch events, and advisor
> updates. For now, persist traces manually via `TraceWriter`.

Planned topics:
- `fit.trace.created` — fired after each `Session.run()`
- `fit.trace.batch` — fired when batch ingestion completes
- `fit.advisor.updated` — fired when advisor redeployed

Placeholder for planned API:

```python
# (future)
from fit.event_bus import EventBus

bus = EventBus()

def on_trace_created(trace):
    # Store, alert, etc.
    pass

bus.subscribe("fit.trace.created", on_trace_created)
```

---

## Multi-Language Support

fit is polyglot. All ports implement the same specs:

| Language | Package | Status |
|----------|---------|--------|
| Python   | `fit` on PyPI | Production |
| Go       | `hop.top/fit` | Production |
| TypeScript | `@hop/fit` on npm | Production |
| Rust     | `fit` on crates.io | Production |
| PHP      | `hop/fit` on Packagist | Production |

All ports share:
- `advice-format-v1.md` — Advice JSON schema
- `reward-schema-v1.md` — Reward structure
- `trace-format-v1.md` — xrr-compatible trace format
- `session-protocol.md` — Session state machine

Cross-language example:

```python
# Train in Python
python -m examples.train_advisor --traces ./traces

# Serve in Go
go run github.com/hop-top/fit/cmd/fit serve --config config.yaml

# Consume in TypeScript/Node.js
import { RemoteAdvisor } from "@hop/fit";
const advisor = new RemoteAdvisor("http://localhost:8080");

# Consume in Rust
let advisor = RemoteAdvisor::from_endpoint("http://localhost:8080");
```

---

## API Reference

### fit.types

#### Advice

```python
@dataclass(frozen=True)
class Advice:
    domain: str
    steering_text: str
    confidence: float
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
```

#### Reward

```python
@dataclass(frozen=True)
class Reward:
    score: float | None
    breakdown: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### Trace

```python
@dataclass(frozen=True)
class Trace:
    id: str
    session_id: str
    timestamp: str
    input: dict[str, Any]
    advice: Advice
    frontier: dict[str, Any]
    reward: Reward
    metadata: dict[str, Any] = field(default_factory=dict)
```

### fit.advisor

#### Advisor (ABC)

```python
class Advisor(ABC):
    @abstractmethod
    def generate_advice(self, context: dict[str, Any]) -> Advice:
        """Generate steering advice for the given context."""

    @abstractmethod
    def model_id(self) -> str:
        """Return advisor model identifier."""
```

#### RemoteAdvisor

```python
class RemoteAdvisor(Advisor):
    def __init__(self, endpoint: str, timeout_ms: int = 5000):
        """
        Args:
            endpoint: Base URL of advisor HTTP service
            timeout_ms: Request timeout in milliseconds
        """

    @classmethod
    def from_endpoint(cls, url: str) -> RemoteAdvisor:
        """Convenience factory."""
```

### fit.session

#### SessionConfig

```python
@dataclass
class SessionConfig:
    mode: str = "one-shot"
    max_steps: int = 10
    reward_threshold: float = 1.0
```

#### Session

```python
class Session:
    def __init__(
        self,
        advisor: Advisor,
        adapter: Any,
        scorer: RewardScorer,
        config: SessionConfig | None = None,
    ):
        """
        Args:
            advisor: Generates per-request advice
            adapter: Calls frontier LLM
            scorer: Scores frontier output
            config: Session configuration
        """

    def run(
        self, 
        prompt: str, 
        context: dict[str, Any] | None = None
    ) -> tuple[str, Reward, Trace]:
        """
        Run one-shot inference.
        
        Args:
            prompt: User input
            context: Optional context dict passed to advisor/scorer
            
        Returns:
            (output, reward, trace)
        """
```

### fit.reward

#### RewardScorer (ABC)

```python
class RewardScorer(ABC):
    @abstractmethod
    def score(self, output: str, context: dict[str, Any]) -> Reward:
        """Score the frontier output given context."""
```

#### DimensionScorer

```python
class DimensionScorer(RewardScorer):
    def __init__(self, dimension: str):
        """Creates stub scorer (returns 0.5)."""

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        """Returns neutral Reward(score=0.5, breakdown={dimension: 0.5})."""
```

#### CompositeScorer

```python
class CompositeScorer(RewardScorer):
    def __init__(
        self,
        scorers: Sequence[RewardScorer],
        weights: Sequence[float] | None = None,
    ):
        """
        Args:
            scorers: List of scorers to combine
            weights: Optional weights (defaults to equal weighting)
        """

    @classmethod
    def composite(cls, names: list[str]) -> CompositeScorer:
        """Create from dimension names (creates DimensionScorers)."""

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        """Returns weighted average + merged breakdown."""
```

### fit.adapters

#### Adapter (ABC)

```python
class Adapter(ABC):
    @abstractmethod
    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        """Call frontier LLM. Returns (output_text, metadata)."""
```

#### AnthropicAdapter

```python
class AnthropicAdapter(Adapter):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        client: Any | None = None,
    ):
        """
        Args:
            model: Claude model ID
            api_key: API key (defaults to ANTHROPIC_API_KEY env)
            client: Pre-initialized anthropic.Anthropic client
        """
```

#### OpenAIAdapter

```python
class OpenAIAdapter(Adapter):
    def __init__(
        self,
        model: str = "gpt-5",
        api_key: str | None = None,
        client: Any | None = None,
    ):
        """
        Args:
            model: GPT model ID
            api_key: API key (defaults to OPENAI_API_KEY env)
            client: Pre-initialized openai.OpenAI client
        """
```

#### OllamaAdapter

```python
class OllamaAdapter(Adapter):
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        http_client: Any | None = None,
    ):
        """
        Args:
            model: Ollama model name
            base_url: Ollama server URL
            http_client: Pre-initialized httpx client
        """
```

### fit.trace

#### TraceWriter

```python
class TraceWriter:
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory to write YAML traces to
        """

    def write(self, trace: Trace, step: int = 1) -> Path:
        """
        Write trace as YAML cassette.
        
        Creates: output_dir/{session_id}/step-{step:03d}.yaml
        
        Returns:
            Path to written file
        """
```

#### TraceReader

```python
class TraceReader:
    def __init__(self, output_dir: str):
        """Args: output_dir with session subdirs."""

    def list_sessions(self) -> list[str]:
        """Returns list of session IDs in directory."""

    def read(self, session_id: str, step: int = 1) -> dict[str, Any]:
        """Read trace YAML as dict."""
```

### fit.training

#### TraceIngester

```python
class TraceIngester:
    def __init__(self, config: TraceIngestConfig | None = None):
        """Initialize with optional config."""

    def load_batch(self, paths: list[str | Path]) -> None:
        """Load traces from files or directories."""

    def filter(self, domain: str | None = None) -> TraceIngester:
        """Filter by domain."""

    def to_trace_records(self) -> list[TraceRecord]:
        """Convert to normalized TraceRecords."""
```

#### DatasetBuilder

```python
class DatasetBuilder:
    def __init__(self, records: list[TraceRecord]):
        """Initialize with trace records."""

    def build(
        self,
        normalize_rewards: bool = True,
        group_by_session: bool = True,
    ) -> FitDataset:
        """Build training dataset."""
```

#### FitDataset

```python
class FitDataset:
    def __init__(self, examples: list[TrainingExample]):
        """Initialize with examples."""

    def split(
        self,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[FitDataset, FitDataset]:
        """Split into (train, val)."""

    def reward_stats(self) -> dict[str, float]:
        """Compute mean, std, min, max, count."""

    def __len__(self) -> int
    def __getitem__(self, idx: int) -> TrainingExample
    def __iter__(self)
```

#### GRPOTrainer

```python
class GRPOTrainer:
    def __init__(
        self,
        config: GRPOConfig,
        reward_fn: RewardFn | None = None,
    ):
        """Initialize trainer."""

    def train(self, dataset: FitDataset) -> TrainingResult:
        """Train advisor model. Returns TrainingResult."""
```

#### GRPOConfig

```python
@dataclass
class GRPOConfig:
    base_model: str = "Qwen/Qwen2-0.5B"
    learning_rate: float = 1e-5
    epochs: int = 3
    batch_size: int = 8
    max_seq_length: int = 512
    reward_shaping: str = "linear"
    output_dir: str = "./advisor-output"
    use_trl: bool = True
    seed: int = 42
    beta: float = 0.1
    clip_range: float = 0.2
```

#### ModelExporter

```python
class ModelExporter:
    def __init__(self, model_path: str):
        """Initialize exporter."""

    def to_safetensors(self, output_dir: str) -> Path:
        """Export to safetensors format."""

    def to_gguf(self, output_dir: str) -> Path:
        """Export to GGUF (quantized)."""

    def to_onnx(self, output_dir: str) -> Path:
        """Export to ONNX format."""

    def generate_model_card(self, result: TrainingResult) -> dict:
        """Generate metadata card."""
```

---

## Summary

fit enables steering black-box LLMs via trained advisor models, without
modifying the frontier model. Integrate by:

1. Choosing/implementing an adapter for your frontier LLM
2. Loading or training an advisor model
3. Creating a reward scorer
4. Running inference via `Session.run()`
5. Collecting traces for continuous improvement

All ports (Python, Go, TypeScript, Rust, PHP) share the same specs and
can interoperate via HTTP. Start with the Quick Start example and scale
to production with custom adapters, scorers, and reward functions.
