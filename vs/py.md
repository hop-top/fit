# Python: pip install and quickstart

## Installation

```bash
pip install fit
```

With adapter extras:

```bash
pip install "fit[adapters]"  # anthropic, openai SDKs
pip install "fit[dev]"       # pytest, ruff, mypy
```

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
output, reward, trace = session.run(
    "What is the standard deduction?",
    context={"jurisdiction": "US", "filing_status": "single"},
)

print(f"Output: {output}")
print(f"Reward: {reward.score:.2f}")
```

## Adapter configuration

Three adapters ship with the package:

```python
from fit.adapters import AnthropicAdapter, OpenAIAdapter, OllamaAdapter

# Anthropic (Claude) — pass API key explicitly
adapter = AnthropicAdapter(api_key="sk-ant-...")

# OpenAI (GPT)
adapter = OpenAIAdapter()

# Ollama (local)
adapter = OllamaAdapter()     # defaults to localhost:11434
```

Custom adapters implement `call(prompt, advice) -> (output, meta)`:

```python
from fit.types import Advice
from typing import Any

class MyAdapter:
    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        system = f"[Advisor Guidance]\n{advice.steering_text}"
        # Call your LLM here
        output = call_llm(system, prompt)
        return output, {"model": "my-model", "provider": "custom"}
```

## Custom reward functions

Subclass `RewardScorer` for domain-specific scoring:

```python
from fit.reward import RewardScorer
from fit.types import Reward

class TaxAccuracyScorer(RewardScorer):
    def score(self, output: str, context: dict) -> Reward:
        accuracy = compute_accuracy(output, context)
        return Reward(
            score=accuracy,
            breakdown={
                "accuracy": accuracy,
                "relevance": 0.9,
                "safety": 1.0,
                "efficiency": 0.8,
            },
        )
```

Combine multiple scorers with weights:

```python
from fit.reward import CompositeScorer

scorer = CompositeScorer(
    scorers=[TaxAccuracyScorer(), SafetyScorer()],
    weights=[0.7, 0.3],
)
```

Or use the convenience factory:

```python
scorer = CompositeScorer.composite(["accuracy", "relevance", "safety"])
```

## Trace handling

```python
from fit.trace import TraceWriter, TraceReader

writer = TraceWriter("./traces")
writer.write(trace, step=1)

reader = TraceReader("./traces")
sessions = reader.list_sessions()
data = reader.read("sess_abc123", step=1)
```

Traces are xrr-compatible YAML cassettes:

```
traces/
  {session_id}/
    step-001.yaml
    step-002.yaml
```

## Multi-turn sessions

```python
from fit.session import Session, SessionConfig

config = SessionConfig(mode="multi-turn", max_steps=10, reward_threshold=0.95)
session = Session(advisor=advisor, adapter=adapter, scorer=scorer, config=config)
```

## FastAPI integration

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AskRequest(BaseModel):
    prompt: str
    context: dict = {}

@app.post("/ask")
async def ask(req: AskRequest):
    output, reward, trace = session.run(req.prompt, req.context)
    return {"output": output, "reward": {"score": reward.score}}
```
