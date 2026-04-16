# Reward Schema v1

Language-agnostic spec for reward scoring.

## Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["score", "breakdown"],
  "properties": {
    "score": {
      "type": ["number", "null"],
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Aggregate reward scalar. null indicates scorer failure (see metadata.error)"
    },
    "breakdown": {
      "type": "object",
      "properties": {
        "accuracy": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "relevance": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "safety": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "efficiency": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        }
      },
      "additionalProperties": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0
      }
    },
    "metadata": {
      "type": "object",
      "additionalProperties": true
    }
  }
}
```

## Examples

### Exact match reward
```json
{
  "score": 1.0,
  "breakdown": {
    "accuracy": 1.0,
    "relevance": 1.0,
    "safety": 1.0,
    "efficiency": 1.0
  }
}
```

### Partial credit reward
```json
{
  "score": 0.62,
  "breakdown": {
    "accuracy": 0.7,
    "relevance": 0.8,
    "safety": 1.0,
    "efficiency": 0.0
  },
  "metadata": {
    "scorer": "rubric-judge-v2",
    "notes": "correct conclusion but verbose path"
  }
}
```

### Scorer failure
```json
{
  "score": null,
  "breakdown": {},
  "metadata": {
    "error": "scorer_timeout",
    "error_detail": "rubric-judge-v2 timed out after 5000ms"
  }
}
```

### Agent efficiency reward
```json
{
  "score": 0.85,
  "breakdown": {
    "accuracy": 1.0,
    "relevance": 0.9,
    "safety": 1.0,
    "efficiency": 0.5
  },
  "metadata": {
    "steps_taken": 12,
    "steps_optimal": 6,
    "issue_resolved": true
  }
}
```

## Composite scoring

When multiple scorers contribute:

```
final_score = weighted_sum(scorer_i.score, weight_i)
```

Default weights: equal. Configurable per domain.

## Determinism

For conformance testing: given identical (input, advice, output),
the reward MUST be deterministic. Scorers that use LLM judges
MUST pin model version + temperature=0.
