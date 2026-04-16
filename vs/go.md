# Go: Integrating fit into an HTTP Service

## Installation

```bash
go get hop.top/fit
```

## Basic usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    fit "hop.top/fit"
)

func main() {
    ctx := context.Background()

    // Connect to advisor endpoint
    advisor := fit.NewRemoteAdvisor("http://localhost:8080")

    // Pick a frontier adapter
    adapter := fit.NewAnthropicAdapter(os.Getenv("ANTHROPIC_API_KEY"))

    // Configure reward scoring
    scorer := fit.NewCompositeScorer(/* your scorers */)

    // Create session
    session := fit.NewSession(advisor, adapter, scorer)

    // Run one-shot session
    result, err := session.Run(ctx, "What is the standard deduction?", map[string]any{
        "jurisdiction":    "US",
        "filing_status":   "single",
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("Output:", result.Output)
    fmt.Printf("Reward: %.2f\n", result.Reward.Score)
}
```

## Adapter configuration

Three frontier adapters ship out of the box:

```go
// Anthropic (Claude)
anthropic := fit.NewAnthropicAdapter(apiKey)

// OpenAI (GPT)
openai := fit.NewOpenAIAdapter(apiKey)

// Ollama (local models)
ollama := fit.NewOllamaAdapter() // defaults to localhost:11434
```

Each adapter injects advice into the system prompt as hidden context:
the frontier model receives guidance, but end users never see it.

Custom adapters implement the `Adapter` interface:

```go
type Adapter interface {
    Call(ctx context.Context, prompt string, advice *fit.Advice) (
        string, map[string]any, error,
    )
}
```

## Custom reward functions

Implement `RewardScorer` for domain-specific scoring:

```go
type TaxAccuracyScorer struct{}

func (s *TaxAccuracyScorer) Score(
    output string,
    context map[string]any,
) (*fit.Reward, error) {
    // Your scoring logic here
    score := computeAccuracy(output, context)
    return &fit.Reward{
        Score:     score,
        Breakdown: map[string]float64{
            "accuracy":   score,
            "relevance":  0.9,
            "safety":     1.0,
            "efficiency": 0.8,
        },
    }, nil
}
```

Combine multiple scorers with `CompositeScorer`:

```go
scorer := fit.NewCompositeScorer(
    []fit.RewardScorer{&TaxAccuracyScorer{}, &SafetyScorer{}},
    []float64{0.7, 0.3}, // weights
)
```

## Trace handling

```go
writer := fit.NewTraceWriter("./traces")

// After session run:
err := writer.Write(result.Trace, 1)
if err != nil {
    log.Fatal(err)
}

// Traces stored as:
// ./traces/{session_id}/step-001.yaml
```

Traces are xrr-compatible YAML cassettes. Load them for replay
or advisor training:

```go
reader := fit.NewTraceReader("./traces")
sessions := reader.ListSessions()
```

## Multi-turn sessions

```go
session := fit.NewSession(advisor, adapter, scorer)
session.Config.Mode = "multi-turn"
session.Config.MaxSteps = 10
session.Config.RewardThreshold = 0.95
```

## HTTP service integration

```go
func handleRequest(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    var req struct {
        Prompt   string         `json:"prompt"`
        Context  map[string]any `json:"context"`
    }
    json.Unmarshal(body, &req)

    result, err := session.Run(r.Context(), req.Prompt, req.Context)
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }

    json.NewEncoder(w).Encode(map[string]any{
        "output": result.Output,
        "reward": result.Reward,
    })
}
```
