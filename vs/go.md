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
    "os"

    fit "hop.top/fit"
)

func main() {
    ctx := context.Background()

    // Implement the Advisor interface for your backend
    advisor := &MyRemoteAdvisor{endpoint: "http://localhost:8080"}

    // Implement the Adapter interface for your frontier LLM
    adapter := &MyAnthropicAdapter{apiKey: os.Getenv("ANTHROPIC_API_KEY")}

    // Implement the RewardScorer interface for your domain
    scorer := &MyCompositeScorer{}

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

Implement the `Adapter` interface for your frontier LLM:

```go
// AnthropicAdapter calls the Anthropic Messages API.
type AnthropicAdapter struct {
    apiKey string
    model  string
}

func (a *AnthropicAdapter) Call(
    ctx context.Context, prompt string, advice *fit.Advice,
) (string, map[string]any, error) {
    system := "[Advisor Guidance]\n" + advice.SteeringText
    // Call Anthropic API with system prompt containing advice
    // ...
    return output, meta, nil
}
```

Each adapter injects advice into the system prompt as hidden context:
the frontier model receives guidance, but end users never see it.

The `Adapter` interface:

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

Combine multiple scorers with a weighted composite:

```go
type CompositeScorer struct {
    scorers []fit.RewardScorer
    weights []float64
}

func (c *CompositeScorer) Score(
    output string, context map[string]any,
) (*fit.Reward, error) {
    totalWeight := 0.0
    combined := 0.0
    for i, s := range c.scorers {
        r, err := s.Score(output, context)
        if err != nil {
            return nil, err
        }
        combined += r.Score * c.weights[i]
        totalWeight += c.weights[i]
    }
    return &fit.Reward{Score: combined / totalWeight}, nil
}
```

## Trace handling

```go
writer := fit.NewTraceWriter("./traces")

// Trace structs are produced by Session.Run:
trace := result.Trace
// Write trace to xrr-compatible YAML (implement your own writer
// or use the TraceWriter as a starting point)
```

Traces are xrr-compatible YAML cassettes. The `Trace` struct
contains all session data for replay or advisor training.

## Multi-turn sessions

> **Note:** Multi-turn sessions are not yet implemented in the Go port.
> The `SessionConfig` fields are reserved for future use.

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
