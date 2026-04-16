package fit

import "context"

// Adapter is the frontier LLM adapter interface.
type Adapter interface {
	Call(ctx context.Context, prompt string, advice *Advice) (string, map[string]any, error)
}
