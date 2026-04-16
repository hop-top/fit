package fit

import "context"

// Advice represents advisor model output (advice-format-v1).
type Advice struct {
	Domain       string            `json:"domain" yaml:"domain"`
	SteeringText string            `json:"steering_text" yaml:"steering_text"`
	Confidence   float64           `json:"confidence" yaml:"confidence"`
	Constraints  []string          `json:"constraints,omitempty" yaml:"constraints,omitempty"`
	Metadata     map[string]any    `json:"metadata,omitempty" yaml:"metadata,omitempty"`
	Version      string            `json:"version" yaml:"version"`
}

// Advisor generates per-instance steering advice.
type Advisor interface {
	GenerateAdvice(ctx context.Context, input map[string]any) (*Advice, error)
	ModelID() string
}
