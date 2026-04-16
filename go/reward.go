package fit

// Reward represents a scoring result (reward-schema-v1).
type Reward struct {
	Score     float64        `json:"score" yaml:"score"`
	Breakdown map[string]float64 `json:"breakdown" yaml:"breakdown"`
	Metadata  map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// RewardScorer scores frontier LLM output.
type RewardScorer interface {
	Score(output string, context map[string]any) (*Reward, error)
}
