package fit

// Reward represents a scoring result (reward-schema-v1).
// Score is *float64 (nullable) so failures serialize as JSON null (not NaN).
type Reward struct {
	Score     *float64           `json:"score" yaml:"score"`
	Breakdown map[string]float64 `json:"breakdown" yaml:"breakdown"`
	Metadata  map[string]any     `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// Float64Ptr returns a pointer to the given float64 value.
func Float64Ptr(v float64) *float64 { return &v }

// RewardScorer scores frontier LLM output.
type RewardScorer interface {
	Score(output string, context map[string]any) (*Reward, error)
}
