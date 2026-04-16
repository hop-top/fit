package fit

import "testing"

func TestTypesExist(t *testing.T) {
	_ = Advice{Domain: "test", SteeringText: "test", Confidence: 0.5}
	_ = Reward{Score: 0.9, Breakdown: map[string]float64{"accuracy": 1.0}}
	_ = SessionConfig{Mode: "one-shot", MaxSteps: 10}
}
