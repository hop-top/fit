package fit

import (
	"context"
	"testing"
)

// PR#16 regression: frontier failure must return nil error so callers
// don't drop the partial trace.

func TestFrontierFailureReturnsNilError(t *testing.T) {
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &errorAdapter{},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)
	if err != nil {
		t.Fatalf("expected nil error on frontier failure, got: %v", err)
	}
	if result == nil {
		t.Fatal("result is nil, expected partial result")
	}
}

func TestFrontierFailureEmptyOutput(t *testing.T) {
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &errorAdapter{},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, _ := session.Run(context.Background(), "test", nil)
	if result.Output != "" {
		t.Errorf("Output = %q, want empty string on frontier failure", result.Output)
	}
}

func TestFrontierFailurePartialTraceReward(t *testing.T) {
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &errorAdapter{},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, _ := session.Run(context.Background(), "test", nil)
	if result.Reward == nil {
		t.Fatal("Reward is nil")
	}
	if result.Reward.Score != nil {
		t.Errorf("Reward.Score = %v, want nil", result.Reward.Score)
	}
	if result.Reward.Metadata == nil {
		t.Fatal("Reward.Metadata is nil")
	}
	metaErr, ok := result.Reward.Metadata["error"].(string)
	if !ok || metaErr != "frontier_failure" {
		t.Errorf("Reward.Metadata[\"error\"] = %v, want \"frontier_failure\"", result.Reward.Metadata["error"])
	}
}

// leakyAdapter returns non-empty output alongside an error.
type leakyAdapter struct{}

func (l *leakyAdapter) Call(_ context.Context, _ string, _ *Advice) (string, map[string]any, error) {
	return "leaked-output", nil, errAdapterFailed
}

func TestFrontierFailureLeakyOutputSanitized(t *testing.T) {
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &leakyAdapter{},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)
	if err != nil {
		t.Fatalf("expected nil error, got: %v", err)
	}
	if result.Output != "" {
		t.Errorf("Output = %q, want empty string (adapter output must not leak)", result.Output)
	}
}
