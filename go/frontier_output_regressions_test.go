package fit

import (
	"context"
	"testing"
)

// PR#19 regression: spec/trace-format-v1.md requires frontier.output
// in every trace. The session must inject adapter output into
// frontierMeta even when the adapter omits it from its metadata.

// metaNoOutputAdapter returns output but its meta has no "output" key.
type metaNoOutputAdapter struct{}

func (m *metaNoOutputAdapter) Call(_ context.Context, _ string, _ *Advice) (string, map[string]any, error) {
	return "hello", map[string]any{"model": "test"}, nil
}

func TestFrontierOutputInjectedWhenAdapterOmitsFromMeta(t *testing.T) {
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &metaNoOutputAdapter{},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output, ok := result.Trace.Frontier["output"].(string)
	if !ok {
		t.Fatal("trace.Frontier missing 'output' key or wrong type")
	}
	if output != "hello" {
		t.Errorf("trace.Frontier['output'] = %q, want %q", output, "hello")
	}
}

func TestFrontierOutputInjectedOnAdapterFailure(t *testing.T) {
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &errorAdapter{},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output, ok := result.Trace.Frontier["output"].(string)
	if !ok {
		t.Fatal("trace.Frontier missing 'output' key or wrong type on adapter failure")
	}
	if output != "" {
		t.Errorf("trace.Frontier['output'] = %q, want empty string on adapter failure", output)
	}
}
