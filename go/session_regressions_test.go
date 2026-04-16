package fit

import (
	"context"
	"testing"
)

// --- Stub implementations ---

type stubAdvisor struct {
	advice *Advice
}

func (s *stubAdvisor) GenerateAdvice(_ context.Context, _ map[string]any) (*Advice, error) {
	return s.advice, nil
}

func (s *stubAdvisor) ModelID() string { return "stub-advisor" }

type stubAdapter struct {
	output string
}

func (s *stubAdapter) Call(_ context.Context, _ string, _ *Advice) (string, map[string]any, error) {
	return s.output, map[string]any{"provider": "stub"}, nil
}

type stubScorer struct {
	score float64
}

func (s *stubScorer) Score(_ string, _ map[string]any) (*Reward, error) {
	return &Reward{Score: s.score, Breakdown: map[string]float64{}}, nil
}

type errorAdvisor struct{}

func (e *errorAdvisor) GenerateAdvice(_ context.Context, _ map[string]any) (*Advice, error) {
	return nil, errOops
}

func (e *errorAdvisor) ModelID() string { return "error-advisor" }

var errOops = &simpleError{"advisor failed"}

type simpleError struct {
	msg string
}

func (e *simpleError) Error() string { return e.msg }

// --- Regression tests ---

func TestFallbackAdviceHasVersion(t *testing.T) {
	// Regression: fallback advice on advisor error must have
	// Version="1.0" and non-nil Metadata.
	session := &Session{
		Advisor: &errorAdvisor{},
		Adapter: &stubAdapter{output: "out"},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Trace == nil || result.Trace.Advice == nil {
		t.Fatal("trace or advice is nil")
	}
	a := result.Trace.Advice

	if a.Version != "1.0" {
		t.Errorf("fallback advice Version = %q, want \"1.0\"", a.Version)
	}
	if a.Metadata == nil {
		t.Error("fallback advice Metadata is nil, want empty map")
	}
}

func TestFallbackAdviceFieldsComplete(t *testing.T) {
	session := &Session{
		Advisor: &errorAdvisor{},
		Adapter: &stubAdapter{output: "out"},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	a := result.Trace.Advice

	if a.Domain != "unknown" {
		t.Errorf("domain = %q, want unknown", a.Domain)
	}
	if a.Confidence != 0 {
		t.Errorf("confidence = %f, want 0", a.Confidence)
	}
	if a.SteeringText != "" {
		t.Errorf("steering_text = %q, want empty", a.SteeringText)
	}
}

// PR#6 Item 6 regression: nil contextMap must not panic
// A scorer that writes to contextMap would panic on nil map.
func TestNilContextMapNoPanic(t *testing.T) {
	writingScorer := &contextWritingScorer{}
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &stubAdapter{output: "out"},
		Scorer:  writingScorer,
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("result is nil")
	}
	// Verify the scorer received a non-nil map it could write to
	if !writingScorer.receivedContext {
		t.Error("scorer did not receive context")
	}
}

// contextWritingScorer writes to the context map to prove nil guard works.
type contextWritingScorer struct {
	receivedContext bool
}

func (c *contextWritingScorer) Score(_ string, ctx map[string]any) (*Reward, error) {
	c.receivedContext = true
	ctx["_written"] = true // would panic if ctx is nil
	return &Reward{Score: 0.5, Breakdown: map[string]float64{}}, nil
}
