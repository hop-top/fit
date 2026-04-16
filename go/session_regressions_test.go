package fit

import (
	"bytes"
	"context"
	"encoding/json"
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
	return &Reward{Score: Float64Ptr(s.score), Breakdown: map[string]float64{}}, nil
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
	return &Reward{Score: Float64Ptr(0.5), Breakdown: map[string]float64{}}, nil
}

// PR#11 regression: adapter failure must produce a partial trace, not nil.
func TestAdapterFailureProducesPartialTrace(t *testing.T) {
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &errorAdapter{},
		Scorer:  &stubScorer{score: 0.5},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)

	// PR#16: frontier failure must return nil error (partial trace, not error).
	if err != nil {
		t.Fatalf("expected nil error from frontier failure, got: %v", err)
	}

	// Partial result must be returned (not nil).
	if result == nil {
		t.Fatal("result is nil, expected partial result with trace")
	}

	// Trace must be present.
	if result.Trace == nil {
		t.Fatal("result.Trace is nil, expected partial trace")
	}

	// Reward score must be nil (null in JSON, per reward-schema-v1).
	if result.Reward.Score != nil {
		t.Errorf("reward.Score = %v, want nil", result.Reward.Score)
	}

	// Frontier must contain the error.
	if result.Trace.Frontier == nil {
		t.Fatal("trace.Frontier is nil, expected error info")
	}
	frontierErr, _ := result.Trace.Frontier["error"].(string)
	if frontierErr == "" {
		t.Error("trace.Frontier missing 'error' key")
	}
}

// errorAdapter always returns an error from Call().
type errorAdapter struct{}

func (e *errorAdapter) Call(_ context.Context, _ string, _ *Advice) (string, map[string]any, error) {
	return "", nil, errAdapterFailed
}

var errAdapterFailed = &simpleError{"adapter failed"}

// PR#15 regression: Reward with nil Score must serialize to JSON null.
func TestRewardNilScoreSerializesNull(t *testing.T) {
	r := &Reward{Score: nil, Breakdown: map[string]float64{}}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	if !bytes.Contains(data, []byte(`"score":null`)) {
		t.Errorf("JSON = %s, want score:null", data)
	}
	// Round-trip must parse back to nil
	var r2 Reward
	if err := json.Unmarshal(data, &r2); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if r2.Score != nil {
		t.Errorf("round-trip Score = %v, want nil", r2.Score)
	}
}

// PR#15 regression: Reward with numeric Score must round-trip through JSON.
func TestRewardNumericScoreRoundTrip(t *testing.T) {
	r := &Reward{Score: Float64Ptr(0.75), Breakdown: map[string]float64{"acc": 0.7}}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	if !bytes.Contains(data, []byte(`"score":0.75`)) {
		t.Errorf("JSON = %s, want score:0.75", data)
	}
	var r2 Reward
	if err := json.Unmarshal(data, &r2); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if r2.Score == nil || *r2.Score != 0.75 {
		t.Errorf("round-trip Score = %v, want 0.75", r2.Score)
	}
}

// PR#15 regression: scorer failure must produce nil score with error metadata.
func TestScorerFailureProducesNilScore(t *testing.T) {
	session := &Session{
		Advisor: &stubAdvisor{advice: &Advice{Domain: "test", Version: "1.0"}},
		Adapter: &stubAdapter{output: "out"},
		Scorer:  &errorScorer{},
		Config:  SessionConfig{Mode: "one-shot"},
	}

	result, err := session.Run(context.Background(), "test", nil)
	// Session should not return an error (scorer failure is handled).
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Reward.Score != nil {
		t.Errorf("Score = %v, want nil", result.Reward.Score)
	}
	if result.Reward.Metadata == nil || result.Reward.Metadata["error"] != "scorer_failure" {
		t.Errorf("Metadata = %v, want error=scorer_failure", result.Reward.Metadata)
	}
}

type errorScorer struct{}

func (e *errorScorer) Score(_ string, _ map[string]any) (*Reward, error) {
	return nil, &simpleError{"scorer failed"}
}
