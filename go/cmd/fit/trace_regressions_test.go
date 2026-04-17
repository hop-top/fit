package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hop.top/fit"
	"hop.top/kit/cli"
)

// Regression: trace list must report error when a session directory is
// unreadable, not silently ignore it.
//
// Before fix: os.ReadDir errors for per-session directories were discarded
// via `steps, _ := os.ReadDir(...)`, hiding permission/IO problems and
// misreporting step counts. After fix: a warning is printed and the
// session is shown with "(steps: ?)".
func TestTraceListUnreadableSessionDir(t *testing.T) {
	// Set up traces dir with one good and one unreadable session.
	tracesDir := t.TempDir()
	goodDir := filepath.Join(tracesDir, "session-good")
	if err := os.MkdirAll(goodDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Put a step file in the good session so ReadDir returns 1 entry.
	if err := os.WriteFile(filepath.Join(goodDir, "step-001.yaml"), []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}

	badDir := filepath.Join(tracesDir, "session-bad")
	if err := os.MkdirAll(badDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Remove read/execute permissions to make ReadDir fail.
	if err := os.Chmod(badDir, 0o000); err != nil {
		t.Skipf("cannot revoke permissions (may need root): %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chmod(badDir, 0o755)
	})

	var stdout bytes.Buffer

	root := cli.New(cli.Config{Name: "fit", Version: "test", Short: "test"})
	listCmd := traceListCmd(root)
	listCmd.SetOut(&stdout)
	listCmd.SetArgs([]string{"--dir", tracesDir})

	err := listCmd.Execute()
	if err != nil {
		t.Fatalf("unexpected error from list command: %v", err)
	}

	out := stdout.String()

	// The good session must list normally with correct step count.
	if !strings.Contains(out, "session-good  (1 steps)") {
		t.Errorf("expected good session line, got %q", out)
	}

	// The bad session must appear with "(steps: ?)".
	if !strings.Contains(out, "session-bad  (steps: ?)") {
		t.Errorf("expected '(steps: ?)' for unreadable session, got %q", out)
	}
}

// Regression: trace show --format json must not fail when reward contains
// NaN or Inf floats. encoding/json rejects non-finite floats, so we
// sanitize them before encoding.
//
// Before fix: json.Encoder.Encode would error with "unsupported value"
// when Reward.Score was NaN or Inf. After fix: non-finite values are
// replaced with 0 and a metadata flag is set.
func TestTraceShowJSONSanitizesNaN(t *testing.T) {
	trace := fit.Trace{
		ID:        "t1",
		SessionID: "s1",
		Timestamp: "2025-01-01T00:00:00Z",
		Reward: &fit.Reward{
			Score:     fit.Float64Ptr(math.NaN()),
			Breakdown: map[string]float64{"accuracy": math.Inf(1), "safety": 0.9},
		},
	}
	sanitizeNonFinite(&trace)

	// Score should be 0, not NaN.
	if trace.Reward.Score == nil || *trace.Reward.Score != 0 {
		t.Errorf("Score = %v, want 0", trace.Reward.Score)
	}

	// Metadata must record the sanitization.
	if trace.Reward.Metadata["scorer_error"] != "non-finite score sanitized to 0" {
		t.Errorf("Metadata[scorer_error] = %v, want sanitization note", trace.Reward.Metadata["scorer_error"])
	}

	// Breakdown accuracy should be 0, not +Inf.
	if trace.Reward.Breakdown["accuracy"] != 0 {
		t.Errorf("Breakdown[accuracy] = %v, want 0", trace.Reward.Breakdown["accuracy"])
	}

	// Safety should be untouched.
	if trace.Reward.Breakdown["safety"] != 0.9 {
		t.Errorf("Breakdown[safety] = %v, want 0.9", trace.Reward.Breakdown["safety"])
	}

	// Full JSON encoding must succeed.
	var buf bytes.Buffer
	err := json.NewEncoder(&buf).Encode(trace)
	if err != nil {
		t.Fatalf("json.Encode failed: %v", err)
	}
}

func TestTraceShowJSONSanitizesInf(t *testing.T) {
	trace := fit.Trace{
		ID:        "t2",
		SessionID: "s2",
		Timestamp: "2025-01-01T00:00:00Z",
		Reward: &fit.Reward{
			Score:     fit.Float64Ptr(math.Inf(-1)),
			Breakdown: map[string]float64{},
		},
	}
	sanitizeNonFinite(&trace)

	if trace.Reward.Score == nil || *trace.Reward.Score != 0 {
		t.Errorf("Score = %v, want 0", trace.Reward.Score)
	}
	if trace.Reward.Metadata["scorer_error"] != "non-finite score sanitized to 0" {
		t.Errorf("Metadata[scorer_error] = %v, want sanitization note", trace.Reward.Metadata["scorer_error"])
	}

	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(trace); err != nil {
		t.Fatalf("json.Encode failed: %v", err)
	}
}

func TestTraceShowJSONNilRewardNoPanic(t *testing.T) {
	trace := fit.Trace{
		ID:        "t3",
		SessionID: "s3",
		Timestamp: "2025-01-01T00:00:00Z",
		Reward:    nil,
	}
	// Must not panic on nil reward.
	sanitizeNonFinite(&trace)

	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(trace); err != nil {
		t.Fatalf("json.Encode failed: %v", err)
	}
}

func TestTraceShowJSONFiniteUntouched(t *testing.T) {
	trace := fit.Trace{
		ID:        "t4",
		SessionID: "s4",
		Timestamp: "2025-01-01T00:00:00Z",
		Reward: &fit.Reward{
			Score:     fit.Float64Ptr(0.75),
			Breakdown: map[string]float64{"a": 0.5},
			Metadata:  map[string]any{"scorer": "test"},
		},
	}
	sanitizeNonFinite(&trace)

	if trace.Reward.Score == nil || *trace.Reward.Score != 0.75 {
		t.Errorf("Score = %v, want 0.75", trace.Reward.Score)
	}
	if trace.Reward.Breakdown["a"] != 0.5 {
		t.Errorf("Breakdown[a] = %v, want 0.5", trace.Reward.Breakdown["a"])
	}
	// Existing metadata must not be overwritten.
	if trace.Reward.Metadata["scorer"] != "test" {
		t.Errorf("Metadata[scorer] = %v, want 'test'", trace.Reward.Metadata["scorer"])
	}
	// No scorer_error key should be added for finite values.
	if _, ok := trace.Reward.Metadata["scorer_error"]; ok {
		t.Error("scorer_error should not be set for finite scores")
	}
}
