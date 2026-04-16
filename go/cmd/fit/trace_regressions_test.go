package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

	var stdout, stderr bytes.Buffer

	listCmd := traceListCmd()
	listCmd.SetOut(&stdout)
	listCmd.SetErr(&stderr)
	listCmd.SetArgs([]string{"--dir", tracesDir})

	err := listCmd.Execute()
	if err != nil {
		t.Fatalf("unexpected error from list command: %v", err)
	}

	combined := stdout.String() + stderr.String()

	// The good session must list normally with correct step count.
	if !strings.Contains(combined, "session-good  (1 steps)") {
		t.Errorf("expected good session line, got stdout=%q stderr=%q", stdout.String(), stderr.String())
	}

	// The bad session must appear with "(steps: ?)".
	if !strings.Contains(combined, "session-bad  (steps: ?)") {
		t.Errorf("expected '(steps: ?)' for unreadable session, got stdout=%q stderr=%q", stdout.String(), stderr.String())
	}

	// A warning about the unreadable session must be emitted.
	if !strings.Contains(combined, "warning") || !strings.Contains(combined, "session-bad") {
		t.Errorf("expected warning about session-bad, got stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}
