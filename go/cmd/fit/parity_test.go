//go:build parity

// Package main contains CLI parity tests for fit.
//
// Currently Go-only. When TS and Python CLIs are implemented as
// standalone binaries, add them to parityBinaries() following the
// kit/cli/parity_test.go pattern:
//
//   - TS: build via esbuild in TestMain, add binary{lang:"ts", ...}
//   - Py: locate venv entry point, add binary{lang:"py", ...}
//
// Run with:
//
//	cd go && go test -tags parity ./cmd/fit/... -v -run TestParity -count=1
package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ── TestMain — build fit binary once ────────────────────────────────────────

var (
	parityBin string
	parityTmp string
)

func TestMain(m *testing.M) {
	root := findModuleRoot()
	if root == "" {
		fmt.Fprintln(os.Stderr, "parity: could not find go.mod")
		os.Exit(1)
	}

	tmp, err := os.MkdirTemp("", "fit-parity-*")
	if err != nil {
		fmt.Fprintln(os.Stderr, "parity: MkdirTemp:", err)
		os.Exit(1)
	}
	parityTmp = tmp

	parityBin = filepath.Join(tmp, "fit")
	if runtime.GOOS == "windows" {
		parityBin += ".exe"
	}

	build := exec.Command("go", "build", "-buildvcs=false",
		"-o", parityBin, filepath.Join(root, "cmd", "fit"))
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "parity: go build failed: %v\n%s", err, out)
		os.Exit(1)
	}

	code := m.Run()
	os.RemoveAll(tmp)
	os.Exit(code)
}

func findModuleRoot() string {
	dir, err := filepath.Abs(".")
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// ── Harness ─────────────────────────────────────────────────────────────────

type parityResult struct {
	stdout string
	stderr string
	code   int
}

func (r parityResult) combined() string { return r.stdout + r.stderr }

func run(args ...string) parityResult {
	cmd := exec.Command(parityBin, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	done := make(chan error, 1)
	_ = cmd.Start()
	go func() { done <- cmd.Wait() }()

	var runErr error
	select {
	case runErr = <-done:
	case <-time.After(10 * time.Second):
		_ = cmd.Process.Kill()
		runErr = cmd.Wait()
	}

	code := 0
	if runErr != nil {
		if exitErr, ok := runErr.(*exec.ExitError); ok {
			code = exitErr.ExitCode()
		} else {
			code = 1
		}
	}

	return parityResult{
		stdout: stdout.String(),
		stderr: stderr.String(),
		code:   code,
	}
}

// ── Contract tests ──────────────────────────────────────────────────────────

func TestParityHelp(t *testing.T) {
	r := run("--help")
	if r.code != 0 {
		t.Fatalf("--help exit %d, want 0\nstderr: %s", r.code, r.stderr)
	}
	out := r.combined()
	if !strings.Contains(strings.ToLower(out), "advisor") {
		t.Errorf("--help output must contain 'advisor'\ngot: %s", out)
	}
}

func TestParityVersion(t *testing.T) {
	r := run("--version")
	if r.code != 0 {
		t.Fatalf("--version exit %d, want 0\nstderr: %s", r.code, r.stderr)
	}
	out := r.combined()
	if !strings.Contains(out, "0.1.0") {
		t.Errorf("--version output must contain version string\ngot: %s", out)
	}
}

func TestParityServeHelp(t *testing.T) {
	r := run("serve", "--help")
	if r.code != 0 {
		t.Fatalf("serve --help exit %d, want 0\nstderr: %s", r.code, r.stderr)
	}
	out := r.combined()
	for _, flag := range []string{"--addr", "--model", "--timeout"} {
		if !strings.Contains(out, flag) {
			t.Errorf("serve --help must list %s\ngot: %s", flag, out)
		}
	}
}

func TestParityEvalHelp(t *testing.T) {
	r := run("eval", "--help")
	if r.code != 0 {
		t.Fatalf("eval --help exit %d, want 0\nstderr: %s", r.code, r.stderr)
	}
	out := r.combined()
	if !strings.Contains(out, "--dataset") {
		t.Errorf("eval --help must list --dataset flag\ngot: %s", out)
	}
}

func TestParityTraceHelp(t *testing.T) {
	r := run("trace", "--help")
	if r.code != 0 {
		t.Fatalf("trace --help exit %d, want 0\nstderr: %s", r.code, r.stderr)
	}
	out := r.combined()
	for _, sub := range []string{"list", "show"} {
		if !strings.Contains(out, sub) {
			t.Errorf("trace --help must list subcommand %q\ngot: %s", sub, out)
		}
	}
}

// TestParityUnknownCommand verifies unknown subcommands don't route to
// a domain handler. Cobra exits 0 and shows root help; other frameworks
// may exit non-zero. The contract: output must not be empty and must
// not contain domain-specific data (advisor, evaluation results, etc.).
func TestParityUnknownCommand(t *testing.T) {
	r := run("nonexistent-cmd-xyz")
	out := r.combined()
	if strings.TrimSpace(out) == "" {
		t.Errorf("nonexistent command must produce output")
	}
	// Must not route to a real command handler.
	lower := strings.ToLower(out)
	if strings.Contains(lower, "generating advice") ||
		strings.Contains(lower, "evaluation complete") {
		t.Errorf("nonexistent command must not route to domain handler\ngot: %s", out)
	}
}
