package main

import (
	"testing"

	"github.com/spf13/cobra"
	"hop.top/kit/cli"
)

func TestGuardCmd_ReadNoFlag(t *testing.T) {
	cmd := &cobra.Command{Use: "noop"}
	guardCmd(cmd, cli.SafetyRead)

	if cmd.Flags().Lookup("force") != nil {
		t.Fatal("SafetyRead should not add --force flag")
	}
	if cmd.PersistentPreRunE != nil {
		t.Fatal("SafetyRead should not set PersistentPreRunE")
	}
}

func TestGuardCmd_CautionAddsFlag(t *testing.T) {
	cmd := &cobra.Command{Use: "noop"}
	guardCmd(cmd, cli.SafetyCaution)

	f := cmd.Flags().Lookup("force")
	if f == nil {
		t.Fatal("SafetyCaution should add --force flag")
	}
	if f.Shorthand != "f" {
		t.Fatalf("expected shorthand 'f', got %q", f.Shorthand)
	}
}

func TestGuardCmd_CautionBlocksNonTTY(t *testing.T) {
	cmd := &cobra.Command{Use: "noop", RunE: func(*cobra.Command, []string) error { return nil }}
	guardCmd(cmd, cli.SafetyCaution)

	// Test env stdin is not a TTY, so guard should block.
	err := cmd.PersistentPreRunE(cmd, nil)
	if err == nil {
		t.Fatal("caution should block in non-TTY without --force")
	}
}

func TestGuardCmd_CautionForceBypass(t *testing.T) {
	cmd := &cobra.Command{Use: "noop", RunE: func(*cobra.Command, []string) error { return nil }}
	guardCmd(cmd, cli.SafetyCaution)

	_ = cmd.Flags().Set("force", "true")
	if err := cmd.PersistentPreRunE(cmd, nil); err != nil {
		t.Fatalf("caution with --force should pass, got: %v", err)
	}
}

func TestGuardCmd_ChainsExistingPreRunE(t *testing.T) {
	called := false
	cmd := &cobra.Command{
		Use: "noop",
		PersistentPreRunE: func(*cobra.Command, []string) error {
			called = true
			return nil
		},
	}
	guardCmd(cmd, cli.SafetyCaution)

	_ = cmd.Flags().Set("force", "true")
	if err := cmd.PersistentPreRunE(cmd, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("existing PersistentPreRunE should be chained")
	}
}
