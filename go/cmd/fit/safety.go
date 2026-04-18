package main

import (
	"github.com/spf13/cobra"
	"hop.top/kit/cli"
)

// guardCmd attaches a --force flag and wires SafetyGuard as a PersistentPreRunE
// for the given command at the specified safety level.
//
// Usage:
//
//	cmd := &cobra.Command{...}
//	guardCmd(cmd, cli.SafetyCaution)
//
// Current commands and their levels:
//
//	fit serve         → SafetyRead  (no guard needed)
//	fit eval          → SafetyRead  (no guard needed)
//	fit trace list    → SafetyRead  (no guard needed)
//	fit trace show    → SafetyRead  (no guard needed)
//
// Wire SafetyCaution on any future command that writes/overwrites files
// (e.g. trace convert --overwrite, train, export).
// Wire SafetyDangerous for irreversible operations (e.g. purge, reset).
func guardCmd(cmd *cobra.Command, level cli.SafetyLevel) {
	if level == cli.SafetyRead {
		return
	}

	cmd.Flags().BoolP("force", "f", false, "skip safety confirmation")

	existing := cmd.PersistentPreRunE
	cmd.PersistentPreRunE = func(cmd *cobra.Command, args []string) error {
		if err := cli.SafetyGuard(cmd, level); err != nil {
			return err
		}
		if existing != nil {
			return existing(cmd, args)
		}
		return nil
	}
}
