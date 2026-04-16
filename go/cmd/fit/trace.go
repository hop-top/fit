package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"hop.top/fit"
)

func traceCmd() *cobra.Command {
	var (
		sessionID string
		step      int
		format    string
	)

	cmd := &cobra.Command{
		Use:   "trace",
		Short: "Inspect/convert trace files",
		Long: `Inspect and convert xrr-compatible trace cassettes.

Supports listing sessions, reading specific steps, and converting
between YAML and JSON formats.`,
	}

	// Subcommands
	cmd.AddCommand(traceListCmd())
	cmd.AddCommand(traceShowCmd())

	return cmd
}

func traceListCmd() *cobra.Command {
	var tracesDir string

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List trace sessions",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			entries, err := os.ReadDir(tracesDir)
			if err != nil {
				if os.IsNotExist(err) {
					fmt.Fprintln(cmd.OutOrStdout(), "no traces directory")
					return nil
				}
				return err
			}

			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				// Count steps in this session
				steps, _ := os.ReadDir(filepath.Join(tracesDir, e.Name()))
				fmt.Fprintf(cmd.OutOrStdout(), "%s  (%d steps)\n", e.Name(), len(steps))
			}

			return nil
		},
	}

	cmd.Flags().StringVarP(&tracesDir, "dir", "d", "./traces", "traces directory")

	return cmd
}

func traceShowCmd() *cobra.Command {
	var (
		tracesDir string
		format    string
	)

	cmd := &cobra.Command{
		Use:   "show <session-id> [step]",
		Short: "Show a specific trace step",
		Args:  cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			sessionID := args[0]
			step := 1
			if len(args) > 1 {
				_, err := fmt.Sscanf(args[1], "%d", &step)
				if err != nil {
					return fmt.Errorf("invalid step number: %s", args[1])
				}
			}

			stepFile := filepath.Join(
				tracesDir, sessionID,
				fmt.Sprintf("step-%03d.yaml", step),
			)

			data, err := os.ReadFile(stepFile)
			if err != nil {
				return fmt.Errorf("read trace: %w", err)
			}

			if format == "json" {
				// Parse YAML and re-emit as JSON
				var trace fit.Trace
				if err := yamlUnmarshal(data, &trace); err != nil {
					return fmt.Errorf("parse trace: %w", err)
				}
				enc := json.NewEncoder(cmd.OutOrStdout())
				enc.SetIndent("", "  ")
				return enc.Encode(trace)
			}

			// Default: emit raw YAML
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}

	cmd.Flags().StringVarP(&tracesDir, "dir", "d", "./traces", "traces directory")
	cmd.Flags().StringVarP(&format, "format", "f", "yaml", "output format (yaml|json)")

	return cmd
}
