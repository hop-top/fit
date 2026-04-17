package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"hop.top/fit"
	"hop.top/kit/cli"
	"hop.top/kit/log"
)

func traceCmd(root *cli.Root) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "trace",
		Short: "Inspect/convert trace files",
		Long: `Inspect and convert xrr-compatible trace cassettes.

Supports listing sessions, reading specific steps, and converting
between YAML and JSON formats.`,
	}

	// Subcommands
	cmd.AddCommand(traceListCmd(root))
	cmd.AddCommand(traceShowCmd())

	return cmd
}

func traceListCmd(root *cli.Root) *cobra.Command {
	cfg := Config()
	var tracesDir string
	logger := log.New(root.Viper)

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
				steps, err := os.ReadDir(filepath.Join(tracesDir, e.Name()))
				if err != nil {
					logger.Warn("cannot read session", "session", e.Name(), "err", err)
					fmt.Fprintf(cmd.OutOrStdout(), "%s  (steps: ?)\n", e.Name())
					continue
				}
				fmt.Fprintf(cmd.OutOrStdout(), "%s  (%d steps)\n", e.Name(), len(steps))
			}

			return nil
		},
	}

	cmd.Flags().StringVarP(&tracesDir, "dir", "d", cfg.TracesDir, "traces directory")

	return cmd
}

func traceShowCmd() *cobra.Command {
	cfg := Config()
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
				sanitizeNonFinite(&trace)
				enc := json.NewEncoder(cmd.OutOrStdout())
				enc.SetIndent("", "  ")
				return enc.Encode(trace)
			}

			// Default: emit raw YAML
			_, err = cmd.OutOrStdout().Write(data)
			return err
		},
	}

	cmd.Flags().StringVarP(&tracesDir, "dir", "d", cfg.TracesDir, "traces directory")
	cmd.Flags().StringVarP(&format, "format", "f", "yaml", "output format (yaml|json)")

	return cmd
}

// sanitizeNonFinite replaces NaN/Inf floats in the trace with JSON-safe
// values so encoding/json does not error. Nil score is already JSON-safe.
// Non-nil score with NaN/Inf is set to 0 with a metadata flag.
// Breakdown NaN/Inf values are set to 0.
func sanitizeNonFinite(t *fit.Trace) {
	if t.Reward == nil {
		return
	}
	r := t.Reward
	if r.Score != nil && (math.IsNaN(*r.Score) || math.IsInf(*r.Score, 0)) {
		r.Score = fit.Float64Ptr(0)
		if r.Metadata == nil {
			r.Metadata = make(map[string]any)
		}
		r.Metadata["scorer_error"] = "non-finite score sanitized to 0"
	}
	for k, v := range r.Breakdown {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			r.Breakdown[k] = 0
		}
	}
}
