package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"hop.top/fit"
	"hop.top/kit/cli"
	"hop.top/kit/output"
)

// evalRow is a single evaluation result for structured output.
type evalRow struct {
	Score  float64 `json:"score" yaml:"score" table:"Score"`
	Domain string  `json:"domain" yaml:"domain" table:"Domain"`
	Prompt string  `json:"prompt" yaml:"prompt" table:"Prompt"`
}

// evalSummary is the aggregated evaluation summary.
type evalSummary struct {
	Cases    int     `json:"cases" yaml:"cases" table:"Cases"`
	AvgScore float64 `json:"avg_score" yaml:"avg_score" table:"Avg Score"`
}

// evalOutput wraps results + summary for single-document structured output.
type evalOutput struct {
	Results []evalRow   `json:"results" yaml:"results"`
	Summary evalSummary `json:"summary" yaml:"summary"`
}

func evalCmd(root *cli.Root) *cobra.Command {
	var (
		datasetPath string
	)

	cmd := &cobra.Command{
		Use:   "eval",
		Short: "Run evaluation against a dataset",
		Long: `Evaluate advisor performance against a dataset of prompts.

Loads test cases from the dataset file, runs each through an advisor session,
and prints scores to stdout.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if datasetPath == "" {
				return fmt.Errorf("--dataset is required")
			}

			cases, err := loadDataset(datasetPath)
			if err != nil {
				return fmt.Errorf("load dataset: %w", err)
			}

			advisor := &stubAdvisor{model: "eval-advisor"}
			scorer := &stubScorer{}
			adapter := &evalAdapter{}

			format := root.Viper.GetString("format")
			var totalScore float64
			var rows []evalRow

			for _, tc := range cases {
				session := fit.NewSession(advisor, adapter, scorer)
				result, err := session.Run(cmd.Context(), tc.Prompt, tc.Context)
				if err != nil {
					fmt.Fprintf(cmd.ErrOrStderr(), "FAIL: %s: %v\n", tc.Prompt, err)
					continue
				}
				scoreVal := 0.0
				if result.Reward.Score != nil {
					scoreVal = *result.Reward.Score
					totalScore += scoreVal
				}
				rows = append(rows, evalRow{
					Score:  scoreVal,
					Domain: result.Trace.Advice.Domain,
					Prompt: tc.Prompt,
				})
			}

			summary := evalSummary{}
			if len(rows) > 0 {
				summary = evalSummary{
					Cases:    len(rows),
					AvgScore: totalScore / float64(len(rows)),
				}
			}

			// For structured formats, emit a single document.
			if format != output.Table {
				return output.Render(cmd.OutOrStdout(), format, evalOutput{
					Results: rows,
					Summary: summary,
				})
			}

			// Table format: render rows then summary separately.
			if err := output.Render(cmd.OutOrStdout(), format, rows); err != nil {
				return err
			}
			if len(rows) > 0 {
				fmt.Fprintln(cmd.OutOrStdout())
				return output.Render(cmd.OutOrStdout(), format, summary)
			}
			return nil
		},
	}

	cmd.Flags().StringVarP(&datasetPath, "dataset", "d", "", "path to dataset JSON file (required)")

	_ = cmd.MarkFlagRequired("dataset")

	return cmd
}

// evalCase is a single test case in the dataset.
type evalCase struct {
	Prompt  string         `json:"prompt" yaml:"prompt"`
	Context map[string]any `json:"context,omitempty" yaml:"context,omitempty"`
}

// evalAdapter is a minimal adapter for evaluation runs.
type evalAdapter struct{}

func (e *evalAdapter) Call(_ context.Context, prompt string, advice *fit.Advice) (string, map[string]any, error) {
	const output = "eval output"
	return output, map[string]any{
		"model":    "eval",
		"provider": "local",
		"output":   output,
		"usage": map[string]int{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
		},
	}, nil
}

func loadDataset(path string) ([]evalCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	ext := filepath.Ext(path)
	switch ext {
	case ".json":
		var cases []evalCase
		if err := json.Unmarshal(data, &cases); err != nil {
			return nil, err
		}
		return cases, nil
	default:
		return nil, fmt.Errorf("unsupported dataset format: %s (use .json)", ext)
	}
}

// stubScorer is a placeholder scorer for evaluation runs.
type stubScorer struct{}

func (s *stubScorer) Score(_ string, _ map[string]any) (*fit.Reward, error) {
	return &fit.Reward{
		Score:     fit.Float64Ptr(0.5),
		Breakdown: map[string]float64{"accuracy": 0.5, "relevance": 0.5},
	}, nil
}
