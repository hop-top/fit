package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"hop.top/fit"
)

func evalCmd() *cobra.Command {
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

			var totalScore float64
			var count int

			for _, tc := range cases {
				session := fit.NewSession(advisor, adapter, scorer)
				result, err := session.Run(cmd.Context(), tc.Prompt, tc.Context)
				if err != nil {
					fmt.Fprintf(cmd.OutOrStdout(), "FAIL: %s: %v\n", tc.Prompt, err)
					continue
				}
				totalScore += result.Reward.Score
				count++

				fmt.Fprintf(cmd.OutOrStdout(),
					"score=%.2f domain=%s prompt=%q\n",
					result.Reward.Score,
					result.Trace.Advice.Domain,
					tc.Prompt,
				)
			}

			if count > 0 {
				fmt.Fprintf(cmd.OutOrStdout(),
					"\n--- summary ---\ncases: %d  avg_score: %.3f\n",
					count, totalScore/float64(count),
				)
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
		Score:     0.5,
		Breakdown: map[string]float64{"accuracy": 0.5, "relevance": 0.5},
	}, nil
}
