package main

import (
	"github.com/spf13/cobra"
	"hop.top/kit/cli/completion"
)

// registerCompletions wires shell completions for flags with known value sets.
func registerCompletions(root *cobra.Command, cmds ...*cobra.Command) {
	// Root --format (persistent, registered by kit/output).
	completion.BindFlag(root, "format",
		completion.StaticValues("table", "json", "yaml"),
	)

	for _, cmd := range cmds {
		switch cmd.Name() {
		case "eval":
			completion.BindFlag(cmd, "dataset", completion.File(".json"))

		case "trace":
			for _, sub := range cmd.Commands() {
				switch sub.Name() {
				case "list":
					completion.BindFlag(sub, "dir", completion.Dir())
				case "show":
					completion.BindFlag(sub, "dir", completion.Dir())
					completion.BindFlag(sub, "format",
						completion.StaticValues("yaml", "json"),
					)
				}
			}
		}
	}
}
