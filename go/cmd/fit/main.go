// Package main provides the fit CLI — a tool for training small advisor
// models to steer black-box LLMs.
//
// Commands:
//
//	fit serve  — start advisor HTTP server
//	fit eval   — run evaluation against a dataset
//	fit trace  — inspect/convert trace files
package main

import (
	"context"
	"os"

	"hop.top/kit/cli"
	"hop.top/kit/log"
)

func main() {
	root := cli.New(cli.Config{
		Name:    "fit",
		Version: "0.1.0",
		Short:   "Train small advisor models to steer black-box LLMs",
	})

	// Load layered config; errors are non-fatal (defaults apply).
	if err := LoadConfig(); err != nil {
		logger := log.New(root.Viper)
		logger.Warn("config load failed, using defaults", "err", err)
	}

	root.Cmd.AddCommand(serveCmd(root))
	root.Cmd.AddCommand(evalCmd(root))
	root.Cmd.AddCommand(traceCmd(root))

	if err := root.Execute(context.Background()); err != nil {
		os.Exit(1)
	}
}
