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
)

func main() {
	root := cli.New(cli.Config{
		Name:    "fit",
		Version: "0.1.0",
		Short:   "Train small advisor models to steer black-box LLMs",
	})

	root.Cmd.AddCommand(serveCmd())
	root.Cmd.AddCommand(evalCmd())
	root.Cmd.AddCommand(traceCmd())

	if err := root.Execute(context.Background()); err != nil {
		os.Exit(1)
	}
}
