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

	"hop.top/kit/bus"
	"hop.top/kit/cli"
	"hop.top/kit/log"
)

// appBus is the application-wide event bus for trace lifecycle events.
var appBus bus.Bus

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

	// Init app-wide event bus (in-memory by default).
	appBus = bus.New()

	serve := serveCmd(root)
	eval := evalCmd(root)
	trace := traceCmd(root)

	root.Cmd.AddCommand(serve)
	root.Cmd.AddCommand(eval)
	root.Cmd.AddCommand(trace)

	registerCompletions(root.Cmd, serve, eval, trace)

	loadAliases(root)

	if err := root.Execute(context.Background()); err != nil {
		_ = appBus.Close(context.Background())
		os.Exit(1)
	}
	_ = appBus.Close(context.Background())
}
