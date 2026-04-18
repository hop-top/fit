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
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"hop.top/kit/bus"
	"hop.top/kit/cli"
	"hop.top/kit/log"
	"hop.top/kit/xdg"
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

	// Persistent flags for bus adapter selection.
	cfg := Config()
	root.Cmd.PersistentFlags().String("bus", cfg.Bus,
		`event bus adapter: "memory" or "sqlite"`)
	root.Cmd.PersistentFlags().String("bus-path", cfg.BusPath,
		"SQLite bus database path (default ~/.local/share/fit/events.db)")

	// Init app-wide event bus after flags are parsed.
	root.Cmd.PersistentPreRunE = func(cmd *cobra.Command, args []string) error {
		var err error
		appBus, err = initBus(cmd)
		return err
	}

	serve := serveCmd(root)
	eval := evalCmd(root)
	trace := traceCmd(root)

	root.Cmd.AddCommand(serve)
	root.Cmd.AddCommand(eval)
	root.Cmd.AddCommand(trace)

	registerCompletions(root.Cmd, serve, eval, trace)

	loadAliases(root)

	if err := root.Execute(context.Background()); err != nil {
		if appBus != nil {
			_ = appBus.Close(context.Background())
		}
		os.Exit(1)
	}
	if appBus != nil {
		_ = appBus.Close(context.Background())
	}
}

// initBus creates the event bus from config and flags.
func initBus(cmd *cobra.Command) (bus.Bus, error) {
	cfg := Config()
	adapter := cfg.Bus

	if cmd.Flags().Changed("bus") {
		adapter, _ = cmd.Flags().GetString("bus")
	}

	switch adapter {
	case "sqlite":
		path := cfg.BusPath
		if cmd.Flags().Changed("bus-path") {
			path, _ = cmd.Flags().GetString("bus-path")
		}
		if path == "" {
			path = defaultBusPath()
		}
		return newSQLiteBus(path)
	case "memory", "":
		return bus.New(), nil
	default:
		return nil, fmt.Errorf("unknown bus adapter %q (want memory or sqlite)", adapter)
	}
}

// newSQLiteBus creates a bus backed by SQLiteAdapter, ensuring the
// parent directory exists.
func newSQLiteBus(path string) (bus.Bus, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("creating bus dir: %w", err)
	}
	sa, err := bus.NewSQLiteAdapter(path)
	if err != nil {
		return nil, fmt.Errorf("sqlite bus at %s: %w", path, err)
	}
	return bus.New(bus.WithAdapter(sa)), nil
}

// defaultBusPath returns ~/.local/share/fit/events.db via XDG.
func defaultBusPath() string {
	dir, err := xdg.DataDir("fit")
	if err != nil || dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".local", "share", "fit")
	}
	return filepath.Join(dir, "events.db")
}
