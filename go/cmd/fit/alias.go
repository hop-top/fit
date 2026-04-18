package main

import (
	"path/filepath"

	"hop.top/kit/alias"
	"hop.top/kit/cli"
	"hop.top/kit/xdg"
)

// aliasFile returns the default alias store path:
// ~/.config/fit/aliases.yaml
func aliasFile() string {
	dir, _ := xdg.ConfigDir("fit")
	if dir == "" {
		return "aliases.yaml"
	}
	return filepath.Join(dir, "aliases.yaml")
}

// loadAliases creates an alias.Store, loads persisted aliases, registers
// them as cobra commands, and adds the "alias" management subcommand.
func loadAliases(root *cli.Root) {
	store := alias.NewStore(aliasFile())
	_ = store.Load() // missing file is fine

	// Register persisted aliases as cobra commands.
	_ = root.LoadAliasStore(store)

	// Add "alias" management command (list/add/remove).
	cmd := root.AliasCmd(store)
	cmd.GroupID = "management"
	root.Cmd.AddCommand(cmd)
}
