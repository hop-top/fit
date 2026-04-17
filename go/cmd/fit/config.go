package main

import (
	"os"
	"path/filepath"
	"strconv"
	"time"

	"hop.top/kit/config"
	"hop.top/kit/xdg"
)

// FitConfig holds runtime configuration for the fit CLI.
type FitConfig struct {
	Addr      string `yaml:"addr"`
	TracesDir string `yaml:"traces-dir"`
	Timeout   int    `yaml:"timeout"`
	Model     string `yaml:"model"`
}

// globalConfig is loaded once at startup and shared across subcommands.
var globalConfig = FitConfig{
	Addr:      ":8080",
	TracesDir: "./traces",
	Timeout:   5000,
	Model:     "advisor-v1",
}

// Config returns the loaded configuration.
func Config() FitConfig { return globalConfig }

// LoadConfig loads layered YAML configuration into globalConfig:
// ~/.config/fit/config.yaml → .fit.yaml → env overrides.
func LoadConfig() error {
	userDir, _ := xdg.ConfigDir("fit")
	var userPath string
	if userDir != "" {
		userPath = filepath.Join(userDir, "config.yaml")
	}

	return config.Load(&globalConfig, config.Options{
		UserConfigPath:    userPath,
		ProjectConfigPath: ".fit.yaml",
		EnvOverride:       envOverride,
	})
}

func envOverride(dst any) {
	cfg, ok := dst.(*FitConfig)
	if !ok {
		return
	}
	if v := os.Getenv("FIT_ADDR"); v != "" {
		cfg.Addr = v
	}
	if v := os.Getenv("FIT_TRACES_DIR"); v != "" {
		cfg.TracesDir = v
	}
	if v := os.Getenv("FIT_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.Timeout = n
		}
	}
	if v := os.Getenv("FIT_MODEL"); v != "" {
		cfg.Model = v
	}
}

// TimeoutDuration returns the timeout as a time.Duration.
func (c FitConfig) TimeoutDuration() time.Duration {
	return time.Duration(c.Timeout) * time.Millisecond
}
