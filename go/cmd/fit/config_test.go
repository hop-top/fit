package main

import (
	"os"
	"path/filepath"
	"testing"

	"hop.top/kit/config"
)

func TestFitConfigDefaults(t *testing.T) {
	cfg := Config()
	if cfg.Addr != ":8080" {
		t.Errorf("Addr = %q, want :8080", cfg.Addr)
	}
	if cfg.TracesDir != "./traces" {
		t.Errorf("TracesDir = %q, want ./traces", cfg.TracesDir)
	}
	if cfg.Timeout != 5000 {
		t.Errorf("Timeout = %d, want 5000", cfg.Timeout)
	}
	if cfg.Model != "advisor-v1" {
		t.Errorf("Model = %q, want advisor-v1", cfg.Model)
	}
}

func TestFitConfigLoadMergesFile(t *testing.T) {
	// Write a temp config file and load it via config.Load directly.
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(cfgPath, []byte("addr: \":9090\"\nmodel: \"custom\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := FitConfig{
		Addr:      ":8080",
		TracesDir: "./traces",
		Timeout:   5000,
		Model:     "advisor-v1",
	}

	err := config.Load(&cfg, config.Options{
		UserConfigPath: cfgPath,
	})
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	// File values override defaults.
	if cfg.Addr != ":9090" {
		t.Errorf("Addr = %q, want :9090", cfg.Addr)
	}
	if cfg.Model != "custom" {
		t.Errorf("Model = %q, want custom", cfg.Model)
	}
	// Unset fields keep defaults.
	if cfg.TracesDir != "./traces" {
		t.Errorf("TracesDir = %q, want ./traces", cfg.TracesDir)
	}
	if cfg.Timeout != 5000 {
		t.Errorf("Timeout = %d, want 5000", cfg.Timeout)
	}
}

func TestFitConfigEnvOverride(t *testing.T) {
	cfg := FitConfig{
		Addr:      ":8080",
		TracesDir: "./traces",
		Timeout:   5000,
		Model:     "advisor-v1",
	}

	t.Setenv("FIT_ADDR", ":7070")
	t.Setenv("FIT_TRACES_DIR", "/tmp/traces")

	envOverride(&cfg)

	if cfg.Addr != ":7070" {
		t.Errorf("Addr = %q, want :7070", cfg.Addr)
	}
	if cfg.TracesDir != "/tmp/traces" {
		t.Errorf("TracesDir = %q, want /tmp/traces", cfg.TracesDir)
	}
	// Unset env vars keep previous values.
	if cfg.Timeout != 5000 {
		t.Errorf("Timeout = %d, want 5000", cfg.Timeout)
	}
}

func TestFitConfigLayerOrder(t *testing.T) {
	// File sets addr=:9090, env overrides to :7070.
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(cfgPath, []byte("addr: \":9090\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FIT_ADDR", ":7070")

	cfg := FitConfig{Addr: ":8080", TracesDir: "./traces", Timeout: 5000, Model: "advisor-v1"}
	err := config.Load(&cfg, config.Options{
		UserConfigPath: cfgPath,
		EnvOverride:    envOverride,
	})
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	// Env override wins over file.
	if cfg.Addr != ":7070" {
		t.Errorf("Addr = %q, want :7070 (env overrides file)", cfg.Addr)
	}
}
