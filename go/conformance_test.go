package fit

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

func fixturesDir() string {
	return filepath.Join("..", "spec", "fixtures")
}

func loadYAML(name string, dest any) error {
	data, err := os.ReadFile(filepath.Join(fixturesDir(), name))
	if err != nil {
		return err
	}
	return yaml.Unmarshal(data, dest)
}

func loadJSON(name string, dest any) error {
	data, err := os.ReadFile(filepath.Join(fixturesDir(), name))
	if err != nil {
		return err
	}
	return json.Unmarshal(data, dest)
}

// --- Advice conformance ---

func TestAdviceParseYAML(t *testing.T) {
	var a Advice
	if err := loadYAML("advice-v1.yaml", &a); err != nil {
		t.Fatalf("parse yaml: %v", err)
	}
	if a.Domain != "tax-compliance" {
		t.Errorf("domain = %q, want tax-compliance", a.Domain)
	}
	if a.Confidence < 0 || a.Confidence > 1 {
		t.Errorf("confidence %f out of [0,1]", a.Confidence)
	}
	if len(a.Constraints) != 3 {
		t.Errorf("constraints len = %d, want 3", len(a.Constraints))
	}
	if a.Version != "1.0" {
		t.Errorf("version = %q, want 1.0", a.Version)
	}
}

func TestAdviceParseJSON(t *testing.T) {
	var a Advice
	if err := loadJSON("advice-v1.json", &a); err != nil {
		t.Fatalf("parse json: %v", err)
	}
	if a.Domain != "tax-compliance" {
		t.Errorf("domain = %q, want tax-compliance", a.Domain)
	}
}

func TestAdviceYAMLJSONEquivalence(t *testing.T) {
	var yml, jsn Advice
	if err := loadYAML("advice-v1.yaml", &yml); err != nil {
		t.Fatal(err)
	}
	if err := loadJSON("advice-v1.json", &jsn); err != nil {
		t.Fatal(err)
	}
	if yml.Domain != jsn.Domain {
		t.Errorf("domain mismatch: yaml=%q json=%q", yml.Domain, jsn.Domain)
	}
	if yml.Confidence != jsn.Confidence {
		t.Errorf("confidence mismatch: yaml=%f json=%f", yml.Confidence, jsn.Confidence)
	}
}

func TestAdviceRoundTripYAML(t *testing.T) {
	var a Advice
	if err := loadYAML("advice-v1.yaml", &a); err != nil {
		t.Fatal(err)
	}
	out, err := yaml.Marshal(&a)
	if err != nil {
		t.Fatal(err)
	}
	var a2 Advice
	if err := yaml.Unmarshal(out, &a2); err != nil {
		t.Fatal(err)
	}
	if a2.Domain != a.Domain {
		t.Errorf("round-trip domain: %q != %q", a2.Domain, a.Domain)
	}
	if a2.Confidence != a.Confidence {
		t.Errorf("round-trip confidence: %f != %f", a2.Confidence, a.Confidence)
	}
}

// --- Reward conformance ---

func TestRewardParseJSON(t *testing.T) {
	var r Reward
	if err := loadJSON("reward-v1.json", &r); err != nil {
		t.Fatalf("parse json: %v", err)
	}
	if r.Score < 0 || r.Score > 1 {
		t.Errorf("score %f out of [0,1]", r.Score)
	}
	if r.Breakdown["accuracy"] < 0 || r.Breakdown["accuracy"] > 1 {
		t.Errorf("accuracy breakdown out of range")
	}
	if r.Metadata["scorer"] != "rubric-judge-v2" {
		t.Errorf("scorer metadata mismatch")
	}
}

func TestRewardRoundTripJSON(t *testing.T) {
	var r Reward
	if err := loadJSON("reward-v1.json", &r); err != nil {
		t.Fatal(err)
	}
	out, err := json.Marshal(&r)
	if err != nil {
		t.Fatal(err)
	}
	var r2 Reward
	if err := json.Unmarshal(out, &r2); err != nil {
		t.Fatal(err)
	}
	if r2.Score != r.Score {
		t.Errorf("round-trip score: %f != %f", r2.Score, r.Score)
	}
}

// --- Trace conformance ---

func TestTraceParseYAML(t *testing.T) {
	var tr Trace
	if err := loadYAML("trace-v1.yaml", &tr); err != nil {
		t.Fatalf("parse yaml: %v", err)
	}
	if tr.ID != "550e8400-e29b-41d4-a716-446655440000" {
		t.Errorf("id = %q", tr.ID)
	}
	if tr.SessionID != "sess_abc123" {
		t.Errorf("session_id = %q", tr.SessionID)
	}
	if tr.Advice == nil || tr.Advice.Domain != "tax-compliance" {
		t.Error("advice domain mismatch")
	}
	if tr.Reward == nil {
		t.Error("reward is nil")
	}
}

func TestTraceRequiredFields(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(fixturesDir(), "trace-v1.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := yaml.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"id", "session_id", "timestamp", "input", "advice", "frontier", "reward"} {
		if _, ok := m[key]; !ok {
			t.Errorf("missing required field: %s", key)
		}
	}
}

func TestTraceRoundTripYAML(t *testing.T) {
	var tr Trace
	if err := loadYAML("trace-v1.yaml", &tr); err != nil {
		t.Fatal(err)
	}
	out, err := yaml.Marshal(&tr)
	if err != nil {
		t.Fatal(err)
	}
	var tr2 Trace
	if err := yaml.Unmarshal(out, &tr2); err != nil {
		t.Fatal(err)
	}
	if tr2.ID != tr.ID {
		t.Errorf("round-trip id: %q != %q", tr2.ID, tr.ID)
	}
}

// --- Multi-turn session conformance ---

func TestSessionMultiParse(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(fixturesDir(), "session-multi.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := yaml.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if m["mode"] != "multi-turn" {
		t.Errorf("mode = %v, want multi-turn", m["mode"])
	}
	steps, ok := m["steps"].([]any)
	if !ok {
		t.Fatal("steps is not a list")
	}
	if len(steps) != 3 {
		t.Errorf("steps len = %d, want 3", len(steps))
	}
}
