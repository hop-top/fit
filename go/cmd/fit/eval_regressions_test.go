package main

import (
	"context"
	"testing"

	"hop.top/fit"
)

// Regression: evalAdapter.Call() must set meta["output"] to the same
// value as the returned output string, not to the prompt.
//
// Before fix: Call() returned ("eval output", ...) but set
// meta["output"] = prompt. The metadata output must be consistent
// with the returned output.
func TestEvalAdapterMetaOutputMatchesReturn(t *testing.T) {
	adapter := &evalAdapter{}

	output, meta, err := adapter.Call(context.Background(), "test prompt", &fit.Advice{})
	if err != nil {
		t.Fatalf("Call() returned error: %v", err)
	}

	metaOutput, ok := meta["output"]
	if !ok {
		t.Fatal("meta missing 'output' key")
	}

	metaOutputStr, ok := metaOutput.(string)
	if !ok {
		t.Fatalf("meta['output'] is %T, want string", metaOutput)
	}

	if metaOutputStr != output {
		t.Errorf("meta['output'] = %q, want %q (returned output)", metaOutputStr, output)
	}
}
