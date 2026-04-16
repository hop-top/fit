package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"hop.top/fit"
)

// stubAdvisorForTest returns fixed advice for handler tests.
type stubAdvisorForTest struct{}

func (a *stubAdvisorForTest) GenerateAdvice(_ context.Context, input map[string]any) (*fit.Advice, error) {
	return &fit.Advice{
		Domain:       "test",
		SteeringText: "test advice",
		Confidence:   0.9,
		Version:      "1.0",
		Metadata:     input,
	}, nil
}

func (a *stubAdvisorForTest) ModelID() string { return "test-advisor" }

// Regression: handleAdvise must work with advisor only (no scorer param).
//
// Before fix: handleAdvise accepted a scorer fit.RewardScorer parameter
// that was never used in the handler body. The fix removes the dead
// scorer parameter entirely. This test verifies the /advise endpoint
// still returns correct advice JSON after the removal.
func TestHandleAdviseWithoutScorer(t *testing.T) {
	advisor := &stubAdvisorForTest{}
	handler := handleAdvise(advisor, 5000)

	body := `{"prompt":"hello","context":{"key":"value"}}`
	req := httptest.NewRequest(http.MethodPost, "/advise", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var advice fit.Advice
	if err := json.Unmarshal(rec.Body.Bytes(), &advice); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if advice.Domain != "test" {
		t.Errorf("expected domain=test, got %s", advice.Domain)
	}
	if advice.SteeringText != "test advice" {
		t.Errorf("expected steering_text='test advice', got %s", advice.SteeringText)
	}
	if advice.Confidence != 0.9 {
		t.Errorf("expected confidence=0.9, got %f", advice.Confidence)
	}
	if advice.Version != "1.0" {
		t.Errorf("expected version=1.0, got %s", advice.Version)
	}
}

// Regression: handleAdvise returns 400 for invalid JSON.
func TestHandleAdviseInvalidJSON(t *testing.T) {
	advisor := &stubAdvisorForTest{}
	handler := handleAdvise(advisor, 5000)

	req := httptest.NewRequest(http.MethodPost, "/advise", strings.NewReader("not json"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}
