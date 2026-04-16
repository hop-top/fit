package fit

import (
	"context"
	"math"
	"time"

	"github.com/google/uuid"
)

// SessionState represents the session lifecycle state.
type SessionState string

const (
	StateInit     SessionState = "init"
	StateAdvise   SessionState = "advise"
	StateFrontier SessionState = "frontier"
	StateScore    SessionState = "score"
	StateTrace    SessionState = "trace"
	StateDone     SessionState = "done"
)

// SessionConfig holds session configuration.
type SessionConfig struct {
	Mode            string  `json:"mode" yaml:"mode"`
	MaxSteps        int     `json:"max_steps" yaml:"max_steps"`
	RewardThreshold float64 `json:"reward_threshold" yaml:"reward_threshold"`
}

// SessionResult is the output of a completed session.
type SessionResult struct {
	Output string
	Reward *Reward
	Trace  *Trace
}

// Session orchestrates the advisor → frontier → reward → trace cycle.
type Session struct {
	Advisor Advisor
	Adapter Adapter
	Scorer  RewardScorer
	Config  SessionConfig
	Tracer  *TraceWriter
}

// NewSession creates a new session with defaults.
func NewSession(advisor Advisor, adapter Adapter, scorer RewardScorer) *Session {
	return &Session{
		Advisor: advisor,
		Adapter: adapter,
		Scorer:  scorer,
		Config:  SessionConfig{Mode: "one-shot", MaxSteps: 10, RewardThreshold: 1.0},
	}
}

// Run executes a one-shot session cycle.
func (s *Session) Run(ctx context.Context, prompt string, contextMap map[string]any) (*SessionResult, error) {
	sessionID := uuid.New().String()
	input := map[string]any{
		"prompt":  prompt,
		"context": contextMap,
	}

	// Advise
	advice, err := s.Advisor.GenerateAdvice(ctx, input)
	if err != nil {
		advice = &Advice{Domain: "unknown", SteeringText: "", Confidence: 0}
	}

	// Frontier
	output, frontierMeta, err := s.Adapter.Call(ctx, prompt, advice)
	if err != nil {
		return nil, err
	}

	// Score
	reward, err := s.Scorer.Score(output, contextMap)
	if err != nil {
		reward = &Reward{Score: math.NaN(), Breakdown: map[string]float64{}}
	}

	// Trace
	trace := &Trace{
		ID:        uuid.New().String(),
		SessionID: sessionID,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Input:     input,
		Advice:    advice,
		Frontier:  frontierMeta,
		Reward:    reward,
	}

	return &SessionResult{Output: output, Reward: reward, Trace: trace}, nil
}
