package fit

// Trace represents an xrr-compatible session trace (trace-format-v1).
type Trace struct {
	ID        string         `json:"id" yaml:"id"`
	SessionID string         `json:"session_id" yaml:"session_id"`
	Timestamp string         `json:"timestamp" yaml:"timestamp"`
	Input     map[string]any `json:"input" yaml:"input"`
	Advice    *Advice        `json:"advice" yaml:"advice"`
	Frontier  map[string]any `json:"frontier" yaml:"frontier"`
	Reward    *Reward        `json:"reward" yaml:"reward"`
	Metadata  map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty"`
}

// TraceWriter writes xrr-compatible YAML cassettes.
type TraceWriter struct {
	outputDir string
}

// NewTraceWriter creates a trace writer for the given directory.
func NewTraceWriter(outputDir string) *TraceWriter {
	return &TraceWriter{outputDir: outputDir}
}
