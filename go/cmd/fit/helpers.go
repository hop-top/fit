package main

import (
	"context"
	"encoding/json"
	"io"

	"gopkg.in/yaml.v3"
)

// jsonDecode reads JSON from r into v.
func jsonDecode(r io.Reader, v any) error {
	return json.NewDecoder(r).Decode(v)
}

// jsonEncode writes v as JSON to w.
func jsonEncode(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

// yamlUnmarshal parses YAML data into v.
func yamlUnmarshal(data []byte, v any) error {
	return yaml.Unmarshal(data, v)
}
