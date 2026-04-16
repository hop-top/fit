# Advice Format v1

Language-agnostic spec for advisor model output.

## Schema

```yaml
# Required fields
domain: string          # Target domain (tax, support, coding, etc.)
steering_text: string   # Human-readable steering instruction
confidence: float       # 0.0–1.0, advisor's confidence in advice

# Optional fields
constraints:
  - string              # Hard constraints to enforce
metadata:
  key: value            # Arbitrary k/v pairs

# Versioning
version: "1.0"
```

## Example: Tax reasoning

```yaml
version: "1.0"
domain: tax-compliance
steering_text: |
  Double-check filing status assumptions before calculating.
  Prioritize rules explicitly stated in supplied instructions.
  If ambiguity remains, surface it instead of guessing.
  Show calculation steps only when confidence exceeds 0.8.
confidence: 0.87
constraints:
  - never fabricate tax code sections
  - cite specific IRS publication numbers
  - flag state-specific rules
metadata:
  model: advisor-tax-v2.3
  latency_ms: 42
```

## Example: Coding agent

```yaml
version: "1.0"
domain: code-agent
steering_text: |
  Search symbol references before editing.
  Prefer minimal patch strategy.
  Run tests after each edit cycle.
  Avoid broad file listing — use grep directly.
confidence: 0.72
constraints:
  - do not modify files outside src/
  - preserve existing test cases
metadata:
  model: advisor-swe-v1.1
  step: 3
```

## Example: Support copilot

```yaml
version: "1.0"
domain: enterprise-support
steering_text: |
  Customer is on enterprise plan — avoid speculative language.
  Acknowledge issue before proposing solution.
  Mention SLA boundaries explicitly.
  Prioritize steps that do not require admin privileges.
confidence: 0.94
constraints:
  - no external links without verification
  - respect data residency rules
metadata:
  model: advisor-support-v3.0
  tier: enterprise
  account_id: acct_abc123
```

## Encoding rules

- YAML serialization (JSON is valid YAML subset)
- `steering_text` may be multiline; preserved as-is
- `confidence` must be in [0.0, 1.0]
- `constraints` is optional; defaults to empty list
- `metadata` is optional; defaults to empty map
- All ports must parse both YAML and JSON forms
