# Rust: Adding fit as a dependency

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
fit = "0.1"
```

Or via cargo add:

```bash
cargo add fit
```

## Basic usage

```rust
use std::collections::HashMap;

use fit::{
    Advisor, Adapter, RewardScorer,
    RemoteAdvisor, Session, SessionConfig, SessionMode,
    AnthropicAdapter, CompositeScorer,
};

#[tokio::main]
async fn main() -> Result<(), fit::FitError> {
    let advisor = RemoteAdvisor::from_endpoint("http://localhost:8080");
    let adapter = AnthropicAdapter::new(&std::env::var("ANTHROPIC_API_KEY").unwrap());
    let scorer = CompositeScorer::from_dimensions(&["accuracy", "relevance", "safety"]);

    let context_map: HashMap<&str, &str> = HashMap::from([
        ("jurisdiction", "US"),
        ("filing_status", "single"),
    ]);

    let session = Session::new(advisor, adapter, scorer);
    let result = session.run(
        "What is the standard deduction?",
        context_map,
    ).await?;

    println!("Output: {}", result.output);
    println!("Reward: {:.2}", result.reward.score);

    Ok(())
}
```

## Adapter configuration

Three adapters included:

```rust
use fit::{AnthropicAdapter, OpenAIAdapter, OllamaAdapter};

// Anthropic (Claude)
let anthropic = AnthropicAdapter::new(&api_key)
    .with_model("claude-sonnet-4-6");

// OpenAI (GPT)
let openai = OpenAIAdapter::new(&api_key)
    .with_model("gpt-5");

// Ollama (local)
let ollama = OllamaAdapter::new()
    .with_model("llama3")
    .with_base_url("http://localhost:11434");
```

Custom adapters implement the `Adapter` trait:

```rust
use fit::{Adapter, Advice, FitError};
use std::collections::HashMap;

struct MyAdapter;

#[async_trait::async_trait]
impl Adapter for MyAdapter {
    async fn call(
        &self,
        prompt: &str,
        advice: &Advice,
    ) -> Result<(String, HashMap<String, serde_yaml::Value>), FitError> {
        let system = format!("[Advisor Guidance]\n{}", advice.steering_text);
        let output = call_llm(&system, prompt).await;
        let mut meta = HashMap::new();
        meta.insert("model".into(), serde_yaml::Value::String("my-model".into()));
        Ok((output, meta))
    }
}
```

## Custom reward functions

Implement `RewardScorer`:

```rust
use fit::{Reward, RewardScorer, FitError};
use std::collections::HashMap;

struct TaxAccuracyScorer;

impl RewardScorer for TaxAccuracyScorer {
    fn score(
        &self,
        output: &str,
        context: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let accuracy = compute_accuracy(output, context);
        let mut breakdown = HashMap::new();
        breakdown.insert("accuracy".to_string(), accuracy);
        breakdown.insert("safety".to_string(), 1.0);

        Ok(Reward::new(accuracy, breakdown))
    }
}
```

Combine scorers with weights:

```rust
let scorer = CompositeScorer::new(
    vec![Box::new(TaxAccuracyScorer), Box::new(SafetyScorer)],
    vec![0.7, 0.3],
);
```

## Trace handling

```rust
use fit::TraceWriter;

let writer = TraceWriter::new("./traces");
writer.write(&result.trace, 1)?; // step number

let sessions = writer.list_sessions()?;
let trace = writer.read("sess_abc123", 1)?;
```

Traces are xrr-compatible YAML cassettes:

```text
traces/
  {session_id}/
    step-001.yaml
```

## Multi-turn sessions

```rust
let config = SessionConfig {
    mode: SessionMode::MultiTurn,
    max_steps: 10,
    reward_threshold: 0.95,
};
let session = Session::new_with_config(advisor, adapter, scorer, config);
```

## Axum integration

```rust
use axum::{Json, Router, routing::post};
use serde::Deserialize;

#[derive(Deserialize)]
struct AskRequest {
    prompt: String,
    #[serde(default)]
    context: HashMap<String, serde_yaml::Value>,
}

async fn handle_ask(
    State(session): State<Session<...>>,
    Json(req): Json<AskRequest>,
) -> Json<serde_json::Value> {
    let result = session.run(&req.prompt, &req.context).await.unwrap();
    Json(serde_json::json!({
        "output": result.output,
        "reward": { "score": result.reward.score }
    }))
}
```
