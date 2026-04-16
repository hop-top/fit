use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Advisor output (advice-format-v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Advice {
    pub domain: String,
    pub steering_text: String,
    pub confidence: f64,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_yaml::Value>,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "1.0".to_string()
}

/// Advisor trait — generates per-instance steering advice.
#[async_trait::async_trait]
pub trait Advisor: Send + Sync {
    async fn generate_advice(
        &self,
        context: HashMap<String, serde_yaml::Value>,
    ) -> Result<Advice, Box<dyn std::error::Error + Send + Sync>>;

    fn model_id(&self) -> String;
}
