use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::FitError;

/// Advisor output (advice-format-v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Advice {
    pub domain: String,
    pub steering_text: String,
    pub confidence: f64,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_yaml::Value>,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "1.0".to_string()
}

impl Advice {
    /// Create advice with defaults for optional fields.
    pub fn new(domain: &str, steering_text: &str, confidence: f64) -> Self {
        Self {
            domain: domain.to_string(),
            steering_text: steering_text.to_string(),
            confidence,
            constraints: vec![],
            metadata: BTreeMap::new(),
            version: "1.0".to_string(),
        }
    }

    /// Parse advice from YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, FitError> {
        serde_yaml::from_str(yaml).map_err(FitError::Yaml)
    }

    /// Parse advice from JSON string.
    pub fn from_json(json: &str) -> Result<Self, FitError> {
        serde_json::from_str(json).map_err(FitError::Json)
    }

    /// Serialize to YAML string.
    pub fn to_yaml(&self) -> Result<String, FitError> {
        serde_yaml::to_string(self).map_err(FitError::Yaml)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, FitError> {
        serde_json::to_string_pretty(self).map_err(FitError::Json)
    }
}

/// Advisor trait -- generates per-instance steering advice.
#[async_trait::async_trait]
pub trait Advisor: Send + Sync {
    async fn generate_advice(
        &self,
        context: BTreeMap<String, serde_yaml::Value>,
    ) -> Result<Advice, FitError>;

    fn model_id(&self) -> String;
}

/// HTTP-backed advisor that calls a remote /advise endpoint.
pub struct RemoteAdvisor {
    endpoint: String,
    timeout_ms: u64,
    client: reqwest::Client,
}

impl RemoteAdvisor {
    pub fn new(endpoint: &str, timeout_ms: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .build()
            .unwrap_or_default();
        Self {
            endpoint: endpoint.to_string(),
            timeout_ms,
            client,
        }
    }

    pub fn from_endpoint(url: &str) -> Self {
        Self::new(url, 5000)
    }
}

#[async_trait::async_trait]
impl Advisor for RemoteAdvisor {
    async fn generate_advice(
        &self,
        context: BTreeMap<String, serde_yaml::Value>,
    ) -> Result<Advice, FitError> {
        let resp = self
            .client
            .post(format!("{}/advise", self.endpoint))
            .json(&context)
            .send()
            .await
            .map_err(|e| FitError::Http(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(FitError::Http(format!(
                "advisor returned {}",
                resp.status()
            )));
        }

        let advice: Advice = resp
            .json()
            .await
            .map_err(|e| FitError::Http(e.to_string()))?;

        Ok(advice)
    }

    fn model_id(&self) -> String {
        format!("remote:{}", self.endpoint)
    }
}

/// Stub advisor for testing -- returns fixed advice.
pub struct StubAdvisor {
    model: String,
}

impl StubAdvisor {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl Advisor for StubAdvisor {
    async fn generate_advice(
        &self,
        _context: BTreeMap<String, serde_yaml::Value>,
    ) -> Result<Advice, FitError> {
        Ok(Advice::new("generic", "Stub advice.", 0.5))
    }

    fn model_id(&self) -> String {
        self.model.clone()
    }
}
