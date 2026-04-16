use crate::advisor::Advice;
use crate::error::FitError;
use std::collections::BTreeMap;

/// Frontier LLM adapter trait.
#[async_trait::async_trait]
pub trait Adapter: Send + Sync {
    async fn call(
        &self,
        prompt: &str,
        advice: &Advice,
    ) -> Result<(String, BTreeMap<String, serde_yaml::Value>), FitError>;
}

/// Anthropic API adapter.
pub struct AnthropicAdapter {
    model: String,
    api_key: String,
    client: reqwest::Client,
}

impl AnthropicAdapter {
    pub fn new(api_key: &str) -> Self {
        Self {
            model: "claude-sonnet-4-6".to_string(),
            api_key: api_key.to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    fn build_system_prompt(&self, advice: &Advice) -> String {
        format!(
            "[Advisor Guidance]\n{}\n\nConstraints: {}",
            advice.steering_text,
            advice.constraints.join("; ")
        )
    }
}

#[async_trait::async_trait]
impl Adapter for AnthropicAdapter {
    async fn call(
        &self,
        prompt: &str,
        advice: &Advice,
    ) -> Result<(String, BTreeMap<String, serde_yaml::Value>), FitError> {
        let system = self.build_system_prompt(advice);

        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        });

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| FitError::Http(e.to_string()))?;

        let status = resp.status();
        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| FitError::Http(e.to_string()))?;

        if !status.is_success() {
            return Err(FitError::Http(format!(
                "anthropic error {}: {}",
                status, data
            )));
        }

        let output = data["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let mut meta = BTreeMap::new();
        meta.insert(
            "model".into(),
            serde_yaml::Value::String(self.model.clone()),
        );
        meta.insert(
            "provider".into(),
            serde_yaml::Value::String("anthropic".into()),
        );
        meta.insert("output".into(), serde_yaml::to_value(&output).unwrap_or_default());

        if let Some(usage) = data.get("usage") {
            meta.insert("usage".into(), serde_yaml::to_value(usage).unwrap_or_default());
        }

        Ok((output, meta))
    }
}

/// OpenAI API adapter.
pub struct OpenAIAdapter {
    model: String,
    api_key: String,
    client: reqwest::Client,
}

impl OpenAIAdapter {
    pub fn new(api_key: &str) -> Self {
        Self {
            model: "gpt-5".to_string(),
            api_key: api_key.to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

#[async_trait::async_trait]
impl Adapter for OpenAIAdapter {
    async fn call(
        &self,
        prompt: &str,
        advice: &Advice,
    ) -> Result<(String, BTreeMap<String, serde_yaml::Value>), FitError> {
        let system = format!(
            "[Advisor Guidance]\n{}\n\nConstraints: {}",
            advice.steering_text,
            advice.constraints.join("; ")
        );

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| FitError::Http(e.to_string()))?;

        let status = resp.status();
        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| FitError::Http(e.to_string()))?;

        if !status.is_success() {
            return Err(FitError::Http(format!(
                "openai error {}: {}",
                status, data
            )));
        }

        let output = data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let mut meta = BTreeMap::new();
        meta.insert(
            "model".into(),
            serde_yaml::Value::String(self.model.clone()),
        );
        meta.insert(
            "provider".into(),
            serde_yaml::Value::String("openai".into()),
        );
        meta.insert("output".into(), serde_yaml::to_value(&output).unwrap_or_default());

        if let Some(usage) = data.get("usage") {
            meta.insert("usage".into(), serde_yaml::to_value(usage).unwrap_or_default());
        }

        Ok((output, meta))
    }
}

/// Ollama local adapter.
pub struct OllamaAdapter {
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl OllamaAdapter {
    pub fn new() -> Self {
        Self {
            model: "llama3".to_string(),
            base_url: "http://localhost:11434".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.to_string();
        self
    }
}

impl Default for OllamaAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Adapter for OllamaAdapter {
    async fn call(
        &self,
        prompt: &str,
        advice: &Advice,
    ) -> Result<(String, BTreeMap<String, serde_yaml::Value>), FitError> {
        let system = format!(
            "[Advisor Guidance]\n{}\n\nConstraints: {}",
            advice.steering_text,
            advice.constraints.join("; ")
        );

        let body = serde_json::json!({
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": false,
        });

        let resp = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| FitError::Http(e.to_string()))?;

        let status = resp.status();
        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| FitError::Http(e.to_string()))?;

        if !status.is_success() {
            return Err(FitError::Http(format!(
                "ollama error {}: {}",
                status, data
            )));
        }

        let output = data["response"].as_str().unwrap_or("").to_string();

        let mut meta = BTreeMap::new();
        meta.insert(
            "model".into(),
            serde_yaml::Value::String(self.model.clone()),
        );
        meta.insert(
            "provider".into(),
            serde_yaml::Value::String("ollama".into()),
        );
        meta.insert("output".into(), serde_yaml::to_value(&output).unwrap_or_default());

        Ok((output, meta))
    }
}
