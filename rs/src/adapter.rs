use crate::advisor::Advice;
use std::collections::HashMap;

/// Frontier LLM adapter trait.
#[async_trait::async_trait]
pub trait Adapter: Send + Sync {
    async fn call(
        &self,
        prompt: &str,
        advice: &Advice,
    ) -> Result<(String, HashMap<String, serde_yaml::Value>), Box<dyn std::error::Error + Send + Sync>>;
}
