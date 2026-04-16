use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reward scoring result (reward-schema-v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reward {
    pub score: f64,
    pub breakdown: HashMap<String, f64>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_yaml::Value>,
}

/// Reward scorer trait.
pub trait RewardScorer: Send + Sync {
    fn score(
        &self,
        output: &str,
        context: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Reward, Box<dyn std::error::Error + Send + Sync>>;
}
