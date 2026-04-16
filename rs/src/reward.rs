use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::FitError;

/// Reward scoring result (reward-schema-v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reward {
    pub score: f64,
    pub breakdown: HashMap<String, f64>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_yaml::Value>,
}

impl Reward {
    /// Create a new reward with the given score and breakdown.
    pub fn new(score: f64, breakdown: HashMap<String, f64>) -> Self {
        Self {
            score,
            breakdown,
            metadata: HashMap::new(),
        }
    }

    /// Parse reward from JSON string.
    pub fn from_json(json: &str) -> Result<Self, FitError> {
        serde_json::from_str(json).map_err(FitError::Json)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, FitError> {
        serde_json::to_string_pretty(self).map_err(FitError::Json)
    }
}

/// Reward scorer trait.
pub trait RewardScorer: Send + Sync {
    fn score(
        &self,
        output: &str,
        context: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError>;
}

/// Composite scorer combining multiple scorers with weights.
///
/// final_score = weighted_sum(scorer_i.score, weight_i) / total_weight
pub struct CompositeScorer {
    scorers: Vec<Box<dyn RewardScorer>>,
    weights: Vec<f64>,
}

impl CompositeScorer {
    pub fn new(scorers: Vec<Box<dyn RewardScorer>>, weights: Vec<f64>) -> Self {
        let weights = if weights.is_empty() {
            let n = scorers.len() as f64;
            vec![1.0 / n; scorers.len()]
        } else {
            weights
        };
        Self { scorers, weights }
    }

    /// Convenience: create equal-weight composite from dimension names.
    pub fn from_dimensions(names: &[&str]) -> Self {
        let scorers: Vec<Box<dyn RewardScorer>> = names
            .iter()
            .map(|n| Box::new(DimensionScorer::new(n)) as Box<dyn RewardScorer>)
            .collect();
        Self::new(scorers, vec![])
    }
}

impl RewardScorer for CompositeScorer {
    fn score(
        &self,
        output: &str,
        context: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let rewards: Vec<Reward> = self
            .scorers
            .iter()
            .map(|s| s.score(output, context))
            .collect::<Result<Vec<_>, _>>()?;

        let total_weight: f64 = self.weights.iter().sum();
        if total_weight == 0.0 {
            return Ok(Reward::new(0.0, HashMap::new()));
        }

        let combined: f64 = rewards
            .iter()
            .zip(self.weights.iter())
            .map(|(r, w)| r.score * w)
            .sum();

        let mut meta = HashMap::new();
        meta.insert(
            "scorers".into(),
            serde_yaml::Value::Number((rewards.len() as u64).into()),
        );

        Ok(Reward {
            score: combined / total_weight,
            breakdown: rewards
                .first()
                .map(|r| r.breakdown.clone())
                .unwrap_or_default(),
            metadata: meta,
        })
    }
}

/// Single-dimension stub scorer for testing.
pub struct DimensionScorer {
    dimension: String,
}

impl DimensionScorer {
    pub fn new(dimension: &str) -> Self {
        Self {
            dimension: dimension.to_string(),
        }
    }
}

impl RewardScorer for DimensionScorer {
    fn score(
        &self,
        _output: &str,
        _context: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let mut breakdown = HashMap::new();
        breakdown.insert(self.dimension.clone(), 0.5);
        Ok(Reward::new(0.5, breakdown))
    }
}

/// Exact-match scorer: returns 1.0 if output matches expected, else 0.0.
pub struct ExactMatchScorer {
    expected: String,
}

impl ExactMatchScorer {
    pub fn new(expected: &str) -> Self {
        Self {
            expected: expected.to_string(),
        }
    }
}

impl RewardScorer for ExactMatchScorer {
    fn score(
        &self,
        output: &str,
        _context: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let score = if output.trim() == self.expected.trim() {
            1.0
        } else {
            0.0
        };

        let mut breakdown = HashMap::new();
        breakdown.insert("accuracy".to_string(), score);
        breakdown.insert("relevance".to_string(), score);
        breakdown.insert("safety".to_string(), 1.0);
        breakdown.insert("efficiency".to_string(), 1.0);

        Ok(Reward::new(score, breakdown))
    }
}
