use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::FitError;

/// Reward scoring result (reward-schema-v1).
/// `score` is `Option<f64>` so failures serialize as JSON `null` (not NaN).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reward {
    pub score: Option<f64>,
    pub breakdown: BTreeMap<String, f64>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_yaml::Value>,
}

impl Reward {
    /// Create a new reward with the given score and breakdown.
    pub fn new(score: f64, breakdown: BTreeMap<String, f64>) -> Self {
        Self {
            score: Some(score),
            breakdown,
            metadata: BTreeMap::new(),
        }
    }

    /// Create a reward representing a failure (null score).
    pub fn null(breakdown: BTreeMap<String, f64>) -> Self {
        Self {
            score: None,
            breakdown,
            metadata: BTreeMap::new(),
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
        context: &BTreeMap<String, serde_yaml::Value>,
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
    pub fn new(
        scorers: Vec<Box<dyn RewardScorer>>,
        weights: Vec<f64>,
    ) -> Result<Self, FitError> {
        let weights = if scorers.is_empty() {
            vec![]
        } else if weights.is_empty() {
            let n = scorers.len() as f64;
            vec![1.0 / n; scorers.len()]
        } else {
            weights
        };

        if weights.len() != scorers.len() {
            return Err(FitError::Scoring(format!(
                "weights/scorers length mismatch: {} scorers but {} weights",
                scorers.len(),
                weights.len(),
            )));
        }

        Ok(Self { scorers, weights })
    }

    /// Convenience: create equal-weight composite from dimension names.
    pub fn from_dimensions(names: &[&str]) -> Result<Self, FitError> {
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
        context: &BTreeMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let rewards: Vec<Reward> = self
            .scorers
            .iter()
            .map(|s| s.score(output, context))
            .collect::<Result<Vec<_>, _>>()?;

        let total_weight: f64 = self.weights.iter().sum();
        if total_weight == 0.0 {
            return Ok(Reward::new(0.0, BTreeMap::new()));
        }

        // If any scorer returns None score, propagate None.
        if rewards.iter().any(|r| r.score.is_none()) {
            let mut meta = BTreeMap::new();
            meta.insert(
                "scorers".into(),
                serde_yaml::Value::Number((rewards.len() as u64).into()),
            );
            meta.insert(
                "error".into(),
                serde_yaml::Value::String("child_score_is_null".into()),
            );
            let merged_breakdown: BTreeMap<String, f64> = rewards
                .iter()
                .flat_map(|r| r.breakdown.clone())
                .collect();
            return Ok(Reward {
                score: None,
                breakdown: merged_breakdown,
                metadata: meta,
            });
        }

        let combined: f64 = rewards
            .iter()
            .zip(self.weights.iter())
            .map(|(r, w)| r.score.unwrap_or(0.0) * w)
            .sum();

        let mut meta = BTreeMap::new();
        meta.insert(
            "scorers".into(),
            serde_yaml::Value::Number((rewards.len() as u64).into()),
        );

        let merged_breakdown: BTreeMap<String, f64> = rewards
            .iter()
            .flat_map(|r| r.breakdown.clone())
            .collect();

        Ok(Reward {
            score: Some(combined / total_weight),
            breakdown: merged_breakdown,
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
        _context: &BTreeMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let mut breakdown = BTreeMap::new();
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
        _context: &BTreeMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let score = if output.trim() == self.expected.trim() {
            1.0
        } else {
            0.0
        };

        let mut breakdown = BTreeMap::new();
        breakdown.insert("accuracy".to_string(), score);
        breakdown.insert("relevance".to_string(), score);
        breakdown.insert("safety".to_string(), 1.0);
        breakdown.insert("efficiency".to_string(), 1.0);

        Ok(Reward::new(score, breakdown))
    }
}
