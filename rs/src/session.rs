use crate::adapter::Adapter;
use crate::advisor::{Advice, Advisor};
use crate::reward::RewardScorer;
use crate::trace::Trace;
use std::collections::HashMap;

/// Session configuration.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub mode: String,
    pub max_steps: u32,
    pub reward_threshold: f64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            mode: "one-shot".to_string(),
            max_steps: 10,
            reward_threshold: 1.0,
        }
    }
}

/// Session orchestrator.
pub struct Session<A, S, R>
where
    A: Advisor,
    S: Adapter,
    R: RewardScorer,
{
    advisor: A,
    adapter: S,
    scorer: R,
    config: SessionConfig,
}

/// Session result.
pub struct SessionResult {
    pub output: String,
    pub reward: crate::reward::Reward,
    pub trace: Trace,
}

impl<A, S, R> Session<A, S, R>
where
    A: Advisor,
    S: Adapter,
    R: RewardScorer,
{
    pub fn new(advisor: A, adapter: S, scorer: R) -> Self {
        Self {
            advisor,
            adapter,
            scorer,
            config: SessionConfig::default(),
        }
    }

    pub async fn run(
        &self,
        prompt: &str,
        context: HashMap<String, serde_yaml::Value>,
    ) -> Result<SessionResult, Box<dyn std::error::Error + Send + Sync>> {
        let session_id = uuid::Uuid::new_v4().to_string();

        // Advise
        let advice = self.advisor.generate_advice(context.clone()).await.unwrap_or(
            Advice {
                domain: "unknown".to_string(),
                steering_text: String::new(),
                confidence: 0.0,
                constraints: vec![],
                metadata: HashMap::new(),
                version: "1.0".to_string(),
            },
        );

        // Frontier
        let (output, frontier_meta) = self.adapter.call(prompt, &advice).await?;

        // Score
        let reward = self.scorer.score(&output, &context).unwrap_or(
            crate::reward::Reward {
                score: 0.0,
                breakdown: HashMap::new(),
                metadata: HashMap::new(),
            },
        );

        let trace = Trace {
            id: uuid::Uuid::new_v4().to_string(),
            session_id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            input: context,
            advice,
            frontier: frontier_meta,
            reward: reward.clone(),
            metadata: HashMap::new(),
        };

        Ok(SessionResult {
            output,
            reward,
            trace,
        })
    }
}
