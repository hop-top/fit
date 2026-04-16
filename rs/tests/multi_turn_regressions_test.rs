use std::collections::HashMap;

use fit::{
    Adapter, Advice, FitError, Reward, RewardScorer, Session, SessionConfig, SessionMode,
};
use async_trait::async_trait;

/// Stub adapter that echoes the prompt back.
struct EchoAdapter;

#[async_trait]
impl Adapter for EchoAdapter {
    async fn call(
        &self,
        prompt: &str,
        _advice: &Advice,
    ) -> Result<(String, HashMap<String, serde_yaml::Value>), FitError> {
        let mut meta = HashMap::new();
        meta.insert(
            "provider".into(),
            serde_yaml::Value::String("echo".into()),
        );
        Ok((prompt.to_string(), meta))
    }
}

/// Scorer that always returns 0.3 (below threshold) to force multiple steps.
struct LowScorer;

impl RewardScorer for LowScorer {
    fn score(
        &self,
        _output: &str,
        _context: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let mut breakdown = HashMap::new();
        breakdown.insert("accuracy".to_string(), 0.3);
        Ok(Reward::new(0.3, breakdown))
    }
}

/// Regression: run_multi_turn must use a single session_id across all steps.
///
/// Before fix: run_multi_turn called run() in a loop, and run() generated
/// a fresh uuid::Uuid::new_v4() each time. Every step got a different
/// session_id, violating the session protocol spec.
#[tokio::test]
async fn multi_turn_same_session_id() {
    let advisor = fit::StubAdvisor::new("stub");
    let adapter = EchoAdapter;
    let scorer = LowScorer;

    let config = SessionConfig {
        mode: SessionMode::MultiTurn,
        max_steps: 3,
        reward_threshold: 1.0, // never reached since scorer returns 0.3
    };

    let mut session = Session::new(advisor, adapter, scorer).with_config(config);
    let results = session
        .run_multi_turn("test prompt", HashMap::new())
        .await
        .expect("multi-turn should succeed");

    // Should have exactly 3 steps (max_steps limit)
    assert_eq!(results.len(), 3, "expected 3 steps");

    // All steps must share the same session_id
    let first_sid = &results[0].trace.session_id;
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            &result.trace.session_id,
            first_sid,
            "step {i} has different session_id"
        );
    }

    // Verify session_id is not empty
    assert!(!first_sid.is_empty(), "session_id must not be empty");
}

/// Regression: run_multi_turn must reset step counter to 0 at start.
///
/// Before fix: self.step was not reset at the beginning of run_multi_turn,
/// so if a Session was reused, steps would continue from the previous count.
#[tokio::test]
async fn multi_turn_resets_step_counter() {
    let advisor = fit::StubAdvisor::new("stub");
    let adapter = EchoAdapter;
    let scorer = LowScorer;

    let config = SessionConfig {
        mode: SessionMode::MultiTurn,
        max_steps: 2,
        reward_threshold: 1.0,
    };

    let mut session = Session::new(advisor, adapter, scorer).with_config(config);

    // First multi-turn run: should produce 2 steps (step 0 and step 1)
    let results = session
        .run_multi_turn("prompt1", HashMap::new())
        .await
        .expect("first run");
    assert_eq!(results.len(), 2);

    // Second multi-turn run on same session: step should reset, still produce 2
    let results2 = session
        .run_multi_turn("prompt2", HashMap::new())
        .await
        .expect("second run");
    assert_eq!(
        results2.len(),
        2,
        "second run should also produce 2 steps (step counter must reset)"
    );

    // Session IDs should differ between runs
    assert_ne!(
        results[0].trace.session_id, results2[0].trace.session_id,
        "different multi-turn runs must have different session_ids"
    );
}
