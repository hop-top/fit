use std::collections::BTreeMap;

use hop_top_fit::{
    Adapter, Advice, FitError, Reward, RewardScorer, Session, SessionState,
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
    ) -> Result<(String, BTreeMap<String, serde_yaml::Value>), FitError> {
        let mut meta = BTreeMap::new();
        meta.insert(
            "provider".into(),
            serde_yaml::Value::String("echo".into()),
        );
        Ok((prompt.to_string(), meta))
    }
}

/// Scorer that always returns a fixed score.
struct FixedScorer(f64);

impl RewardScorer for FixedScorer {
    fn score(
        &self,
        _output: &str,
        _context: &BTreeMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let mut breakdown = BTreeMap::new();
        breakdown.insert("accuracy".to_string(), self.0);
        Ok(Reward::new(self.0, breakdown))
    }
}

/// Regression: calling run() twice must not fail with InvalidTransition.
///
/// Before fix: after the first run(), session state is Done. A second
/// call to run() tries Done->Advise which is not a valid transition
/// (the valid table only has Init->Advise and Trace->Advise). The fix
/// resets state/step/traces at the start of run(), mirroring what
/// run_multi_turn() already does.
#[tokio::test]
async fn test_run_twice_resets_state() {
    let advisor = hop_top_fit::StubAdvisor::new("stub");
    let adapter = EchoAdapter;
    let scorer = FixedScorer(0.8);

    let mut session = Session::new(advisor, adapter, scorer);

    // First run must succeed and end in Done
    let result1 = session
        .run("first prompt", BTreeMap::new())
        .await
        .expect("first run() should succeed");
    assert_eq!(result1.state, SessionState::Done);
    assert_eq!(*session.state(), SessionState::Done);

    // Second run must also succeed — NOT fail with InvalidTransition
    let result2 = session
        .run("second prompt", BTreeMap::new())
        .await
        .expect("second run() should succeed (state must be reset)");
    assert_eq!(result2.state, SessionState::Done);
    assert_eq!(*session.state(), SessionState::Done);

    // Traces should only contain the second run (reset clears previous)
    assert_eq!(session.traces().len(), 1, "traces should be reset between runs");
}
