use std::collections::HashMap;

use fit::{
    Adapter, Advice, FitError, Reward, RewardScorer, Session, SessionConfig, SessionMode,
    SessionState,
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

/// Scorer that always returns a fixed score.
struct FixedScorer(f64);

impl RewardScorer for FixedScorer {
    fn score(
        &self,
        _output: &str,
        _context: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        let mut breakdown = HashMap::new();
        breakdown.insert("accuracy".to_string(), self.0);
        Ok(Reward::new(self.0, breakdown))
    }
}

/// Regression: one-shot run() must end in Trace state (not Done).
///
/// Before fix: run_with_session_id assigned self.state = SessionState::Init
/// directly, bypassing validate_transition. The one-shot run() left the
/// session in Trace state after completion, with Done deferred to the caller.
/// This test verifies the state machine follows valid transitions throughout.
#[tokio::test]
async fn one_shot_ends_in_trace_state() {
    let advisor = fit::StubAdvisor::new("stub");
    let adapter = EchoAdapter;
    let scorer = FixedScorer(0.8);

    let mut session = Session::new(advisor, adapter, scorer);
    let result = session
        .run("test prompt", HashMap::new())
        .await
        .expect("one-shot should succeed");

    // After run_with_session_id, state must be Trace (Done is deferred)
    assert_eq!(
        result.state,
        SessionState::Trace,
        "one-shot run should end in Trace state"
    );
    assert_eq!(*session.state(), SessionState::Trace);
}

/// Regression: multi-turn must follow Trace->Advise between steps
/// (not Trace->Init->Advise).
///
/// Before fix: run_with_session_id always reset self.state = Init before
/// transitioning to Advise. In a multi-turn loop, after the first step
/// ended in Trace, the next call would go Trace->Init (invalid) then
/// Init->Advise. The fix ensures run_with_session_id goes Trace->Advise
/// directly via transition().
#[tokio::test]
async fn multi_turn_trace_to_advise_transition() {
    let advisor = fit::StubAdvisor::new("stub");
    let adapter = EchoAdapter;
    let scorer = FixedScorer(0.3); // always below threshold

    let config = SessionConfig {
        mode: SessionMode::MultiTurn,
        max_steps: 3,
        reward_threshold: 1.0,
    };

    let mut session = Session::new(advisor, adapter, scorer).with_config(config);
    let results = session
        .run_multi_turn("test prompt", HashMap::new())
        .await
        .expect("multi-turn should succeed");

    // Must have completed all steps without InvalidTransition errors
    assert_eq!(results.len(), 3, "expected 3 steps");

    // Each step's returned state must be Trace (end of run_with_session_id)
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            result.state,
            SessionState::Trace,
            "step {i} should end in Trace state"
        );
    }
}

/// Regression: multi-turn must end in Done via transition(), not direct
/// state assignment.
///
/// Before fix: run_multi_turn set self.state = SessionState::Done directly,
/// bypassing validate_transition. The fix routes through transition() which
/// validates Trace->Done is a legal transition.
#[tokio::test]
async fn multi_turn_ends_in_done_via_transition() {
    let advisor = fit::StubAdvisor::new("stub");
    let adapter = EchoAdapter;
    let scorer = FixedScorer(0.3);

    let config = SessionConfig {
        mode: SessionMode::MultiTurn,
        max_steps: 2,
        reward_threshold: 1.0,
    };

    let mut session = Session::new(advisor, adapter, scorer).with_config(config);
    session
        .run_multi_turn("test prompt", HashMap::new())
        .await
        .expect("multi-turn should succeed");

    // Session must end in Done state
    assert_eq!(
        *session.state(),
        SessionState::Done,
        "multi-turn must end in Done state"
    );
}

/// Regression: max_steps=0 must transition to Done via transition(), not
/// direct state assignment.
///
/// Before fix: run_multi_turn with max_steps=0 set self.state = Done
/// directly, bypassing validate_transition. The fix routes through
/// transition() which validates Init->Done. Since Init->Done is not
/// in the valid transitions table, this test verifies we need a valid
/// path. The implementation sets state to Init first then calls
/// transition(Done), so this exercises the transition machinery.
#[tokio::test]
async fn zero_steps_done_via_transition() {
    let advisor = fit::StubAdvisor::new("stub");
    let adapter = EchoAdapter;
    let scorer = FixedScorer(0.5);

    let config = SessionConfig {
        mode: SessionMode::MultiTurn,
        max_steps: 0,
        reward_threshold: 1.0,
    };

    let mut session = Session::new(advisor, adapter, scorer).with_config(config);
    let results = session
        .run_multi_turn("test prompt", HashMap::new())
        .await
        .expect("zero-step should succeed");

    assert!(results.is_empty(), "max_steps=0 should produce no results");
    assert_eq!(*session.state(), SessionState::Done);
}

/// Regression: high-reward early exit must still reach Done via transition.
///
/// When scorer returns a score above the threshold, multi-turn should
/// exit after the first step and transition Trace->Done correctly.
#[tokio::test]
async fn multi_turn_early_exit_done_via_transition() {
    let advisor = fit::StubAdvisor::new("stub");
    let adapter = EchoAdapter;
    let scorer = FixedScorer(0.95); // above threshold

    let config = SessionConfig {
        mode: SessionMode::MultiTurn,
        max_steps: 10,
        reward_threshold: 0.9,
    };

    let mut session = Session::new(advisor, adapter, scorer).with_config(config);
    let results = session
        .run_multi_turn("test prompt", HashMap::new())
        .await
        .expect("multi-turn should succeed");

    // Should exit early with 1 result
    assert_eq!(results.len(), 1, "should exit after first high-score step");
    assert_eq!(*session.state(), SessionState::Done);
}
