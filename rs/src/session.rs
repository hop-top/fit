use crate::adapter::Adapter;
use crate::advisor::{Advice, Advisor};
use crate::error::FitError;
use crate::reward::{Reward, RewardScorer};
use crate::trace::{Trace, TraceWriter};
use std::collections::HashMap;

/// Session lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionState {
    Init,
    Advise,
    Frontier,
    Score,
    Trace,
    Done,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionState::Init => write!(f, "init"),
            SessionState::Advise => write!(f, "advise"),
            SessionState::Frontier => write!(f, "frontier"),
            SessionState::Score => write!(f, "score"),
            SessionState::Trace => write!(f, "trace"),
            SessionState::Done => write!(f, "done"),
        }
    }
}

/// Session configuration.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub mode: SessionMode,
    pub max_steps: u32,
    pub reward_threshold: f64,
}

/// Session mode: one-shot or multi-turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionMode {
    OneShot,
    MultiTurn,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            mode: SessionMode::OneShot,
            max_steps: 10,
            reward_threshold: 1.0,
        }
    }
}

/// Session result.
pub struct SessionResult {
    pub output: String,
    pub reward: Reward,
    pub trace: Trace,
    pub state: SessionState,
}

/// Session orchestrator with full state machine.
///
/// States: Init -> Advise -> Frontier -> Score -> Trace -> (Advise | Done)
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
    state: SessionState,
    step: u32,
    traces: Vec<Trace>,
    writer: Option<TraceWriter>,
}

impl<A, S, R> Session<A, S, R>
where
    A: Advisor,
    S: Adapter,
    R: RewardScorer,
{
    /// Create a new session with default config.
    pub fn new(advisor: A, adapter: S, scorer: R) -> Self {
        Self {
            advisor,
            adapter,
            scorer,
            config: SessionConfig::default(),
            state: SessionState::Init,
            step: 0,
            traces: vec![],
            writer: None,
        }
    }

    /// Set session configuration.
    pub fn with_config(mut self, config: SessionConfig) -> Self {
        self.config = config;
        self
    }

    /// Set trace writer for persistent output.
    pub fn with_writer(mut self, writer: TraceWriter) -> Self {
        self.writer = Some(writer);
        self
    }

    /// Get current state.
    pub fn state(&self) -> &SessionState {
        &self.state
    }

    /// Get all recorded traces.
    pub fn traces(&self) -> &[Trace] {
        &self.traces
    }

    /// Validate a state transition.
    fn validate_transition(&self, to: &SessionState) -> Result<(), FitError> {
        let valid = match (&self.state, to) {
            (SessionState::Init, SessionState::Advise) => true,
            (SessionState::Advise, SessionState::Frontier) => true,
            (SessionState::Frontier, SessionState::Score) => true,
            (SessionState::Score, SessionState::Trace) => true,
            (SessionState::Trace, SessionState::Advise) => true,
            (SessionState::Trace, SessionState::Done) => true,
            _ => false,
        };

        if !valid {
            return Err(FitError::InvalidTransition {
                from: self.state.to_string(),
                to: to.to_string(),
            });
        }

        Ok(())
    }

    /// Transition to a new state.
    fn transition(&mut self, to: SessionState) -> Result<(), FitError> {
        self.validate_transition(&to)?;
        self.state = to;
        Ok(())
    }

    /// Run a one-shot session: Init -> Advise -> Frontier -> Score -> Trace -> Done
    pub async fn run(
        &mut self,
        prompt: &str,
        context: HashMap<String, serde_yaml::Value>,
    ) -> Result<SessionResult, FitError> {
        let session_id = uuid::Uuid::new_v4().to_string();

        // Init -> Advise
        self.transition(SessionState::Advise)?;

        // Build input map
        let mut input = HashMap::new();
        input.insert(
            "prompt".to_string(),
            serde_yaml::Value::String(prompt.to_string()),
        );
        for (k, v) in &context {
            input.insert(k.clone(), v.clone());
        }

        // Advise: generate advice, fallback to empty on failure
        let advice = match self.advisor.generate_advice(input.clone()).await {
            Ok(a) => a,
            Err(_) => Advice::new("unknown", "", 0.0),
        };

        // Advise -> Frontier
        self.transition(SessionState::Frontier)?;

        // Frontier: call adapter
        let (output, frontier_meta) = self.adapter.call(prompt, &advice).await?;

        // Frontier -> Score
        self.transition(SessionState::Score)?;

        // Score: evaluate output, fallback to 0 on failure
        let reward = match self.scorer.score(&output, &context) {
            Ok(r) => r,
            Err(_) => Reward::new(0.0, HashMap::new()),
        };

        // Score -> Trace
        self.transition(SessionState::Trace)?;
        self.step += 1;

        let trace = Trace {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            input,
            advice,
            frontier: frontier_meta,
            reward: reward.clone(),
            metadata: {
                let mut m = HashMap::new();
                m.insert(
                    "trace_version".into(),
                    serde_yaml::Value::String("1.0".into()),
                );
                m
            },
        };

        // Write trace if writer is configured
        if let Some(ref writer) = self.writer {
            writer.write(&trace, self.step)?;
        }

        self.traces.push(trace.clone());

        // Trace -> Done (one-shot always completes after first cycle)
        self.transition(SessionState::Done)?;

        Ok(SessionResult {
            output,
            reward,
            trace,
            state: self.state.clone(),
        })
    }

    /// Run a multi-turn session with max_steps limit.
    pub async fn run_multi_turn(
        &mut self,
        prompt: &str,
        mut context: HashMap<String, serde_yaml::Value>,
    ) -> Result<Vec<SessionResult>, FitError> {
        self.config.mode = SessionMode::MultiTurn;
        let mut results = vec![];
        let mut current_prompt = prompt.to_string();

        loop {
            let result = self.run(&current_prompt, context.clone()).await?;
            let done = result.reward.score >= self.config.reward_threshold
                || self.step >= self.config.max_steps;

            // Collect observations for next turn
            context.insert(
                "step".into(),
                serde_yaml::Value::Number(self.step.into()),
            );

            results.push(result);

            if done {
                break;
            }
        }

        Ok(results)
    }
}
