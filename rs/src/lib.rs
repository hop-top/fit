pub mod adapter;
pub mod advisor;
pub mod error;
pub mod reward;
pub mod session;
pub mod trace;

pub use adapter::{Adapter, AnthropicAdapter, OllamaAdapter, OpenAIAdapter};
pub use advisor::{Advice, Advisor, RemoteAdvisor, StubAdvisor};
pub use error::FitError;
pub use reward::{CompositeScorer, DimensionScorer, ExactMatchScorer, Reward, RewardScorer};
pub use session::{Session, SessionConfig, SessionMode, SessionResult, SessionState};
pub use trace::{Trace, TraceWriter};
