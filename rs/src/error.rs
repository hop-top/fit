use thiserror::Error;

/// Unified error type for the fit library.
#[derive(Debug, Error)]
pub enum FitError {
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("http error: {0}")]
    Http(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("session error: {0}")]
    Session(String),

    #[error("invalid state transition: from {from} to {to}")]
    InvalidTransition { from: String, to: String },

    #[error("scoring error: {0}")]
    Scoring(String),
}
