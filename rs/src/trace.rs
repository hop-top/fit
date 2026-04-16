use crate::advisor::Advice;
use crate::error::FitError;
use crate::reward::Reward;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Session trace (trace-format-v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub id: String,
    pub session_id: String,
    pub timestamp: String,
    pub input: HashMap<String, serde_yaml::Value>,
    pub advice: Advice,
    pub frontier: HashMap<String, serde_yaml::Value>,
    pub reward: Reward,
    #[serde(default)]
    pub metadata: HashMap<String, serde_yaml::Value>,
}

impl Trace {
    /// Parse trace from YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, FitError> {
        serde_yaml::from_str(yaml).map_err(FitError::Yaml)
    }

    /// Parse trace from JSON string.
    pub fn from_json(json: &str) -> Result<Self, FitError> {
        serde_json::from_str(json).map_err(FitError::Json)
    }

    /// Serialize to YAML (xrr-compatible).
    pub fn to_yaml(&self) -> Result<String, FitError> {
        serde_yaml::to_string(self).map_err(FitError::Yaml)
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, FitError> {
        serde_json::to_string_pretty(self).map_err(FitError::Json)
    }
}

/// Writes xrr-compatible YAML cassettes.
///
/// Traces stored as:
/// ```text
/// {output_dir}/{session_id}/step-001.yaml
/// ```
pub struct TraceWriter {
    output_dir: String,
}

impl TraceWriter {
    pub fn new(output_dir: &str) -> Self {
        Self {
            output_dir: output_dir.to_string(),
        }
    }

    /// Write a trace to disk as a YAML cassette.
    pub fn write(&self, trace: &Trace, step: u32) -> Result<(), FitError> {
        let session_dir = Path::new(&self.output_dir).join(&trace.session_id);
        std::fs::create_dir_all(&session_dir)?;

        let path = session_dir.join(format!("step-{step:03}.yaml"));
        let yaml = serde_yaml::to_string(trace).map_err(FitError::Yaml)?;

        std::fs::write(path, yaml)?;
        Ok(())
    }

    /// Read a trace from disk.
    pub fn read(&self, session_id: &str, step: u32) -> Result<Trace, FitError> {
        let path = Path::new(&self.output_dir)
            .join(session_id)
            .join(format!("step-{step:03}.yaml"));

        let data = std::fs::read_to_string(path)?;
        Trace::from_yaml(&data)
    }

    /// List all session IDs in the output directory.
    pub fn list_sessions(&self) -> Result<Vec<String>, FitError> {
        let dir = Path::new(&self.output_dir);
        if !dir.exists() {
            return Ok(vec![]);
        }

        let mut sessions = vec![];
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    sessions.push(name.to_string());
                }
            }
        }

        sessions.sort();
        Ok(sessions)
    }
}
