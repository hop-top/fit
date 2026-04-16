use crate::advisor::Advice;
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

/// Writes xrr-compatible YAML cassettes.
pub struct TraceWriter {
    output_dir: String,
}

impl TraceWriter {
    pub fn new(output_dir: &str) -> Self {
        Self {
            output_dir: output_dir.to_string(),
        }
    }

    pub fn write(&self, trace: &Trace, step: u32) -> std::io::Result<()> {
        let session_dir = Path::new(&self.output_dir).join(&trace.session_id);
        std::fs::create_dir_all(&session_dir)?;
        let path = session_dir.join(format!("step-{step:03}.yaml"));
        let yaml = serde_yaml::to_string(trace)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, yaml)
    }
}
