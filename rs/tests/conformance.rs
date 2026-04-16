use std::fs;
use std::path::PathBuf;

use fit::{Advice, Reward, Trace};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("spec")
        .join("fixtures")
}

fn load_yaml_str(name: &str) -> String {
    fs::read_to_string(fixtures_dir().join(name)).expect(&format!("read {name}"))
}

fn load_json_str(name: &str) -> String {
    fs::read_to_string(fixtures_dir().join(name)).expect(&format!("read {name}"))
}

// --- Advice conformance ---

#[test]
fn advice_parse_yaml() {
    let yaml = load_yaml_str("advice-v1.yaml");
    let a = Advice::from_yaml(&yaml).expect("parse yaml");
    assert_eq!(a.domain, "tax-compliance");
    assert!((a.confidence - 0.87).abs() < 1e-10);
    assert_eq!(a.constraints.len(), 3);
    assert_eq!(a.version, "1.0");
}

#[test]
fn advice_parse_json() {
    let json = load_json_str("advice-v1.json");
    let a = Advice::from_json(&json).expect("parse json");
    assert_eq!(a.domain, "tax-compliance");
    assert!((a.confidence - 0.87).abs() < 1e-10);
}

#[test]
fn advice_yaml_json_equivalence() {
    let yml = Advice::from_yaml(&load_yaml_str("advice-v1.yaml")).expect("yaml");
    let jsn = Advice::from_json(&load_json_str("advice-v1.json")).expect("json");
    assert_eq!(yml.domain, jsn.domain);
    assert!((yml.confidence - jsn.confidence).abs() < 1e-10);
    assert_eq!(yml.constraints, jsn.constraints);
}

#[test]
fn advice_round_trip_yaml() {
    let a = Advice::from_yaml(&load_yaml_str("advice-v1.yaml")).expect("parse");
    let out = a.to_yaml().expect("serialize");
    let a2 = Advice::from_yaml(&out).expect("re-parse");
    assert_eq!(a2.domain, a.domain);
    assert!((a2.confidence - a.confidence).abs() < 1e-10);
}

#[test]
fn advice_round_trip_json() {
    let a = Advice::from_json(&load_json_str("advice-v1.json")).expect("parse");
    let out = a.to_json().expect("serialize");
    let a2 = Advice::from_json(&out).expect("re-parse");
    assert_eq!(a2.domain, a.domain);
}

// --- Reward conformance ---

#[test]
fn reward_parse_json() {
    let json = load_json_str("reward-v1.json");
    let r = Reward::from_json(&json).expect("parse json");
    let score = r.score.expect("score should be Some");
    assert!((score - 0.62).abs() < 1e-10);
    let acc = r.breakdown.get("accuracy").expect("accuracy");
    assert!((acc - 0.7).abs() < 1e-10);
    assert!((r.breakdown.get("safety").unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn reward_score_in_range() {
    let r = Reward::from_json(&load_json_str("reward-v1.json")).expect("parse");
    let score = r.score.expect("score should be Some");
    assert!(score >= 0.0 && score <= 1.0);
    for v in r.breakdown.values() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn reward_round_trip_json() {
    let r = Reward::from_json(&load_json_str("reward-v1.json")).expect("parse");
    let out = r.to_json().expect("serialize");
    let r2 = Reward::from_json(&out).expect("re-parse");
    assert!((r2.score.unwrap() - r.score.unwrap()).abs() < 1e-10);
}

// --- Trace conformance ---

#[test]
fn trace_parse_yaml() {
    let yaml = load_yaml_str("trace-v1.yaml");
    let t = Trace::from_yaml(&yaml).expect("parse yaml");
    assert_eq!(t.id, "550e8400-e29b-41d4-a716-446655440000");
    assert_eq!(t.session_id, "sess_abc123");
    assert_eq!(t.advice.domain, "tax-compliance");
    let score = t.reward.score.expect("score should be Some");
    assert!((score - 0.95).abs() < 1e-10);
}

#[test]
fn trace_round_trip_yaml() {
    let t = Trace::from_yaml(&load_yaml_str("trace-v1.yaml")).expect("parse");
    let out = t.to_yaml().expect("serialize");
    let t2 = Trace::from_yaml(&out).expect("re-parse");
    assert_eq!(t2.id, t.id);
    assert_eq!(t2.session_id, t.session_id);
}

#[test]
fn trace_required_fields_present() {
    let yaml = load_yaml_str("trace-v1.yaml");
    let t = Trace::from_yaml(&yaml).expect("parse");
    assert!(!t.id.is_empty());
    assert!(!t.session_id.is_empty());
    assert!(!t.timestamp.is_empty());
    assert!(!t.frontier.is_empty());
}

// --- Multi-turn session conformance ---

#[test]
fn session_multi_parse() {
    let yaml = load_yaml_str("session-multi.yaml");
    // Parse as generic yaml to verify structure
    let doc: serde_yaml::Value = serde_yaml::from_str(&yaml).expect("parse multi");
    let mapping = doc.as_mapping().expect("mapping");
    assert_eq!(
        mapping.get(&serde_yaml::Value::String("mode".into()))
            .expect("mode")
            .as_str(),
        Some("multi-turn")
    );
    let steps = mapping
        .get(&serde_yaml::Value::String("steps".into()))
        .expect("steps")
        .as_sequence()
        .expect("sequence");
    assert_eq!(steps.len(), 3);
}

#[test]
fn session_multi_session_id_consistent() {
    let yaml = load_yaml_str("session-multi.yaml");
    let doc: serde_yaml::Value = serde_yaml::from_str(&yaml).expect("parse");
    let mapping = doc.as_mapping().unwrap();
    let sid = mapping
        .get(&serde_yaml::Value::String("session_id".into()))
        .unwrap()
        .as_str()
        .unwrap();
    let steps = mapping
        .get(&serde_yaml::Value::String("steps".into()))
        .unwrap()
        .as_sequence()
        .unwrap();
    for step in steps {
        let step_sid = step
            .get(&serde_yaml::Value::String("session_id".into()))
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(step_sid, sid);
    }
}
