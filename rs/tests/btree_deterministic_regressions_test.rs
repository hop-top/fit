use std::collections::BTreeMap;

use hop_top_fit::{Advice, Reward, Trace};

/// Regression: Reward breakdown must serialize keys in alphabetical order.
///
/// Before fix: breakdown used HashMap whose iteration order is
/// nondeterministic. Serializing the same struct twice could produce
/// different JSON/YAML key orders, causing noisy diffs in trace
/// cassettes. BTreeMap guarantees sorted (alphabetical) key order.
#[test]
fn reward_breakdown_deterministic_key_order() {
    let mut breakdown = BTreeMap::new();
    // Insert in non-alphabetical order
    breakdown.insert("zebra".to_string(), 0.1);
    breakdown.insert("alpha".to_string(), 0.2);
    breakdown.insert("middle".to_string(), 0.3);

    let reward = Reward::new(0.6, breakdown);
    let json = reward.to_json().expect("serialize json");

    // Keys must appear in alphabetical order in the JSON string
    let alpha_pos = json.find("\"alpha\"").expect("alpha key present");
    let middle_pos = json.find("\"middle\"").expect("middle key present");
    let zebra_pos = json.find("\"zebra\"").expect("zebra key present");

    assert!(
        alpha_pos < middle_pos,
        "alpha must come before middle in JSON"
    );
    assert!(
        middle_pos < zebra_pos,
        "middle must come before zebra in JSON"
    );
}

/// Regression: Reward breakdown round-trip preserves alphabetical order.
#[test]
fn reward_breakdown_roundtrip_order() {
    let mut breakdown = BTreeMap::new();
    breakdown.insert("delta".to_string(), 0.4);
    breakdown.insert("bravo".to_string(), 0.2);
    breakdown.insert("charlie".to_string(), 0.3);
    breakdown.insert("alpha".to_string(), 0.1);

    let reward = Reward::new(0.25, breakdown);
    let json = reward.to_json().expect("serialize");
    let reward2 = Reward::from_json(&json).expect("deserialize");

    // Collect keys and verify alphabetical
    let keys: Vec<&String> = reward2.breakdown.keys().collect();
    assert_eq!(keys, vec!["alpha", "bravo", "charlie", "delta"]);
}

/// Regression: Advice metadata must serialize keys in alphabetical order.
#[test]
fn advice_metadata_deterministic_key_order() {
    let mut advice = Advice::new("test", "steer", 0.5);
    advice.metadata.insert(
        "zoo".into(),
        serde_yaml::Value::String("z-val".into()),
    );
    advice.metadata.insert(
        "apple".into(),
        serde_yaml::Value::String("a-val".into()),
    );
    advice.metadata.insert(
        "mango".into(),
        serde_yaml::Value::String("m-val".into()),
    );

    let json = advice.to_json().expect("serialize json");

    let apple_pos = json.find("\"apple\"").expect("apple present");
    let mango_pos = json.find("\"mango\"").expect("mango present");
    let zoo_pos = json.find("\"zoo\"").expect("zoo present");

    assert!(apple_pos < mango_pos, "apple before mango");
    assert!(mango_pos < zoo_pos, "mango before zoo");
}

/// Regression: Trace input/frontier/metadata maps must be deterministic.
#[test]
fn trace_maps_deterministic_key_order() {
    let mut input = BTreeMap::new();
    input.insert(
        "z-prompt".to_string(),
        serde_yaml::Value::String("last".into()),
    );
    input.insert(
        "a-prompt".to_string(),
        serde_yaml::Value::String("first".into()),
    );

    let mut frontier = BTreeMap::new();
    frontier.insert(
        "z-meta".into(),
        serde_yaml::Value::String("last".into()),
    );
    frontier.insert(
        "a-meta".into(),
        serde_yaml::Value::String("first".into()),
    );

    let mut metadata = BTreeMap::new();
    metadata.insert(
        "z-ver".into(),
        serde_yaml::Value::String("last".into()),
    );
    metadata.insert(
        "a-ver".into(),
        serde_yaml::Value::String("first".into()),
    );

    let trace = Trace {
        id: "test-id".to_string(),
        session_id: "test-session".to_string(),
        timestamp: "2025-01-01T00:00:00Z".to_string(),
        input,
        advice: Advice::new("test", "steer", 0.5),
        frontier,
        reward: Reward::new(0.9, BTreeMap::new()),
        metadata,
    };

    let json = trace.to_json().expect("serialize json");

    // Verify at least one map has sorted keys (input map)
    let a_prompt_pos = json.find("\"a-prompt\"").expect("a-prompt present");
    let z_prompt_pos = json.find("\"z-prompt\"").expect("z-prompt present");
    assert!(a_prompt_pos < z_prompt_pos, "a-prompt before z-prompt");
}
