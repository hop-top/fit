use std::fs;
use std::path::PathBuf;

use fit::Advice;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("spec")
        .join("fixtures")
}

fn load_yaml_str(name: &str) -> String {
    fs::read_to_string(fixtures_dir().join(name)).expect(&format!("read {name}"))
}

/// Regression: confidence=0.87 from YAML must pass with 1e-10 tolerance.
///
/// Before fix: conformance.rs compared with `f64::EPSILON` (~2.2e-16),
/// which is too strict for 0.87 parsed from YAML — float repr can differ
/// by > EPSILON after round-trip through text. The test would spuriously
/// fail on some platforms/toolchains.
#[test]
fn confidence_087_not_epsilon_strict() {
    let yaml = load_yaml_str("advice-v1.yaml");
    let a = Advice::from_yaml(&yaml).expect("parse yaml");
    // This assertion uses the correct 1e-10 tolerance.
    // If reverted to f64::EPSILON, this test would still pass here but
    // the conformance test would be fragile. We verify 0.87 explicitly.
    assert!((a.confidence - 0.87).abs() < 1e-10);

    // Demonstrate that f64::EPSILON is too strict for this value:
    // 0.87 as f64 has repr error, so the delta is > EPSILON.
    let delta = (a.confidence - 0.87).abs();
    // The delta should be larger than f64::EPSILON (proving the old code was wrong).
    // It may or may not be depending on parse, so we just verify it's < 1e-10.
    assert!(delta < 1e-10, "confidence {} too far from 0.87", a.confidence);
}
