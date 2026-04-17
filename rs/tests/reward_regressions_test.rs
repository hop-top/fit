use hop_top_fit::{CompositeScorer, DimensionScorer, FitError, Reward, RewardScorer};
use std::collections::BTreeMap;

/// Stub scorer that always returns a null (failure) score.
struct NullScorer;

impl RewardScorer for NullScorer {
    fn score(
        &self,
        _output: &str,
        _context: &BTreeMap<String, serde_yaml::Value>,
    ) -> Result<Reward, FitError> {
        Ok(Reward::null(BTreeMap::new()))
    }
}

/// Regression: CompositeScorer must set metadata.error = "child_score_is_null"
/// when a child scorer returns a null score (reward-schema-v1).
///
/// Before fix: metadata used key "null_reason" with value "child_scorer_null",
/// which did not conform to reward-schema-v1's error convention.
#[test]
fn composite_scorer_null_score_has_error_metadata() {
    let scorers: Vec<Box<dyn RewardScorer>> = vec![
        Box::new(NullScorer),
        Box::new(DimensionScorer::new("ok")),
    ];
    let composite = CompositeScorer::new(scorers, vec![0.5, 0.5])
        .expect("valid weights");
    let result = composite
        .score("test output", &BTreeMap::new())
        .expect("scoring should not error");
    assert!(
        result.score.is_none(),
        "score should be null when a child returns null"
    );
    let error_val = result.metadata.get("error").expect("metadata.error missing");
    assert_eq!(
        error_val.as_str(),
        Some("child_score_is_null"),
        "metadata.error should be \"child_score_is_null\""
    );
}

/// Regression: CompositeScorer must reject weights/scorers length mismatch.
///
/// Before fix: zip silently truncated when weights.len() != scorers.len(),
/// silently ignoring some scorers or weights.
#[test]
fn composite_scorer_rejects_length_mismatch() {
    let scorers: Vec<Box<dyn RewardScorer>> = vec![
        Box::new(DimensionScorer::new("a")),
        Box::new(DimensionScorer::new("b")),
    ];
    // 2 scorers but 3 weights — should fail
    let result = CompositeScorer::new(scorers, vec![0.5, 0.3, 0.2]);
    match result {
        Err(FitError::Scoring(msg)) => {
            assert!(
                msg.contains("mismatch"),
                "error msg should mention mismatch: {msg}"
            );
        }
        Err(other) => panic!("expected Scoring error, got: {other}"),
        Ok(_) => panic!("should reject mismatched weights length"),
    }
}
