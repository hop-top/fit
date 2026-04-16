use fit::{CompositeScorer, DimensionScorer, FitError, RewardScorer};

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
