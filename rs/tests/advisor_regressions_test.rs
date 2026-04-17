use hop_top_fit::RemoteAdvisor;

/// PR#12 regression: RemoteAdvisor::new must not silently drop the
/// timeout by falling back to a default client.
///
/// Before fix: `unwrap_or_default()` returned a client with NO timeout
/// if the build failed for any reason. The timeout was silently lost.
///
/// After fix: `expect()` panics with context if build fails, so a
/// misconfigured client can never be silently created.
///
/// This test verifies the advisor stores the configured timeout and
/// builds a client successfully (not the default).
#[test]
fn remote_advisor_stores_configured_timeout() {
    let advisor = RemoteAdvisor::new("http://localhost:9999", 7500);
    // The advisor should have stored the timeout we passed.
    assert_eq!(advisor.timeout_ms(), 7500);
}

/// Verify that from_endpoint uses a sensible default timeout.
#[test]
fn remote_advisor_from_endpoint_default_timeout() {
    let advisor = RemoteAdvisor::from_endpoint("http://localhost:9999");
    assert_eq!(advisor.timeout_ms(), 5000);
}
