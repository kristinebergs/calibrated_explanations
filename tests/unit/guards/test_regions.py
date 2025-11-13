import numpy as np
from calibrated_explanations.guards import ConformalRegionOracle
from calibrated_explanations.guards.intervals import union_intervals


def test_intervals_clip_to_global_bounds():
    # Create synthetic data with known global bounds [0, 1] for both features
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.9, 0.1],
    ])
    y = np.zeros(len(X), dtype=int)

    # Use a single cluster so the computed interval around the center can be large
    guard = ConformalRegionOracle(alpha=0.5, n_clusters=1)
    guard.fit(X, y)

    # Choose an instance near the upper edge so unclipped high would exceed 1.0
    x_inst = np.array([0.95, 0.95])
    ctx = 0
    intervals = guard.intervals(x_inst, ctx)

    # Intervals are lists of (rel_low, rel_high) tuples per feature
    assert len(intervals) == 2

    for j, feats in enumerate(intervals):
        for rel_low, rel_high in feats:
            # After clipping, the absolute bounds must lie within [0, 1]
            abs_low = x_inst[j] + rel_low
            abs_high = x_inst[j] + rel_high
            assert abs_low >= 0 - 1e-12
            assert abs_high <= 1 + 1e-12


def test_union_intervals_overlapping():
    # Overlapping and touching intervals should be merged
    intervals = [(0.0, 0.5), (0.4, 1.0), (1.0, 1.5), (2.0, 2.5)]
    merged = union_intervals(intervals)
    assert merged == [(0.0, 1.5), (2.0, 2.5)]


def test_union_intervals_empty():
    assert union_intervals([]) == []


def test_icp_data_splitting():
    """Test that ICP splits data correctly."""
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)

    guard = ConformalRegionOracle(prop_size=0.6, random_state=42)
    guard.fit(X, y)

    # Check that we have stored calibration scores
    assert hasattr(guard, '_cal_scores')
    assert len(guard._cal_scores) > 0


def test_knn_nonconformity():
    """Test k-NN based nonconformity measure."""
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)

    guard = ConformalRegionOracle(ncm_method="knn", k=3, prop_size=0.5, random_state=42)
    guard.fit(X, y)

    # Check that NN models are stored
    assert hasattr(guard, '_nn_models')
    assert len(guard._nn_models) > 0

    # Test acceptance
    x_test = X[0]
    label_ctx = y[0]
    result = guard.accept(x_test, label_ctx)
    assert isinstance(result, bool)


def test_pvalue_computation():
    """Test p-value computation."""
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)

    guard = ConformalRegionOracle(prop_size=0.5, random_state=42)
    guard.fit(X, y)

    x_test = X[0]
    label_ctx = y[0]
    p_val = guard.pvalue(x_test, label_ctx)
    assert 0.0 <= p_val <= 1.0


def test_epsilon_based_acceptance():
    """Test epsilon-based acceptance."""
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)

    guard = ConformalRegionOracle(prop_size=0.5, epsilon=0.1, random_state=42)
    guard.fit(X, y)

    x_test = X[0]
    label_ctx = y[0]
    result = guard.accept(x_test, label_ctx)
    assert isinstance(result, bool)


def test_coverage_verification():
    """Test that empirical coverage matches significance level for synthetic data."""
    np.random.seed(42)
    # Generate data from two well-separated Gaussians
    n_samples = 200
    X_class0 = np.random.randn(n_samples//2, 2) + np.array([2, 2])
    X_class1 = np.random.randn(n_samples//2, 2) - np.array([2, 2])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))

    guard = ConformalRegionOracle(alpha=0.1, prop_size=0.5, random_state=42)
    guard.fit(X, y)

    # Generate test points from the same distribution
    X_test_class0 = np.random.randn(50, 2) + np.array([2, 2])
    X_test_class1 = np.random.randn(50, 2) - np.array([2, 2])
    X_test = np.vstack([X_test_class0, X_test_class1])
    y_test = np.array([0] * 50 + [1] * 50)

    accepted = 0
    for x_test, y_test_val in zip(X_test, y_test):
        if guard.accept(x_test, y_test_val):
            accepted += 1

    coverage = accepted / len(X_test)
    # Should be approximately 1 - alpha = 0.9
    assert 0.8 <= coverage <= 1.0  # Allow some tolerance
