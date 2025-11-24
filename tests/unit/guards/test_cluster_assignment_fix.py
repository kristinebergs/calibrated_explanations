"""Test for cluster calibration point assignment bug fix.

This test verifies that calibration points are correctly assigned to clusters
in the augmented feature space [x || prediction], not just the base feature space.
"""

import numpy as np
import pytest

from calibrated_explanations.core.explain.guards.regions import ConformalRegionOracle


class MockIntervalLearnerWithVaryingPredictions:
    """Mock interval learner with varying predictions to test augmented space clustering."""
    
    def __init__(self, prediction_values=None):
        """Initialize with optional prediction values for different samples."""
        self.prediction_values = prediction_values
    
    def predict(self, x_arr, uq_interval=False):
        """Return predictions with optional interval bounds.
        
        If prediction_values is provided, use those; otherwise generate based on x.
        This allows testing scenarios where predictions vary significantly.
        """
        n_samples = len(x_arr)
        
        if self.prediction_values is not None:
            preds = np.asarray(self.prediction_values[:n_samples])
        else:
            # Generate predictions that vary based on input
            preds = x_arr.sum(axis=1)
        
        if uq_interval:
            # Generate varying interval widths
            lower = preds - 10.0
            upper = preds + 10.0
            return preds, (lower, upper)
        return preds


def test_cluster_assignment_uses_augmented_space():
    """Test that calibration points are assigned to clusters using augmented space.
    
    This test verifies the fix for the bug where calibration points were being
    assigned to clusters using only base features, not augmented [x || pred] space.
    
    Before the fix: cluster_cal_counts=[0, 0, 0, 0, 0]
    After the fix: cluster_cal_counts should have non-zero values
    """
    # Create data with clear structure
    n_samples = 500
    n_features = 20
    n_clusters = 5
    
    rng = np.random.default_rng(42)
    x_arr = rng.standard_normal((n_samples, n_features))
    y_arr = x_arr.sum(axis=1)
    
    # Create interval learner with varying predictions
    interval_learner = MockIntervalLearnerWithVaryingPredictions()
    
    # Create and fit oracle
    oracle = ConformalRegionOracle(
        alpha=0.05,
        n_clusters=n_clusters,
        random_state=42
    )
    oracle.fit(x_arr, y_arr, interval_learner=interval_learner)
    
    # Check that clusters were assigned properly
    # pylint: disable=protected-access
    assert oracle._cal_nearest is not None
    assert len(oracle._cal_nearest) > 0
    
    # Verify that calibration points are distributed across clusters
    # (not all assigned to cluster 0 or all unassigned)
    cluster_assignments = oracle._cal_nearest
    unique_clusters = np.unique(cluster_assignments)
    
    # At least 2 clusters should have calibration points
    assert len(unique_clusters) >= 2, (
        f"Expected at least 2 clusters with calibration points, "
        f"but got {len(unique_clusters)}: {unique_clusters}"
    )
    
    # Count calibration points per cluster
    cluster_counts = np.bincount(cluster_assignments, minlength=n_clusters)
    
    # Verify no cluster has 0 points when we have enough calibration data
    # (with 500 calibration samples and 5 clusters, each should get some points)
    assert np.all(cluster_counts > 0), (
        f"Expected all clusters to have calibration points, "
        f"but got counts: {cluster_counts}"
    )
    
    # Verify the counts sum to the total calibration set size
    # prop_size=0.75 means 75% goes to proper set, 25% to calibration set
    # so cal_size = n_samples * (1 - 0.75) = 500 * 0.25 = 125
    expected_cal_size = int(n_samples * 0.25)
    assert np.sum(cluster_counts) == expected_cal_size, (
        f"Expected {expected_cal_size} calibration points total, "
        f"but got {np.sum(cluster_counts)}"
    )


def test_cluster_assignment_with_20_clusters():
    """Test cluster assignment with 20 clusters as mentioned in the bug report.
    
    The bug report specifically mentioned problems with 20 clusters and 500 calibration
    instances, where all clusters showed 0 calibration points.
    """
    n_samples = 2000  # Total samples (will be split 75/25 for proper/cal)
    n_features = 20
    n_clusters = 20
    
    rng = np.random.default_rng(42)
    x_arr = rng.standard_normal((n_samples, n_features))
    y_arr = x_arr.sum(axis=1)
    
    # Create interval learner with varying predictions
    interval_learner = MockIntervalLearnerWithVaryingPredictions()
    
    # Create and fit oracle with 20 clusters
    oracle = ConformalRegionOracle(
        alpha=0.05,
        n_clusters=n_clusters,
        random_state=42
    )
    oracle.fit(x_arr, y_arr, interval_learner=interval_learner)
    
    # Check cluster assignments
    # pylint: disable=protected-access
    cluster_assignments = oracle._cal_nearest
    cluster_counts = np.bincount(cluster_assignments, minlength=n_clusters)
    
    # With 500 calibration samples (2000 * 0.25) and 20 clusters,
    # we expect roughly 25 points per cluster on average
    expected_cal_size = int(n_samples * 0.25)
    assert np.sum(cluster_counts) == expected_cal_size
    
    # Most clusters should have at least a few points
    # (some clusters might have very few due to data distribution)
    clusters_with_points = np.sum(cluster_counts > 0)
    assert clusters_with_points >= 15, (
        f"Expected at least 15/20 clusters to have calibration points, "
        f"but only {clusters_with_points} clusters have points. "
        f"Counts: {cluster_counts}"
    )


def test_cluster_centers_dimensionality():
    """Test that cluster centers have correct dimensionality in augmented space."""
    n_samples = 200
    n_features = 10
    n_clusters = 5
    
    rng = np.random.default_rng(42)
    x_arr = rng.standard_normal((n_samples, n_features))
    y_arr = x_arr.sum(axis=1)
    
    interval_learner = MockIntervalLearnerWithVaryingPredictions()
    
    oracle = ConformalRegionOracle(
        alpha=0.05,
        n_clusters=n_clusters,
        random_state=42
    )
    oracle.fit(x_arr, y_arr, interval_learner=interval_learner)
    
    # pylint: disable=protected-access
    # Cluster centers should be in augmented space: n_features + 1 (for prediction)
    assert oracle._cluster_centers.shape == (n_clusters, n_features + 1), (
        f"Expected cluster centers shape ({n_clusters}, {n_features + 1}), "
        f"but got {oracle._cluster_centers.shape}"
    )
    
    # Base feature count should be stored
    assert oracle._n_features_base == n_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
