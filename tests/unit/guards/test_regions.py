"""Unit tests for ConformalRegionOracle.

These tests are intentionally small and deterministic to exercise the
algorithmic properties of the guard implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.guards.regions import ConformalRegionOracle


class MockIntervalLearner:  # pylint: disable=too-few-public-methods
    """Mock interval learner for testing."""

    def predict(self, x_arr):
        """Return constant-width intervals."""
        n_samples = len(x_arr)
        lower = np.zeros(n_samples)
        upper = np.ones(n_samples) * 0.5
        return list(zip(lower, upper))


class TestConformalRegionOracleInit:
    """Test initialization and parameter validation."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        oracle = ConformalRegionOracle()
        assert oracle.alpha == 0.1
        assert oracle.n_clusters == 5
        assert oracle.relaxation_factor == 1.0
        assert oracle.prop_size == 0.75
        assert oracle.random_state is None
        # pylint: disable=protected-access
        assert not oracle._fitted

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        oracle = ConformalRegionOracle(
            alpha=0.05,
            n_clusters=10,
            relaxation_factor=2.0,
            prop_size=0.8,
            random_state=42,
        )
        assert oracle.alpha == 0.05
        assert oracle.n_clusters == 10
        assert oracle.relaxation_factor == 2.0
        assert oracle.prop_size == 0.8
        assert oracle.random_state == 42

    def test_init_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalRegionOracle(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalRegionOracle(alpha=1.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalRegionOracle(alpha=-0.1)

    def test_init_invalid_n_clusters(self):
        """Test that invalid n_clusters raises ValueError."""
        with pytest.raises(ValueError, match="n_clusters must be"):
            ConformalRegionOracle(n_clusters=0)

    def test_init_invalid_prop_size(self):
        """Test that invalid prop_size raises ValueError."""
        with pytest.raises(ValueError, match="prop_size must be"):
            ConformalRegionOracle(prop_size=0.0)
        with pytest.raises(ValueError, match="prop_size must be"):
            ConformalRegionOracle(prop_size=1.5)

    def test_init_invalid_relaxation_factor(self):
        """Test that invalid relaxation_factor raises ValueError."""
        with pytest.raises(ValueError, match="relaxation_factor must be"):
            ConformalRegionOracle(relaxation_factor=-1.0)


class TestConformalRegionOracleFit:
    """Test fitting of ConformalRegionOracle."""

    @staticmethod
    def _make_simple_data(n_samples=100, n_features=2, random_state=42):
        """Generate simple synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_fit_basic(self):
        """Test basic fitting."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()

        result = oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Should return self
        assert result is oracle
        # pylint: disable=protected-access
        assert oracle._fitted

        # Should have cluster centers and radii
        # pylint: disable=protected-access
        assert oracle._cluster_centers is not None
        # pylint: disable=protected-access
        assert oracle._cluster_centers.shape[0] == 3
        # pylint: disable=protected-access
        assert oracle._cluster_centers.shape[1] == 2
        # pylint: disable=protected-access
        assert oracle._cluster_radii is not None
        # pylint: disable=protected-access
        assert len(oracle._cluster_radii) == 3

    def test_fit_with_interval_learner(self):
        """Test fitting with interval learner."""

        x_arr, y_arr = self._make_simple_data(100)
        interval_learner = MockIntervalLearner()
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        assert oracle._fitted
        # pylint: disable=protected-access
        assert oracle._width_min == 0.5
        # pylint: disable=protected-access
        assert oracle._width_max == 0.5

    def test_fit_too_small_dataset(self):
        """Test that fitting small datasets raises ValueError."""
        x_arr = np.random.default_rng(0).standard_normal((5, 2))
        y_arr = x_arr.sum(axis=1)
        oracle = ConformalRegionOracle(n_clusters=10)
        interval_learner = MockIntervalLearner()

        with pytest.raises(ValueError, match="Training set too small"):
            oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

    def test_fit_proper_calibration_split(self):
        """Test that proper/calibration split works correctly."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, prop_size=0.6, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Check that clusters were fitted on proper set
        # pylint: disable=protected-access
        assert oracle._cluster_centers is not None
        # Check that radius was computed
        # pylint: disable=protected-access
        assert oracle._cluster_radii is not None
        # pylint: disable=protected-access
        assert np.all(oracle._cluster_radii > 0)

    def test_fit_stores_global_bounds(self):
        """Test that global min/max are stored."""
        x_arr = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]])
        y_arr = np.array([1.0, 1.0, 0.5, 0.8])
        oracle = ConformalRegionOracle(n_clusters=2, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Global bounds should match data bounds
        # pylint: disable=protected-access
        np.testing.assert_array_almost_equal(oracle._global_mins, [0.0, 0.0])
        # pylint: disable=protected-access
        np.testing.assert_array_almost_equal(oracle._global_maxs, [1.0, 1.0])


class TestConformalRegionOracleAccept:
    """Test acceptance logic."""

    @staticmethod
    def _make_simple_data(n_samples=100, n_features=2, random_state=42):
        """Generate simple synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_accept_not_fitted(self):
        """Test that accept raises error if not fitted."""
        oracle = ConformalRegionOracle()
        x_point = np.array([0.0, 0.0])

        with pytest.raises(RuntimeError, match="not fitted"):
            oracle.accept(x_point)

    def test_accept_fitted_point(self):
        """Test accepting a point from training distribution."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Accept a point from training set (should be accepted with high prob)
        result = oracle.accept(x_arr[0])
        assert isinstance(result, (bool, np.bool_))

    def test_accept_with_calibrated_prediction(self):
        """Test accept with calibrated prediction for modulation."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Accept with high-confidence prediction (narrow interval)
        result_narrow = oracle.accept(x_arr[0], calibrated_prediction=(0.5, (0.4, 0.6)))
        assert isinstance(result_narrow, (bool, np.bool_))

        # Accept with low-confidence prediction (wide interval)
        result_wide = oracle.accept(x_arr[0], calibrated_prediction=(0.5, (0.0, 1.0)))
        assert isinstance(result_wide, (bool, np.bool_))

    def test_accept_1d_input(self):
        """Test accept with 1D input (single feature)."""
        x_arr = np.random.default_rng(0).standard_normal((50, 1))
        y_arr = x_arr.ravel()
        oracle = ConformalRegionOracle(n_clusters=2, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Should handle 1D input
        result = oracle.accept(np.array([0.5]))
        assert isinstance(result, (bool, np.bool_))


class TestConformalRegionOracleAcceptBatch:
    """Test batch acceptance."""

    @staticmethod
    def _make_simple_data(n_samples=100, n_features=2, random_state=42):
        """Generate simple synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_accept_batch(self):
        """Test batch acceptance."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Test batch
        x_test = x_arr[:10]
        results = oracle.accept_batch(x_test)

        assert isinstance(results, np.ndarray)
        assert results.dtype == bool
        assert len(results) == 10

    def test_accept_batch_with_predictions(self):
        """Test batch with calibrated predictions."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_test = x_arr[:5]
        preds = [(0.5, (0.4, 0.6))] * 5
        results = oracle.accept_batch(x_test, preds)

        assert isinstance(results, np.ndarray)
        assert results.dtype == bool
        assert len(results) == 5


class TestConformalRegionOracleIntervals:
    """Test interval computation."""

    @staticmethod
    def _make_simple_data(n_samples=100, n_features=2, random_state=42):
        """Generate simple synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_intervals_not_fitted(self):
        """Test that intervals raises error if not fitted."""
        oracle = ConformalRegionOracle()
        x_point = np.array([0.0, 0.0])

        with pytest.raises(RuntimeError, match="not fitted"):
            oracle.intervals(x_point)

    def test_intervals_basic(self):
        """Test interval computation."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_orig = x_arr[0]
        intervals = oracle.intervals(x_orig)

        # Should return list of intervals per feature
        assert isinstance(intervals, list)
        assert len(intervals) == 2  # 2 features

        # Each interval is a list of tuples
        for interval_j in intervals:
            assert isinstance(interval_j, list)
            for low, high in interval_j:
                assert low <= high

    def test_intervals_clipped_to_bounds(self):
        """Test that intervals are clipped to global bounds."""
        # Create data with known bounds
        x_arr = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
        y_arr = np.zeros(len(x_arr))

        oracle = ConformalRegionOracle(n_clusters=1, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Point near corner should have clipped intervals
        x_orig = np.array([0.95, 0.95])
        intervals = oracle.intervals(x_orig)

        # Check that intervals are within [0, 1]
        for interval_j in intervals:
            for low, high in interval_j:
                assert low >= -1e-12  # Allow small numerical error
                assert high <= 1.0 + 1e-12

    def test_intervals_with_calibrated_prediction(self):
        """Test interval computation with modulation."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_orig = x_arr[0]

        # Intervals with narrow prediction (high confidence)
        intervals_narrow = oracle.intervals(x_orig, calibrated_prediction=(0.5, (0.4, 0.6)))

        # Intervals with wide prediction (low confidence)
        intervals_wide = oracle.intervals(x_orig, calibrated_prediction=(0.5, (0.0, 1.0)))

        # Both should be valid
        assert len(intervals_narrow) == 2
        assert len(intervals_wide) == 2


class TestConformalRegionOracleNumericalStability:
    """Test numerical stability edge cases."""

    def test_single_cluster_data(self):
        """Test fitting with single cluster."""
        x_arr = np.ones((20, 2))  # All same point
        y_arr = np.ones(20)

        oracle = ConformalRegionOracle(n_clusters=1, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        assert oracle._fitted
        result = oracle.accept(np.array([1.0, 1.0]))
        assert isinstance(result, (bool, np.bool_))

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        n_features = 20
        x_arr = np.random.default_rng(0).standard_normal((50, n_features))
        y_arr = x_arr.sum(axis=1)

        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        assert oracle._fitted
        result = oracle.accept(x_arr[0])
        assert isinstance(result, (bool, np.bool_))

    def test_acceptance_rate_reasonable(self):
        """Test that acceptance rate is reasonable."""
        x_arr = np.random.default_rng(0).standard_normal((100, 2))
        y_arr = x_arr.sum(axis=1)

        oracle = ConformalRegionOracle(alpha=0.1, n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Test on training set; should have ~(1-alpha) acceptance
        results = oracle.accept_batch(x_arr)
        acceptance_rate = np.mean(results)

        # Should be roughly >= 1 - alpha, but not all accepted
        assert 0.5 < acceptance_rate <= 1.0
