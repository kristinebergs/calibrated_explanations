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
        assert oracle.prop_size == 0.75
        assert oracle.random_state is None
        # pylint: disable=protected-access
        assert not oracle._fitted

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        oracle = ConformalRegionOracle(
            alpha=0.05,
            n_clusters=10,
            prop_size=0.8,
            random_state=42,
        )
        assert oracle.alpha == 0.05
        assert oracle.n_clusters == 10
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

    # Note: relaxation_factor removed from API; tests for invalid relaxation removed

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
        # Interval learner that produces variable widths so confidence
        # modulation is active during fit.
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            def predict(self, x_arr):
                n = len(x_arr)
                lowers = np.zeros(n)
                # create slightly varying upper bounds so width_max > width_min
                uppers = 0.1 + (np.arange(n) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()

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
        # Interval learner that produces variable widths so confidence
        # modulation is active during fit.
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            def predict(self, x_arr):
                n = len(x_arr)
                lowers = np.zeros(n)
                uppers = 0.1 + (np.arange(n) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Accept a point from training set (should be accepted with high prob)
        result = oracle.accept(x_arr[0])
        assert isinstance(result, (bool, np.bool_))

    def test_accept_with_calibrated_prediction(self):
        """Test accept with calibrated prediction for modulation."""
        x_arr, y_arr = self._make_simple_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        # Create an interval learner with variable widths so modulation is active
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            """Interval learner producing varying upper bounds for widths."""

            def predict(self, x_arr):
                n_samples = len(x_arr)
                lowers = np.zeros(n_samples)
                uppers = 0.1 + (np.arange(n_samples) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()
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
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            """Produces varying interval widths for modulation tests."""

            def predict(self, x_arr):
                n = len(x_arr)
                lowers = np.zeros(n)
                uppers = 0.1 + (np.arange(n) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()
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
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            """Produces varying interval widths for modulation tests."""

            def predict(self, x_arr):
                n = len(x_arr)
                lowers = np.zeros(n)
                uppers = 0.1 + (np.arange(n) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()
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
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            """Produces varying interval widths for modulation tests."""

            def predict(self, x_arr):
                n = len(x_arr)
                lowers = np.zeros(n)
                uppers = 0.1 + (np.arange(n) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()
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
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            """Produces varying interval widths for modulation tests."""

            def predict(self, x_arr):
                n = len(x_arr)
                lowers = np.zeros(n)
                uppers = 0.1 + (np.arange(n) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()
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
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            """Produces varying interval widths for modulation tests."""

            def predict(self, x_arr):
                n = len(x_arr)
                lowers = np.zeros(n)
                uppers = 0.1 + (np.arange(n) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()
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


class TestConformalRegionOracleDynamic:
    """Test dynamic updates to alpha and width-based modulation."""

    def test_width_normalization_modulates_radius(self):
        """Width-normalization (normalized conformal regression).

        Verify that normalized conformal regression uses interval width as a
        difficulty estimate: the effective radius is proportional to the
        interval width (r_eff = q_norm * width). Therefore, a narrow
        interval yields a smaller effective radius than a wide interval.
        """
        rng = np.random.default_rng(42)
        x_arr = rng.standard_normal((200, 2))
        y_arr = x_arr.sum(axis=1)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        class VariableIntervalLearner:  # pylint: disable=too-few-public-methods
            """Produces varying interval widths for modulation tests."""

            def predict(self, x_arr):
                n = len(x_arr)
                lowers = np.zeros(n)
                uppers = 0.1 + (np.arange(n) % 5) * 0.01
                return list(zip(lowers, uppers))

        interval_learner = VariableIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Pick a point and its nearest cluster base radius
        x_point = x_arr[0]
        distances = np.linalg.norm(oracle._cluster_centers - x_point, axis=1)
        nearest = int(np.argmin(distances))
        base_r = float(oracle._cluster_radii[nearest])

        # Compare narrow vs wide intervals: with width-based normalization
        # narrow intervals (more confident) should yield a larger effective
        # radius (less restrictive) than wide intervals (more uncertain).
        cal_pred_wide = (0.5, (0.0, 1.0))
        cal_pred_narrow = (0.5, (0.0, 0.1))

        r_eff_wide = oracle._compute_effective_radius(base_r, cal_pred_wide)
        r_eff_narrow = oracle._compute_effective_radius(base_r, cal_pred_narrow)

        # Under normalized conformal regression, r_eff is proportional to width
        assert r_eff_narrow < r_eff_wide

    def test_set_alpha_recomputes_radii_and_requires_fit(self):
        """set_alpha should recompute cluster radii from cached calibration scores.

        Also verify that calling set_alpha before fit raises a clear error.
        """
        # calling before fit should raise
        oracle_unfitted = ConformalRegionOracle()
        with pytest.raises(RuntimeError):
            oracle_unfitted.set_alpha(0.2)

        # After fit, changing alpha should update radii
        rng = np.random.default_rng(42)
        x_arr = rng.standard_normal((200, 2))
        y_arr = x_arr.sum(axis=1)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        old_radii = oracle._cluster_radii.copy()
        # Pick a very different alpha to force a change
        oracle.set_alpha(0.9, per_cluster=False)
        new_radii = oracle._cluster_radii

        # Radii should have changed (global quantile changed)
        assert not np.allclose(old_radii, new_radii)
