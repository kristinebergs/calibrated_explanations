"""Unit tests for ConformalRegionOracle with augmented space clustering.

These tests exercise the new augmented space implementation where clustering
uses [x || calibrated_prediction] as input.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibrated_explanations.core.explain.guards.regions import ConformalRegionOracle


class MockIntervalLearner:  # pylint: disable=too-few-public-methods
    """Mock interval learner supporting uq_interval parameter."""

    def predict(self, x_arr, uq_interval=False):
        """Return predictions with optional interval bounds."""
        n_samples = len(x_arr)
        preds = np.ones(n_samples) * 0.5
        if uq_interval:
            lower = np.zeros(n_samples)
            upper = np.ones(n_samples)
            return preds, (lower, upper)
        return preds


class TestConformalRegionOracleBasics:
    """Test basic initialization and parameter validation."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        oracle = ConformalRegionOracle()
        assert oracle.alpha == 0.1
        assert oracle.n_clusters == 5
        # pylint: disable=protected-access
        assert not oracle._fitted

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        oracle = ConformalRegionOracle(alpha=0.05, n_clusters=10, random_state=42)
        assert oracle.alpha == 0.05
        assert oracle.n_clusters == 10
        assert oracle.random_state == 42


class TestConformalRegionOracleFitting:
    """Test fitting with augmented space."""

    @staticmethod
    def _make_data(n_samples=100, n_features=2, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_fit_returns_self(self):
        """Test that fit returns the oracle instance."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()

        result = oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        assert result is oracle
        # pylint: disable=protected-access
        assert oracle._fitted

    def test_fit_creates_augmented_cluster_centers(self):
        """Test that cluster centers are in augmented space."""
        x_arr, y_arr = self._make_data(100, n_features=2)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()

        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        centers = oracle._cluster_centers
        assert centers is not None
        assert centers.shape[0] == 3  # n_clusters
        # Augmented space: 2 features + 1 prediction = 3 dimensions
        assert centers.shape[1] == 3

    def test_fit_requires_interval_learner(self):
        """Test that fit raises error without interval_learner."""
        x_arr, y_arr = self._make_data(50)
        oracle = ConformalRegionOracle(n_clusters=2)

        with pytest.raises(ValueError, match="interval_learner must be provided"):
            oracle.fit(x_arr, y_arr, interval_learner=None)

    def test_fit_stores_base_feature_count(self):
        """Test that base feature count is stored."""
        x_arr, y_arr = self._make_data(100, n_features=5)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()

        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        assert oracle._n_features_base == 5

    def test_fit_computes_width_statistics(self):
        """Test that width statistics are computed."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)

        class VariableWidthLearner:  # pylint: disable=too-few-public-methods
            """Interval learner with varying widths."""

            def predict(self, x_arr, uq_interval=False):  # pylint: disable=missing-function-docstring
                n = len(x_arr)
                preds = np.ones(n) * 0.5
                if uq_interval:
                    lower = np.zeros(n)
                    upper = 0.2 + (np.arange(n) % 5) * 0.1
                    return preds, (lower, upper)
                return preds

        interval_learner = VariableWidthLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        assert oracle._width_min is not None
        assert oracle._width_max is not None
        assert oracle._width_max > oracle._width_min


class TestConformalRegionOracleAccept:
    """Test accept with mandatory calibrated_prediction."""

    @staticmethod
    def _make_data(n_samples=100, n_features=2, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr


class TestConformalRegionOracleAcceptBatch:
    """Test batch acceptance with mandatory predictions."""

    @staticmethod
    def _make_data(n_samples=100, n_features=2, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr


class TestConformalRegionOracleIntervals:
    """Test intervals computation in augmented space."""

    @staticmethod
    def _make_data(n_samples=100, n_features=2, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_intervals_requires_calibrated_prediction(self):
        """Test that intervals requires calibrated_prediction."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_point = x_arr[0]

        with pytest.raises(ValueError, match="calibrated_prediction is mandatory"):
            oracle.intervals(x_point, calibrated_prediction=None)

    def test_intervals_returns_correct_shape(self):
        """Test that intervals returns correct number of intervals."""
        x_arr, y_arr = self._make_data(100, n_features=3)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = MockIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_point = x_arr[0]
        calibrated_pred = (0.5, (0.0, 1.0))

        intervals = oracle.intervals(x_point, calibrated_prediction=calibrated_pred)

        # Should have 3 intervals (one per feature)
        assert len(intervals) == 3
        # Each interval is a list of (low, high) tuples
        for interval_j in intervals:
            if interval_j:  # Some may be empty if at boundary
                assert len(interval_j) == 1
                low, high = interval_j[0]
                assert low <= high
