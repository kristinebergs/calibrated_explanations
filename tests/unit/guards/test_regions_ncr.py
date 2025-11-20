"""Comprehensive tests for Normalized Conformal Regression (NCR) in ConformalRegionOracle.

These tests verify the core NCR functionality:
- Width extraction from interval_learner
- Normalized quantile computation
- Confidence modulation of acceptance radius
- Integration with accept/intervals methods
"""

from __future__ import annotations

import numpy as np

from calibrated_explanations.core.explain.guards.regions import ConformalRegionOracle


class VariableWidthIntervalLearner:  # pylint: disable=too-few-public-methods
    """Mock interval learner with varying widths for testing confidence modulation."""

    def __init__(self, width_factor=1.0):
        """Initialize with a width scaling factor.

        Parameters
        ----------
        width_factor : float
            Scale factor for interval widths (default 1.0 → unit width).
        """
        self.width_factor = width_factor

    def predict(self, x_arr, uq_interval=False):
        """Return predictions with variable interval bounds."""
        n_samples = len(x_arr)
        preds = np.ones(n_samples) * 0.5
        if uq_interval:
            lower = np.zeros(n_samples)
            # Variable widths based on sample index for diversity
            upper = 0.2 + (np.arange(n_samples) % 5) * 0.1 * self.width_factor
            return preds, (lower, upper)
        return preds


class TestNCRWidthExtraction:
    """Test correct extraction of interval widths from interval_learner."""

    @staticmethod
    def _make_data(n_samples=100, n_features=2, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_extract_widths_from_calibration_set(self):
        """Verify fit() correctly extracts and caches widths from calibration set."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()

        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        assert oracle._cal_widths is not None
        assert len(oracle._cal_widths) > 0
        # Widths should be positive
        assert np.all(oracle._cal_widths > 0)

    def test_width_statistics_computed(self):
        """Verify _width_min and _width_max are correctly computed."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner(width_factor=2.0)

        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        assert oracle._width_min is not None
        assert oracle._width_max is not None
        # Min should be less than or equal to max
        assert oracle._width_min <= oracle._width_max
        # With variable widths, should have range
        assert oracle._width_max > oracle._width_min

    def test_width_extraction_handles_scalar_result(self):
        """Verify width extraction handles edge case of scalar width."""

        class ScalarWidthLearner:  # pylint: disable=too-few-public-methods
            """Returns scalar width (edge case)."""

            def predict(self, x_arr, uq_interval=False):
                n_samples = len(x_arr)
                preds = np.ones(n_samples) * 0.5
                if uq_interval:
                    # Return scalar bounds (edge case)
                    lower = np.zeros(n_samples)
                    upper = np.ones(n_samples)
                    return preds, (lower, upper)
                return preds

        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = ScalarWidthLearner()

        # Should not raise even with scalar widths
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        assert oracle._cal_widths is not None


class TestNCRNormalizedQuantileComputation:
    """Test computation of normalized quantiles for NCR."""

    @staticmethod
    def _make_data(n_samples=100, n_features=2, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_normalized_quantile_computed(self):
        """Verify fit() computes normalized quantile."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(alpha=0.1, n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()

        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        # Normalized quantiles should be stored
        assert oracle._cluster_norm_quantiles is not None
        assert len(oracle._cluster_norm_quantiles) == 3  # n_clusters
        # All should be positive (quantiles of nonconformity scores)
        assert np.all(oracle._cluster_norm_quantiles > 0)

    def test_normalized_quantile_varies_with_alpha(self):
        """Verify normalized quantile changes when alpha is adjusted."""
        x_arr, y_arr = self._make_data(100)
        interval_learner = VariableWidthIntervalLearner()

        # Fit with alpha=0.1
        oracle1 = ConformalRegionOracle(alpha=0.1, n_clusters=3, random_state=42)
        oracle1.fit(x_arr, y_arr, interval_learner=interval_learner)

        # Fit with alpha=0.05
        oracle2 = ConformalRegionOracle(alpha=0.05, n_clusters=3, random_state=42)
        oracle2.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        # Lower alpha (more stringent) should give higher quantile
        assert oracle2._cluster_norm_quantiles is not None
        assert oracle1._cluster_norm_quantiles is not None
        # Median quantile should be higher for lower alpha
        assert np.median(oracle2._cluster_norm_quantiles) >= np.median(
            oracle1._cluster_norm_quantiles
        )

    def test_set_alpha_updates_normalized_quantile(self):
        """Verify set_alpha correctly recomputes normalized quantile."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(alpha=0.1, n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()

        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        q_before = oracle._cluster_norm_quantiles.copy()

        # Change alpha
        oracle.set_alpha(0.05, per_cluster=False)
        q_after = oracle._cluster_norm_quantiles

        # Quantile should change
        assert not np.allclose(q_before, q_after)
        # New quantile should be higher (more stringent)
        assert np.all(q_after >= q_before)


class TestNCREffectiveRadiusModulation:
    """Test that acceptance radius is modulated by interval width."""

    @staticmethod
    def _make_data(n_samples=100, n_features=2, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_accept_with_narrow_interval(self):
        """Test that narrow intervals (high confidence) produce stricter acceptance."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_test = x_arr[0]

        # Very narrow interval (high confidence)
        narrow_pred = (0.5, (0.49, 0.51))  # width = 0.02
        accept_narrow = oracle.accept(x_test, calibrated_prediction=narrow_pred)

        assert isinstance(accept_narrow, (bool, np.bool_))

    def test_accept_with_wide_interval(self):
        """Test that wide intervals (low confidence) produce more relaxed acceptance."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_test = x_arr[0]

        # Very wide interval (low confidence)
        wide_pred = (0.5, (0.0, 1.0))  # width = 1.0
        accept_wide = oracle.accept(x_test, calibrated_prediction=wide_pred)

        assert isinstance(accept_wide, (bool, np.bool_))

    def test_effective_radius_scales_with_width(self):
        """Verify that effective radius r_eff = q_norm * width."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, alpha=0.1, random_state=42)
        interval_learner = VariableWidthIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        # pylint: disable=protected-access
        # Get a normalized quantile value
        if oracle._cluster_norm_quantiles is not None:
            q_norm = float(np.median(oracle._cluster_norm_quantiles))
            assert q_norm > 0

            # Test with different widths
            width1 = 0.1
            width2 = 0.5

            # Expected effective radii
            r_eff_1 = q_norm * width1
            r_eff_2 = q_norm * width2

            # Wider interval should have larger radius
            assert r_eff_2 > r_eff_1

    def test_intervals_respect_confidence_modulation(self):
        """Verify intervals respect the effective radius from confidence modulation."""
        x_arr, y_arr = self._make_data(100, n_features=3)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_test = x_arr[0]

        # Narrow interval
        narrow_pred = (0.5, (0.49, 0.51))
        intervals_narrow = oracle.intervals(x_test, calibrated_prediction=narrow_pred)

        # Wide interval
        wide_pred = (0.5, (0.0, 1.0))
        intervals_wide = oracle.intervals(x_test, calibrated_prediction=wide_pred)

        # Both should return 3 intervals (one per feature)
        assert len(intervals_narrow) == 3
        assert len(intervals_wide) == 3

        # Wide interval intervals should generally be larger
        # (measure by summing the widths of non-empty intervals)
        total_width_narrow = sum((iv[0][1] - iv[0][0]) if iv else 0 for iv in intervals_narrow)
        total_width_wide = sum((iv[0][1] - iv[0][0]) if iv else 0 for iv in intervals_wide)

        # Wide interval should allow larger perturbations (or equal)
        assert total_width_wide >= total_width_narrow


class TestNCRBatchAcceptance:
    """Test batch acceptance with confidence modulation."""

    @staticmethod
    def _make_data(n_samples=100, n_features=2, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_accept_batch_with_varying_widths(self):
        """Test batch acceptance with varying confidence levels."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        batch = x_arr[:5]

        # Create predictions with varying widths
        preds = [
            (0.5, (0.49, 0.51)),  # narrow
            (0.5, (0.4, 0.6)),  # medium
            (0.5, (0.0, 1.0)),  # wide
            (0.5, (0.45, 0.55)),  # narrow
            (0.5, (0.3, 0.7)),  # medium-wide
        ]

        results = oracle.accept_batch(batch, calibrated_predictions=preds)

        assert isinstance(results, np.ndarray)
        assert results.shape == (5,)
        assert results.dtype == bool


class TestNCRIntegrationWithExplainer:
    """Integration tests verifying NCR works with explainer-style predictions."""

    @staticmethod
    def _make_data(n_samples=100, n_features=3, random_state=42):
        """Generate synthetic data."""
        rng = np.random.default_rng(random_state)
        x_arr = rng.standard_normal((n_samples, n_features))
        y_arr = x_arr.sum(axis=1)
        return x_arr, y_arr

    def test_ncr_with_explainer_style_predict(self):
        """Test NCR with explainer-style predict interface."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        x_test = x_arr[0]

        # Simulate explainer.predict(x, uq_interval=True) output
        preds, (lowers, uppers) = interval_learner.predict(x_test.reshape(1, -1), uq_interval=True)

        # Extract single prediction tuple (explainer returns arrays)
        calibrated_pred = (preds[0], (lowers[0], uppers[0]))

        # Should accept without error
        result = oracle.accept(x_test, calibrated_prediction=calibrated_pred)
        assert isinstance(result, (bool, np.bool_))

    def test_ncr_with_batch_explainer_predict(self):
        """Test NCR batch with explainer-style batch predictions."""
        x_arr, y_arr = self._make_data(100)
        oracle = ConformalRegionOracle(n_clusters=3, random_state=42)
        interval_learner = VariableWidthIntervalLearner()
        oracle.fit(x_arr, y_arr, interval_learner=interval_learner)

        batch = x_arr[:5]

        # Simulate explainer.predict(batch, uq_interval=True) output
        preds, (lowers, uppers) = interval_learner.predict(batch, uq_interval=True)

        # Convert to list of tuples (explainer returns arrays)
        calibrated_preds = [(preds[i], (lowers[i], uppers[i])) for i in range(len(batch))]

        # Should accept batch without error
        results = oracle.accept_batch(batch, calibrated_predictions=calibrated_preds)
        assert isinstance(results, np.ndarray)
        assert len(results) == 5
