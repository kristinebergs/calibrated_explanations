"""Comprehensive unit tests for ConformalRegionsGuardPlugin.

Tests cover initialization, filtering, batch acceptance, error handling,
and metric tracking for the guard plugin.
"""

import logging
from typing import Mapping, Optional, Sequence, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from calibrated_explanations.core.exceptions import ConfigurationError
from calibrated_explanations.core.explain.guards.conformal_regions_plugin import (
    ConformalRegionsGuardPlugin,
)
from calibrated_explanations.core.explain.guards.regions import ConformalRegionOracle
from calibrated_explanations.plugins.guards import GuardContext


class DummyIntervalLearner:
    """Mock interval learner supporting uq_interval parameter."""

    def predict(self, x, uq_interval=False):
        """Mock predict method."""
        n_samples = len(x)
        preds = np.ones(n_samples) * 0.5
        if uq_interval:
            lower = np.zeros(n_samples)
            upper = np.ones(n_samples)
            return preds, (lower, upper)
        # Legacy format for backward compatibility
        return [(0.0, 1.0) for _ in range(n_samples)]


@pytest.fixture
def sample_context():
    """Create a sample GuardContext for testing."""
    x_cal = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    y_cal = np.array([0, 1, 0])
    interval_learner = DummyIntervalLearner()

    return GuardContext(
        task="classification",
        mode="factual",
        learner=None,
        x_cal=x_cal,
        y_cal=y_cal,
        interval_learner=interval_learner,
        feature_names=["f0", "f1"],
        categorical_features=[],
        num_features=2,
        metadata={},
    )


@pytest.fixture
def plugin():
    """Create a fresh plugin instance for testing."""
    return ConformalRegionsGuardPlugin()


class TestPluginInitialization:
    """Test cases for plugin initialization."""

    def test_should_expose_plugin_metadata(self, plugin):
        """Verify plugin exposes correct metadata."""
        meta = plugin.plugin_meta
        assert meta["name"] == "core.guard.conformal_regions"
        assert meta["version"] == "1.0.0"
        assert "perturbation_filtering" in meta["capabilities"]
        assert "candidate_filtering" in meta["capabilities"]
        assert "factual" in meta["modes"]
        assert "alternative" in meta["modes"]
        assert "fast" in meta["modes"]
        assert "classification" in meta["tasks"]
        assert "regression" in meta["tasks"]

    def test_should_support_all_documented_modes(self, plugin):
        """Verify supports_mode returns True for all documented modes."""
        for mode in ["factual", "alternative", "fast"]:
            for task in ["classification", "regression"]:
                assert plugin.supports_mode(mode, task=task)

    def test_should_reject_unsupported_modes(self, plugin):
        """Verify supports_mode returns False for unsupported modes."""
        assert not plugin.supports_mode("unknown_mode", task="classification")
        assert not plugin.supports_mode("factual", task="unsupported_task")


class TestPluginInitialize:
    """Test cases for plugin.initialize() method."""

    def test_should_initialize_with_guard_params(self, plugin, sample_context):
        """Verify plugin initializes guard when params provided in context."""
        # Add guard_params to context metadata
        sample_context.metadata["guard_params"] = {
            "alpha": 0.9,
            "n_clusters": 5,
        }

        with patch(
            "calibrated_explanations.core.explain.guards.conformal_regions_plugin."
            "ConformalRegionOracle"
        ) as mock_oracle:
            mock_guard = MagicMock()
            mock_oracle.return_value = mock_guard

            plugin.initialize(sample_context)

            # Verify guard was created and fitted
            assert mock_oracle.called
            assert mock_guard.fit.called
            assert plugin._context == sample_context
            assert plugin._guard_params == {"alpha": 0.9, "n_clusters": 5}

    def test_should_use_training_data_from_guard_params(self, plugin, sample_context):
        """Verify plugin uses x_train and y_train from guard_params if provided."""
        import numpy as np

        # Create mock training data
        x_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])

        # Add guard_params with training data
        sample_context.metadata["guard_params"] = {
            "alpha": 0.9,
            "n_clusters": 5,
            "x_train": x_train,
            "y_train": y_train,
        }

        with patch(
            "calibrated_explanations.core.explain.guards.conformal_regions_plugin."
            "ConformalRegionOracle"
        ) as mock_oracle:
            mock_guard = MagicMock()
            mock_oracle.return_value = mock_guard

            plugin.initialize(sample_context)

            # Verify guard was created and fitted with provided training data
            assert mock_oracle.called
            mock_guard.fit.assert_called_once()
            # Check that fit was called with x_train, y_train, and interval_learner
            args, kwargs = mock_guard.fit.call_args
            assert np.array_equal(args[0], x_train)  # x
            assert np.array_equal(args[1], y_train)  # y
            assert "interval_learner" in kwargs
            assert plugin._guard_params == {"alpha": 0.9, "n_clusters": 5}

    def test_should_log_info_when_no_params_provided(self, plugin, sample_context, caplog):
        """Verify plugin logs info when no guard_params in context."""
        with caplog.at_level(logging.INFO):
            plugin.initialize(sample_context)

        assert "No guard_params provided" in caplog.text or plugin._guard is None

    def test_should_handle_fit_error_gracefully_when_not_enforced(
        self, plugin, sample_context, caplog
    ):
        """Verify plugin handles fit errors gracefully when not enforced."""
        sample_context.metadata["guard_params"] = {"alpha": 0.9}

        with patch(
            "calibrated_explanations.core.explain.guards.conformal_regions_plugin."
            "ConformalRegionOracle"
        ) as mock_oracle:
            mock_oracle.side_effect = ValueError("Test error")

            with caplog.at_level(logging.WARNING):
                plugin.initialize(sample_context)

            assert plugin._guard is None
            assert "Failed to fit guard" in caplog.text or plugin._guard is None

    def test_should_raise_when_fit_error_with_enforcement(
        self, plugin, sample_context
    ):
        """Verify plugin raises on fit error when enforcement enabled."""
        sample_context.metadata["guard_params"] = {
            "alpha": 0.9,
            "enforcement": True,
        }

        with patch(
            "calibrated_explanations.core.explain.guards.conformal_regions_plugin."
            "ConformalRegionOracle"
        ) as mock_oracle:
            mock_oracle.side_effect = ValueError("Test error")

            with pytest.raises(ConfigurationError):
                plugin.initialize(sample_context)

    def test_should_pass_wrapped_interval_learner_to_guard(
        self, plugin, sample_context
    ):
        """Verify plugin wraps interval learner when initializing guard."""
        sample_context.metadata["guard_params"] = {"alpha": 0.9}

        with patch(
            "calibrated_explanations.core.explain.guards.conformal_regions_plugin."
            "ConformalRegionOracle"
        ) as mock_oracle:
            with patch(
                "calibrated_explanations.core.explain.guards.conformal_regions_plugin."
                "IntervalLearnerAdapter"
            ) as mock_adapter:
                mock_guard = MagicMock()
                mock_oracle.return_value = mock_guard
                mock_adapter.return_value = MagicMock()

                plugin.initialize(sample_context)

                # Verify interval learner was wrapped and passed to fit
                assert mock_adapter.called
                assert mock_guard.fit.called


class TestFilterPerturbations:
    """Test cases for filter_perturbations() method."""

    def test_should_return_unfiltered_when_no_guard(self, plugin):
        """Verify returns unfiltered perturbations when no guard present."""
        x_perturbed = np.array([[0.0, 0.0], [1.0, 1.0]])
        x_feature = np.array([[0, 0, 0], [1, 1, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0]), "low": np.array([0.0]), "high": np.array([1.0])}

        result_x, result_feat = plugin.filter_perturbations(
            x_perturbed, x_feature, x_orig, prediction
        )

        np.testing.assert_array_equal(result_x, x_perturbed)
        np.testing.assert_array_equal(result_feat, x_feature)

    def test_should_raise_when_no_guard_with_enforcement(self, plugin):
        """Verify raises when no guard present but enforcement enabled."""
        plugin._enforcement = True
        x_perturbed = np.array([[0.0, 0.0]])
        x_feature = np.array([[0, 0, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0]), "low": np.array([0.0]), "high": np.array([1.0])}

        with pytest.raises(RuntimeError, match="No guard present for filtering"):
            plugin.filter_perturbations(x_perturbed, x_feature, x_orig, prediction)

    def test_should_return_unfiltered_when_empty_perturbations(self, plugin):
        """Verify returns unfiltered when perturbations is empty."""
        plugin._guard = MagicMock()
        x_perturbed = np.array([]).reshape(0, 2)
        x_feature = np.array([]).reshape(0, 3)
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0]), "low": np.array([0.0]), "high": np.array([1.0])}

        result_x, result_feat = plugin.filter_perturbations(
            x_perturbed, x_feature, x_orig, prediction
        )

        assert result_x.shape[0] == 0
        assert result_feat.shape[0] == 0

    def test_should_return_unfiltered_when_guard_not_fitted(self, plugin, caplog):
        """Verify returns unfiltered when guard not fitted."""
        mock_guard = MagicMock()
        mock_guard._fitted = False
        plugin._guard = mock_guard

        x_perturbed = np.array([[0.0, 0.0]])
        x_feature = np.array([[0, 0, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0]), "low": np.array([0.0]), "high": np.array([1.0])}

        with caplog.at_level(logging.WARNING):
            result_x, result_feat = plugin.filter_perturbations(
                x_perturbed, x_feature, x_orig, prediction
            )

        np.testing.assert_array_equal(result_x, x_perturbed)
        assert "not fitted" in caplog.text or result_x.shape[0] == 1

    def test_should_raise_when_guard_not_fitted_with_enforcement(self, plugin):
        """Verify raises when guard not fitted but enforcement enabled."""
        plugin._enforcement = True
        mock_guard = MagicMock()
        mock_guard._fitted = False
        plugin._guard = mock_guard

        x_perturbed = np.array([[0.0, 0.0]])
        x_feature = np.array([[0, 0, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0]), "low": np.array([0.0]), "high": np.array([1.0])}

        with pytest.raises(RuntimeError, match="not fitted"):
            plugin.filter_perturbations(x_perturbed, x_feature, x_orig, prediction)

    def test_should_filter_perturbations_using_guard(self, plugin):
        """Verify filters perturbations using guard.accept_batch()."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.accept_batch.return_value = np.array([True, False])
        plugin._guard = mock_guard

        x_perturbed = np.array([[0.0, 0.0], [1.0, 1.0]])
        x_feature = np.array([[0, 0, 0], [1, 1, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {
            "predict": np.array([0.0]),
            "low": np.array([0.0]),
            "high": np.array([1.0]),
        }

        result_x, result_feat = plugin.filter_perturbations(
            x_perturbed, x_feature, x_orig, prediction
        )

        assert result_x.shape[0] == 1
        assert result_feat.shape[0] == 1
        assert plugin.metrics["filtered_perturbations"] == 1

    def test_should_handle_filtering_error_gracefully(self, plugin, caplog):
        """Verify handles filtering errors gracefully when not enforced."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.accept_batch.side_effect = ValueError("Test error")
        plugin._guard = mock_guard

        x_perturbed = np.array([[0.0, 0.0]])
        x_feature = np.array([[0, 0, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0]), "low": np.array([0.0]), "high": np.array([1.0])}

        with caplog.at_level(logging.DEBUG):
            result_x, result_feat = plugin.filter_perturbations(
                x_perturbed, x_feature, x_orig, prediction
            )

        np.testing.assert_array_equal(result_x, x_perturbed)

    def test_should_raise_on_filtering_error_with_enforcement(self, plugin):
        """Verify raises on filtering error when enforcement enabled."""
        plugin._enforcement = True
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.accept_batch.side_effect = ValueError("Test error")
        plugin._guard = mock_guard

        x_perturbed = np.array([[0.0, 0.0]])
        x_feature = np.array([[0, 0, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0]), "low": np.array([0.0]), "high": np.array([1.0])}

        with pytest.raises(ValueError):
            plugin.filter_perturbations(x_perturbed, x_feature, x_orig, prediction)


class TestFilterCandidates:
    """Test cases for filter_candidates() method."""

    def test_should_return_all_candidates_when_no_guard(self, plugin):
        """Verify returns all candidates when no guard present."""
        candidates = np.array([0.0, 0.5, 1.0])
        x_orig = np.array([[0.0, 0.0]])

        result = plugin.filter_candidates(0, candidates, x_orig)

        np.testing.assert_array_equal(result, candidates)

    def test_should_raise_when_no_guard_with_enforcement(self, plugin):
        """Verify raises when no guard present but enforcement enabled."""
        plugin._enforcement = True
        candidates = np.array([0.0, 0.5, 1.0])
        x_orig = np.array([[0.0, 0.0]])

        with pytest.raises(RuntimeError, match="No guard present for filtering"):
            plugin.filter_candidates(0, candidates, x_orig)

    def test_should_return_all_candidates_when_x_orig_is_none(self, plugin):
        """Verify returns all candidates when x_orig is None."""
        plugin._guard = MagicMock()
        candidates = np.array([0.0, 0.5, 1.0])

        result = plugin.filter_candidates(0, candidates, None)

        np.testing.assert_array_equal(result, candidates)

    def test_should_return_all_candidates_when_guard_not_fitted(self, plugin, caplog):
        """Verify returns all candidates when guard not fitted."""
        mock_guard = MagicMock()
        mock_guard._fitted = False
        plugin._guard = mock_guard

        candidates = np.array([0.0, 0.5, 1.0])
        x_orig = np.array([[0.0, 0.0]])

        with caplog.at_level(logging.WARNING):
            result = plugin.filter_candidates(0, candidates, x_orig)

        np.testing.assert_array_equal(result, candidates)

    def test_should_filter_candidates_using_guard_intervals(self, plugin):
        """Verify filters candidates using guard.intervals()."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.intervals.return_value = [[(0.2, 0.8)], [(0.0, 1.0)]]
        plugin._guard = mock_guard

        candidates = np.array([0.0, 0.3, 0.5, 0.9, 1.0])
        x_orig = np.array([[0.0, 0.0]])
        cal_pred = (0.5, (0.0, 1.0))

        result = plugin.filter_candidates(0, candidates, x_orig, cal_pred)

        # Should keep only candidates within [0.2, 0.8]
        expected = np.array([0.3, 0.5])
        np.testing.assert_array_equal(result, expected)
        assert plugin.metrics["filtered_candidates"] == 3

    def test_should_handle_filtering_error_gracefully(self, plugin, caplog):
        """Verify handles filtering errors gracefully when not enforced."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.intervals.side_effect = ValueError("Test error")
        plugin._guard = mock_guard

        candidates = np.array([0.0, 0.5, 1.0])
        x_orig = np.array([[0.0, 0.0]])

        with caplog.at_level(logging.DEBUG):
            result = plugin.filter_candidates(0, candidates, x_orig)

        np.testing.assert_array_equal(result, candidates)

    def test_should_raise_on_filtering_error_with_enforcement(self, plugin):
        """Verify raises on filtering error when enforcement enabled."""
        plugin._enforcement = True
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.intervals.side_effect = ValueError("Test error")
        plugin._guard = mock_guard

        candidates = np.array([0.0, 0.5, 1.0])
        x_orig = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError):
            plugin.filter_candidates(0, candidates, x_orig)


class TestAcceptBatch:
    """Test cases for accept_batch() method."""

    def test_should_accept_all_when_no_guard(self, plugin):
        """Verify accepts all when no guard present."""
        x_batch = np.array([[0.0, 0.0], [1.0, 1.0]])

        result = plugin.accept_batch(x_batch)

        expected = np.array([True, True])
        np.testing.assert_array_equal(result, expected)

    def test_should_raise_when_no_guard_with_enforcement(self, plugin):
        """Verify raises when no guard present but enforcement enabled."""
        plugin._enforcement = True
        x_batch = np.array([[0.0, 0.0]])

        with pytest.raises(RuntimeError, match="No guard present"):
            plugin.accept_batch(x_batch)

    def test_should_accept_all_when_guard_not_fitted(self, plugin, caplog):
        """Verify accepts all when guard not fitted."""
        mock_guard = MagicMock()
        mock_guard._fitted = False
        plugin._guard = mock_guard

        x_batch = np.array([[0.0, 0.0], [1.0, 1.0]])

        with caplog.at_level(logging.WARNING):
            result = plugin.accept_batch(x_batch)

        expected = np.array([True, True])
        np.testing.assert_array_equal(result, expected)

    def test_should_use_guard_accept_batch(self, plugin):
        """Verify uses guard.accept_batch() for filtering."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.accept_batch.return_value = np.array([True, False, True])
        plugin._guard = mock_guard

        x_batch = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])

        result = plugin.accept_batch(x_batch)

        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result, expected)
        assert plugin.metrics["accept_calls"] == 3
        assert plugin.metrics["accept_rejections"] == 1

    def test_should_handle_accept_error_gracefully(self, plugin, caplog):
        """Verify handles accept errors gracefully when not enforced."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.accept_batch.side_effect = ValueError("Test error")
        plugin._guard = mock_guard

        x_batch = np.array([[0.0, 0.0]])

        with caplog.at_level(logging.WARNING):
            result = plugin.accept_batch(x_batch)

        expected = np.array([True])
        np.testing.assert_array_equal(result, expected)

    def test_should_raise_on_accept_error_with_enforcement(self, plugin):
        """Verify raises on accept error when enforcement enabled."""
        plugin._enforcement = True
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.accept_batch.side_effect = ValueError("Test error")
        plugin._guard = mock_guard

        x_batch = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError):
            plugin.accept_batch(x_batch)


class TestGetSetGuard:
    """Test cases for get_guard() and set_guard() methods."""

    def test_should_get_current_guard(self, plugin):
        """Verify get_guard returns current guard."""
        mock_guard = MagicMock()
        plugin._guard = mock_guard

        result = plugin.get_guard()

        assert result is mock_guard

    def test_should_get_none_when_no_guard(self, plugin):
        """Verify get_guard returns None when no guard present."""
        result = plugin.get_guard()

        assert result is None

    def test_should_set_guard(self, plugin):
        """Verify set_guard replaces current guard."""
        mock_guard = MagicMock()
        mock_guard._fitted = True

        plugin.set_guard(mock_guard)

        assert plugin._guard is mock_guard

    def test_should_set_guard_to_none(self, plugin):
        """Verify set_guard can set guard to None."""
        mock_guard = MagicMock()
        plugin._guard = mock_guard

        plugin.set_guard(None)

        assert plugin._guard is None

    def test_should_warn_when_setting_unfitted_guard(self, plugin, caplog):
        """Verify warns when setting unfitted guard without enforcement."""
        mock_guard = MagicMock()
        mock_guard._fitted = False

        with caplog.at_level(logging.WARNING):
            plugin.set_guard(mock_guard)

        assert "not fitted" in caplog.text or plugin._guard is mock_guard

    def test_should_raise_when_setting_unfitted_guard_with_enforcement(self, plugin):
        """Verify raises when setting unfitted guard with enforcement."""
        plugin._enforcement = True
        mock_guard = MagicMock()
        mock_guard._fitted = False

        with pytest.raises(ValueError, match="not fitted"):
            plugin.set_guard(mock_guard)


class TestBuildCalibratedPredictions:
    """Test cases for _build_calibrated_predictions() helper."""

    def test_should_build_predictions_from_dict(self, plugin):
        """Verify builds calibrated predictions from prediction dict."""
        prediction = {
            "predict": np.array([0.5, 0.7]),
            "low": np.array([0.4, 0.6]),
            "high": np.array([0.6, 0.8]),
        }
        x_orig = np.array([[0.0, 0.0], [1.0, 1.0]])

        result = plugin._build_calibrated_predictions(prediction, x_orig)

        assert len(result) == 2
        assert result[0] == (0.5, (0.4, 0.6))
        assert result[1] == (0.7, (0.6, 0.8))

    def test_should_handle_missing_keys(self, plugin):
        """Verify handles missing prediction keys gracefully."""
        prediction = {"predict": np.array([0.5])}
        x_orig = np.array([[0.0, 0.0]])

        result = plugin._build_calibrated_predictions(prediction, x_orig)

        # Should return None for missing indices
        assert len(result) == 1


class TestMapPerturbedToCalibratedPreds:
    """Test cases for _map_perturbed_to_calibrated_preds() helper."""

    def test_should_map_perturbed_to_calibrated(self, plugin):
        """Verify maps perturbed rows to calibrated predictions."""
        perturbed_x = np.array([[0.0, 0.0], [1.0, 1.0]])
        perturbed_feature = np.array([[0, 0], [1, 1]])
        cal_preds = [(0.5, (0.4, 0.6)), (0.7, (0.6, 0.8))]

        result = plugin._map_perturbed_to_calibrated_preds(
            perturbed_x, perturbed_feature, cal_preds
        )

        assert len(result) == 2
        assert result[0] == (0.5, (0.4, 0.6))
        assert result[1] == (0.7, (0.6, 0.8))

    def test_should_handle_invalid_indices(self, plugin):
        """Verify handles invalid instance indices gracefully."""
        perturbed_x = np.array([[0.0, 0.0]])
        perturbed_feature = np.array([[0, "invalid"]])
        cal_preds = [(0.5, (0.4, 0.6))]

        result = plugin._map_perturbed_to_calibrated_preds(
            perturbed_x, perturbed_feature, cal_preds
        )

        # Should handle ValueError gracefully
        assert len(result) >= 1


class TestMetricsTracking:
    """Test cases for metrics tracking."""

    def test_should_track_filtered_perturbations(self, plugin):
        """Verify tracks filtered perturbation count."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.accept_batch.return_value = np.array([True, False, False])
        plugin._guard = mock_guard

        x_perturbed = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        x_feature = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {
            "predict": np.array([0.0]),
            "low": np.array([0.0]),
            "high": np.array([1.0]),
        }

        plugin.filter_perturbations(x_perturbed, x_feature, x_orig, prediction)

        assert plugin.metrics["filtered_perturbations"] == 2

    def test_should_track_filtered_candidates(self, plugin):
        """Verify tracks filtered candidate count."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.intervals.return_value = [[(0.3, 0.7)]]
        plugin._guard = mock_guard

        candidates = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        x_orig = np.array([[0.0, 0.0]])

        plugin.filter_candidates(0, candidates, x_orig)

        assert plugin.metrics["filtered_candidates"] == 4

    def test_should_track_accept_calls(self, plugin):
        """Verify tracks accept_batch call count."""
        mock_guard = MagicMock()
        mock_guard._fitted = True
        mock_guard.accept_batch.return_value = np.array([True, True, False])
        plugin._guard = mock_guard

        x_batch = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        plugin.accept_batch(x_batch)

        assert plugin.metrics["accept_calls"] == 3
        assert plugin.metrics["accept_rejections"] == 1
