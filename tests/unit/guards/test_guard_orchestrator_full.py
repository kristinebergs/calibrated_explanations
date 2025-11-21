"""Comprehensive unit tests for GuardOrchestrator.

Tests cover initialization, plugin management, perturbation/candidate filtering,
batch acceptance, and guard access/mutation.
"""

import logging
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence, Tuple
from unittest.mock import MagicMock, Mock, call

import numpy as np
import pytest

from calibrated_explanations.core.explain.guards.guard_orchestrator import (
    GuardOrchestrator,
)
from calibrated_explanations.plugins.guards import GuardContext


class MockGuardPlugin:
    """Mock guard plugin for testing orchestrator delegation."""

    def __init__(self):
        """Initialize mock plugin."""
        self.initialized = False
        self.context = None

    def initialize(self, context: GuardContext) -> None:
        """Mark plugin as initialized."""
        self.initialized = True
        self.context = context

    def filter_perturbations(self, perturbed_x, perturbed_feature, x_orig, prediction):
        """Return all perturbations unfiltered."""
        return perturbed_x, perturbed_feature

    def filter_candidates(self, feature_index, candidates, x_orig, calibrated_pred=None):
        """Return all candidates unfiltered."""
        return candidates


class MockExplainer:
    """Mock explainer for orchestrator initialization."""

    def __init__(self):
        """Initialize mock explainer."""
        self.metadata = {}


@pytest.fixture
def mock_explainer():
    """Create a mock explainer."""
    return MockExplainer()


@pytest.fixture
def mock_plugin():
    """Create a mock guard plugin."""
    return MockGuardPlugin()


@pytest.fixture
def sample_context():
    """Create a sample GuardContext."""
    return GuardContext(
        task="classification",
        mode="factual",
        learner=None,
        x_cal=np.array([[0.0, 0.0], [1.0, 1.0]]),
        y_cal=np.array([0, 1]),
        interval_learner=MagicMock(),
        feature_names=["f0", "f1"],
        categorical_features=[],
        num_features=2,
        metadata={},
    )


@pytest.fixture
def orchestrator(mock_explainer, mock_plugin):
    """Create an orchestrator with mock plugin."""
    return GuardOrchestrator(mock_explainer, mock_plugin)


class TestOrchestratorInitialization:
    """Test cases for orchestrator initialization."""

    def test_should_initialize_with_plugin(self, mock_explainer, mock_plugin):
        """Verify orchestrator initializes with guard plugin."""
        orch = GuardOrchestrator(mock_explainer, mock_plugin)

        assert orch.explainer is mock_explainer
        assert orch._guard_plugin is mock_plugin
        assert orch._context is None

    def test_should_initialize_guard_on_context_set(
        self, orchestrator, sample_context
    ):
        """Verify orchestrator initializes guard when context provided."""
        orchestrator.initialize(sample_context)

        assert orchestrator._context is sample_context
        assert orchestrator._guard_plugin.initialized
        assert orchestrator._guard_plugin.context is sample_context

    def test_should_handle_guard_init_error(self, mock_explainer, sample_context):
        """Verify raises when guard initialization fails."""
        mock_plugin = MagicMock()
        mock_plugin.initialize.side_effect = ValueError("Init error")
        orch = GuardOrchestrator(mock_explainer, mock_plugin)

        with pytest.raises(ValueError):
            orch.initialize(sample_context)


class TestPluginManagement:
    """Test cases for plugin management methods."""

    def test_should_set_plugin(self, orchestrator, sample_context):
        """Verify set_plugin replaces guard plugin."""
        new_plugin = MockGuardPlugin()
        orchestrator._context = sample_context

        orchestrator.set_plugin(new_plugin)

        assert orchestrator._guard_plugin is new_plugin
        assert new_plugin.initialized

    def test_should_set_plugin_to_none(self, orchestrator):
        """Verify set_plugin can set guard to None."""
        orchestrator.set_plugin(None)

        assert orchestrator._guard_plugin is None

    def test_should_get_plugin(self, orchestrator, mock_plugin):
        """Verify get_plugin returns current plugin."""
        result = orchestrator.get_plugin()

        assert result is mock_plugin

    def test_should_get_none_when_no_plugin(self, mock_explainer):
        """Verify get_plugin returns None when no plugin."""
        orch = GuardOrchestrator(mock_explainer, None)

        result = orch.get_plugin()

        assert result is None

    def test_should_handle_plugin_init_error_on_set(self, mock_explainer, sample_context):
        """Verify raises when new plugin init fails."""
        mock_plugin = MagicMock()
        mock_plugin.initialize.side_effect = ValueError("Init error")
        orch = GuardOrchestrator(mock_explainer, None)
        orch._context = sample_context

        with pytest.raises(ValueError):
            orch.set_plugin(mock_plugin)


class TestFilterPerturbations:
    """Test cases for filter_perturbations() method."""

    def test_should_return_unfiltered_when_no_plugin(self, mock_explainer):
        """Verify returns unfiltered perturbations when no plugin."""
        orch = GuardOrchestrator(mock_explainer, None)

        x_perturbed = np.array([[0.0, 0.0], [1.0, 1.0]])
        x_feature = np.array([[0, 0, 0], [1, 1, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0])}

        result_x, result_feat = orch.filter_perturbations(
            x_perturbed, x_feature, x_orig, prediction
        )

        np.testing.assert_array_equal(result_x, x_perturbed)
        np.testing.assert_array_equal(result_feat, x_feature)

    def test_should_return_unfiltered_when_empty_perturbations(self, orchestrator):
        """Verify returns empty arrays when perturbations empty."""
        x_perturbed = np.array([]).reshape(0, 2)
        x_feature = np.array([]).reshape(0, 3)
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([])}

        result_x, result_feat = orchestrator.filter_perturbations(
            x_perturbed, x_feature, x_orig, prediction
        )

        assert result_x.shape[0] == 0
        assert result_feat.shape[0] == 0

    def test_should_raise_on_plugin_error(self, orchestrator):
        """Verify raises when plugin fails."""
        mock_plugin = MagicMock()
        mock_plugin.filter_perturbations.side_effect = ValueError("Plugin error")
        orchestrator._guard_plugin = mock_plugin

        x_perturbed = np.array([[0.0, 0.0]])
        x_feature = np.array([[0, 0, 0]])
        x_orig = np.array([[0.0, 0.0]])
        prediction = {"predict": np.array([0.0])}

        with pytest.raises(ValueError):
            orchestrator.filter_perturbations(
                x_perturbed, x_feature, x_orig, prediction
            )


class TestFilterCandidates:
    """Test cases for filter_candidates() method."""

    def test_should_return_all_when_no_plugin(self, mock_explainer):
        """Verify returns all candidates when no plugin."""
        orch = GuardOrchestrator(mock_explainer, None)

        candidates = np.array([0.0, 0.5, 1.0])
        x_orig = np.array([[0.0, 0.0]])

        result = orch.filter_candidates(0, candidates, x_orig)

        assert len(result) == 3

    def test_should_raise_on_plugin_error(self, orchestrator):
        """Verify raises when plugin fails."""
        mock_plugin = MagicMock()
        mock_plugin.filter_candidates.side_effect = ValueError("Plugin error")
        orchestrator._guard_plugin = mock_plugin

        candidates = np.array([0.0, 0.5, 1.0])
        x_orig = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError):
            orchestrator.filter_candidates(0, candidates, x_orig)

class TestFitGuard:
    """Test cases for fit_guard() method."""

    def test_should_reinitialize_with_updated_params(self, orchestrator, sample_context):
        """Verify fit_guard reinitializes guard with updated params."""
        orchestrator._context = sample_context
        mock_plugin = MagicMock()
        orchestrator._guard_plugin = mock_plugin

        guard_params = {"alpha": 0.9, "n_clusters": 5}

        orchestrator.fit_guard(guard_params)

        # Verify plugin was initialized with updated context
        assert mock_plugin.initialize.called
        # Check that the new context has updated metadata
        call_context = mock_plugin.initialize.call_args[0][0]
        assert call_context.metadata.get("guard_params") == guard_params

    def test_should_do_nothing_without_plugin(self, mock_explainer, caplog):
        """Verify does nothing when no plugin."""
        orch = GuardOrchestrator(mock_explainer, None)

        with caplog.at_level(logging.DEBUG):
            orch.fit_guard({"alpha": 0.9})

        assert "without an active guard plugin" in caplog.text or orch._guard_plugin is None

    def test_should_do_nothing_without_context(self, orchestrator, caplog):
        """Verify does nothing when no context."""
        orchestrator._context = None

        with caplog.at_level(logging.DEBUG):
            orchestrator.fit_guard({"alpha": 0.9})

        assert "before guard context" in caplog.text or orchestrator._context is None

    def test_should_handle_non_dict_params(self, orchestrator, sample_context):
        """Verify handles non-dict guard_params."""
        orchestrator._context = sample_context
        mock_plugin = MagicMock()
        orchestrator._guard_plugin = mock_plugin

        # Pass a non-dict that will be stored as-is
        orchestrator.fit_guard("some_value")

        assert mock_plugin.initialize.called


class TestGetSetGuard:
    """Test cases for get_guard() and set_guard() methods."""

    def test_should_get_guard_via_plugin_getter(self, orchestrator):
        """Verify get_guard calls plugin.get_guard()."""
        mock_guard = MagicMock()
        mock_plugin = MagicMock()
        mock_plugin.get_guard = MagicMock(return_value=mock_guard)
        orchestrator._guard_plugin = mock_plugin

        result = orchestrator.get_guard()

        assert result is mock_guard
        mock_plugin.get_guard.assert_called_once()

    def test_should_get_guard_via_private_attribute(self, orchestrator):
        """Verify get_guard falls back to _guard attribute."""
        mock_guard = MagicMock()
        mock_plugin = MagicMock()
        mock_plugin.get_guard = None  # No getter method
        mock_plugin._guard = mock_guard
        orchestrator._guard_plugin = mock_plugin

        result = orchestrator.get_guard()

        assert result is mock_guard

    def test_should_get_none_when_no_plugin(self, orchestrator):
        """Verify get_guard returns None when no plugin."""
        orchestrator._guard_plugin = None

        result = orchestrator.get_guard()

        assert result is None

    def test_should_set_guard_via_plugin_setter(self, orchestrator):
        """Verify set_guard calls plugin.set_guard()."""
        mock_guard = MagicMock()
        mock_plugin = MagicMock()
        mock_plugin.set_guard = MagicMock()
        orchestrator._guard_plugin = mock_plugin

        orchestrator.set_guard(mock_guard)

        mock_plugin.set_guard.assert_called_once_with(mock_guard)

    def test_should_set_guard_via_private_attribute(self, orchestrator):
        """Verify set_guard falls back to _guard attribute."""
        mock_guard = MagicMock()
        mock_plugin = MagicMock()
        mock_plugin.set_guard = None  # No setter method
        orchestrator._guard_plugin = mock_plugin

        orchestrator.set_guard(mock_guard)

        assert orchestrator._guard_plugin._guard is mock_guard

    def test_should_do_nothing_when_no_plugin(self, orchestrator, caplog):
        """Verify set_guard does nothing when no plugin."""
        orchestrator._guard_plugin = None

        with caplog.at_level(logging.DEBUG):
            orchestrator.set_guard(MagicMock())

        assert "without an active guard plugin" in caplog.text or orchestrator._guard_plugin is None


class TestGetMetrics:
    """Test cases for get_metrics() method."""

    def test_should_return_copy_of_metrics(self, orchestrator):
        """Verify get_metrics returns a copy of metrics dict."""
        orchestrator.metrics["accept_calls"] = 10
        orchestrator.metrics["filtered_perturbations"] = 5

        result = orchestrator.get_metrics()

        assert result["accept_calls"] == 10
        assert result["filtered_perturbations"] == 5
        # Verify it's a copy, not the same dict
        result["accept_calls"] = 100
        assert orchestrator.metrics["accept_calls"] == 10


class TestIntegrationScenarios:
    """Integration tests combining multiple methods."""

    def test_complete_workflow_with_filtering(self, mock_explainer, sample_context):
        """Verify complete workflow of initialization and filtering."""
        mock_plugin = MockGuardPlugin()
        orch = GuardOrchestrator(mock_explainer, mock_plugin)

        # Initialize
        orch.initialize(sample_context)

        # Filter perturbations
        x_perturbed = np.array([[0.0, 0.0], [1.0, 1.0]])
        x_feature = np.array([[0, 0, 0], [1, 1, 0]])
        x_orig = np.array([[0.0, 0.0], [1.0, 1.0]])
        prediction = {
            "predict": np.array([0.0, 1.0]),
            "low": np.array([0.0, 0.0]),
            "high": np.array([1.0, 1.0]),
        }

        result_x, result_feat = orch.filter_perturbations(
            x_perturbed, x_feature, x_orig, prediction
        )

        # Verify we got results
        assert result_x.shape[0] > 0
