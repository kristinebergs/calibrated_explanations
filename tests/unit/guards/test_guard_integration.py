"""Integration tests for Guard Plugin Architecture with CalibratedExplainer.

Tests verify that:
1. Guard can be configured via guard_params kwarg passed to CalibratedExplainer
2. Guard plugin is properly initialized through PluginManager during explainer init
3. Guard is optional (explainer works without guard_params)
4. Guard filtering is accessible through ExplanationContext during explanation

Note: Direct `.guard` property and `.set_guard()` method were removed in Phase 7.
Guard state is now managed exclusively through the plugin system.
Ref: improvement_docs/ignore/guards/IMPLEMENTATION_CHECKLIST.md Phase 7-8
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.core.explain.guards.regions import ConformalRegionOracle


class TestGuardPluginIntegration:
    """Test guard plugin integration with CalibratedExplainer.

    These tests verify that the guard plugin system works correctly.
    They focus on user-facing behavior rather than internal implementation.
    Per test guidelines (TEST_GUIDELINES_ENHANCED.md), tests should verify:
    - Domain invariants (e.g., "guard filtering reduces candidate set")
    - Behavior contracts (e.g., "explainer works with/without guard")
    - Not internal mechanisms (e.g., private property access)
    """

    @pytest.fixture(scope="class")
    def classification_data(self):
        """Generate classification dataset."""
        x_data, y_data = make_classification(
            n_samples=100, n_features=5, n_informative=3, random_state=42
        )
        split_idx = 50
        x_train, x_cal = x_data[:split_idx], x_data[split_idx:]
        y_train, y_cal = y_data[:split_idx], y_data[split_idx:]
        return x_train, x_cal, y_train, y_cal

    @pytest.fixture(scope="class")
    def fitted_classifier(self, classification_data):
        """Fit a classifier on training data."""
        x_train, _, y_train, _ = classification_data
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(x_train, y_train)
        return clf

    @pytest.fixture(scope="class")
    def multiclass_data(self):
        """Generate multiclass classification dataset."""
        x_data, y_data = make_classification(
            n_samples=100, n_features=5, n_informative=3, n_classes=3, random_state=42
        )
        split_idx = 50
        x_train, x_cal = x_data[:split_idx], x_data[split_idx:]
        y_train, y_cal = y_data[:split_idx], y_data[split_idx:]
        return x_train, x_cal, y_train, y_cal

    @pytest.fixture(scope="class")
    def fitted_multiclass_classifier(self, multiclass_data):
        """Fit a multiclass classifier on training data."""
        x_train, _, y_train, _ = multiclass_data
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(x_train, y_train)
        return clf

    @pytest.fixture(scope="class")
    def regression_data(self):
        """Generate regression dataset."""
        x_data, y_data = make_regression(
            n_samples=100, n_features=5, noise=0.1, random_state=42
        )
        split_idx = 50
        x_train, x_cal = x_data[:split_idx], x_data[split_idx:]
        y_train, y_cal = y_data[:split_idx], y_data[split_idx:]
        return x_train, x_cal, y_train, y_cal

    @pytest.fixture(scope="class")
    def fitted_regressor(self, regression_data):
        """Fit a regressor on training data."""
        x_train, _, y_train, _ = regression_data
        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(x_train, y_train)
        return reg

    def test_explainer_without_guard__should_initialize_successfully(
        self, classification_data, fitted_classifier
    ):
        """Verify that explainer works without explicit guard_params.

        Domain Rule: Guard is optional; explanations should be generated
        without explicit guard filtering if no guard_params provided.
        By default, guard is initialized if guard_enabled=True.
        """
        _, x_cal, _, y_cal = classification_data

        # Should not raise TypeError about missing guard_params
        explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
        )

        # Guard orchestrator should exist
        guard_orch = explainer._plugin_manager.guard_orchestrator
        assert guard_orch is not None
        # By default, if guard_enabled=True, guard plugin should be initialized
        # (unless explicitly disabled)

    def test_explainer_with_guard_params__should_initialize_guard_plugin(
        self, classification_data, fitted_classifier
    ):
        """Verify that guard_params are passed to PluginManager and guard is initialized.

        Domain Rule: When guard_params provided, the plugin system should:
        1. Accept the parameters
        2. Initialize guard plugin with those parameters
        3. Make guard available through plugin manager
        """
        _, x_cal, _, y_cal = classification_data

        explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Guard orchestrator should be initialized
        guard_orch = explainer._plugin_manager.guard_orchestrator
        assert guard_orch is not None

        # Guard plugin should be configured (not None)
        assert guard_orch._guard_plugin is not None

    def test_explainer_explain_without_guard__should_generate_explanations(
        self, classification_data, fitted_classifier
    ):
        """Verify that explanation generation works without guard.

        Behavior: Explanations should be generated successfully
        whether or not a guard is configured.
        """
        _, x_cal, _, y_cal = classification_data
        x_test = x_cal[:1]

        explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
        )

        # Should not raise
        explanation = explainer.explain_factual(x_test)
        assert explanation is not None
        assert len(explanation) > 0

    def test_explainer_explain_with_guard_params__should_generate_explanations(
        self, classification_data, fitted_classifier
    ):
        """Verify that explanation generation works with guard configured.

        Behavior: Guard configuration should not break explanation pipeline;
        explanations should be generated with guard filtering active.
        """
        _, x_cal, _, y_cal = classification_data
        x_test = x_cal[:1]

        explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Should not raise
        explanation = explainer.explain_factual(x_test)
        assert explanation is not None
        assert len(explanation) > 0

    def test_explainer_with_guard_params_multiclass__should_initialize_guard_plugin(
        self, multiclass_data, fitted_multiclass_classifier
    ):
        """Verify that guard_params work for multiclass classification."""
        _, x_cal, _, y_cal = multiclass_data

        explainer = CalibratedExplainer(
            learner=fitted_multiclass_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 5},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Guard orchestrator should be initialized
        guard_orch = explainer._plugin_manager.guard_orchestrator
        assert guard_orch is not None

        # Guard plugin should be configured (not None)
        assert guard_orch._guard_plugin is not None

    def test_explainer_explain_with_guard_params_multiclass__should_generate_explanations(
        self, multiclass_data, fitted_multiclass_classifier
    ):
        """Verify that explanation generation works with guard for multiclass."""
        _, x_cal, _, y_cal = multiclass_data
        x_test = x_cal[:1]

        explainer = CalibratedExplainer(
            learner=fitted_multiclass_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 5},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Should not raise
        explanation = explainer.explain_factual(x_test)
        assert explanation is not None
        assert len(explanation) > 0

    def test_explainer_with_guard_params_regression__should_initialize_guard_plugin(
        self, regression_data, fitted_regressor
    ):
        """Verify that guard_params work for regression."""
        _, x_cal, _, y_cal = regression_data

        explainer = CalibratedExplainer(
            learner=fitted_regressor,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="regression",
            guard_params={"alpha": 0.1, "n_clusters": 5},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Guard orchestrator should be initialized
        guard_orch = explainer._plugin_manager.guard_orchestrator
        assert guard_orch is not None

        # Guard plugin should be configured (not None)
        assert guard_orch._guard_plugin is not None

    def test_explainer_explain_with_guard_params_regression__should_generate_explanations(
        self, regression_data, fitted_regressor
    ):
        """Verify that explanation generation works with guard for regression."""
        _, x_cal, _, y_cal = regression_data
        x_test = x_cal[:1]

        explainer = CalibratedExplainer(
            learner=fitted_regressor,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="regression",
            guard_params={"alpha": 0.1, "n_clusters": 5},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Should not raise
        explanation = explainer.explain_factual(x_test)
        assert explanation is not None
        assert len(explanation) > 0

    def test_guard_orchestrator__should_be_accessible_from_plugin_manager(
        self, classification_data, fitted_classifier
    ):
        """Verify that guard orchestrator is accessible through PluginManager.

        Behavior: After explainer initialization, guard orchestrator should be
        available via plugin manager for testing/diagnostics.
        """
        _, x_cal, _, y_cal = classification_data

        explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Guard orchestrator should be accessible
        guard_orch = explainer._plugin_manager.guard_orchestrator
        assert guard_orch is not None

        # Should have filter_perturbations method (guard plugin protocol)
        assert hasattr(guard_orch, "filter_perturbations")
        assert callable(guard_orch.filter_perturbations)

    def test_oracle__should_have_intervals_method(self, classification_data, fitted_classifier):
        """Verify that ConformalRegionOracle.intervals() works.

        Behavior: Guard oracle should have intervals() method that returns
        candidate intervals for a given instance.
        """
        from calibrated_explanations.core.explain.guards.interval_learner_adapter import (
            IntervalLearnerAdapter,
        )

        x_train, _, y_train, _ = classification_data

        # Create oracle and fit it
        temp_explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_train,
            y_cal=y_train,
            mode="classification",
        )

        wrapped_learner = IntervalLearnerAdapter(temp_explainer.interval_learner)
        oracle = ConformalRegionOracle(alpha=0.1, n_clusters=3)
        oracle.fit(x_train, y_train, interval_learner=wrapped_learner)

        # Test that oracle can return intervals
        x_test = x_train[:1]
        calibrated_pred = (0.5, (0.0, 1.0))
        intervals = oracle.intervals(x_test[0], calibrated_prediction=calibrated_pred)

        # Should return list of intervals
        assert isinstance(intervals, list)
        assert len(intervals) > 0

    def test_explainer_with_guard_params_multiclass__should_initialize_guard_plugin(
        self, multiclass_data, fitted_multiclass_classifier
    ):
        """Verify that guard_params work for multiclass classification.

        Domain Rule: Guard should initialize correctly for multiclass tasks.
        """
        _, x_cal, _, y_cal = multiclass_data

        explainer = CalibratedExplainer(
            learner=fitted_multiclass_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Guard orchestrator should be initialized
        guard_orch = explainer._plugin_manager.guard_orchestrator
        assert guard_orch is not None

    def test_explainer_explain_with_guard_params_multiclass__should_generate_explanations(
        self, multiclass_data, fitted_multiclass_classifier
    ):
        """Verify that explanation generation works with guard for multiclass.

        Behavior: Guard configuration should work for multiclass classification.
        """
        _, x_cal, _, y_cal = multiclass_data
        x_test = x_cal[:1]

        explainer = CalibratedExplainer(
            learner=fitted_multiclass_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Should not raise
        explanation = explainer.explain_factual(x_test)
        assert explanation is not None
        assert len(explanation) > 0

    def test_explainer_with_guard_params_regression__should_initialize_guard_plugin(
        self, regression_data, fitted_regressor
    ):
        """Verify that guard_params work for regression.

        Domain Rule: Guard should initialize correctly for regression tasks.
        """
        _, x_cal, _, y_cal = regression_data

        explainer = CalibratedExplainer(
            learner=fitted_regressor,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="regression",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Guard orchestrator should be initialized
        guard_orch = explainer._plugin_manager.guard_orchestrator
        assert guard_orch is not None

    def test_explainer_explain_with_guard_params_regression__should_generate_explanations(
        self, regression_data, fitted_regressor
    ):
        """Verify that explanation generation works with guard for regression.

        Behavior: Guard configuration should work for regression.
        """
        _, x_cal, _, y_cal = regression_data
        x_test = x_cal[:1]

        explainer = CalibratedExplainer(
            learner=fitted_regressor,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="regression",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )
        assert explainer.guard is not None and explainer.guard._fitted

        # Should not raise
        explanation = explainer.explain_factual(x_test)
        assert explanation is not None
        assert len(explanation) > 0

        # Should not raise
        explanation = explainer.explain_factual(x_test, threshold=0.5)
        assert explanation is not None
        assert len(explanation) > 0


__all__ = [
    "TestGuardPluginIntegration",
]


class TestConformalRegionOracleMetrics:
    """Unit tests for ConformalRegionOracle distance metric handling.

    These tests focus on the nonconformity_metric configuration and
    basic behavioral differences between supported metrics.
    """

    def test_should_raise_when_nonconformity_metric_invalid(self):
        """Constructor should validate nonconformity_metric choices.

        Guardrails: invalid metric names must fail fast with ValueError.
        """

        with pytest.raises(ValueError):
            ConformalRegionOracle(nonconformity_metric="invalid-metric")
