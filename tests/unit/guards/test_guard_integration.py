"""Integration tests for ConformalRegionOracle with CalibratedExplainer.

Tests verify that:
1. Guard can be initialized via guard_params in CalibratedExplainer constructor
2. Guard is fitted automatically during explainer initialization
3. Guard is optional (explainer works without guard_params)
4. Guard.set_guard() allows replacing guard after initialization
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression

from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.guards import ConformalRegionOracle
from calibrated_explanations.core.exceptions import NotFittedError


class TestGuardIntegration:
    """Test guard integration with CalibratedExplainer."""

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
    def fitted_explainer_no_guard(self, classification_data, fitted_classifier):
        """Create a fitted explainer without guard."""
        _, x_cal, _, y_cal = classification_data
        explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
        )
        return explainer

    @pytest.fixture(scope="class")
    def fitted_oracle(self, classification_data, fitted_classifier):
        """Create and fit a ConformalRegionOracle (for manual testing)."""
        from calibrated_explanations.guards.interval_learner_adapter import IntervalLearnerAdapter
        
        x_train, _, y_train, _ = classification_data

        # Create a temporary explainer to get the interval_learner
        temp_explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_train,
            y_cal=y_train,
            mode="classification",
        )

        # Wrap the interval learner to support uq_interval parameter
        wrapped_learner = IntervalLearnerAdapter(temp_explainer.interval_learner)

        oracle = ConformalRegionOracle(alpha=0.1, n_clusters=3)
        oracle.fit(x_train, y_train, interval_learner=wrapped_learner)
        return oracle

    def test_no_guard_by_default(self, fitted_explainer_no_guard):
        """Test that guard is None when guard_params not provided."""
        assert fitted_explainer_no_guard.guard is None

    def test_guard_params_creates_fitted_guard(self, classification_data, fitted_classifier):
        """Test that guard_params creates and fits a guard automatically."""
        _, x_cal, _, y_cal = classification_data

        explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )

        # Guard should be created and fitted
        assert explainer.guard is not None

    def test_set_guard_with_fitted_oracle(self, fitted_explainer_no_guard, fitted_oracle):
        """Test set_guard() method with a fitted oracle."""
        fitted_explainer_no_guard.set_guard(fitted_oracle)
        assert fitted_explainer_no_guard.guard is fitted_oracle

    def test_set_guard_unfitted_raises_error(self, fitted_explainer_no_guard):
        """Test that set_guard() raises error for unfitted oracle."""
        unfitted_oracle = ConformalRegionOracle(alpha=0.1)
        with pytest.raises(NotFittedError):
            fitted_explainer_no_guard.set_guard(unfitted_oracle)

    def test_set_guard_none_clears_guard(self, fitted_explainer_no_guard, fitted_oracle):
        """Test that set_guard(None) clears the guard."""
        fitted_explainer_no_guard.set_guard(fitted_oracle)
        assert fitted_explainer_no_guard.guard is fitted_oracle

        fitted_explainer_no_guard.set_guard(None)
        assert fitted_explainer_no_guard.guard is None

    def test_explain_without_guard(self, fitted_explainer_no_guard, classification_data):
        """Test that explainer works without guard."""
        _, x_cal, _, _ = classification_data
        x_test = x_cal[:1]

        # Should not raise
        explanation = fitted_explainer_no_guard.explain_factual(x_test)
        assert explanation is not None

    def test_explain_with_guard_params(self, classification_data, fitted_classifier):
        """Test that explainer works with guard_params."""
        _, x_cal, _, y_cal = classification_data

        explainer = CalibratedExplainer(
            learner=fitted_classifier,
            x_cal=x_cal,
            y_cal=y_cal,
            mode="classification",
            guard_params={"alpha": 0.1, "n_clusters": 3},
        )

        x_test = x_cal[:1]
        # Should not raise
        explanation = explainer.explain_factual(x_test)
        assert explanation is not None

    def test_guard_accept_method_callable(self, fitted_oracle, classification_data):
        """Test that oracle.accept() is callable with calibrated prediction."""
        x_train, _, _, _ = classification_data
        x_test = x_train[:1]

        # Must provide calibrated_prediction (mandatory parameter)
        calibrated_pred = (0.5, (0.0, 1.0))
        result = fitted_oracle.accept(x_test[0], calibrated_prediction=calibrated_pred)
        assert isinstance(result, (bool, np.bool_))

    def test_guard_intervals_method_callable(self, fitted_oracle, classification_data):
        """Test that oracle.intervals() is callable with calibrated prediction."""
        x_train, _, _, _ = classification_data
        x_test = x_train[:1]

        # Must provide calibrated_prediction (mandatory parameter)
        calibrated_pred = (0.5, (0.0, 1.0))
        # Should return intervals
        intervals = fitted_oracle.intervals(x_test[0], calibrated_prediction=calibrated_pred)
        assert isinstance(intervals, list)
        assert len(intervals) > 0


__all__ = [
    "TestGuardIntegration",
]

