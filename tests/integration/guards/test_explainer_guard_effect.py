"""Test guard filtering effects on explanation output.

Verifies that:
- Guard params are accepted by explainer
- Explanations are generated successfully with guards
- Guard configuration doesn't break the explanation pipeline

Note: This test focuses on behavioral effects of guard configuration,
not on specific internal guard mechanics.
"""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from calibrated_explanations import CalibratedExplainer


@pytest.mark.integration
def test_explainer_with_guard_params__should_generate_valid_explanations():
    """Verify that explainer with guard params generates valid explanations.

    Behavior: Guard configuration should not prevent explanation generation.
    Explanations should be valid regardless of guard settings.
    """
    # Use a larger dataset for stability
    x_data, y_data = make_classification(
        n_samples=1000, n_features=6, n_informative=4, random_state=42
    )
    split = 800
    x_train, x_cal = x_data[:split], x_data[split:]
    y_train, y_cal = y_data[:split], y_data[split:]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    # Test instances from calibration holdout
    x_test = x_cal[:10]

    # Explainer without guard
    expl_no_guard = CalibratedExplainer(
        learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
    )
    explanations_no_guard = expl_no_guard.explain_factual(x_test)

    # Should generate valid explanations
    assert explanations_no_guard is not None
    assert len(explanations_no_guard) == 10

    # Explainer with guard: using loose alpha so filtering is not too aggressive
    guard_params = {"alpha": 0.9, "n_clusters": 8}
    expl_with_guard = CalibratedExplainer(
        learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification", guard_params=guard_params
    )
    explanations_with_guard = expl_with_guard.explain_factual(x_test)

    # Should also generate valid explanations
    assert explanations_with_guard is not None
    assert len(explanations_with_guard) == 10

    # Both should have feature weights
    for exp in explanations_no_guard:
        assert hasattr(exp, "feature_weights")
        assert exp.feature_weights is not None

    for exp in explanations_with_guard:
        assert hasattr(exp, "feature_weights")
        assert exp.feature_weights is not None


@pytest.mark.integration
def test_guard_with_alternative_mode__should_generate_explanations():
    """Verify that guard works in alternative explanation mode.

    Behavior: Guard filtering should work across all explanation modes
    (factual, alternative, fast).
    """
    x_data, y_data = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=42
    )
    split = 150
    x_train, x_cal = x_data[:split], x_data[split:]
    y_train, y_cal = y_data[:split], y_data[split:]

    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(x_train, y_train)

    x_test = x_cal[:3]

    # Create explainer with guard
    explainer = CalibratedExplainer(
        learner=clf,
        x_cal=x_cal,
        y_cal=y_cal,
        mode="classification",
        guard_params={"alpha": 0.8, "n_clusters": 5},
    )

    # Alternative mode should also work with guard
    explanations = explainer.explore_alternatives(x_test)

    # Should generate valid explanations
    assert explanations is not None
    assert len(explanations) == 3


@pytest.mark.integration
def test_guard_with_fast_mode__should_generate_explanations():
    """Verify that guard works in fast explanation mode.

    Behavior: Guard filtering should work across all explanation modes
    (factual, alternative, fast).
    """
    x_data, y_data = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=42
    )
    split = 150
    x_train, x_cal = x_data[:split], x_data[split:]
    y_train, y_cal = y_data[:split], y_data[split:]

    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(x_train, y_train)

    x_test = x_cal[:3]

    # Create explainer with guard
    explainer = CalibratedExplainer(
        learner=clf,
        x_cal=x_cal,
        y_cal=y_cal,
        mode="classification",
        guard_params={"alpha": 0.8, "n_clusters": 5},
    )

    # Fast mode should also work with guard
    explanations = explainer.explain_fast(x_test)

    # Should generate valid explanations
    assert explanations is not None
    assert len(explanations) == 3

