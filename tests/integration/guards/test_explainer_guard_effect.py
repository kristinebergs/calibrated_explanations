import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from calibrated_explanations import CalibratedExplainer


@pytest.mark.integration
def test_guard_reduces_candidate_rules():
    """Verify that a fitted ConformalRegionOracle guard reduces or equals the
    number of rules produced by explain_factual compared to no-guard.

    The test uses a deterministic small dataset and a strict guard (higher
    alpha, no relaxation) to make conformal regions tighter and thus filter
    more perturbations.
    """
    rng = np.random.RandomState(42)
    x_data, y_data = make_classification(
        n_samples=120, n_features=6, n_informative=4, random_state=42
    )
    split = 80
    x_train, x_cal = x_data[:split], x_data[split:]
    y_train, y_cal = y_data[:split], y_data[split:]

    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(x_train, y_train)

    # Choose a single test instance from calibration holdout
    x_test = x_cal[:1]

    # Explainer without guard
    expl_no_guard = CalibratedExplainer(
        learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
    )
    expl_no_guard.set_seed(42)
    exp_no = expl_no_guard.explain_factual(x_test)
    # Compute rules count for first instance
    rules_no = len(exp_no[0])

    # Explainer with a strict guard (higher alpha -> smaller conformal radii)
    guard_params = {"alpha": 0.5, "n_clusters": 3, "relaxation_factor": 0.0}
    expl_guard = CalibratedExplainer(
        learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification", guard_params=guard_params
    )
    expl_guard.set_seed(42)
    exp_guard = expl_guard.explain_factual(x_test)
    rules_guard = len(exp_guard[0])

    # Guard should not increase the number of rules; typically it will reduce it.
    # This test is intentionally conservative: require non-increase (<=).
    # A strict reduction (<) can be flaky on small synthetic data/configs,
    # so we avoid asserting strict inequality here.
    assert rules_guard <= rules_no
