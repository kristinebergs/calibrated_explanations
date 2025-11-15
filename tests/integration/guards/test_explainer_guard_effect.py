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
    # Use a larger dataset to make the conformal regions and classifier
    # behavior more stable and deterministic.
    x_data, y_data = make_classification(
        n_samples=1000, n_features=6, n_informative=4, random_state=42
    )
    split = 800
    x_train, x_cal = x_data[:split], x_data[split:]
    y_train, y_cal = y_data[:split], y_data[split:]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)

    # Choose several test instances from calibration holdout to make the
    # comparison more robust (aggregate rule counts across instances).
    x_test = x_cal[:10]

    # Explainer without guard
    expl_no_guard = CalibratedExplainer(
        learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification"
    )
    expl_no_guard.set_seed(42)
    exp_no = expl_no_guard.explain_factual(x_test)
    # Compute aggregate rules count across test instances
    rules_no = sum(len(r) for r in exp_no)

    # Explainer with a strict guard: higher alpha and more clusters make
    # conformal regions tighter and more local, increasing filtering power.
    guard_params = {"alpha": 0.9, "n_clusters": 8}
    expl_guard = CalibratedExplainer(
        learner=clf, x_cal=x_cal, y_cal=y_cal, mode="classification", guard_params=guard_params
    )
    expl_guard.set_seed(42)
    exp_guard = expl_guard.explain_factual(x_test)
    rules_guard = sum(len(r) for r in exp_guard)

    # Guard should not increase the number of rules; typically it will reduce it.
    # Empirically this is usually a reduction, but strict reduction can be
    # dataset- and config-dependent; require non-increase to keep the test
    # stable while the deterministic unit-level test verifies strict filtering
    # behavior on crafted OOD perturbations.
    assert rules_guard <= rules_no
