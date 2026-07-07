"""Capability contract tests for factual and alternative explanations.

Requirements verified:
  CE-REQ-EXPL-API-001     — Factual explanation API contract (CE-CAP-EXPL-001)
  CE-REQ-EXPL-RETURN-001  — Factual explanation return contract (CE-CAP-EXPL-001)
  CE-REQ-EXPL-API-002     — Alternative explanation API contract (CE-CAP-EXPL-002)
  CE-REQ-EXPL-ALT-RETURN-001 — Alternative explanation return contract (CE-CAP-EXPL-002)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove statistical validity, coverage guarantees, or calibration accuracy.
See the requirement files under development/capabilities/requirements/ for the full
assumption boundary.
"""

from __future__ import annotations

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from tif_explanation import run_factual_tif_scenario, run_alternative_tif_scenario

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@pytest.fixture(scope="module")
def binary_classification_explainer():
    """Return a fitted and calibrated WrapCalibratedExplainer for binary classification.

    Uses a deterministic synthetic dataset. Splits into proper-train / calibration / test.
    """
    X, y = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=_RNG_SEED,
    )
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X, y, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestClassifier(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal)
    return explainer, X_test, y_test


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-API-001 — Factual explanation API contract
# ---------------------------------------------------------------------------


def test_should_produce_factual_explanations_when_fitted_and_calibrated(
    binary_classification_explainer,
):
    """Verify CE-REQ-EXPL-API-001: explain_factual returns a non-empty, indexable result.

    Acceptance criteria (from CE-REQ-EXPL-API-001):
    - explain_factual(X_test) completes without error.
    - len(result) == len(X_test).
    - result[0] is not None.
    """
    explainer, X_test, _ = binary_classification_explainer

    result = explainer.explain_factual(X_test)

    assert result is not None, "explain_factual must return a non-None object"
    assert len(result) == len(
        X_test
    ), f"CE-REQ-EXPL-API-001: len(result)={len(result)} != len(X_test)={len(X_test)}"
    assert result[0] is not None, "CE-REQ-EXPL-API-001: result[0] must not be None"


def test_should_produce_factual_explanations_for_each_instance(
    binary_classification_explainer,
):
    """Verify CE-REQ-EXPL-API-001: every instance in X_test yields an explanation.

    Checks the per-instance indexing contract for all test rows.
    """
    explainer, X_test, _ = binary_classification_explainer

    result = explainer.explain_factual(X_test)

    for i in range(len(X_test)):
        assert result[i] is not None, f"CE-REQ-EXPL-API-001: result[{i}] must not be None"


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-API-002 — Alternative explanation API contract
# ---------------------------------------------------------------------------


def test_should_produce_alternative_explanations_when_fitted_and_calibrated(
    binary_classification_explainer,
):
    """Verify CE-REQ-EXPL-API-002: explore_alternatives returns a non-empty, indexable result.

    Acceptance criteria (from CE-REQ-EXPL-API-002):
    - explore_alternatives(X_test) completes without error.
    - len(result) == len(X_test).
    - result[0] is not None.
    """
    explainer, X_test, _ = binary_classification_explainer

    result = explainer.explore_alternatives(X_test)

    assert result is not None, "explore_alternatives must return a non-None object"
    assert len(result) == len(
        X_test
    ), f"CE-REQ-EXPL-API-002: len(result)={len(result)} != len(X_test)={len(X_test)}"
    assert result[0] is not None, "CE-REQ-EXPL-API-002: result[0] must not be None"


def test_should_produce_alternative_explanations_for_each_instance(
    binary_classification_explainer,
):
    """Verify CE-REQ-EXPL-API-002: every instance in X_test yields an alternative explanation.

    Checks the per-instance indexing contract for all test rows.
    """
    explainer, X_test, _ = binary_classification_explainer

    result = explainer.explore_alternatives(X_test)

    for i in range(len(X_test)):
        assert result[i] is not None, f"CE-REQ-EXPL-API-002: result[{i}] must not be None"


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-RETURN-001 — Factual explanation return contract (via TIF)
# ---------------------------------------------------------------------------


def test_should_preserve_cardinality_when_factual_explain():
    """Verify CE-REQ-EXPL-RETURN-001: explain_factual returns len(result) == len(X_test).

    Acceptance criteria (from CE-REQ-EXPL-RETURN-001):
    - result is not None.
    - len(result) == n_instances (cardinality preserved).
    - result[0] is not None.

    Uses CE-TIF-EXPL-001 scenario: run_factual_tif_scenario().
    """
    obs = run_factual_tif_scenario()

    assert (
        not obs.exception_raised
    ), f"CE-REQ-EXPL-RETURN-001: explain_factual raised {obs.exception_type}"
    assert not obs.result_is_none, "CE-REQ-EXPL-RETURN-001: result must not be None"
    assert (
        obs.result_len == obs.n_instances
    ), f"CE-REQ-EXPL-RETURN-001: len(result)={obs.result_len} != n_instances={obs.n_instances}"
    assert not obs.first_item_is_none, "CE-REQ-EXPL-RETURN-001: result[0] must not be None"


def test_should_return_accessible_feature_weights_when_factual_explain():
    """Verify CE-REQ-EXPL-RETURN-001: result[0].feature_weights is accessible and non-None.

    Acceptance criteria (from CE-REQ-EXPL-RETURN-001):
    - result[0].feature_weights is not None.

    Uses CE-TIF-EXPL-001 scenario: run_factual_tif_scenario().
    """
    obs = run_factual_tif_scenario()

    assert (
        not obs.exception_raised
    ), f"CE-REQ-EXPL-RETURN-001: explain_factual raised {obs.exception_type}"
    assert (
        obs.feature_weights_accessible
    ), "CE-REQ-EXPL-RETURN-001: result[0].feature_weights must be accessible and non-None"


# ---------------------------------------------------------------------------
# CE-REQ-EXPL-ALT-RETURN-001 — Alternative explanation return contract (via TIF)
# ---------------------------------------------------------------------------


def test_should_preserve_cardinality_when_alternative_explain():
    """Verify CE-REQ-EXPL-ALT-RETURN-001: explore_alternatives returns len(result) == len(X_test).

    Acceptance criteria (from CE-REQ-EXPL-ALT-RETURN-001):
    - result is not None.
    - len(result) == n_instances (cardinality preserved).
    - result[0] is not None.

    Uses CE-TIF-EXPL-001 scenario: run_alternative_tif_scenario().
    """
    obs = run_alternative_tif_scenario()

    assert (
        not obs.exception_raised
    ), f"CE-REQ-EXPL-ALT-RETURN-001: explore_alternatives raised {obs.exception_type}"
    assert not obs.result_is_none, "CE-REQ-EXPL-ALT-RETURN-001: result must not be None"
    assert (
        obs.result_len == obs.n_instances
    ), f"CE-REQ-EXPL-ALT-RETURN-001: len(result)={obs.result_len} != n_instances={obs.n_instances}"
    assert not obs.first_item_is_none, "CE-REQ-EXPL-ALT-RETURN-001: result[0] must not be None"


def test_should_return_alternative_explanations_type_when_explore_alternatives():
    """Verify CE-REQ-EXPL-ALT-RETURN-001: result type is AlternativeExplanations.

    Acceptance criteria (from CE-REQ-EXPL-ALT-RETURN-001):
    - type(result).__name__ == "AlternativeExplanations".

    Uses CE-TIF-EXPL-001 scenario: run_alternative_tif_scenario().
    """
    obs = run_alternative_tif_scenario()

    assert (
        not obs.exception_raised
    ), f"CE-REQ-EXPL-ALT-RETURN-001: explore_alternatives raised {obs.exception_type}"
    assert obs.result_type_name == "AlternativeExplanations", (
        f"CE-REQ-EXPL-ALT-RETURN-001: expected type 'AlternativeExplanations', "
        f"got '{obs.result_type_name}'"
    )
