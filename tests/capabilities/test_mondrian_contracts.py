"""Capability contract tests for Mondrian conditional calibration.

Requirements verified:
  CE-REQ-MOND-API-001 — Mondrian calibration API contract (CE-CAP-MOND-001)

These tests verify the observable public-API behavior stated in those requirements.
They do not prove conditional validity guarantees within each Mondrian category.
See development/capabilities/requirements/CE-REQ-MOND-API-001.md for the full
assumption boundary.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer
from calibrated_explanations.utils.exceptions import (
    ConfigurationError,
    DataShapeError,
    ValidationError,
)

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@pytest.fixture
def mondrian_setup():
    """Return fitted wrapper, calibration/test data, and a simple Mondrian function."""
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

    # Mondrian categorizer: partition by sign of first feature (2 categories: 0, 1)
    def mondrian_fn(x):
        return (np.asarray(x)[:, 0] >= 0).astype(int)

    return explainer, X_cal, y_cal, X_test, y_test, mondrian_fn


@pytest.fixture
def regression_setup():
    """Return a fitted regression wrapper and deterministic Mondrian labels."""
    rng = np.random.default_rng(_RNG_SEED)
    X = rng.normal(size=(_N_SAMPLES, _N_FEATURES))
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(scale=0.1, size=_N_SAMPLES)
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X, y, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestRegressor(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)

    def mondrian_fn(x):
        return (np.asarray(x)[:, 0] >= 0).astype(int)

    return explainer, X_cal, y_cal, X_test, y_test, mondrian_fn


# ---------------------------------------------------------------------------
# CE-REQ-MOND-API-001 — Mondrian calibration API contract
# ---------------------------------------------------------------------------


def test_should_calibrate_when_mondrian_categorizer_provided(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-API-001: calibrate with mc= completes and reports calibrated=True.

    Acceptance criteria (from CE-REQ-MOND-API-001):
    - calibrate(X_cal, y_cal, mc=mondrian_fn) completes without error.
    - After calibration, wrapper.calibrated is True.
    """
    explainer, X_cal, y_cal, _X_test, _y_test, mondrian_fn = mondrian_setup

    explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)

    assert (
        explainer.calibrated is True
    ), "CE-REQ-MOND-API-001: explainer.calibrated must be True after Mondrian calibration"


@pytest.mark.parametrize("method_name", ["predict", "predict_proba", "explain_factual"])
def test_should_raise_validation_error_when_bins_calibrated_inference_omits_bins(
    mondrian_setup,
    method_name,
):
    """Verify CE-REQ-MOND-CONS-001: inline bins calibration requires inference bins."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal))

    with pytest.raises(ValidationError, match="calibrated with Mondrian bins") as exc_info:
        getattr(explainer, method_name)(X_test)

    assert exc_info.value.details["n_instances"] == len(X_test)


def test_should_raise_validation_error_when_regression_bins_calibrated_predict_omits_bins(
    regression_setup,
):
    """Verify CE-REQ-MOND-CONS-001 for regression percentile predictions."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = regression_setup
    explainer.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal))

    with pytest.raises(ValidationError, match="calibrated with Mondrian bins"):
        explainer.predict(X_test)


def test_should_raise_configuration_error_when_global_inference_receives_bins(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-CONS-001: global calibration cannot accept conditional bins."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal)

    with pytest.raises(ConfigurationError, match="not calibrated with Mondrian bins") as exc_info:
        explainer.predict_proba(X_test, bins=mondrian_fn(X_test))

    assert "conditional calibration required" in exc_info.value.details["requirement"]


def test_should_raise_configuration_error_when_mc_inference_receives_explicit_bins(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-CONS-001: mc-derived bins and explicit bins are exclusive."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)

    with pytest.raises(ConfigurationError, match="categorizer derives bins automatically"):
        explainer.predict_proba(X_test, bins=mondrian_fn(X_test))


def test_should_raise_validation_error_when_inference_bins_include_unknown_label(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-VAL-001: unseen Mondrian labels fail fast."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal))
    bins_test = mondrian_fn(X_test)
    bins_test[0] = 99

    with pytest.raises(ValidationError, match="not seen during calibration") as exc_info:
        explainer.predict_proba(X_test, bins=bins_test)

    assert exc_info.value.details["unknown_labels"] == [99]


def test_should_raise_data_shape_error_when_inference_bins_length_mismatches_samples(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-VAL-001: inference bins must align with x."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal))

    with pytest.raises(DataShapeError, match="length of Mondrian bins") as exc_info:
        explainer.predict_proba(X_test, bins=np.array([0]))

    assert exc_info.value.details == {"bins_length": 1, "n_samples": len(X_test)}


def test_should_raise_data_shape_error_when_calibration_bins_length_mismatches_samples(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-VAL-001: calibration bins must align with calibration data."""
    explainer, X_cal, y_cal, _X_test, _y_test, _mondrian_fn = mondrian_setup

    with pytest.raises(DataShapeError, match="length of Mondrian bins") as exc_info:
        explainer.calibrate(X_cal, y_cal, bins=np.array([0]))

    assert exc_info.value.details == {"bins_length": 1, "n_samples": len(X_cal)}


def test_should_raise_data_shape_error_when_mc_derives_wrong_length_bins(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-VAL-001 for mc-derived bins that do not match x."""
    explainer, X_cal, y_cal, _X_test, _y_test, _mondrian_fn = mondrian_setup

    def wrong_length_mc(x):
        return np.zeros(1, dtype=int)

    with pytest.raises(DataShapeError, match="length of Mondrian bins") as exc_info:
        explainer.calibrate(X_cal, y_cal, mc=wrong_length_mc)

    assert exc_info.value.details == {"bins_length": 1, "n_samples": len(X_cal)}


def test_should_raise_data_shape_error_when_core_explainer_bins_length_mismatches_samples(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-VAL-001 at the CalibratedExplainer public boundary."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal))

    with pytest.raises(DataShapeError, match="length of Mondrian bins"):
        explainer.explainer.explain_factual(X_test, bins=np.array([0]))


def test_should_reset_conditional_state_when_recalibrated_without_channel(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-LIFE-001: plain recalibration returns to global state."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)

    explainer.calibrate(X_cal, y_cal)
    predictions = explainer.predict_proba(X_test)

    assert explainer.mc is None
    assert explainer.explainer.is_mondrian() is False
    assert predictions.shape[0] == len(X_test)


def test_should_reuse_conditional_categorizer_when_requested(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-LIFE-001: reuse_conditional applies the stored categorizer."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)

    explainer.calibrate(X_cal, y_cal, reuse_conditional=True)
    predictions = explainer.predict_proba(X_test)

    assert explainer.mc is mondrian_fn
    assert explainer.explainer.is_mondrian() is True
    assert predictions.shape[0] == len(X_test)


def test_should_raise_validation_error_when_calibrate_receives_multiple_conditional_channels(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-LIFE-001: calibrate accepts one conditional channel."""
    explainer, X_cal, y_cal, _X_test, _y_test, mondrian_fn = mondrian_setup

    with pytest.raises(ValidationError, match="exactly one conditional calibration channel"):
        explainer.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal), mc=mondrian_fn)


def test_should_raise_validation_error_when_reuse_conditional_has_no_stored_mc(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-LIFE-001: inline bins cannot be reused implicitly."""
    explainer, X_cal, y_cal, _X_test, _y_test, _mondrian_fn = mondrian_setup

    with pytest.raises(ValidationError, match="requires a stored Mondrian categorizer"):
        explainer.calibrate(X_cal, y_cal, reuse_conditional=True)


def test_should_warn_and_require_explicit_bins_when_pickled_mc_wrapper_is_loaded(
    mondrian_setup,
    enable_fallbacks,
):
    """Verify CE-REQ-MOND-SER-001: pickle loudly drops mc and keeps bins semantics."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)

    with pytest.warns(UserWarning, match="drops the configured Mondrian categorizer"):
        restored = pickle.loads(pickle.dumps(explainer))

    with pytest.raises(ValidationError, match="calibrated with Mondrian bins"):
        restored.predict_proba(X_test)


def test_should_warn_and_require_explicit_bins_when_saved_mc_wrapper_is_loaded(
    mondrian_setup,
    tmp_path,
    enable_fallbacks,
):
    """Verify CE-REQ-MOND-SER-001 for save_state/load_state persistence."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, mc=mondrian_fn)
    state_path = tmp_path / "mondrian-state"

    with pytest.warns(UserWarning, match="drops the configured Mondrian categorizer"):
        explainer.save_state(state_path)
    restored = WrapCalibratedExplainer.load_state(state_path)

    with pytest.raises(ValidationError, match="calibrated with Mondrian bins"):
        restored.predict_proba(X_test)


def test_should_predict_when_inline_bins_match_calibration_vocabulary(
    mondrian_setup,
):
    """Verify CE-REQ-MOND-API-001: correct inline bins remain supported."""
    explainer, X_cal, y_cal, X_test, _y_test, mondrian_fn = mondrian_setup
    explainer.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal))

    predictions = explainer.predict_proba(X_test, bins=mondrian_fn(X_test))

    assert predictions.shape == (len(X_test), 2)
