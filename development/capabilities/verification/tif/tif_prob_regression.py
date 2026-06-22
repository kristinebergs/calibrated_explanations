"""TIF verification interface for CE probabilistic regression threshold query capabilities.

TIF ID: CE-TIF-PRED-PROB-001

Requirements served:
  CE-REQ-PRED-PROB-API-001    — predict_proba with threshold API contract
  CE-REQ-PRED-PROB-BOUNDS-001 — returned probability values bounded in [0, 1]

Tests call run_prob_regression_tif_scenario() and assert on the returned
ProbRegressionObservation against acceptance criteria from the requirement files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 150
_N_FEATURES = 4
_N_TEST = 5


@dataclass
class ProbRegressionObservation:
    """Structured observation returned by probabilistic regression TIF scenarios.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised.
    exception_type : str or None
        Exception class name if raised; None otherwise.
    result_is_none : bool
        Whether the result is None.
    proba_len : int or None
        Length of the returned probability array.
    proba_min : float or None
        Minimum value in the probability array.
    proba_max : float or None
        Maximum value in the probability array.
    threshold : float
        The threshold used.
    n_instances : int
        Number of test instances.
    """

    exception_raised: bool
    exception_type: Optional[str]
    result_is_none: bool
    proba_len: Optional[int]
    proba_min: Optional[float]
    proba_max: Optional[float]
    threshold: float
    n_instances: int


def _build_regression_explainer() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer for regression."""
    X_all, y_all = make_regression(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        random_state=_RNG_SEED,
        noise=10.0,
    )
    X_train_cal, X_test, y_train_cal, _ = train_test_split(
        X_all, y_all, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestRegressor(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal)

    assert explainer.fitted, "TIF sanity: explainer must be fitted"
    assert explainer.calibrated, "TIF sanity: explainer must be calibrated"

    return explainer, X_test, y_cal


def run_prob_regression_tif_scenario(
    *,
    threshold: float = 0.0,
) -> ProbRegressionObservation:
    """Stimulate CE-REQ-PRED-PROB-API-001 and CE-REQ-PRED-PROB-BOUNDS-001.

    TIF ID: CE-TIF-PRED-PROB-001

    Requirements served:
      CE-REQ-PRED-PROB-API-001    (observation: exception_raised, proba_len)
      CE-REQ-PRED-PROB-BOUNDS-001 (observation: proba_min, proba_max)

    Parameters
    ----------
    threshold : float
        Scalar threshold for P(Y > threshold | X). Default 0.0.

    Returns
    -------
    ProbRegressionObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test, _ = _build_regression_explainer()
    n_instances = len(X_test)

    exception_raised = False
    exception_type = None
    result_is_none = True
    proba_len = None
    proba_min = None
    proba_max = None

    try:
        result = explainer.predict_proba(X_test, threshold=threshold)
    except Exception as exc:
        exception_raised = True
        exception_type = type(exc).__name__
        return ProbRegressionObservation(
            exception_raised=True,
            exception_type=exception_type,
            result_is_none=True,
            proba_len=None,
            proba_min=None,
            proba_max=None,
            threshold=threshold,
            n_instances=n_instances,
        )

    result_is_none = result is None

    if result is not None:
        proba_arr = np.asarray(result)
        proba_len = len(proba_arr)
        proba_min = float(np.min(proba_arr))
        proba_max = float(np.max(proba_arr))

    return ProbRegressionObservation(
        exception_raised=exception_raised,
        exception_type=exception_type,
        result_is_none=result_is_none,
        proba_len=proba_len,
        proba_min=proba_min,
        proba_max=proba_max,
        threshold=threshold,
        n_instances=n_instances,
    )
