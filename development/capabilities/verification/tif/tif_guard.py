"""TIF verification interface for CE guarded explanation capabilities.

TIF ID: CE-TIF-GUARD-001

Requirements served:
  CE-REQ-GUARD-API-001 — explain_factual with GuardedOptions API contract

Tests call run_guard_tif_scenario() and assert on the returned
GuardObservation against acceptance criteria from the requirement files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import GuardedOptions
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@dataclass
class GuardObservation:
    """Structured observation returned by guarded explanation TIF scenarios.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised during the call.
    exception_type : str or None
        Exception class name if raised; None otherwise.
    result_is_none : bool
        Whether the result is None.
    result_len : int or None
        len(result) if result supports __len__; None otherwise.
    n_instances : int
        Number of test instances.
    """

    exception_raised: bool
    exception_type: Optional[str]
    result_is_none: bool
    result_len: Optional[int]
    n_instances: int


def _build_guard_explainer() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer for guard tests."""
    X_all, y_all = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=_RNG_SEED,
    )
    X_train_cal, X_test, y_train_cal, _ = train_test_split(
        X_all, y_all, test_size=_N_TEST, random_state=_RNG_SEED
    )
    X_proper, X_cal, y_proper, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestClassifier(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(X_proper, y_proper)
    explainer.calibrate(X_cal, y_cal)

    assert explainer.fitted, "TIF sanity: explainer must be fitted"
    assert explainer.calibrated, "TIF sanity: explainer must be calibrated"

    return explainer, X_test


def run_guard_tif_scenario() -> GuardObservation:
    """Stimulate CE-REQ-GUARD-API-001 through WrapCalibratedExplainer with GuardedOptions.

    TIF ID: CE-TIF-GUARD-001

    Requirements served:
      CE-REQ-GUARD-API-001 (observation: exception_raised, result_is_none, result_len)

    Returns
    -------
    GuardObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test = _build_guard_explainer()
    n_instances = len(X_test)

    try:
        result = explainer.explain_factual(X_test, guarded_options=GuardedOptions())
    except Exception as exc:
        return GuardObservation(
            exception_raised=True,
            exception_type=type(exc).__name__,
            result_is_none=True,
            result_len=None,
            n_instances=n_instances,
        )

    result_len = None
    if result is not None:
        import contextlib

        with contextlib.suppress(TypeError):
            result_len = len(result)

    return GuardObservation(
        exception_raised=False,
        exception_type=None,
        result_is_none=result is None,
        result_len=result_len,
        n_instances=n_instances,
    )
