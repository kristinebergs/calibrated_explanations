"""TIF verification interface for CE uncertainty prediction interval capabilities.

TIF ID: CE-TIF-PRED-001

Requirements served:
  CE-REQ-PRED-API-001              — predict with uq_interval=True API contract
  CE-REQ-PRED-INTERVAL-BOUNDS-001  — low_high_percentiles parameter and interval semantics

This module stimulates CE through WrapCalibratedExplainer only.

Tests call run_prediction_tif_scenario() and assert on the returned
PredictionObservation against acceptance criteria from the requirement files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class PredictionObservation:
    """Structured observation returned by prediction TIF scenarios.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised during the predict call.
    exception_type : str or None
        Class name of the exception if raised; None otherwise.
    result_is_none : bool
        Whether the result is None.
    y_hat_len : int or None
        Length of the point prediction array; None if exception.
    low_is_none : bool
        Whether the lower bound array is None.
    high_is_none : bool
        Whether the upper bound array is None.
    bounds_ordered : bool
        Whether low[i] <= high[i] for all i. False if exception or None bounds.
    low_lte_yhat : bool or None
        Whether low[i] <= y_hat[i] for all i (regression point estimate ordering).
        None when not applicable.
    low_high_percentiles : tuple or None
        The percentile tuple used, or None for default.
    n_instances : int
        Number of test instances.
    low_values : list or None
        The actual lower bound values, for narrowing assertion checks.
    high_values : list or None
        The actual upper bound values.
    y_hat_values : list or None
        The point prediction values.
    """

    exception_raised: bool
    exception_type: Optional[str]
    result_is_none: bool
    y_hat_len: Optional[int]
    low_is_none: bool
    high_is_none: bool
    bounds_ordered: bool
    low_lte_yhat: Optional[bool]
    low_high_percentiles: Optional[tuple]
    n_instances: int
    low_values: Optional[list] = field(default=None)
    high_values: Optional[list] = field(default=None)
    y_hat_values: Optional[list] = field(default=None)


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

    return explainer, X_test


def run_prediction_tif_scenario(
    *,
    low_high_percentiles: Optional[tuple] = None,
) -> PredictionObservation:
    """Stimulate CE-REQ-PRED-API-001 and CE-REQ-PRED-INTERVAL-BOUNDS-001.

    TIF ID: CE-TIF-PRED-001

    Requirements served:
      CE-REQ-PRED-API-001             (observation: exception_raised, y_hat_len,
                                        low_is_none, high_is_none, bounds_ordered)
      CE-REQ-PRED-INTERVAL-BOUNDS-001 (observation: bounds_ordered, low_values, high_values)

    Parameters
    ----------
    low_high_percentiles : tuple or None
        If provided, passed as low_high_percentiles to predict(uq_interval=True).
        If None, uses the default (5, 95).

    Returns
    -------
    PredictionObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test = _build_regression_explainer()
    n_instances = len(X_test)

    exception_raised = False
    exception_type = None
    result_is_none = True
    y_hat_len = None
    low_is_none = True
    high_is_none = True
    bounds_ordered = False
    low_lte_yhat = None
    low_values = None
    high_values = None
    y_hat_values = None

    predict_kwargs: dict = {"uq_interval": True}
    if low_high_percentiles is not None:
        predict_kwargs["low_high_percentiles"] = low_high_percentiles

    try:
        result = explainer.predict(X_test, **predict_kwargs)
    except Exception as exc:
        exception_raised = True
        exception_type = type(exc).__name__
        return PredictionObservation(
            exception_raised=True,
            exception_type=exception_type,
            result_is_none=True,
            y_hat_len=None,
            low_is_none=True,
            high_is_none=True,
            bounds_ordered=False,
            low_lte_yhat=None,
            low_high_percentiles=low_high_percentiles,
            n_instances=n_instances,
        )

    result_is_none = result is None

    if result is not None:
        try:
            y_hat, (low, high) = result
            y_hat_len = len(y_hat) if y_hat is not None else None
            low_is_none = low is None
            high_is_none = high is None

            if low is not None and high is not None:
                low_arr = np.asarray(low)
                high_arr = np.asarray(high)
                y_hat_arr = np.asarray(y_hat)
                bounds_ordered = bool(np.all(low_arr <= high_arr))
                low_lte_yhat = bool(np.all(low_arr <= y_hat_arr))
                low_values = low_arr.tolist()
                high_values = high_arr.tolist()
                y_hat_values = y_hat_arr.tolist()
        except (ValueError, TypeError):
            pass

    return PredictionObservation(
        exception_raised=exception_raised,
        exception_type=exception_type,
        result_is_none=result_is_none,
        y_hat_len=y_hat_len,
        low_is_none=low_is_none,
        high_is_none=high_is_none,
        bounds_ordered=bounds_ordered,
        low_lte_yhat=low_lte_yhat,
        low_high_percentiles=low_high_percentiles,
        n_instances=n_instances,
        low_values=low_values,
        high_values=high_values,
        y_hat_values=y_hat_values,
    )


_DATASET_ID = (
    "sklearn make_regression n_samples=150 n_features=4 n_informative=3 noise=10 random_seed=42"
)


def build_evidence_payload(
    *,
    commit_sha: str,
    timestamp: str,
    date_suffix: str,
    package_version: str,
    python_version: str,
    platform_str: str,
    spec_claim_ids: list,
    spec_requirement_ids: list,
    spec_adr_refs: list,
    spec_tif_id: str,
    spec_verification_type: str,
    spec_evidence_key: str,
) -> dict:
    """Build a complete evidence payload for CE-TIF-PRED-001.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    Envelope metadata is injected from the TIF spec by the caller.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    default = run_prediction_tif_scenario()
    custom = run_prediction_tif_scenario(low_high_percentiles=(10, 90))
    scenarios = [
        scenario_entry(
            "predict_uq_interval_default",
            obs_to_dict(default),
            [
                acceptance_entry(
                    "CE-REQ-PRED-API-001", "exception_raised", False, default.exception_raised
                ),
                acceptance_entry(
                    "CE-REQ-PRED-API-001",
                    "y_hat_len == n_instances",
                    True,
                    default.y_hat_len == default.n_instances,
                ),
                acceptance_entry("CE-REQ-PRED-API-001", "low_is_none", False, default.low_is_none),
                acceptance_entry(
                    "CE-REQ-PRED-API-001", "high_is_none", False, default.high_is_none
                ),
                acceptance_entry(
                    "CE-REQ-PRED-INTERVAL-BOUNDS-001",
                    "bounds_ordered",
                    True,
                    default.bounds_ordered,
                ),
            ],
        ),
        scenario_entry(
            "predict_uq_interval_percentiles_10_90",
            obs_to_dict(custom),
            [
                acceptance_entry(
                    "CE-REQ-PRED-INTERVAL-BOUNDS-001",
                    "exception_raised",
                    False,
                    custom.exception_raised,
                ),
                acceptance_entry(
                    "CE-REQ-PRED-INTERVAL-BOUNDS-001", "bounds_ordered", True, custom.bounds_ordered
                ),
            ],
        ),
    ]
    return build_payload(
        spec_evidence_key,
        claim_ids=spec_claim_ids,
        requirement_ids=spec_requirement_ids,
        adr_refs=spec_adr_refs,
        tif_ids=[spec_tif_id],
        verification_type=spec_verification_type,
        dataset_id=_DATASET_ID,
        scenarios=scenarios,
        commit_sha=commit_sha,
        timestamp=timestamp,
        date_suffix=date_suffix,
        package_version=package_version,
        python_version=python_version,
        platform_str=platform_str,
    )
