"""TIF verification interface for CE uncertainty prediction interval capabilities.

TIF ID: CE-TIF-PRED-001

Requirements served:
  CE-REQ-PRED-API-001              — predict with uq_interval=True API contract
  CE-REQ-PRED-INTERVAL-BOUNDS-001  — low_high_percentiles parameter and interval semantics
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
    """Stimulate CE-REQ-PRED-API-001 and CE-REQ-PRED-INTERVAL-BOUNDS-001."""
    explainer, X_test = _build_regression_explainer()
    n_instances = len(X_test)

    predict_kwargs: dict = {"uq_interval": True}
    if low_high_percentiles is not None:
        predict_kwargs["low_high_percentiles"] = low_high_percentiles

    try:
        result = explainer.predict(X_test, **predict_kwargs)
    except Exception as exc:
        return PredictionObservation(
            exception_raised=True,
            exception_type=type(exc).__name__,
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
    y_hat_len = None
    low_is_none = True
    high_is_none = True
    bounds_ordered = False
    low_lte_yhat = None
    low_values = None
    high_values = None
    y_hat_values = None

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
        exception_raised=False,
        exception_type=None,
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
    "sklearn make_regression n_samples=150 n_features=4 "
    "n_informative=3 noise=10 random_seed=42"
)


def build_evidence_payload(
    *,
    commit_sha: str,
    timestamp: str,
    date_suffix: str,
    package_version: str,
    python_version: str,
    platform_str: str,
) -> dict:
    """Build a complete evidence payload for CE-TIF-PRED-001."""
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
                acceptance_entry("CE-REQ-PRED-API-001", "exception_raised", False, default.exception_raised),
                acceptance_entry("CE-REQ-PRED-API-001", "y_hat_len == n_instances", True, default.y_hat_len == default.n_instances),
                acceptance_entry("CE-REQ-PRED-API-001", "low_is_none", False, default.low_is_none),
                acceptance_entry("CE-REQ-PRED-API-001", "high_is_none", False, default.high_is_none),
                acceptance_entry("CE-REQ-PRED-INTERVAL-BOUNDS-001", "bounds_ordered", True, default.bounds_ordered),
            ],
        ),
        scenario_entry(
            "predict_uq_interval_percentiles_10_90",
            obs_to_dict(custom),
            [
                acceptance_entry("CE-REQ-PRED-INTERVAL-BOUNDS-001", "exception_raised", False, custom.exception_raised),
                acceptance_entry("CE-REQ-PRED-INTERVAL-BOUNDS-001", "bounds_ordered", True, custom.bounds_ordered),
            ],
        ),
    ]
    return build_payload(
        "PRED-001",
        claim_ids=["CE-CAP-PRED-001"],
        requirement_ids=["CE-REQ-PRED-API-001", "CE-REQ-PRED-INTERVAL-BOUNDS-001"],
        adr_refs=["ADR-013", "ADR-021"],
        tif_ids=["CE-TIF-PRED-001"],
        verification_type="behavioral_contract",
        dataset_id=_DATASET_ID,
        scenarios=scenarios,
        commit_sha=commit_sha,
        timestamp=timestamp,
        date_suffix=date_suffix,
        package_version=package_version,
        python_version=python_version,
        platform_str=platform_str,
    )
