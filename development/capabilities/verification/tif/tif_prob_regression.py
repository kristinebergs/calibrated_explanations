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


_DATASET_ID = (
    "sklearn make_regression n_samples=150 n_features=4 " "n_informative=3 noise=10 random_seed=42"
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
    """Build a complete evidence payload for CE-TIF-PRED-PROB-001.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    Envelope metadata is injected from the TIF spec by the caller.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    obs = run_prob_regression_tif_scenario(threshold=0.0)
    scenarios = [
        scenario_entry(
            "prob_regression_threshold_0",
            obs_to_dict(obs),
            [
                acceptance_entry(
                    "CE-REQ-PRED-PROB-API-001", "exception_raised", False, obs.exception_raised
                ),
                acceptance_entry(
                    "CE-REQ-PRED-PROB-API-001",
                    "proba_len == n_instances",
                    True,
                    obs.proba_len == obs.n_instances,
                ),
                acceptance_entry(
                    "CE-REQ-PRED-PROB-BOUNDS-001",
                    "proba_min >= 0.0",
                    True,
                    obs.proba_min is not None and obs.proba_min >= 0.0,
                ),
                acceptance_entry(
                    "CE-REQ-PRED-PROB-BOUNDS-001",
                    "proba_max <= 1.0",
                    True,
                    obs.proba_max is not None and obs.proba_max <= 1.0,
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
        configuration={"threshold": 0.0},
    )
