"""TIF verification interface for CE classification prediction capabilities.

TIF ID: CE-TIF-PRED-CLASS-001

Requirements served:
  CE-REQ-PRED-CLASS-API-001    — predict_proba and predict API contract for classification
  CE-REQ-PRED-CLASS-BOUNDS-001 — probability values bounded in [0, 1]

Tests call run_classification_tif_scenario() and assert on the returned
ClassificationObservation against acceptance criteria from the requirement files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 5


@dataclass
class ClassificationObservation:
    """Structured observation returned by classification TIF scenarios.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised.
    exception_type : str or None
        Exception class name if raised; None otherwise.
    proba_is_none : bool
        Whether predict_proba returned None.
    proba_len : int or None
        Length of the probability array.
    proba_min : float or None
        Minimum value in the probability array.
    proba_max : float or None
        Maximum value in the probability array.
    labels_is_none : bool
        Whether predict returned None.
    labels_len : int or None
        Length of the label array.
    n_instances : int
        Number of test instances.
    """

    exception_raised: bool
    exception_type: Optional[str]
    proba_is_none: bool
    proba_len: Optional[int]
    proba_min: Optional[float]
    proba_max: Optional[float]
    labels_is_none: bool
    labels_len: Optional[int]
    n_instances: int


def _build_classification_explainer() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer for binary classification."""
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


def run_classification_tif_scenario() -> ClassificationObservation:
    """Stimulate CE-REQ-PRED-CLASS-API-001 and CE-REQ-PRED-CLASS-BOUNDS-001.

    TIF ID: CE-TIF-PRED-CLASS-001

    Requirements served:
      CE-REQ-PRED-CLASS-API-001    (observation: exception_raised, proba_len, labels_len)
      CE-REQ-PRED-CLASS-BOUNDS-001 (observation: proba_min, proba_max)

    Returns
    -------
    ClassificationObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test = _build_classification_explainer()
    n_instances = len(X_test)

    exception_raised = False
    exception_type = None
    proba_is_none = True
    proba_len = None
    proba_min = None
    proba_max = None
    labels_is_none = True
    labels_len = None

    try:
        probas = explainer.predict_proba(X_test)
        labels = explainer.predict(X_test)
    except Exception as exc:
        exception_raised = True
        exception_type = type(exc).__name__
        return ClassificationObservation(
            exception_raised=True,
            exception_type=exception_type,
            proba_is_none=True,
            proba_len=None,
            proba_min=None,
            proba_max=None,
            labels_is_none=True,
            labels_len=None,
            n_instances=n_instances,
        )

    proba_is_none = probas is None
    labels_is_none = labels is None

    if probas is not None:
        proba_arr = np.asarray(probas)
        proba_len = len(proba_arr)
        proba_min = float(np.min(proba_arr))
        proba_max = float(np.max(proba_arr))

    if labels is not None:
        labels_len = len(labels)

    return ClassificationObservation(
        exception_raised=exception_raised,
        exception_type=exception_type,
        proba_is_none=proba_is_none,
        proba_len=proba_len,
        proba_min=proba_min,
        proba_max=proba_max,
        labels_is_none=labels_is_none,
        labels_len=labels_len,
        n_instances=n_instances,
    )


_DATASET_ID = (
    "sklearn make_classification n_samples=120 n_features=4 "
    "n_informative=3 n_redundant=1 random_seed=42"
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
    """Build a complete evidence payload for CE-TIF-PRED-CLASS-001.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    Envelope metadata is injected from the TIF spec by the caller.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    obs = run_classification_tif_scenario()
    scenarios = [
        scenario_entry(
            "classification_api_and_bounds",
            obs_to_dict(obs),
            [
                acceptance_entry(
                    "CE-REQ-PRED-CLASS-API-001", "exception_raised", False, obs.exception_raised
                ),
                acceptance_entry(
                    "CE-REQ-PRED-CLASS-API-001",
                    "proba_len == n_instances",
                    True,
                    obs.proba_len == obs.n_instances,
                ),
                acceptance_entry(
                    "CE-REQ-PRED-CLASS-API-001",
                    "labels_len == n_instances",
                    True,
                    obs.labels_len == obs.n_instances,
                ),
                acceptance_entry(
                    "CE-REQ-PRED-CLASS-BOUNDS-001",
                    "proba_min >= 0.0",
                    True,
                    obs.proba_min is not None and obs.proba_min >= 0.0,
                ),
                acceptance_entry(
                    "CE-REQ-PRED-CLASS-BOUNDS-001",
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
    )
