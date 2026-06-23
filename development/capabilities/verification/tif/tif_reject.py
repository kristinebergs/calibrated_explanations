"""TIF verification interface for CE reject policy explanation capabilities.

TIF ID: CE-TIF-REJECT-001

Requirements served:
  CE-REQ-REJECT-API-001 — explain_factual with RejectPolicySpec API contract

Tests call run_reject_tif_scenario() and assert on the returned
RejectObservation against acceptance criteria from the requirement files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations import RejectPolicySpec
from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@dataclass
class RejectObservation:
    """Structured observation returned by reject policy TIF scenarios.

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


def _build_reject_explainer() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer for reject tests."""
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


def run_reject_tif_scenario() -> RejectObservation:
    """Stimulate CE-REQ-REJECT-API-001 through WrapCalibratedExplainer with RejectPolicySpec.

    TIF ID: CE-TIF-REJECT-001

    Requirements served:
      CE-REQ-REJECT-API-001 (observation: exception_raised, result_is_none, result_len)

    Returns
    -------
    RejectObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test = _build_reject_explainer()
    n_instances = len(X_test)

    try:
        result = explainer.explain_factual(X_test, reject_policy=RejectPolicySpec.flag())
    except Exception as exc:
        return RejectObservation(
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

    return RejectObservation(
        exception_raised=False,
        exception_type=None,
        result_is_none=result is None,
        result_len=result_len,
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
) -> dict:
    """Build a complete evidence payload for CE-TIF-REJECT-001.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    obs = run_reject_tif_scenario()
    scenarios = [
        scenario_entry(
            "explain_factual_with_reject_policy_flag",
            obs_to_dict(obs),
            [
                acceptance_entry(
                    "CE-REQ-REJECT-API-001", "exception_raised", False, obs.exception_raised
                ),
                acceptance_entry(
                    "CE-REQ-REJECT-API-001", "result_is_none", False, obs.result_is_none
                ),
                acceptance_entry(
                    "CE-REQ-REJECT-API-001",
                    "result_len == n_instances",
                    True,
                    obs.result_len == obs.n_instances,
                ),
            ],
        ),
    ]
    return build_payload(
        "REJECT-001",
        claim_ids=["CE-CAP-REJECT-001"],
        requirement_ids=["CE-REQ-REJECT-API-001"],
        adr_refs=["ADR-029", "ADR-038"],
        tif_ids=["CE-TIF-REJECT-001"],
        verification_type="api_contract",
        dataset_id=_DATASET_ID,
        scenarios=scenarios,
        commit_sha=commit_sha,
        timestamp=timestamp,
        date_suffix=date_suffix,
        package_version=package_version,
        python_version=python_version,
        platform_str=platform_str,
    )
