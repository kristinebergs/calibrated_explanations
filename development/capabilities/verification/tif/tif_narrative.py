"""TIF verification interface for CE narrative explanation output capabilities.

TIF ID: CE-TIF-NARR-001

Requirements served:
  CE-REQ-NARR-API-001 — to_narrative(output_format='text') API contract

Tests call run_narrative_tif_scenario() and assert on the returned
NarrativeObservation against acceptance criteria from the requirement files.

Note: requires pyyaml. If pyyaml is not installed this TIF will set
exception_raised=True with exception_type='ImportError'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3


@dataclass
class NarrativeObservation:
    """Structured observation returned by narrative TIF scenarios.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised during the call.
    exception_type : str or None
        Exception class name if raised; None otherwise.
    result_is_none : bool
        Whether the result is None.
    result_is_str : bool
        Whether isinstance(result, str).
    result_len : int or None
        len(result) if result is a non-None str; None otherwise.
    n_instances : int
        Number of test instances.
    """

    exception_raised: bool
    exception_type: Optional[str]
    result_is_none: bool
    result_is_str: bool
    result_len: Optional[int]
    n_instances: int


def _build_narrative_explainer() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer for narrative tests."""
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


def run_narrative_tif_scenario() -> NarrativeObservation:
    """Stimulate CE-REQ-NARR-API-001 through WrapCalibratedExplainer.explain_factual + to_narrative.

    TIF ID: CE-TIF-NARR-001

    Requirements served:
      CE-REQ-NARR-API-001 (observation: exception_raised, result_is_none, result_is_str, result_len)

    Returns
    -------
    NarrativeObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test = _build_narrative_explainer()
    n_instances = len(X_test)

    try:
        explanations = explainer.explain_factual(X_test)
        result = explanations.to_narrative(output_format="text")
    except Exception as exc:
        return NarrativeObservation(
            exception_raised=True,
            exception_type=type(exc).__name__,
            result_is_none=True,
            result_is_str=False,
            result_len=None,
            n_instances=n_instances,
        )

    result_is_none = result is None
    result_is_str = isinstance(result, str)
    result_len = len(result) if result_is_str else None

    return NarrativeObservation(
        exception_raised=False,
        exception_type=None,
        result_is_none=result_is_none,
        result_is_str=result_is_str,
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
    """Build a complete evidence payload for CE-TIF-NARR-001.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    obs = run_narrative_tif_scenario()
    scenarios = [
        scenario_entry(
            "to_narrative_text_format",
            obs_to_dict(obs),
            [
                acceptance_entry(
                    "CE-REQ-NARR-API-001", "exception_raised", False, obs.exception_raised
                ),
                acceptance_entry(
                    "CE-REQ-NARR-API-001", "result_is_none", False, obs.result_is_none
                ),
                acceptance_entry("CE-REQ-NARR-API-001", "result_is_str", True, obs.result_is_str),
                acceptance_entry(
                    "CE-REQ-NARR-API-001",
                    "result_len > 0",
                    True,
                    obs.result_len is not None and obs.result_len > 0,
                ),
            ],
        ),
    ]
    return build_payload(
        "NARR-001",
        claim_ids=["CE-CAP-NARR-001"],
        requirement_ids=["CE-REQ-NARR-API-001"],
        adr_refs=["ADR-008"],
        tif_ids=["CE-TIF-NARR-001"],
        verification_type="api_contract",
        dataset_id=_DATASET_ID,
        scenarios=scenarios,
        commit_sha=commit_sha,
        timestamp=timestamp,
        date_suffix=date_suffix,
        package_version=package_version,
        python_version=python_version,
        platform_str=platform_str,
        configuration={"output_format": "text"},
    )
