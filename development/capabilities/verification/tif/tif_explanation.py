"""TIF verification interface for CE factual and alternative explanation capabilities.

TIF IDs: CE-TIF-EXPL-001

Requirements served:
  CE-REQ-EXPL-API-001     — explain_factual callable without exception
  CE-REQ-EXPL-RETURN-001  — factual explanation return type and cardinality contract
  CE-REQ-EXPL-API-002     — explore_alternatives callable without exception
  CE-REQ-EXPL-ALT-RETURN-001 — alternative explanation return type and cardinality contract

This module stimulates CE through WrapCalibratedExplainer only. It does not:
  - Construct explanation objects directly
  - Use private/internal CE APIs
  - Import from calibrated_explanations.core.calibrated_explainer
  - Perform final pytest assertions (only local sanity checks)

Tests call run_factual_tif_scenario() or run_alternative_tif_scenario() and assert
on the returned ExplanationObservation against acceptance criteria from the requirement files.
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
_N_INFORMATIVE = 3
_N_REDUNDANT = 1
_N_TEST = 3


@dataclass
class ExplanationObservation:
    """Structured observation returned by explanation TIF scenarios.

    Tests assert on these fields against acceptance criteria from requirement files.
    This dataclass carries observations only; it does not carry acceptance judgements.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised during the explain call.
    exception_type : str or None
        Class name of the exception if raised; None otherwise.
    result_is_none : bool
        Whether the result of the explain call is None.
    result_len : int or None
        len(result) if result supports __len__; None otherwise.
    result_type_name : str or None
        type(result).__name__ if result is not None; None if result is None.
    first_item_is_none : bool
        Whether result[0] is None. Always True when result_is_none is True.
    feature_weights_accessible : bool
        Whether result[0].feature_weights is accessible and not None.
        Only meaningful when first_item_is_none is False and explanation_mode is "factual".
    explanation_mode : str
        The explanation mode used ("factual" or "alternative").
    n_instances : int
        Number of test instances (len(X_test)).
    """

    exception_raised: bool
    exception_type: Optional[str]
    result_is_none: bool
    result_len: Optional[int]
    result_type_name: Optional[str]
    first_item_is_none: bool
    feature_weights_accessible: bool
    explanation_mode: str
    n_instances: int


def _build_explainer_and_data() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer and test data.

    Returns
    -------
    tuple
        (explainer, X_test) where explainer is fitted and calibrated for binary classification.
    """
    X_all, y_all = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=_N_INFORMATIVE,
        n_redundant=_N_REDUNDANT,
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


def run_factual_tif_scenario() -> ExplanationObservation:
    """Stimulate CE-REQ-EXPL-API-001 and CE-REQ-EXPL-RETURN-001 through WrapCalibratedExplainer.

    TIF ID: CE-TIF-EXPL-001

    Requirements served:
      CE-REQ-EXPL-API-001     (observation: exception_raised)
      CE-REQ-EXPL-RETURN-001  (observation: result_len, result_is_none, first_item_is_none,
                                             feature_weights_accessible, result_type_name)

    This function uses the public WrapCalibratedExplainer workflow only:
      1. Creates deterministic fixture data.
      2. Instantiates WrapCalibratedExplainer.
      3. Calls fit().
      4. Calls calibrate().
      5. Calls explain_factual().
      6. Returns an ExplanationObservation with structured observations.

    Returns
    -------
    ExplanationObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test = _build_explainer_and_data()
    n_instances = len(X_test)

    exception_raised = False
    exception_type = None
    result_is_none = True
    result_len = None
    result_type_name = None
    first_item_is_none = True
    feature_weights_accessible = False

    try:
        result = explainer.explain_factual(X_test)
    except Exception as exc:
        exception_raised = True
        exception_type = type(exc).__name__
        return ExplanationObservation(
            exception_raised=exception_raised,
            exception_type=exception_type,
            result_is_none=True,
            result_len=None,
            result_type_name=None,
            first_item_is_none=True,
            feature_weights_accessible=False,
            explanation_mode="factual",
            n_instances=n_instances,
        )

    result_is_none = result is None
    result_type_name = type(result).__name__ if result is not None else None

    if result is not None:
        try:
            result_len = len(result)
        except TypeError:
            result_len = None

        try:
            first_item = result[0]
            first_item_is_none = first_item is None
            if first_item is not None:
                fw = getattr(first_item, "feature_weights", None)
                feature_weights_accessible = fw is not None
        except Exception:
            first_item_is_none = True

    return ExplanationObservation(
        exception_raised=exception_raised,
        exception_type=exception_type,
        result_is_none=result_is_none,
        result_len=result_len,
        result_type_name=result_type_name,
        first_item_is_none=first_item_is_none,
        feature_weights_accessible=feature_weights_accessible,
        explanation_mode="factual",
        n_instances=n_instances,
    )


def run_alternative_tif_scenario() -> ExplanationObservation:
    """Stimulate CE-REQ-EXPL-API-002 and CE-REQ-EXPL-ALT-RETURN-001 through WrapCalibratedExplainer.

    TIF ID: CE-TIF-EXPL-001

    Requirements served:
      CE-REQ-EXPL-API-002        (observation: exception_raised)
      CE-REQ-EXPL-ALT-RETURN-001 (observation: result_len, result_is_none, first_item_is_none,
                                                result_type_name)

    Returns
    -------
    ExplanationObservation
        Structured observations. Tests assert on these fields.
    """
    explainer, X_test = _build_explainer_and_data()
    n_instances = len(X_test)

    exception_raised = False
    exception_type = None
    result_is_none = True
    result_len = None
    result_type_name = None
    first_item_is_none = True

    try:
        result = explainer.explore_alternatives(X_test)
    except Exception as exc:
        exception_raised = True
        exception_type = type(exc).__name__
        return ExplanationObservation(
            exception_raised=exception_raised,
            exception_type=exception_type,
            result_is_none=True,
            result_len=None,
            result_type_name=None,
            first_item_is_none=True,
            feature_weights_accessible=False,
            explanation_mode="alternative",
            n_instances=n_instances,
        )

    result_is_none = result is None
    result_type_name = type(result).__name__ if result is not None else None

    if result is not None:
        try:
            result_len = len(result)
        except TypeError:
            result_len = None

        try:
            first_item = result[0]
            first_item_is_none = first_item is None
        except Exception:
            first_item_is_none = True

    return ExplanationObservation(
        exception_raised=exception_raised,
        exception_type=exception_type,
        result_is_none=result_is_none,
        result_len=result_len,
        result_type_name=result_type_name,
        first_item_is_none=first_item_is_none,
        feature_weights_accessible=False,
        explanation_mode="alternative",
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
    """Build a complete evidence payload for CE-TIF-EXPL-001.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    factual = run_factual_tif_scenario()
    alt = run_alternative_tif_scenario()
    scenarios = [
        scenario_entry(
            "factual_api_contract",
            obs_to_dict(factual),
            [
                acceptance_entry(
                    "CE-REQ-EXPL-API-001", "exception_raised", False, factual.exception_raised
                )
            ],
        ),
        scenario_entry(
            "factual_return_contract",
            obs_to_dict(factual),
            [
                acceptance_entry(
                    "CE-REQ-EXPL-RETURN-001", "result_is_none", False, factual.result_is_none
                ),
                acceptance_entry(
                    "CE-REQ-EXPL-RETURN-001",
                    "result_len == n_instances",
                    True,
                    factual.result_len == factual.n_instances,
                ),
                acceptance_entry(
                    "CE-REQ-EXPL-RETURN-001",
                    "feature_weights_accessible",
                    True,
                    factual.feature_weights_accessible,
                ),
            ],
        ),
        scenario_entry(
            "alternative_api_contract",
            obs_to_dict(alt),
            [
                acceptance_entry(
                    "CE-REQ-EXPL-API-002", "exception_raised", False, alt.exception_raised
                )
            ],
        ),
        scenario_entry(
            "alternative_return_contract",
            obs_to_dict(alt),
            [
                acceptance_entry(
                    "CE-REQ-EXPL-ALT-RETURN-001",
                    "result_type_name",
                    "AlternativeExplanations",
                    alt.result_type_name,
                ),
                acceptance_entry(
                    "CE-REQ-EXPL-ALT-RETURN-001",
                    "result_len == n_instances",
                    True,
                    alt.result_len == alt.n_instances,
                ),
            ],
        ),
    ]
    return build_payload(
        "EXPL-001",
        claim_ids=["CE-CAP-EXPL-001", "CE-CAP-EXPL-002"],
        requirement_ids=[
            "CE-REQ-EXPL-API-001",
            "CE-REQ-EXPL-RETURN-001",
            "CE-REQ-EXPL-API-002",
            "CE-REQ-EXPL-ALT-RETURN-001",
        ],
        adr_refs=["ADR-008", "ADR-015", "ADR-026"],
        tif_ids=["CE-TIF-EXPL-001"],
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
