"""TIF verification interface for CE conjunction capability.

TIF ID: CE-TIF-EXPL-CONJ-001
Specification: development/capabilities/verification/tif/CE-TIF-EXPL-CONJ-001.md

Requirements served:
  CE-REQ-EXPL-CONJ-API-001    — add_conjunctions callable without exception
  CE-REQ-EXPL-CONJ-RETURN-001 — return type and cardinality contract
  CE-REQ-EXPL-CONJ-RULE-001   — multi-feature conjunction rules produced when max_rule_size >= 2
  CE-REQ-EXPL-CONJ-PARAM-001  — max_rule_size=1 suppresses multi-feature conjunction rules

This module stimulates CE through WrapCalibratedExplainer only. It does not:
  - Construct explanation objects directly
  - Use private/internal CE APIs
  - Import from calibrated_explanations.core.calibrated_explainer
  - Perform final pytest assertions (only local sanity checks)

Tests call run_conjunction_tif_scenario() and assert on the returned
ConjunctionObservation against acceptance criteria from the requirement files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class ConjunctionObservation:
    """Structured observation returned by run_conjunction_tif_scenario().

    Tests assert on these fields against acceptance criteria from requirement files.
    This dataclass carries observations only; it does not carry acceptance judgements.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised during add_conjunctions.
    exception_type : str or None
        Class name of the exception if raised; None otherwise.
    result_is_none : bool
        Whether the result of add_conjunctions is None.
    result_len : int or None
        len(result) if result supports __len__; None otherwise.
    result_type_name : str or None
        type(result).__name__ if result is not None; None if result is None.
    any_has_conjunctive_rules : bool
        True if at least one item in the collection has has_conjunctive_rules == True.
        Always False when exception_raised is True.
    object_level : str
        The object_level parameter passed to the scenario.
    max_rule_size : int
        The max_rule_size parameter used.
    n_top_features : int
        The n_top_features parameter used.
    n_instances : int
        Number of test instances (len(X_test)).
    """

    exception_raised: bool
    exception_type: Optional[str]
    result_is_none: bool
    result_len: Optional[int]
    result_type_name: Optional[str]
    any_has_conjunctive_rules: bool
    object_level: str
    max_rule_size: int
    n_top_features: int
    n_instances: int
    explanation_mode: str = field(default="factual")


def _build_explainer_and_data() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer and test data.

    Returns
    -------
    tuple
        (explainer, X_test) where explainer is fitted and calibrated.
    """
    x_all, y_all = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=_N_INFORMATIVE,
        n_redundant=_N_REDUNDANT,
        random_state=_RNG_SEED,
    )
    x_train_cal, x_test, y_train_cal, _ = train_test_split(
        x_all, y_all, test_size=_N_TEST, random_state=_RNG_SEED
    )
    x_proper, x_cal, y_proper, y_cal = train_test_split(
        x_train_cal, y_train_cal, test_size=0.35, random_state=_RNG_SEED
    )
    explainer = WrapCalibratedExplainer(
        RandomForestClassifier(n_estimators=10, random_state=_RNG_SEED)
    )
    explainer.fit(x_proper, y_proper)
    explainer.calibrate(x_cal, y_cal)

    # Sanity check: ensure the explainer is ready before returning observations.
    # This is a local guard, not an assertion on CE behavior.
    assert explainer.fitted, "TIF sanity: explainer must be fitted"
    assert explainer.calibrated, "TIF sanity: explainer must be calibrated"

    return explainer, x_test


def run_conjunction_tif_scenario(
    *,
    explanation_mode: str = "factual",
    object_level: str = "collection",
    n_top_features: int = 5,
    max_rule_size: int = 2,
) -> ConjunctionObservation:
    """Stimulate CE-REQ-EXPL-CONJ-* through WrapCalibratedExplainer.

    TIF ID: CE-TIF-EXPL-CONJ-001

    Requirements served:
      CE-REQ-EXPL-CONJ-API-001    (observation: exception_raised)
      CE-REQ-EXPL-CONJ-RETURN-001 (observation: result_is_none, result_len)
      CE-REQ-EXPL-CONJ-RULE-001   (observation: any_has_conjunctive_rules when max_rule_size >= 2)
      CE-REQ-EXPL-CONJ-PARAM-001  (observation: any_has_conjunctive_rules when max_rule_size == 1)

    This function uses the public WrapCalibratedExplainer workflow only:
      1. Creates deterministic fixture data.
      2. Instantiates WrapCalibratedExplainer.
      3. Calls fit().
      4. Calls calibrate().
      5. Calls explain_factual() or explore_alternatives() per explanation_mode.
      6. Calls add_conjunctions() on the result (collection or individual).
      7. Returns a ConjunctionObservation with structured observations.

    This function does NOT perform final pytest assertions. Tests must assert on
    the returned ConjunctionObservation against acceptance criteria from the
    requirement files.

    Parameters
    ----------
    explanation_mode : str
        "factual" → explain_factual(); "alternative" → explore_alternatives().
    object_level : str
        "collection" → call add_conjunctions on the collection;
        "individual"  → call add_conjunctions on collection[0].
    n_top_features : int
        Passed to add_conjunctions. Default 5.
    max_rule_size : int
        Passed to add_conjunctions. Default 2.

    Returns
    -------
    ConjunctionObservation
        Structured observations. Tests assert on these fields.
    """
    if explanation_mode not in ("factual", "alternative"):
        raise ValueError(
            f"explanation_mode must be 'factual' or 'alternative', got {explanation_mode!r}"
        )
    if object_level not in ("collection", "individual"):
        raise ValueError(f"object_level must be 'collection' or 'individual', got {object_level!r}")

    explainer, x_test = _build_explainer_and_data()
    n_instances = len(x_test)

    if explanation_mode == "factual":
        collection = explainer.explain_factual(x_test)
    else:
        collection = explainer.explore_alternatives(x_test)

    exception_raised = False
    exception_type = None
    result_is_none = True
    result_len = None
    result_type_name = None
    any_has_conjunctive_rules = False

    try:
        if object_level == "collection":
            result = collection.add_conjunctions(
                n_top_features=n_top_features,
                max_rule_size=max_rule_size,
            )
        else:
            result = collection[0].add_conjunctions(
                n_top_features=n_top_features,
                max_rule_size=max_rule_size,
            )
    except Exception as exc:
        exception_raised = True
        exception_type = type(exc).__name__
        return ConjunctionObservation(
            exception_raised=exception_raised,
            exception_type=exception_type,
            result_is_none=True,
            result_len=None,
            result_type_name=None,
            any_has_conjunctive_rules=False,
            object_level=object_level,
            max_rule_size=max_rule_size,
            n_top_features=n_top_features,
            n_instances=n_instances,
            explanation_mode=explanation_mode,
        )

    result_is_none = result is None
    result_type_name = type(result).__name__ if result is not None else None

    if result is not None and object_level == "collection":
        try:
            result_len = len(result)
        except TypeError:
            result_len = None

        # Inspect each item for has_conjunctive_rules (public attribute, ADR-008)
        for i in range(len(collection)):
            item = collection[i]
            if getattr(item, "has_conjunctive_rules", False):
                any_has_conjunctive_rules = True
                break

    elif result is not None and object_level == "individual":
        # For individual: check has_conjunctive_rules on the result itself
        any_has_conjunctive_rules = bool(getattr(result, "has_conjunctive_rules", False))

    return ConjunctionObservation(
        exception_raised=exception_raised,
        exception_type=exception_type,
        result_is_none=result_is_none,
        result_len=result_len,
        result_type_name=result_type_name,
        any_has_conjunctive_rules=any_has_conjunctive_rules,
        object_level=object_level,
        max_rule_size=max_rule_size,
        n_top_features=n_top_features,
        n_instances=n_instances,
        explanation_mode=explanation_mode,
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
    """Build a complete evidence payload for CE-TIF-EXPL-CONJ-001.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    scenarios = []
    for mode in ("factual", "alternative"):
        for level in ("collection", "individual"):
            obs = run_conjunction_tif_scenario(
                explanation_mode=mode, object_level=level, max_rule_size=2, n_top_features=5
            )
            scenarios.append(
                scenario_entry(
                    f"api_{mode}_{level}",
                    obs_to_dict(obs),
                    [
                        acceptance_entry(
                            "CE-REQ-EXPL-CONJ-API-001",
                            "exception_raised",
                            False,
                            obs.exception_raised,
                        )
                    ],
                    {
                        "explanation_mode": mode,
                        "object_level": level,
                        "max_rule_size": 2,
                        "n_top_features": 5,
                    },
                )
            )
    for mode in ("factual", "alternative"):
        obs = run_conjunction_tif_scenario(
            explanation_mode=mode, object_level="collection", max_rule_size=2, n_top_features=5
        )
        scenarios.append(
            scenario_entry(
                f"return_{mode}_collection",
                obs_to_dict(obs),
                [
                    acceptance_entry(
                        "CE-REQ-EXPL-CONJ-RETURN-001", "result_is_none", False, obs.result_is_none
                    ),
                    acceptance_entry(
                        "CE-REQ-EXPL-CONJ-RETURN-001",
                        "result_len == n_instances",
                        True,
                        obs.result_len == obs.n_instances,
                    ),
                ],
                {
                    "explanation_mode": mode,
                    "object_level": "collection",
                    "max_rule_size": 2,
                    "n_top_features": 5,
                },
            )
        )
    for max_rule_size in (2, 3):
        obs = run_conjunction_tif_scenario(
            explanation_mode="factual",
            object_level="collection",
            max_rule_size=max_rule_size,
            n_top_features=5,
        )
        scenarios.append(
            scenario_entry(
                f"rule_factual_collection_max_rule_size_{max_rule_size}",
                obs_to_dict(obs),
                [
                    acceptance_entry(
                        "CE-REQ-EXPL-CONJ-RULE-001",
                        "any_has_conjunctive_rules",
                        True,
                        obs.any_has_conjunctive_rules,
                    )
                ],
                {
                    "explanation_mode": "factual",
                    "object_level": "collection",
                    "max_rule_size": max_rule_size,
                    "n_top_features": 5,
                },
            )
        )
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual", object_level="collection", max_rule_size=1, n_top_features=5
    )
    scenarios.append(
        scenario_entry(
            "param_factual_collection_max_rule_size_1",
            obs_to_dict(obs),
            [
                acceptance_entry(
                    "CE-REQ-EXPL-CONJ-PARAM-001",
                    "any_has_conjunctive_rules",
                    False,
                    obs.any_has_conjunctive_rules,
                ),
                acceptance_entry(
                    "CE-REQ-EXPL-CONJ-PARAM-001", "exception_raised", False, obs.exception_raised
                ),
            ],
            {
                "explanation_mode": "factual",
                "object_level": "collection",
                "max_rule_size": 1,
                "n_top_features": 5,
            },
        )
    )
    return build_payload(
        "EXPL-CONJ-001",
        claim_ids=["CE-CAP-EXPL-CONJ-001"],
        requirement_ids=[
            "CE-REQ-EXPL-CONJ-API-001",
            "CE-REQ-EXPL-CONJ-RETURN-001",
            "CE-REQ-EXPL-CONJ-RULE-001",
            "CE-REQ-EXPL-CONJ-PARAM-001",
        ],
        adr_refs=["ADR-008"],
        tif_ids=["CE-TIF-EXPL-CONJ-001"],
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
