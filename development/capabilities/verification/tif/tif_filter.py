"""TIF verification interface for CE alternative explanation filter capabilities.

TIF ID: CE-TIF-FILTER-001

Requirements served:
  CE-REQ-EXPL-FILTER-SUPER-001   — super_explanations / super() API contract
  CE-REQ-EXPL-FILTER-SEMI-001    — semi_explanations / semi() API contract
  CE-REQ-EXPL-FILTER-COUNTER-001 — counter_explanations / counter() API contract
  CE-REQ-EXPL-FILTER-ENSURED-001 — ensured_explanations / ensured() API contract
  CE-REQ-EXPL-FILTER-PARETO-001  — pareto_explanations / pareto() API contract

Tests call run_filter_tif_scenario(filter_type=...) and assert on the returned
FilterObservation against acceptance criteria from the requirement files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer

_RNG_SEED = 42
_N_SAMPLES = 120
_N_FEATURES = 4
_N_TEST = 3

FilterType = Literal["super", "semi", "counter", "ensured", "pareto"]


@dataclass
class FilterObservation:
    filter_type: str
    exception_raised: bool
    exception_type: Optional[str]
    collection_result_is_none: bool
    collection_result_len: Optional[int]
    individual_result_is_none: bool
    alias_result_is_none: bool
    alias_result_len: Optional[int]
    n_instances: int


def _build_alternatives() -> tuple:
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

    alternatives = explainer.explore_alternatives(X_test)
    return alternatives, X_test


def _safe_len(obj) -> Optional[int]:
    if obj is None:
        return None
    try:
        return len(obj)
    except TypeError:
        return None


def run_filter_tif_scenario(filter_type: FilterType) -> FilterObservation:
    """Stimulate one of the five filter operations through WrapCalibratedExplainer."""
    alternatives, X_test = _build_alternatives()
    n_instances = len(X_test)

    try:
        if filter_type == "super":
            col_result = alternatives.super_explanations()
            ind_result = alternatives[0].super_explanations()
            alias_result = alternatives.super()
        elif filter_type == "semi":
            col_result = alternatives.semi_explanations()
            ind_result = alternatives[0].semi_explanations()
            alias_result = alternatives.semi()
        elif filter_type == "counter":
            col_result = alternatives.counter_explanations()
            ind_result = alternatives[0].counter_explanations()
            alias_result = alternatives.counter()
        elif filter_type == "ensured":
            col_result = alternatives.ensured_explanations()
            ind_result = alternatives[0].ensured_explanations()
            alias_result = alternatives.ensured()
        elif filter_type == "pareto":
            col_result = alternatives.pareto_explanations()
            ind_result = alternatives[0].pareto_explanations()
            alias_result = alternatives.pareto()
        else:
            raise ValueError(f"Unknown filter_type: {filter_type!r}")
    except Exception as exc:
        return FilterObservation(
            filter_type=filter_type,
            exception_raised=True,
            exception_type=type(exc).__name__,
            collection_result_is_none=True,
            collection_result_len=None,
            individual_result_is_none=True,
            alias_result_is_none=True,
            alias_result_len=None,
            n_instances=n_instances,
        )

    return FilterObservation(
        filter_type=filter_type,
        exception_raised=False,
        exception_type=None,
        collection_result_is_none=col_result is None,
        collection_result_len=_safe_len(col_result),
        individual_result_is_none=ind_result is None,
        alias_result_is_none=alias_result is None,
        alias_result_len=_safe_len(alias_result),
        n_instances=n_instances,
    )


_DATASET_ID = (
    "sklearn make_classification n_samples=120 n_features=4 "
    "n_informative=3 n_redundant=1 random_seed=42"
)

_FILTER_REQ_MAP = {
    "super": "CE-REQ-EXPL-FILTER-SUPER-001",
    "semi": "CE-REQ-EXPL-FILTER-SEMI-001",
    "counter": "CE-REQ-EXPL-FILTER-COUNTER-001",
    "ensured": "CE-REQ-EXPL-FILTER-ENSURED-001",
    "pareto": "CE-REQ-EXPL-FILTER-PARETO-001",
}


def build_evidence_payload(
    *,
    commit_sha: str,
    timestamp: str,
    date_suffix: str,
    package_version: str,
    python_version: str,
    platform_str: str,
) -> dict:
    """Build a complete evidence payload for CE-TIF-FILTER-001."""
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    scenarios = []
    for filter_type, req_id in _FILTER_REQ_MAP.items():
        obs = run_filter_tif_scenario(filter_type=filter_type)
        scenarios.append(scenario_entry(
            f"filter_{filter_type}",
            obs_to_dict(obs),
            [
                acceptance_entry(req_id, "exception_raised", False, obs.exception_raised),
                acceptance_entry(req_id, "collection_result_is_none", False, obs.collection_result_is_none),
                acceptance_entry(req_id, "collection_result_len == n_instances", True, obs.collection_result_len == obs.n_instances),
                acceptance_entry(req_id, "individual_result_is_none", False, obs.individual_result_is_none),
                acceptance_entry(req_id, "alias_result_is_none", False, obs.alias_result_is_none),
            ],
        ))
    return build_payload(
        "FILTER-001",
        claim_ids=["CE-CAP-EXPL-FILTER-001"],
        requirement_ids=list(_FILTER_REQ_MAP.values()),
        adr_refs=["ADR-027"],
        tif_ids=["CE-TIF-FILTER-001"],
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
