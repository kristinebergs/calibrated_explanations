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
    """Structured observation returned by filter TIF scenarios.

    Fields
    ------
    filter_type : str
        Which filter operation was exercised.
    exception_raised : bool
        Whether an exception was raised.
    exception_type : str or None
        Exception class name if raised; None otherwise.
    collection_result_is_none : bool
        Whether the collection-level result is None.
    collection_result_len : int or None
        len(result) for collection call; None on exception.
    individual_result_is_none : bool
        Whether the individual-level result is None.
    alias_result_is_none : bool
        Whether the alias method (e.g. .super()) result is None.
    alias_result_len : int or None
        len(alias result) for alias call; None on exception.
    n_instances : int
        Number of test instances.
    """

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
    """Build a deterministic AlternativeExplanations collection."""
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
    """Stimulate one of the five filter operations through WrapCalibratedExplainer.

    TIF ID: CE-TIF-FILTER-001

    Requirements served (by filter_type):
      "super"    → CE-REQ-EXPL-FILTER-SUPER-001
      "semi"     → CE-REQ-EXPL-FILTER-SEMI-001
      "counter"  → CE-REQ-EXPL-FILTER-COUNTER-001
      "ensured"  → CE-REQ-EXPL-FILTER-ENSURED-001
      "pareto"   → CE-REQ-EXPL-FILTER-PARETO-001

    Observations per call:
      collection_result_is_none, collection_result_len — from collection call
      individual_result_is_none                        — from individual call on result[0]
      alias_result_is_none, alias_result_len           — from alias method call

    Parameters
    ----------
    filter_type : {"super", "semi", "counter", "ensured", "pareto"}
        Which filter operation to exercise.

    Returns
    -------
    FilterObservation
        Structured observations. Tests assert on these fields.
    """
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
