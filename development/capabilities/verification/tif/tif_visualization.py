"""TIF verification interface for CE visualization capabilities.

TIF ID: CE-TIF-VIZ-001

Requirements served:
  CE-REQ-VIZ-SMOKE-001 — CalibratedExplanations.plot() no-raise smoke test

Tests call run_visualization_tif_scenario() and assert on the returned
VizObservation against acceptance criteria from the requirement files.

Note: requires matplotlib. If matplotlib is not installed this TIF will set
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
_N_TEST = 2


@dataclass
class VizObservation:
    """Structured observation returned by visualization TIF scenarios.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised during explanations.plot().
    exception_type : str or None
        Exception class name if raised; None otherwise.
    n_instances : int
        Number of test instances.
    """

    exception_raised: bool
    exception_type: Optional[str]
    n_instances: int


def _build_viz_explainer() -> tuple:
    """Build a deterministic fitted+calibrated WrapCalibratedExplainer for visualization tests."""
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


def run_visualization_tif_scenario() -> VizObservation:
    """Stimulate CE-REQ-VIZ-SMOKE-001 through WrapCalibratedExplainer.explain_factual + plot.

    TIF ID: CE-TIF-VIZ-001

    Requirements served:
      CE-REQ-VIZ-SMOKE-001 (observation: exception_raised)

    Uses the Agg backend to avoid display output. Cleans up figure state after the call.

    Returns
    -------
    VizObservation
        Structured observations. Tests assert on these fields.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
    except ImportError as exc:
        return VizObservation(
            exception_raised=True,
            exception_type=type(exc).__name__,
            n_instances=_N_TEST,
        )

    explainer, X_test = _build_viz_explainer()
    n_instances = len(X_test)

    try:
        explanations = explainer.explain_factual(X_test)
        explanations.plot(show=False)
        plt.close("all")
    except Exception as exc:
        plt.close("all")
        return VizObservation(
            exception_raised=True,
            exception_type=type(exc).__name__,
            n_instances=n_instances,
        )

    return VizObservation(
        exception_raised=False,
        exception_type=None,
        n_instances=n_instances,
    )
