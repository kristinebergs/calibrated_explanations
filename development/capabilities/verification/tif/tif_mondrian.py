"""TIF verification interface for CE Mondrian conditional calibration capabilities.

TIF ID: CE-TIF-MOND-001

Requirements served:
  CE-REQ-MOND-API-001 — calibrate with mc= Mondrian categorizer API contract

Tests call run_mondrian_tif_scenario() and assert on the returned
MondrianObservation against acceptance criteria from the requirement files.
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
_N_TEST = 3


@dataclass
class MondrianObservation:
    """Structured observation returned by Mondrian calibration TIF scenarios.

    Fields
    ------
    exception_raised : bool
        Whether an exception was raised during calibrate.
    exception_type : str or None
        Exception class name if raised; None otherwise.
    calibrated : bool
        Whether wrapper.calibrated is True after calibration.
    n_instances : int
        Number of test instances (X_test size; calibration size not reported here).
    """

    exception_raised: bool
    exception_type: Optional[str]
    calibrated: bool
    n_instances: int


def _mondrian_fn(x: np.ndarray) -> np.ndarray:
    """Partition by sign of first feature (2 categories: 0, 1)."""
    return (np.asarray(x)[:, 0] >= 0).astype(int)


def run_mondrian_tif_scenario() -> MondrianObservation:
    """Stimulate CE-REQ-MOND-API-001 through WrapCalibratedExplainer with mc= param.

    TIF ID: CE-TIF-MOND-001

    Requirements served:
      CE-REQ-MOND-API-001 (observation: exception_raised, calibrated)

    Returns
    -------
    MondrianObservation
        Structured observations. Tests assert on these fields.
    """
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

    n_instances = len(X_test)

    try:
        explainer.calibrate(X_cal, y_cal, mc=_mondrian_fn)
    except Exception as exc:
        return MondrianObservation(
            exception_raised=True,
            exception_type=type(exc).__name__,
            calibrated=False,
            n_instances=n_instances,
        )

    return MondrianObservation(
        exception_raised=False,
        exception_type=None,
        calibrated=bool(explainer.calibrated),
        n_instances=n_instances,
    )
