"""TIF verification interface for CE Mondrian conditional calibration capabilities.

TIF ID: CE-TIF-MOND-001

Requirements served:
  CE-REQ-MOND-API-001 — calibrate with mc= Mondrian categorizer API contract

Tests call run_mondrian_tif_scenario() and assert on the returned
MondrianObservation against acceptance criteria from the requirement files.
"""

from __future__ import annotations

import pickle
import warnings
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


def _build_wrapper_fixture():
    """Return a deterministic fitted wrapper and Mondrian fixture data."""
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
    wrapper = WrapCalibratedExplainer(
        RandomForestClassifier(n_estimators=10, random_state=_RNG_SEED)
    )
    wrapper.fit(X_proper, y_proper)
    return wrapper, X_cal, y_cal, X_test, _mondrian_fn


def _observe_exception(operation) -> MondrianObservation:
    """Run an operation and capture its exception outcome."""
    try:
        calibrated = bool(operation())
    except Exception as exc:
        return MondrianObservation(
            exception_raised=True,
            exception_type=type(exc).__name__,
            calibrated=False,
            n_instances=_N_TEST,
        )
    return MondrianObservation(
        exception_raised=False,
        exception_type=None,
        calibrated=calibrated,
        n_instances=_N_TEST,
    )


def run_mondrian_inline_bins_omitted_scenario() -> MondrianObservation:
    """Observe inline bins calibration followed by inference without bins."""
    wrapper, X_cal, y_cal, X_test, mondrian_fn = _build_wrapper_fixture()

    def operation() -> bool:
        wrapper.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal))
        wrapper.predict_proba(X_test)
        return wrapper.calibrated

    return _observe_exception(operation)


def run_mondrian_global_with_bins_scenario() -> MondrianObservation:
    """Observe global calibration followed by conditional inference bins."""
    wrapper, X_cal, y_cal, X_test, mondrian_fn = _build_wrapper_fixture()

    def operation() -> bool:
        wrapper.calibrate(X_cal, y_cal)
        wrapper.predict_proba(X_test, bins=mondrian_fn(X_test))
        return wrapper.calibrated

    return _observe_exception(operation)


def run_mondrian_unknown_label_scenario() -> MondrianObservation:
    """Observe inference with labels outside the calibration vocabulary."""
    wrapper, X_cal, y_cal, X_test, mondrian_fn = _build_wrapper_fixture()

    def operation() -> bool:
        wrapper.calibrate(X_cal, y_cal, bins=mondrian_fn(X_cal))
        bins = mondrian_fn(X_test)
        bins[0] = 99
        wrapper.predict_proba(X_test, bins=bins)
        return wrapper.calibrated

    return _observe_exception(operation)


def run_mondrian_lifecycle_reset_scenario() -> MondrianObservation:
    """Observe that plain recalibration resets conditional state."""
    wrapper, X_cal, y_cal, X_test, mondrian_fn = _build_wrapper_fixture()

    def operation() -> bool:
        wrapper.calibrate(X_cal, y_cal, mc=mondrian_fn)
        wrapper.calibrate(X_cal, y_cal)
        wrapper.predict_proba(X_test)
        return wrapper.calibrated and not wrapper.explainer.is_mondrian()

    return _observe_exception(operation)


def run_mondrian_reuse_without_mc_scenario() -> MondrianObservation:
    """Observe reuse_conditional without a stored categorizer."""
    wrapper, X_cal, y_cal, _X_test, _mondrian_fn = _build_wrapper_fixture()
    return _observe_exception(lambda: wrapper.calibrate(X_cal, y_cal, reuse_conditional=True))


def run_mondrian_pickle_drop_scenario() -> MondrianObservation:
    """Observe that pickle drops mc visibly and loaded inference requires bins."""
    wrapper, X_cal, y_cal, X_test, mondrian_fn = _build_wrapper_fixture()

    def operation() -> bool:
        wrapper.calibrate(X_cal, y_cal, mc=mondrian_fn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            restored = pickle.loads(  # noqa: S301  # nosec B301 - trusted in-process TIF round-trip
                pickle.dumps(wrapper)
            )
        restored.predict_proba(X_test)
        return restored.calibrated

    return _observe_exception(operation)


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
    """Build a complete evidence payload for CE-TIF-MOND-001.

    Called by scripts/generate_tif_evidence.py during dynamic discovery.
    Envelope metadata is injected from the TIF spec by the caller.
    """
    from tif_evidence_helpers import (
        acceptance_entry,
        build_payload,
        obs_to_dict,
        scenario_entry,
    )

    obs = run_mondrian_tif_scenario()
    inline_omitted = run_mondrian_inline_bins_omitted_scenario()
    global_with_bins = run_mondrian_global_with_bins_scenario()
    unknown_label = run_mondrian_unknown_label_scenario()
    lifecycle_reset = run_mondrian_lifecycle_reset_scenario()
    reuse_without_mc = run_mondrian_reuse_without_mc_scenario()
    pickle_drop = run_mondrian_pickle_drop_scenario()
    scenarios = [
        scenario_entry(
            "calibrate_with_mondrian_categorizer",
            obs_to_dict(obs),
            [
                acceptance_entry(
                    "CE-REQ-MOND-API-001", "exception_raised", False, obs.exception_raised
                ),
                acceptance_entry("CE-REQ-MOND-API-001", "calibrated", True, obs.calibrated),
            ],
        ),
        scenario_entry(
            "inline_bins_require_inference_bins",
            obs_to_dict(inline_omitted),
            [
                acceptance_entry(
                    "CE-REQ-MOND-CONS-001",
                    "exception_type",
                    "ValidationError",
                    inline_omitted.exception_type,
                )
            ],
        ),
        scenario_entry(
            "global_calibration_rejects_inference_bins",
            obs_to_dict(global_with_bins),
            [
                acceptance_entry(
                    "CE-REQ-MOND-CONS-001",
                    "exception_type",
                    "ConfigurationError",
                    global_with_bins.exception_type,
                )
            ],
        ),
        scenario_entry(
            "unknown_inference_label_rejected",
            obs_to_dict(unknown_label),
            [
                acceptance_entry(
                    "CE-REQ-MOND-VAL-001",
                    "exception_type",
                    "ValidationError",
                    unknown_label.exception_type,
                )
            ],
        ),
        scenario_entry(
            "plain_recalibration_resets_conditional_state",
            obs_to_dict(lifecycle_reset),
            [
                acceptance_entry(
                    "CE-REQ-MOND-LIFE-001",
                    "exception_raised",
                    False,
                    lifecycle_reset.exception_raised,
                ),
                acceptance_entry(
                    "CE-REQ-MOND-LIFE-001",
                    "calibrated",
                    True,
                    lifecycle_reset.calibrated,
                ),
            ],
        ),
        scenario_entry(
            "reuse_conditional_without_mc_rejected",
            obs_to_dict(reuse_without_mc),
            [
                acceptance_entry(
                    "CE-REQ-MOND-LIFE-001",
                    "exception_type",
                    "ValidationError",
                    reuse_without_mc.exception_type,
                )
            ],
        ),
        scenario_entry(
            "pickle_drop_requires_explicit_bins_after_load",
            obs_to_dict(pickle_drop),
            [
                acceptance_entry(
                    "CE-REQ-MOND-SER-001",
                    "exception_type",
                    "ValidationError",
                    pickle_drop.exception_type,
                )
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
        configuration={"mondrian_fn": "sign of feature 0 (2 categories)"},
    )
