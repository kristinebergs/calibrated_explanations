"""Generate raw evidence records by executing TIF scenarios.

Writes one JSON file per TIF interface into reports/verification/.
Each file records the structured observations from the TIF run against
the acceptance criteria defined in the corresponding requirement files.

Usage:
    python scripts/generate_tif_evidence.py

Output:
    reports/verification/CE-EVID-<AREA>-<NNN>-20260622.json  (one per TIF group)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[1]
_TIF_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"
_OUT_DIR = _REPO_ROOT / "reports" / "verification"

if str(_TIF_DIR) not in sys.path:
    sys.path.insert(0, str(_TIF_DIR))

import calibrated_explanations

_COMMIT_SHA = "ba0f95e1"
_PACKAGE_VERSION = calibrated_explanations.__version__
_TIMESTAMP = datetime.now(timezone.utc).isoformat()
_DATE_SUFFIX = "20260622"
_DATASET_CLF = (
    "sklearn make_classification n_samples=120 n_features=4 "
    "n_informative=3 n_redundant=1 random_seed=42"
)
_DATASET_REG = (
    "sklearn make_regression n_samples=150 n_features=4 "
    "n_informative=3 noise=10 random_seed=42"
)


def _obs(o) -> dict:
    return asdict(o) if hasattr(o, "__dataclass_fields__") else dict(o)


def _write(evidence_id: str, payload: dict) -> Path:
    out = _OUT_DIR / f"{evidence_id}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out.name}")
    return out


# ---------------------------------------------------------------------------
# CE-TIF-EXPL-001 — factual + alternative explanation
# ---------------------------------------------------------------------------
def run_expl():
    from tif_explanation import run_factual_tif_scenario, run_alternative_tif_scenario

    factual = run_factual_tif_scenario()
    alt = run_alternative_tif_scenario()

    # acceptance checks
    def _check(label, field, expected, observed):
        status = "pass" if observed == expected else "fail"
        return {
            "scenario_id": label,
            "result": status,
            "acceptance": {
                "criterion_ref": label.split("_")[0],
                "expected": f"{field} == {expected!r}",
                "observed": f"{field} == {observed!r}",
            },
        }

    scenarios = [
        {
            "scenario_id": "factual_api_contract",
            "observations": _obs(factual),
            "result": "pass" if not factual.exception_raised else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-EXPL-API-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": factual.exception_raised,
                    "result": "pass" if not factual.exception_raised else "fail",
                },
            ],
        },
        {
            "scenario_id": "factual_return_contract",
            "observations": _obs(factual),
            "result": "pass"
            if (
                not factual.result_is_none
                and factual.result_len == factual.n_instances
                and not factual.first_item_is_none
                and factual.feature_weights_accessible
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-EXPL-RETURN-001",
                    "field": "result_is_none",
                    "expected": False,
                    "observed": factual.result_is_none,
                    "result": "pass" if not factual.result_is_none else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-EXPL-RETURN-001",
                    "field": "result_len == n_instances",
                    "expected": True,
                    "observed": factual.result_len == factual.n_instances,
                    "result": "pass" if factual.result_len == factual.n_instances else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-EXPL-RETURN-001",
                    "field": "feature_weights_accessible",
                    "expected": True,
                    "observed": factual.feature_weights_accessible,
                    "result": "pass" if factual.feature_weights_accessible else "fail",
                },
            ],
        },
        {
            "scenario_id": "alternative_api_contract",
            "observations": _obs(alt),
            "result": "pass" if not alt.exception_raised else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-EXPL-API-002",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": alt.exception_raised,
                    "result": "pass" if not alt.exception_raised else "fail",
                },
            ],
        },
        {
            "scenario_id": "alternative_return_contract",
            "observations": _obs(alt),
            "result": "pass"
            if (
                not alt.result_is_none
                and alt.result_len == alt.n_instances
                and alt.result_type_name == "AlternativeExplanations"
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-EXPL-ALT-RETURN-001",
                    "field": "result_type_name",
                    "expected": "AlternativeExplanations",
                    "observed": alt.result_type_name,
                    "result": "pass" if alt.result_type_name == "AlternativeExplanations" else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-EXPL-ALT-RETURN-001",
                    "field": "result_len == n_instances",
                    "expected": True,
                    "observed": alt.result_len == alt.n_instances,
                    "result": "pass" if alt.result_len == alt.n_instances else "fail",
                },
            ],
        },
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-EXPL-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-EXPL-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-EXPL-001", "CE-CAP-EXPL-002"],
            "requirement_ids": [
                "CE-REQ-EXPL-API-001",
                "CE-REQ-EXPL-RETURN-001",
                "CE-REQ-EXPL-API-002",
                "CE-REQ-EXPL-ALT-RETURN-001",
            ],
            "adr_refs": ["ADR-008", "ADR-015", "ADR-026"],
            "tif_ids": ["CE-TIF-EXPL-001"],
            "verification_type": "behavioral_contract",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_CLF,
            "random_seed": 42,
            "configuration": {},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-PRED-001 — uncertainty prediction intervals
# ---------------------------------------------------------------------------
def run_pred():
    from tif_prediction import run_prediction_tif_scenario

    default = run_prediction_tif_scenario()
    custom = run_prediction_tif_scenario(low_high_percentiles=(10, 90))

    scenarios = [
        {
            "scenario_id": "predict_uq_interval_default",
            "observations": _obs(default),
            "result": "pass"
            if (
                not default.exception_raised
                and default.y_hat_len == default.n_instances
                and not default.low_is_none
                and not default.high_is_none
                and default.bounds_ordered
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-PRED-API-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": default.exception_raised,
                    "result": "pass" if not default.exception_raised else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-API-001",
                    "field": "y_hat_len == n_instances",
                    "expected": True,
                    "observed": default.y_hat_len == default.n_instances,
                    "result": "pass" if default.y_hat_len == default.n_instances else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-API-001",
                    "field": "low_is_none",
                    "expected": False,
                    "observed": default.low_is_none,
                    "result": "pass" if not default.low_is_none else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-API-001",
                    "field": "high_is_none",
                    "expected": False,
                    "observed": default.high_is_none,
                    "result": "pass" if not default.high_is_none else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-INTERVAL-BOUNDS-001",
                    "field": "bounds_ordered (low <= high for all i)",
                    "expected": True,
                    "observed": default.bounds_ordered,
                    "result": "pass" if default.bounds_ordered else "fail",
                },
            ],
        },
        {
            "scenario_id": "predict_uq_interval_percentiles_10_90",
            "observations": _obs(custom),
            "result": "pass"
            if (
                not custom.exception_raised
                and custom.bounds_ordered
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-PRED-INTERVAL-BOUNDS-001",
                    "field": "accepts low_high_percentiles without exception",
                    "expected": False,
                    "observed": custom.exception_raised,
                    "result": "pass" if not custom.exception_raised else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-INTERVAL-BOUNDS-001",
                    "field": "bounds_ordered with custom percentiles",
                    "expected": True,
                    "observed": custom.bounds_ordered,
                    "result": "pass" if custom.bounds_ordered else "fail",
                },
            ],
        },
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-PRED-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-PRED-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-PRED-001"],
            "requirement_ids": ["CE-REQ-PRED-API-001", "CE-REQ-PRED-INTERVAL-BOUNDS-001"],
            "adr_refs": ["ADR-013", "ADR-021"],
            "tif_ids": ["CE-TIF-PRED-001"],
            "verification_type": "behavioral_contract",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_REG,
            "random_seed": 42,
            "configuration": {},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-PRED-CLASS-001 — classification predict_proba / predict
# ---------------------------------------------------------------------------
def run_pred_class():
    from tif_classification import run_classification_tif_scenario

    obs = run_classification_tif_scenario()

    scenarios = [
        {
            "scenario_id": "classification_api_and_bounds",
            "observations": _obs(obs),
            "result": "pass"
            if (
                not obs.exception_raised
                and not obs.proba_is_none
                and obs.proba_len == obs.n_instances
                and obs.proba_min is not None and obs.proba_min >= 0.0
                and obs.proba_max is not None and obs.proba_max <= 1.0
                and not obs.labels_is_none
                and obs.labels_len == obs.n_instances
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-PRED-CLASS-API-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": obs.exception_raised,
                    "result": "pass" if not obs.exception_raised else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-CLASS-API-001",
                    "field": "proba_len == n_instances",
                    "expected": True,
                    "observed": obs.proba_len == obs.n_instances,
                    "result": "pass" if obs.proba_len == obs.n_instances else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-CLASS-API-001",
                    "field": "labels_len == n_instances",
                    "expected": True,
                    "observed": obs.labels_len == obs.n_instances,
                    "result": "pass" if obs.labels_len == obs.n_instances else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-CLASS-BOUNDS-001",
                    "field": "proba_min >= 0.0",
                    "expected": True,
                    "observed": obs.proba_min >= 0.0 if obs.proba_min is not None else False,
                    "result": "pass" if (obs.proba_min is not None and obs.proba_min >= 0.0) else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-CLASS-BOUNDS-001",
                    "field": "proba_max <= 1.0",
                    "expected": True,
                    "observed": obs.proba_max <= 1.0 if obs.proba_max is not None else False,
                    "result": "pass" if (obs.proba_max is not None and obs.proba_max <= 1.0) else "fail",
                },
            ],
        }
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-PRED-CLASS-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-PRED-CLASS-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-PRED-CLASS-001"],
            "requirement_ids": ["CE-REQ-PRED-CLASS-API-001", "CE-REQ-PRED-CLASS-BOUNDS-001"],
            "adr_refs": ["ADR-021"],
            "tif_ids": ["CE-TIF-PRED-CLASS-001"],
            "verification_type": "numerical_behavior",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_CLF,
            "random_seed": 42,
            "configuration": {},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-PRED-PROB-001 — probabilistic regression threshold query
# ---------------------------------------------------------------------------
def run_pred_prob():
    from tif_prob_regression import run_prob_regression_tif_scenario

    obs = run_prob_regression_tif_scenario(threshold=0.0)

    scenarios = [
        {
            "scenario_id": "prob_regression_threshold_0",
            "observations": _obs(obs),
            "result": "pass"
            if (
                not obs.exception_raised
                and not obs.result_is_none
                and obs.proba_len == obs.n_instances
                and obs.proba_min is not None and obs.proba_min >= 0.0
                and obs.proba_max is not None and obs.proba_max <= 1.0
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-PRED-PROB-API-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": obs.exception_raised,
                    "result": "pass" if not obs.exception_raised else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-PROB-API-001",
                    "field": "proba_len == n_instances",
                    "expected": True,
                    "observed": obs.proba_len == obs.n_instances,
                    "result": "pass" if obs.proba_len == obs.n_instances else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-PROB-BOUNDS-001",
                    "field": "proba_min >= 0.0",
                    "expected": True,
                    "observed": obs.proba_min >= 0.0 if obs.proba_min is not None else False,
                    "result": "pass" if (obs.proba_min is not None and obs.proba_min >= 0.0) else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-PRED-PROB-BOUNDS-001",
                    "field": "proba_max <= 1.0",
                    "expected": True,
                    "observed": obs.proba_max <= 1.0 if obs.proba_max is not None else False,
                    "result": "pass" if (obs.proba_max is not None and obs.proba_max <= 1.0) else "fail",
                },
            ],
        }
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-PRED-PROB-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-PRED-PROB-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-PRED-PROB-001"],
            "requirement_ids": ["CE-REQ-PRED-PROB-API-001", "CE-REQ-PRED-PROB-BOUNDS-001"],
            "adr_refs": ["ADR-021"],
            "tif_ids": ["CE-TIF-PRED-PROB-001"],
            "verification_type": "numerical_behavior",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_REG,
            "random_seed": 42,
            "configuration": {"threshold": 0.0},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-GUARD-001 — guarded explanation
# ---------------------------------------------------------------------------
def run_guard():
    from tif_guard import run_guard_tif_scenario

    obs = run_guard_tif_scenario()

    scenarios = [
        {
            "scenario_id": "explain_factual_with_guarded_options",
            "observations": _obs(obs),
            "result": "pass"
            if (
                not obs.exception_raised
                and not obs.result_is_none
                and obs.result_len == obs.n_instances
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-GUARD-API-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": obs.exception_raised,
                    "result": "pass" if not obs.exception_raised else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-GUARD-API-001",
                    "field": "result_is_none",
                    "expected": False,
                    "observed": obs.result_is_none,
                    "result": "pass" if not obs.result_is_none else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-GUARD-API-001",
                    "field": "result_len == n_instances",
                    "expected": True,
                    "observed": obs.result_len == obs.n_instances,
                    "result": "pass" if obs.result_len == obs.n_instances else "fail",
                },
            ],
        }
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-GUARD-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-GUARD-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-GUARD-001"],
            "requirement_ids": ["CE-REQ-GUARD-API-001"],
            "adr_refs": ["ADR-032", "ADR-038"],
            "tif_ids": ["CE-TIF-GUARD-001"],
            "verification_type": "api_contract",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_CLF,
            "random_seed": 42,
            "configuration": {},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-REJECT-001 — reject policy
# ---------------------------------------------------------------------------
def run_reject():
    from tif_reject import run_reject_tif_scenario

    obs = run_reject_tif_scenario()

    scenarios = [
        {
            "scenario_id": "explain_factual_with_reject_policy_flag",
            "observations": _obs(obs),
            "result": "pass"
            if (
                not obs.exception_raised
                and not obs.result_is_none
                and obs.result_len == obs.n_instances
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-REJECT-API-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": obs.exception_raised,
                    "result": "pass" if not obs.exception_raised else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-REJECT-API-001",
                    "field": "result_is_none",
                    "expected": False,
                    "observed": obs.result_is_none,
                    "result": "pass" if not obs.result_is_none else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-REJECT-API-001",
                    "field": "result_len == n_instances",
                    "expected": True,
                    "observed": obs.result_len == obs.n_instances,
                    "result": "pass" if obs.result_len == obs.n_instances else "fail",
                },
            ],
        }
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-REJECT-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-REJECT-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-REJECT-001"],
            "requirement_ids": ["CE-REQ-REJECT-API-001"],
            "adr_refs": ["ADR-029", "ADR-038"],
            "tif_ids": ["CE-TIF-REJECT-001"],
            "verification_type": "api_contract",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_CLF,
            "random_seed": 42,
            "configuration": {},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-MOND-001 — Mondrian calibration
# ---------------------------------------------------------------------------
def run_mond():
    from tif_mondrian import run_mondrian_tif_scenario

    obs = run_mondrian_tif_scenario()

    scenarios = [
        {
            "scenario_id": "calibrate_with_mondrian_categorizer",
            "observations": _obs(obs),
            "result": "pass"
            if not obs.exception_raised and obs.calibrated
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-MOND-API-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": obs.exception_raised,
                    "result": "pass" if not obs.exception_raised else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-MOND-API-001",
                    "field": "calibrated",
                    "expected": True,
                    "observed": obs.calibrated,
                    "result": "pass" if obs.calibrated else "fail",
                },
            ],
        }
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-MOND-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-MOND-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-MOND-001"],
            "requirement_ids": ["CE-REQ-MOND-API-001"],
            "adr_refs": ["ADR-013"],
            "tif_ids": ["CE-TIF-MOND-001"],
            "verification_type": "api_contract",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_CLF,
            "random_seed": 42,
            "configuration": {"mondrian_fn": "sign of feature 0 (2 categories)"},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-NARR-001 — narrative output
# ---------------------------------------------------------------------------
def run_narr():
    from tif_narrative import run_narrative_tif_scenario

    obs = run_narrative_tif_scenario()

    scenarios = [
        {
            "scenario_id": "to_narrative_text_format",
            "observations": _obs(obs),
            "result": "pass"
            if (
                not obs.exception_raised
                and not obs.result_is_none
                and obs.result_is_str
                and obs.result_len is not None
                and obs.result_len > 0
            )
            else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-NARR-API-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": obs.exception_raised,
                    "result": "pass" if not obs.exception_raised else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-NARR-API-001",
                    "field": "result_is_str",
                    "expected": True,
                    "observed": obs.result_is_str,
                    "result": "pass" if obs.result_is_str else "fail",
                },
                {
                    "criterion_ref": "CE-REQ-NARR-API-001",
                    "field": "result_len > 0",
                    "expected": True,
                    "observed": obs.result_len > 0 if obs.result_len is not None else False,
                    "result": "pass" if (obs.result_len is not None and obs.result_len > 0) else "fail",
                },
            ],
        }
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-NARR-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-NARR-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-NARR-001"],
            "requirement_ids": ["CE-REQ-NARR-API-001"],
            "adr_refs": ["ADR-008"],
            "tif_ids": ["CE-TIF-NARR-001"],
            "verification_type": "api_contract",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_CLF,
            "random_seed": 42,
            "configuration": {"output_format": "text"},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-VIZ-001 — visualization smoke test
# ---------------------------------------------------------------------------
def run_viz():
    from tif_visualization import run_visualization_tif_scenario

    obs = run_visualization_tif_scenario()

    scenarios = [
        {
            "scenario_id": "plot_no_raise_agg_backend",
            "observations": _obs(obs),
            "result": "pass" if not obs.exception_raised else "fail",
            "acceptance": [
                {
                    "criterion_ref": "CE-REQ-VIZ-SMOKE-001",
                    "field": "exception_raised",
                    "expected": False,
                    "observed": obs.exception_raised,
                    "result": "pass" if not obs.exception_raised else "fail",
                },
            ],
        }
    ]

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-VIZ-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-VIZ-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-VIZ-001"],
            "requirement_ids": ["CE-REQ-VIZ-SMOKE-001"],
            "adr_refs": ["ADR-023", "ADR-036", "ADR-037"],
            "tif_ids": ["CE-TIF-VIZ-001"],
            "verification_type": "empirical_smoke",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_CLF,
            "random_seed": 42,
            "configuration": {"backend": "Agg", "show": False},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# CE-TIF-FILTER-001 — all five filter operations
# ---------------------------------------------------------------------------
def run_filter():
    from tif_filter import run_filter_tif_scenario

    filter_types = ["super", "semi", "counter", "ensured", "pareto"]
    req_map = {
        "super": "CE-REQ-EXPL-FILTER-SUPER-001",
        "semi": "CE-REQ-EXPL-FILTER-SEMI-001",
        "counter": "CE-REQ-EXPL-FILTER-COUNTER-001",
        "ensured": "CE-REQ-EXPL-FILTER-ENSURED-001",
        "pareto": "CE-REQ-EXPL-FILTER-PARETO-001",
    }

    scenarios = []
    for ft in filter_types:
        obs = run_filter_tif_scenario(filter_type=ft)
        req = req_map[ft]
        passed = (
            not obs.exception_raised
            and not obs.collection_result_is_none
            and obs.collection_result_len == obs.n_instances
            and not obs.individual_result_is_none
            and not obs.alias_result_is_none
            and obs.alias_result_len == obs.n_instances
        )
        scenarios.append(
            {
                "scenario_id": f"filter_{ft}",
                "observations": _obs(obs),
                "result": "pass" if passed else "fail",
                "acceptance": [
                    {
                        "criterion_ref": req,
                        "field": "exception_raised",
                        "expected": False,
                        "observed": obs.exception_raised,
                        "result": "pass" if not obs.exception_raised else "fail",
                    },
                    {
                        "criterion_ref": req,
                        "field": "collection_result_is_none",
                        "expected": False,
                        "observed": obs.collection_result_is_none,
                        "result": "pass" if not obs.collection_result_is_none else "fail",
                    },
                    {
                        "criterion_ref": req,
                        "field": "collection_result_len == n_instances",
                        "expected": True,
                        "observed": obs.collection_result_len == obs.n_instances,
                        "result": "pass" if obs.collection_result_len == obs.n_instances else "fail",
                    },
                    {
                        "criterion_ref": req,
                        "field": "individual_result_is_none",
                        "expected": False,
                        "observed": obs.individual_result_is_none,
                        "result": "pass" if not obs.individual_result_is_none else "fail",
                    },
                    {
                        "criterion_ref": req,
                        "field": "alias_result_is_none",
                        "expected": False,
                        "observed": obs.alias_result_is_none,
                        "result": "pass" if not obs.alias_result_is_none else "fail",
                    },
                ],
            }
        )

    overall = "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"

    _write(
        f"CE-EVID-FILTER-001-{_DATE_SUFFIX}",
        {
            "evidence_id": f"CE-EVID-FILTER-001-{_DATE_SUFFIX}",
            "claim_ids": ["CE-CAP-EXPL-FILTER-001"],
            "requirement_ids": [
                "CE-REQ-EXPL-FILTER-SUPER-001",
                "CE-REQ-EXPL-FILTER-SEMI-001",
                "CE-REQ-EXPL-FILTER-COUNTER-001",
                "CE-REQ-EXPL-FILTER-ENSURED-001",
                "CE-REQ-EXPL-FILTER-PARETO-001",
            ],
            "adr_refs": ["ADR-027"],
            "tif_ids": ["CE-TIF-FILTER-001"],
            "verification_type": "api_contract",
            "result": overall,
            "timestamp": _TIMESTAMP,
            "commit_sha": _COMMIT_SHA,
            "package_version": _PACKAGE_VERSION,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "dataset_id": _DATASET_CLF,
            "random_seed": 42,
            "configuration": {},
            "scenarios": scenarios,
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating TIF evidence records -> {_OUT_DIR}")
    print(f"  package: {_PACKAGE_VERSION}  commit: {_COMMIT_SHA}")
    print()

    runners = [
        ("CE-TIF-EXPL-001", run_expl),
        ("CE-TIF-PRED-001", run_pred),
        ("CE-TIF-PRED-CLASS-001", run_pred_class),
        ("CE-TIF-PRED-PROB-001", run_pred_prob),
        ("CE-TIF-GUARD-001", run_guard),
        ("CE-TIF-REJECT-001", run_reject),
        ("CE-TIF-MOND-001", run_mond),
        ("CE-TIF-NARR-001", run_narr),
        ("CE-TIF-VIZ-001", run_viz),
        ("CE-TIF-FILTER-001", run_filter),
    ]

    failed = []
    for tif_id, runner in runners:
        print(f"Running {tif_id}...")
        try:
            runner()
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append(tif_id)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(runners)} TIF evidence records written.")
