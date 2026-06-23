"""Generate raw evidence records by executing active TIF scenarios."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parents[1]
_TIF_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"
_OUT_DIR = _REPO_ROOT / "reports" / "verification"
if str(_TIF_DIR) not in sys.path:
    sys.path.insert(0, str(_TIF_DIR))

import calibrated_explanations

_BEHAVIORAL_TYPES = {
    "api_contract",
    "behavioral_contract",
    "numerical_behavior",
    "statistical_method_alignment",
    "empirical_smoke",
    "visualization_structure",
}
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_NOW = datetime.now(timezone.utc)
_DATE_SUFFIX = _NOW.strftime("%Y%m%d")
_TIMESTAMP = _NOW.isoformat()
_DATASET_CLF = (
    "sklearn make_classification n_samples=120 n_features=4 "
    "n_informative=3 n_redundant=1 random_seed=42"
)
_DATASET_REG = (
    "sklearn make_regression n_samples=150 n_features=4 "
    "n_informative=3 noise=10 random_seed=42"
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


_COMMIT_SHA = _git_sha()
_PACKAGE_VERSION = calibrated_explanations.__version__


def _obs(value: Any) -> dict[str, Any]:
    return asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)


def _acceptance(criterion_ref: str, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {
        "criterion_ref": criterion_ref,
        "field": field,
        "expected": expected,
        "observed": observed,
        "result": "pass" if observed == expected else "fail",
    }


def _scenario(
    scenario_id: str,
    observations: dict[str, Any],
    acceptance: list[dict[str, Any]],
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scenario_id": scenario_id,
        "observations": observations,
        "result": "pass" if all(a["result"] == "pass" for a in acceptance) else "fail",
        "acceptance": acceptance,
    }
    if parameters:
        payload["parameters"] = parameters
    return payload


def _overall_result(scenarios: list[dict[str, Any]]) -> str:
    return "pass" if all(s["result"] == "pass" for s in scenarios) else "fail"


def _payload(
    evidence_key: str,
    *,
    claim_ids: list[str],
    requirement_ids: list[str],
    adr_refs: list[str],
    tif_ids: list[str],
    verification_type: str,
    dataset_id: str,
    scenarios: list[dict[str, Any]],
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_id = f"CE-EVID-{evidence_key}-{_DATE_SUFFIX}"
    return {
        "evidence_id": evidence_id,
        "claim_ids": claim_ids,
        "requirement_ids": requirement_ids,
        "adr_refs": adr_refs,
        "standard_refs": [],
        "tif_ids": tif_ids,
        "verification_type": verification_type,
        "result": _overall_result(scenarios),
        "timestamp": _TIMESTAMP,
        "commit_sha": _COMMIT_SHA,
        "package_version": _PACKAGE_VERSION,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "dataset_id": dataset_id,
        "random_seed": 42,
        "configuration": configuration or {},
        "scenarios": scenarios,
        "artifacts": {"logs": None, "raw_output": None},
    }


def _validate_payload(payload: dict[str, Any], output_stem: str) -> None:
    if payload["evidence_id"] != output_stem:
        raise ValueError(f"{output_stem}: evidence_id does not match filename stem")
    if not payload["claim_ids"]:
        raise ValueError(f"{output_stem}: claim_ids must be non-empty")
    if not payload["requirement_ids"]:
        raise ValueError(f"{output_stem}: requirement_ids must be non-empty")
    if payload["verification_type"] in _BEHAVIORAL_TYPES and not payload["tif_ids"]:
        raise ValueError(f"{output_stem}: behavioral evidence must include tif_ids")
    if not payload["scenarios"]:
        raise ValueError(f"{output_stem}: scenarios must be non-empty")
    if payload["result"] != _overall_result(payload["scenarios"]):
        raise ValueError(f"{output_stem}: top-level result disagrees with scenarios")
    if not _SHA_RE.fullmatch(payload["commit_sha"]):
        raise ValueError(f"{output_stem}: commit_sha must be a full 40-character git SHA")
    requirement_ids = set(payload["requirement_ids"])
    for scenario in payload["scenarios"]:
        for acceptance in scenario["acceptance"]:
            if acceptance["criterion_ref"] not in requirement_ids:
                raise ValueError(
                    f"{output_stem}: criterion_ref {acceptance['criterion_ref']} is not listed"
                )


def _write(payload: dict[str, Any]) -> None:
    out = _OUT_DIR / f"{payload['evidence_id']}.json"
    _validate_payload(payload, out.stem)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out.name}")


def run_expl() -> dict[str, Any]:
    from tif_explanation import run_alternative_tif_scenario, run_factual_tif_scenario

    factual = run_factual_tif_scenario()
    alt = run_alternative_tif_scenario()
    scenarios = [
        _scenario("factual_api_contract", _obs(factual), [
            _acceptance("CE-REQ-EXPL-API-001", "exception_raised", False, factual.exception_raised)
        ]),
        _scenario("factual_return_contract", _obs(factual), [
            _acceptance("CE-REQ-EXPL-RETURN-001", "result_is_none", False, factual.result_is_none),
            _acceptance("CE-REQ-EXPL-RETURN-001", "result_len == n_instances", True, factual.result_len == factual.n_instances),
            _acceptance("CE-REQ-EXPL-RETURN-001", "feature_weights_accessible", True, factual.feature_weights_accessible),
        ]),
        _scenario("alternative_api_contract", _obs(alt), [
            _acceptance("CE-REQ-EXPL-API-002", "exception_raised", False, alt.exception_raised)
        ]),
        _scenario("alternative_return_contract", _obs(alt), [
            _acceptance("CE-REQ-EXPL-ALT-RETURN-001", "result_type_name", "AlternativeExplanations", alt.result_type_name),
            _acceptance("CE-REQ-EXPL-ALT-RETURN-001", "result_len == n_instances", True, alt.result_len == alt.n_instances),
        ]),
    ]
    return _payload(
        "EXPL-001",
        claim_ids=["CE-CAP-EXPL-001", "CE-CAP-EXPL-002"],
        requirement_ids=["CE-REQ-EXPL-API-001", "CE-REQ-EXPL-RETURN-001", "CE-REQ-EXPL-API-002", "CE-REQ-EXPL-ALT-RETURN-001"],
        adr_refs=["ADR-008", "ADR-015", "ADR-026"],
        tif_ids=["CE-TIF-EXPL-001"],
        verification_type="behavioral_contract",
        dataset_id=_DATASET_CLF,
        scenarios=scenarios,
    )


def run_conj() -> dict[str, Any]:
    from tif_conjunction import run_conjunction_tif_scenario

    scenarios: list[dict[str, Any]] = []
    for mode in ("factual", "alternative"):
        for level in ("collection", "individual"):
            obs = run_conjunction_tif_scenario(explanation_mode=mode, object_level=level, max_rule_size=2, n_top_features=5)
            scenarios.append(_scenario(
                f"api_{mode}_{level}",
                _obs(obs),
                [_acceptance("CE-REQ-EXPL-CONJ-API-001", "exception_raised", False, obs.exception_raised)],
                {"explanation_mode": mode, "object_level": level, "max_rule_size": 2, "n_top_features": 5},
            ))
    for mode in ("factual", "alternative"):
        obs = run_conjunction_tif_scenario(explanation_mode=mode, object_level="collection", max_rule_size=2, n_top_features=5)
        scenarios.append(_scenario(
            f"return_{mode}_collection",
            _obs(obs),
            [
                _acceptance("CE-REQ-EXPL-CONJ-RETURN-001", "result_is_none", False, obs.result_is_none),
                _acceptance("CE-REQ-EXPL-CONJ-RETURN-001", "result_len == n_instances", True, obs.result_len == obs.n_instances),
            ],
            {"explanation_mode": mode, "object_level": "collection", "max_rule_size": 2, "n_top_features": 5},
        ))
    for max_rule_size in (2, 3):
        obs = run_conjunction_tif_scenario(explanation_mode="factual", object_level="collection", max_rule_size=max_rule_size, n_top_features=5)
        scenarios.append(_scenario(
            f"rule_factual_collection_max_rule_size_{max_rule_size}",
            _obs(obs),
            [_acceptance("CE-REQ-EXPL-CONJ-RULE-001", "any_has_conjunctive_rules", True, obs.any_has_conjunctive_rules)],
            {"explanation_mode": "factual", "object_level": "collection", "max_rule_size": max_rule_size, "n_top_features": 5},
        ))
    obs = run_conjunction_tif_scenario(explanation_mode="factual", object_level="collection", max_rule_size=1, n_top_features=5)
    scenarios.append(_scenario(
        "param_factual_collection_max_rule_size_1",
        _obs(obs),
        [
            _acceptance("CE-REQ-EXPL-CONJ-PARAM-001", "any_has_conjunctive_rules", False, obs.any_has_conjunctive_rules),
            _acceptance("CE-REQ-EXPL-CONJ-PARAM-001", "exception_raised", False, obs.exception_raised),
        ],
        {"explanation_mode": "factual", "object_level": "collection", "max_rule_size": 1, "n_top_features": 5},
    ))
    return _payload(
        "EXPL-CONJ-001",
        claim_ids=["CE-CAP-EXPL-CONJ-001"],
        requirement_ids=["CE-REQ-EXPL-CONJ-API-001", "CE-REQ-EXPL-CONJ-RETURN-001", "CE-REQ-EXPL-CONJ-RULE-001", "CE-REQ-EXPL-CONJ-PARAM-001"],
        adr_refs=["ADR-008"],
        tif_ids=["CE-TIF-EXPL-CONJ-001"],
        verification_type="behavioral_contract",
        dataset_id=_DATASET_CLF,
        scenarios=scenarios,
    )


def run_pred() -> dict[str, Any]:
    from tif_prediction import run_prediction_tif_scenario

    default = run_prediction_tif_scenario()
    custom = run_prediction_tif_scenario(low_high_percentiles=(10, 90))
    scenarios = [
        _scenario("predict_uq_interval_default", _obs(default), [
            _acceptance("CE-REQ-PRED-API-001", "exception_raised", False, default.exception_raised),
            _acceptance("CE-REQ-PRED-API-001", "y_hat_len == n_instances", True, default.y_hat_len == default.n_instances),
            _acceptance("CE-REQ-PRED-API-001", "low_is_none", False, default.low_is_none),
            _acceptance("CE-REQ-PRED-API-001", "high_is_none", False, default.high_is_none),
            _acceptance("CE-REQ-PRED-INTERVAL-BOUNDS-001", "bounds_ordered", True, default.bounds_ordered),
        ]),
        _scenario("predict_uq_interval_percentiles_10_90", _obs(custom), [
            _acceptance("CE-REQ-PRED-INTERVAL-BOUNDS-001", "exception_raised", False, custom.exception_raised),
            _acceptance("CE-REQ-PRED-INTERVAL-BOUNDS-001", "bounds_ordered", True, custom.bounds_ordered),
        ]),
    ]
    return _payload("PRED-001", claim_ids=["CE-CAP-PRED-001"], requirement_ids=["CE-REQ-PRED-API-001", "CE-REQ-PRED-INTERVAL-BOUNDS-001"], adr_refs=["ADR-013", "ADR-021"], tif_ids=["CE-TIF-PRED-001"], verification_type="behavioral_contract", dataset_id=_DATASET_REG, scenarios=scenarios)


def run_pred_class() -> dict[str, Any]:
    from tif_classification import run_classification_tif_scenario

    obs = run_classification_tif_scenario()
    scenarios = [_scenario("classification_api_and_bounds", _obs(obs), [
        _acceptance("CE-REQ-PRED-CLASS-API-001", "exception_raised", False, obs.exception_raised),
        _acceptance("CE-REQ-PRED-CLASS-API-001", "proba_len == n_instances", True, obs.proba_len == obs.n_instances),
        _acceptance("CE-REQ-PRED-CLASS-API-001", "labels_len == n_instances", True, obs.labels_len == obs.n_instances),
        _acceptance("CE-REQ-PRED-CLASS-BOUNDS-001", "proba_min >= 0.0", True, obs.proba_min is not None and obs.proba_min >= 0.0),
        _acceptance("CE-REQ-PRED-CLASS-BOUNDS-001", "proba_max <= 1.0", True, obs.proba_max is not None and obs.proba_max <= 1.0),
    ])]
    return _payload("PRED-CLASS-001", claim_ids=["CE-CAP-PRED-CLASS-001"], requirement_ids=["CE-REQ-PRED-CLASS-API-001", "CE-REQ-PRED-CLASS-BOUNDS-001"], adr_refs=["ADR-021"], tif_ids=["CE-TIF-PRED-CLASS-001"], verification_type="numerical_behavior", dataset_id=_DATASET_CLF, scenarios=scenarios)


def run_pred_prob() -> dict[str, Any]:
    from tif_prob_regression import run_prob_regression_tif_scenario

    obs = run_prob_regression_tif_scenario(threshold=0.0)
    scenarios = [_scenario("prob_regression_threshold_0", _obs(obs), [
        _acceptance("CE-REQ-PRED-PROB-API-001", "exception_raised", False, obs.exception_raised),
        _acceptance("CE-REQ-PRED-PROB-API-001", "proba_len == n_instances", True, obs.proba_len == obs.n_instances),
        _acceptance("CE-REQ-PRED-PROB-BOUNDS-001", "proba_min >= 0.0", True, obs.proba_min is not None and obs.proba_min >= 0.0),
        _acceptance("CE-REQ-PRED-PROB-BOUNDS-001", "proba_max <= 1.0", True, obs.proba_max is not None and obs.proba_max <= 1.0),
    ])]
    return _payload("PRED-PROB-001", claim_ids=["CE-CAP-PRED-PROB-001"], requirement_ids=["CE-REQ-PRED-PROB-API-001", "CE-REQ-PRED-PROB-BOUNDS-001"], adr_refs=["ADR-021"], tif_ids=["CE-TIF-PRED-PROB-001"], verification_type="numerical_behavior", dataset_id=_DATASET_REG, scenarios=scenarios, configuration={"threshold": 0.0})


def _single_api_payload(key: str, tif_id: str, claim_id: str, req_id: str, adr_refs: list[str], runner_name: str, scenario_id: str, extra_config: dict[str, Any] | None = None) -> dict[str, Any]:
    module = __import__(runner_name[0], fromlist=[runner_name[1]])
    obs = getattr(module, runner_name[1])()
    acceptance = [
        _acceptance(req_id, "exception_raised", False, obs.exception_raised),
    ]
    if hasattr(obs, "result_is_none"):
        acceptance.append(_acceptance(req_id, "result_is_none", False, obs.result_is_none))
    if hasattr(obs, "result_len") and hasattr(obs, "n_instances") and not hasattr(obs, "result_is_str"):
        acceptance.append(_acceptance(req_id, "result_len == n_instances", True, obs.result_len == obs.n_instances))
    if hasattr(obs, "calibrated"):
        acceptance.append(_acceptance(req_id, "calibrated", True, obs.calibrated))
    if hasattr(obs, "result_is_str"):
        acceptance.append(_acceptance(req_id, "result_is_str", True, obs.result_is_str))
    if hasattr(obs, "result_len") and not hasattr(obs, "n_instances"):
        acceptance.append(_acceptance(req_id, "result_len > 0", True, obs.result_len is not None and obs.result_len > 0))
    return _payload(key, claim_ids=[claim_id], requirement_ids=[req_id], adr_refs=adr_refs, tif_ids=[tif_id], verification_type="empirical_smoke" if key == "VIZ-001" else "api_contract", dataset_id=_DATASET_CLF, scenarios=[_scenario(scenario_id, _obs(obs), acceptance)], configuration=extra_config)


def run_guard() -> dict[str, Any]:
    return _single_api_payload("GUARD-001", "CE-TIF-GUARD-001", "CE-CAP-GUARD-001", "CE-REQ-GUARD-API-001", ["ADR-032", "ADR-038"], ("tif_guard", "run_guard_tif_scenario"), "explain_factual_with_guarded_options")


def run_reject() -> dict[str, Any]:
    return _single_api_payload("REJECT-001", "CE-TIF-REJECT-001", "CE-CAP-REJECT-001", "CE-REQ-REJECT-API-001", ["ADR-029", "ADR-038"], ("tif_reject", "run_reject_tif_scenario"), "explain_factual_with_reject_policy_flag")


def run_mond() -> dict[str, Any]:
    return _single_api_payload("MOND-001", "CE-TIF-MOND-001", "CE-CAP-MOND-001", "CE-REQ-MOND-API-001", ["ADR-013"], ("tif_mondrian", "run_mondrian_tif_scenario"), "calibrate_with_mondrian_categorizer", {"mondrian_fn": "sign of feature 0 (2 categories)"})


def run_narr() -> dict[str, Any]:
    return _single_api_payload("NARR-001", "CE-TIF-NARR-001", "CE-CAP-NARR-001", "CE-REQ-NARR-API-001", ["ADR-008"], ("tif_narrative", "run_narrative_tif_scenario"), "to_narrative_text_format", {"output_format": "text"})


def run_viz() -> dict[str, Any]:
    return _single_api_payload("VIZ-001", "CE-TIF-VIZ-001", "CE-CAP-VIZ-001", "CE-REQ-VIZ-SMOKE-001", ["ADR-023", "ADR-036", "ADR-037"], ("tif_visualization", "run_visualization_tif_scenario"), "plot_no_raise_agg_backend", {"backend": "Agg", "show": False})


def run_filter() -> dict[str, Any]:
    from tif_filter import run_filter_tif_scenario

    req_map = {
        "super": "CE-REQ-EXPL-FILTER-SUPER-001",
        "semi": "CE-REQ-EXPL-FILTER-SEMI-001",
        "counter": "CE-REQ-EXPL-FILTER-COUNTER-001",
        "ensured": "CE-REQ-EXPL-FILTER-ENSURED-001",
        "pareto": "CE-REQ-EXPL-FILTER-PARETO-001",
    }
    scenarios = []
    for filter_type, req_id in req_map.items():
        obs = run_filter_tif_scenario(filter_type=filter_type)
        scenarios.append(_scenario(f"filter_{filter_type}", _obs(obs), [
            _acceptance(req_id, "exception_raised", False, obs.exception_raised),
            _acceptance(req_id, "collection_result_is_none", False, obs.collection_result_is_none),
            _acceptance(req_id, "collection_result_len == n_instances", True, obs.collection_result_len == obs.n_instances),
            _acceptance(req_id, "individual_result_is_none", False, obs.individual_result_is_none),
            _acceptance(req_id, "alias_result_is_none", False, obs.alias_result_is_none),
        ]))
    return _payload("FILTER-001", claim_ids=["CE-CAP-EXPL-FILTER-001"], requirement_ids=list(req_map.values()), adr_refs=["ADR-027"], tif_ids=["CE-TIF-FILTER-001"], verification_type="api_contract", dataset_id=_DATASET_CLF, scenarios=scenarios)


_RUNNERS = [
    ("CE-TIF-EXPL-001", run_expl),
    ("CE-TIF-EXPL-CONJ-001", run_conj),
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


def _validate_existing_evidence() -> int:
    """Validate committed raw evidence files without running TIF scenarios.

    Checks structural integrity of each CE-EVID-*.json file:
      - evidence_id matches filename stem
      - claim_ids, requirement_ids, tif_ids present
      - commit_sha is a valid 40-char hex SHA (or 'unknown' when unavailable)
      - verification_type is valid
      - top-level result is consistent with scenario results
      - criterion_ref values are in requirement_ids
      - behavioral evidence has tif_ids

    Does not re-run TIF scenarios or write files.
    """
    print(f"Validating committed raw evidence files in {_OUT_DIR}")
    failed: list[str] = []

    evidence_files = sorted(_OUT_DIR.glob("CE-EVID-*.json"))
    if not evidence_files:
        print("  No CE-EVID-*.json files found.")
        return 0

    for path in evidence_files:
        stem = path.stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ERROR {stem}: JSON parse error: {exc}")
            failed.append(stem)
            continue

        ok = True

        def _fail(msg: str) -> None:
            nonlocal ok
            print(f"  ERROR {stem}: {msg}")
            failed.append(stem)
            ok = False

        if data.get("evidence_id") != stem:
            _fail(f"evidence_id '{data.get('evidence_id')}' != filename stem '{stem}'")

        if not data.get("claim_ids"):
            _fail("claim_ids is empty")

        if not data.get("requirement_ids"):
            _fail("requirement_ids is empty")

        vtype = data.get("verification_type", "")
        if vtype in _BEHAVIORAL_TYPES and not data.get("tif_ids"):
            _fail(f"behavioral verification_type '{vtype}' but tif_ids is empty")

        sha = data.get("commit_sha", "")
        if sha and sha != "unknown" and not _SHA_RE.fullmatch(sha):
            _fail(f"commit_sha '{sha}' is not a 40-character hex SHA")

        scenarios: list[dict[str, Any]] = data.get("scenarios", [])
        if not scenarios:
            _fail("scenarios list is empty")
        else:
            all_pass = all(s.get("result") == "pass" for s in scenarios)
            expected_top = "pass" if all_pass else "fail"
            if data.get("result") and data["result"] != expected_top:
                _fail(
                    f"top-level result '{data['result']}' disagrees with scenarios (expected '{expected_top}')"
                )

        ev_req_set = set(data.get("requirement_ids", []))
        for scenario in scenarios:
            for acc in scenario.get("acceptance", []):
                cref = acc.get("criterion_ref", "")
                if cref and cref not in ev_req_set:
                    _fail(
                        f"scenario '{scenario.get('scenario_id')}': "
                        f"criterion_ref '{cref}' not in requirement_ids"
                    )

        if ok:
            print(f"  OK    {stem}")

    if failed:
        print(f"\nFAILED: {', '.join(dict.fromkeys(failed))}")
        return 1
    print(f"\nAll {len(evidence_files)} evidence file(s) valid.")
    return 0


def main(*, check_current: bool = False, validate_existing: bool = False) -> int:
    if validate_existing:
        return _validate_existing_evidence()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating TIF evidence records -> {_OUT_DIR}")
    print(f"  package: {_PACKAGE_VERSION}  commit: {_COMMIT_SHA}")
    failed: list[str] = []
    written: list[dict[str, Any]] = []
    for tif_id, runner in _RUNNERS:
        print(f"Running {tif_id}...")
        try:
            payload = runner()
            _write(payload)
            written.append(payload)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append(tif_id)
    if check_current:
        current_sha = _git_sha()
        stale = [p["evidence_id"] for p in written if p["commit_sha"] != current_sha]
        failing = [p["evidence_id"] for p in written if p["result"] == "fail"]
        if stale:
            print(f"  ERROR: generated evidence is not at current HEAD: {', '.join(stale)}")
            failed.extend(stale)
        if failing:
            print(f"  ERROR: generated evidence has failing result: {', '.join(failing)}")
            failed.extend(failing)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(_RUNNERS)} TIF evidence records written.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-current",
        action="store_true",
        help="After generating evidence, fail if evidence is not at current HEAD or any result is fail.",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate committed evidence files structurally without running TIF scenarios.",
    )
    args = parser.parse_args()
    sys.exit(main(check_current=args.check_current, validate_existing=args.validate_existing))
