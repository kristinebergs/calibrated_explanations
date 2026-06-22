"""Generate raw capability evidence records for the EXPL-CONJ chain.

Calls TIF scenarios directly, evaluates acceptance criteria from requirement
files, and writes conforming JSON evidence records to reports/verification/.

Usage:
    python scripts/generate_capability_evidence.py
    python scripts/generate_capability_evidence.py --dry-run
    python scripts/generate_capability_evidence.py --out-dir reports/verification/custom/

The script exits with code 1 if any acceptance criterion fails, 0 if all pass.
Evidence files are written only when all scenarios in a requirement pass.

Evidence schema: development/capabilities/evidence/README.md
TIF interface:   development/capabilities/verification/tif/CE-TIF-EXPL-CONJ-001.md
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TIF_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"
_REPORTS_DIR = _REPO_ROOT / "reports" / "verification"

if str(_TIF_DIR) not in sys.path:
    sys.path.insert(0, str(_TIF_DIR))

from tif_conjunction import run_conjunction_tif_scenario  # noqa: E402

# ---------------------------------------------------------------------------
# Environment metadata helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S603, S607
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("calibrated-explanations")
    except Exception:
        return "unknown"


def _env_metadata() -> dict[str, str]:
    return {
        "commit_sha": _git_sha(),
        "package_version": _package_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


# ---------------------------------------------------------------------------
# Acceptance evaluators — one per requirement
# CE-REQ-EXPL-CONJ-API-001, RETURN-001, RULE-001, PARAM-001
# ---------------------------------------------------------------------------

_DATASET_ID = (
    "sklearn make_classification "
    "n_samples=120 n_features=4 n_informative=3 n_redundant=1 random_seed=42"
)
_RANDOM_SEED = 42
_CLAIM_IDS = ["CE-CAP-EXPL-CONJ-001"]
_ADR_REFS = ["ADR-008"]
_TIF_IDS = ["CE-TIF-EXPL-CONJ-001"]


def _evidence_record(
    *,
    evidence_id: str,
    requirement_id: str,
    verification_type: str,
    test_scenarios: list[dict[str, Any]],
    env: dict[str, str],
) -> dict[str, Any]:
    """Build a conforming evidence record dict."""
    all_pass = all(s["result"] == "pass" for s in test_scenarios)
    return {
        "evidence_id": evidence_id,
        "claim_ids": _CLAIM_IDS,
        "requirement_ids": [requirement_id],
        "adr_refs": _ADR_REFS,
        "standard_refs": [],
        "tif_ids": _TIF_IDS,
        "verification_type": verification_type,
        "result": "pass" if all_pass else "fail",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "commit_sha": env["commit_sha"],
        "package_version": env["package_version"],
        "python_version": env["python_version"],
        "platform": env["platform"],
        "dataset_id": _DATASET_ID,
        "random_seed": _RANDOM_SEED,
        "configuration": {},
        "scenarios": test_scenarios,
        "artifacts": {"logs": None, "raw_output": None},
    }


def _run_api_scenarios(env: dict[str, str]) -> dict[str, Any]:
    """CE-REQ-EXPL-CONJ-API-001 — add_conjunctions callable without exception."""
    configs = [
        {"explanation_mode": "factual", "object_level": "collection"},
        {"explanation_mode": "alternative", "object_level": "collection"},
        {"explanation_mode": "factual", "object_level": "individual"},
        {"explanation_mode": "alternative", "object_level": "individual"},
    ]
    scenarios = []
    for cfg in configs:
        obs = run_conjunction_tif_scenario(max_rule_size=2, n_top_features=5, **cfg)
        passed = not obs.exception_raised
        scenarios.append(
            {
                "scenario_id": f"api_{cfg['explanation_mode']}_{cfg['object_level']}",
                "parameters": {**cfg, "max_rule_size": 2, "n_top_features": 5},
                "result": "pass" if passed else "fail",
                "acceptance": {
                    "criterion_ref": "CE-REQ-EXPL-CONJ-API-001",
                    "expected": "exception_raised == False",
                    "observed": f"exception_raised == {obs.exception_raised}"
                    + (f" ({obs.exception_type})" if obs.exception_type else ""),
                },
            }
        )

    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    return _evidence_record(
        evidence_id=f"CE-EVID-EXPL-CONJ-API-001-{date_str}",
        requirement_id="CE-REQ-EXPL-CONJ-API-001",
        verification_type="api_contract",
        test_scenarios=scenarios,
        env=env,
    )


def _run_return_scenarios(env: dict[str, str]) -> dict[str, Any]:
    """CE-REQ-EXPL-CONJ-RETURN-001 — return type and cardinality contract."""
    configs = [
        {"explanation_mode": "factual"},
        {"explanation_mode": "alternative"},
    ]
    scenarios = []
    for cfg in configs:
        obs = run_conjunction_tif_scenario(
            object_level="collection", max_rule_size=2, n_top_features=5, **cfg
        )
        not_none = not obs.result_is_none
        len_ok = obs.result_len == obs.n_instances
        passed = not obs.exception_raised and not_none and len_ok
        scenarios.append(
            {
                "scenario_id": f"return_{cfg['explanation_mode']}_collection",
                "parameters": {
                    **cfg,
                    "object_level": "collection",
                    "max_rule_size": 2,
                    "n_top_features": 5,
                },
                "result": "pass" if passed else "fail",
                "acceptance": {
                    "criterion_ref": "CE-REQ-EXPL-CONJ-RETURN-001",
                    "expected": "result_is_none == False and result_len == n_instances",
                    "observed": (
                        f"result_is_none == {obs.result_is_none}, "
                        f"result_len == {obs.result_len}, "
                        f"n_instances == {obs.n_instances}"
                    ),
                },
            }
        )

    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    return _evidence_record(
        evidence_id=f"CE-EVID-EXPL-CONJ-RETURN-001-{date_str}",
        requirement_id="CE-REQ-EXPL-CONJ-RETURN-001",
        verification_type="api_contract",
        test_scenarios=scenarios,
        env=env,
    )


def _run_rule_scenarios(env: dict[str, str]) -> dict[str, Any]:
    """CE-REQ-EXPL-CONJ-RULE-001 — multi-feature conjunction rules produced (max_rule_size >= 2)."""
    scenarios = []
    for mrs in (2, 3):
        obs = run_conjunction_tif_scenario(
            explanation_mode="factual",
            object_level="collection",
            max_rule_size=mrs,
            n_top_features=5,
        )
        passed = not obs.exception_raised and obs.any_has_conjunctive_rules
        scenarios.append(
            {
                "scenario_id": f"rule_factual_collection_max_rule_size_{mrs}",
                "parameters": {
                    "explanation_mode": "factual",
                    "object_level": "collection",
                    "max_rule_size": mrs,
                    "n_top_features": 5,
                },
                "result": "pass" if passed else "fail",
                "acceptance": {
                    "criterion_ref": "CE-REQ-EXPL-CONJ-RULE-001",
                    "expected": f"any_has_conjunctive_rules == True (when max_rule_size={mrs}, n_informative=3)",
                    "observed": f"any_has_conjunctive_rules == {obs.any_has_conjunctive_rules}",
                },
            }
        )

    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    return _evidence_record(
        evidence_id=f"CE-EVID-EXPL-CONJ-RULE-001-{date_str}",
        requirement_id="CE-REQ-EXPL-CONJ-RULE-001",
        verification_type="behavioral_contract",
        test_scenarios=scenarios,
        env=env,
    )


def _run_param_scenarios(env: dict[str, str]) -> dict[str, Any]:
    """CE-REQ-EXPL-CONJ-PARAM-001 — max_rule_size=1 suppresses multi-feature conjunctions."""
    obs = run_conjunction_tif_scenario(
        explanation_mode="factual",
        object_level="collection",
        max_rule_size=1,
        n_top_features=5,
    )
    passed = not obs.exception_raised and not obs.any_has_conjunctive_rules
    scenarios = [
        {
            "scenario_id": "param_factual_collection_max_rule_size_1",
            "parameters": {
                "explanation_mode": "factual",
                "object_level": "collection",
                "max_rule_size": 1,
                "n_top_features": 5,
            },
            "result": "pass" if passed else "fail",
            "acceptance": {
                "criterion_ref": "CE-REQ-EXPL-CONJ-PARAM-001",
                "expected": "any_has_conjunctive_rules == False (when max_rule_size=1)",
                "observed": (
                    f"any_has_conjunctive_rules == {obs.any_has_conjunctive_rules}, "
                    f"exception_raised == {obs.exception_raised}"
                ),
            },
        }
    ]

    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    return _evidence_record(
        evidence_id=f"CE-EVID-EXPL-CONJ-PARAM-001-{date_str}",
        requirement_id="CE-REQ-EXPL-CONJ-PARAM-001",
        verification_type="behavioral_contract",
        test_scenarios=scenarios,
        env=env,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_RUNNERS = [
    _run_api_scenarios,
    _run_return_scenarios,
    _run_rule_scenarios,
    _run_param_scenarios,
]


def main(dry_run: bool = False, out_dir: Path | None = None) -> int:
    """Run all EXPL-CONJ TIF scenarios and write evidence records.

    Returns 0 on all-pass, 1 if any scenario fails.
    """
    out_dir = out_dir or _REPORTS_DIR
    env = _env_metadata()

    print("CE capability evidence generator — EXPL-CONJ chain")
    print(f"  commit : {env['commit_sha']}")
    print(f"  version: {env['package_version']}")
    print(f"  python : {env['python_version']}")
    print(f"  out_dir: {out_dir}")
    print()

    records: list[dict[str, Any]] = []
    for runner in _RUNNERS:
        record = runner(env)
        records.append(record)
        req_id = record["requirement_ids"][0]
        result = record["result"]
        symbol = "PASS" if result == "pass" else "FAIL"
        failed_scenarios = [s for s in record["scenarios"] if s["result"] != "pass"]
        print(f"  [{symbol}] {req_id}")
        for s in failed_scenarios:
            print(f"      FAIL {s['scenario_id']}: {s['acceptance']['observed']}")

    print()
    all_pass = all(r["result"] == "pass" for r in records)

    if dry_run:
        print("[dry-run] Evidence records (not written):")
        for record in records:
            print(json.dumps(record, indent=2))
        return 0 if all_pass else 1

    if not all_pass:
        print(
            "Some scenarios FAILED — evidence files will not be written for failing requirements."
        )
        print("Fix the failures, then re-run this script to generate evidence.")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record in records:
        if record["result"] != "pass":
            continue
        file_name = f"{record['evidence_id']}.json"
        out_path = out_dir / file_name
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        written.append(out_path)
        print(f"  wrote: {out_path.relative_to(_REPO_ROOT)}")

    print()
    if all_pass:
        print(f"All {len(records)} requirements passed. {len(written)} evidence file(s) written.")
        print()
        print("Next step: review the JSON files above, then write a curated summary to")
        print("  development/capabilities/evidence/evidence_expl_conj_v<version>.md")
    else:
        failing = [r["requirement_ids"][0] for r in records if r["result"] != "pass"]
        print(f"FAIL — {len(failing)} requirement(s) did not pass: {', '.join(failing)}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print evidence records to stdout; do not write files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for evidence JSON files (default: reports/verification/).",
    )
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run, out_dir=args.out_dir))
