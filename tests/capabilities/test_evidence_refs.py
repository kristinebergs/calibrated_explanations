"""Evidence reference validator for raw capability evidence JSON."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from tests.helpers.capability_utils import markdown_table_value

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_DIR = _REPO_ROOT / "reports" / "verification"
_CLAIM_DIR = _REPO_ROOT / "development" / "capabilities" / "claims"
_REQ_DIR = _REPO_ROOT / "development" / "capabilities" / "requirements"
_TIF_SPEC_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"
_TIF_README = _TIF_SPEC_DIR / "README.md"

_REQUIRED_FIELDS = {
    "evidence_id",
    "claim_ids",
    "requirement_ids",
    "tif_ids",
    "verification_type",
    "result",
    "commit_sha",
    "timestamp",
    "package_version",
    "scenarios",
}
_VALID_VERIFICATION_TYPES = {
    "api_contract",
    "behavioral_contract",
    "numerical_behavior",
    "statistical_method_alignment",
    "empirical_smoke",
    "documentation_boundary",
    "visualization_structure",
    "policy_check",
}
_BEHAVIORAL_VERIFICATION_TYPES = {
    "api_contract",
    "behavioral_contract",
    "numerical_behavior",
    "statistical_method_alignment",
    "empirical_smoke",
    "visualization_structure",
}
_VALID_RESULTS = {"pass", "fail"}
_VALID_TIF_EXEMPTIONS = {
    "documentation_boundary",
    "schema_validation",
    "repository_policy",
    "metadata_linkage",
}
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _evidence_files() -> list[Path]:
    if not _EVIDENCE_DIR.exists():
        return []
    return sorted(_EVIDENCE_DIR.glob("*.json"))


def _parametrize_evidence():
    files = _evidence_files()
    if not files:
        return [pytest.param(None, id="no-evidence-files")]
    return [pytest.param(p, id=p.name) for p in files]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_index() -> set[str]:
    return {p.stem for p in _CLAIM_DIR.glob("CE-CAP-*.yaml")}


def _req_index() -> set[str]:
    return {p.stem for p in _REQ_DIR.glob("CE-REQ-*.md")}


def _tif_spec_index() -> set[str]:
    return {p.stem for p in _TIF_SPEC_DIR.glob("CE-TIF-*.md")}


def _requirement_text(requirement_id: str) -> str:
    return (_REQ_DIR / f"{requirement_id}.md").read_text(encoding="utf-8")


def _requirement_strength(requirement_id: str) -> str | None:
    return markdown_table_value(_requirement_text(requirement_id), "verification_strength")


def _requirement_tif_exemption(requirement_id: str) -> str | None:
    text = _requirement_text(requirement_id)
    return markdown_table_value(text, "tif_exemption") or markdown_table_value(
        text, "tif_exemption:"
    )


def _has_valid_tif_exemption(requirement_id: str) -> bool:
    exemption = _requirement_tif_exemption(requirement_id)
    return exemption in _VALID_TIF_EXEMPTIONS


def _acceptance_entries(scenario: dict) -> list[dict]:
    acceptance = scenario.get("acceptance")
    assert isinstance(acceptance, list), "scenario acceptance must be a list"
    return acceptance


@pytest.fixture(scope="module")
def _indexes():
    return _claim_index(), _req_index(), _tif_spec_index()


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_required_fields_present(evidence_path: Path | None):
    """Every evidence JSON must contain all required top-level fields."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    missing = _REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"{evidence_path.name}: missing required fields: {sorted(missing)}"


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_claim_ids_exist(evidence_path: Path | None, _indexes):
    """claim_ids in evidence must reference existing CE-CAP-*.yaml files."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    claims, _, _ = _indexes
    data = _load_json(evidence_path)
    missing = [cid for cid in data.get("claim_ids", []) if cid not in claims]
    assert not missing, f"{evidence_path.name}: missing claims: {missing}"


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_requirement_ids_exist(evidence_path: Path | None, _indexes):
    """requirement_ids in evidence must reference existing CE-REQ-*.md files."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    _, reqs, _ = _indexes
    data = _load_json(evidence_path)
    missing = [rid for rid in data.get("requirement_ids", []) if rid not in reqs]
    assert not missing, f"{evidence_path.name}: missing requirements: {missing}"


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_tif_ids_exist(evidence_path: Path | None, _indexes):
    """tif_ids in evidence must reference existing CE-TIF-*.md spec files."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    _, _, tif_specs = _indexes
    data = _load_json(evidence_path)
    missing = [tid for tid in data.get("tif_ids", []) if tid not in tif_specs]
    assert not missing, f"{evidence_path.name}: missing TIF specs: {missing}"


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_verification_type_is_valid(evidence_path: Path | None):
    """verification_type must be one of the documented allowed values."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    assert data.get("verification_type") in _VALID_VERIFICATION_TYPES


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_result_is_valid(evidence_path: Path | None):
    """result field must be 'pass' or 'fail'."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    assert data.get("result") in _VALID_RESULTS


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_commit_sha_is_not_placeholder(evidence_path: Path | None):
    """commit_sha must be a full 40-character hexadecimal git SHA."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    assert _SHA_RE.fullmatch(data.get("commit_sha", "")), (
        f"{evidence_path.name}: commit_sha must be a full 40-character git SHA"
    )


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_match_evidence_id_to_filename_when_raw_evidence_committed(
    evidence_path: Path | None,
):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    assert _load_json(evidence_path)["evidence_id"] == evidence_path.stem


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_parse_timestamp_when_raw_evidence_committed(evidence_path: Path | None):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    parsed_timestamp = datetime.fromisoformat(_load_json(evidence_path)["timestamp"])
    assert parsed_timestamp.tzinfo is not None, (
        f"{evidence_path.name}: timestamp must include UTC offset"
    )


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_validate_scenario_structure_when_raw_evidence_committed(evidence_path: Path | None):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    assert data["scenarios"], f"{evidence_path.name}: scenarios must be non-empty"
    for scenario in data["scenarios"]:
        assert {"scenario_id", "result", "observations", "acceptance"} <= set(scenario)
        assert scenario["result"] in _VALID_RESULTS
        acceptance = _acceptance_entries(scenario)
        assert acceptance, f"{evidence_path.name}: {scenario['scenario_id']} has no acceptance"
        for entry in acceptance:
            assert {"criterion_ref", "field", "expected", "observed", "result"} <= set(entry)
            assert entry["result"] in _VALID_RESULTS


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_validate_acceptance_requirement_coverage_when_raw_evidence_committed(
    evidence_path: Path | None,
):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    requirement_ids = set(data["requirement_ids"])
    criterion_refs = {
        entry["criterion_ref"]
        for scenario in data["scenarios"]
        for entry in _acceptance_entries(scenario)
    }
    assert criterion_refs <= requirement_ids
    if data["verification_type"] in _BEHAVIORAL_VERIFICATION_TYPES:
        assert requirement_ids <= criterion_refs


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_match_results_to_scenarios_when_raw_evidence_committed(evidence_path: Path | None):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    scenario_results = [scenario["result"] for scenario in data["scenarios"]]
    expected = "pass" if all(result == "pass" for result in scenario_results) else "fail"
    assert data["result"] == expected


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_match_exact_acceptance_results_when_safe_to_compare(evidence_path: Path | None):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    scalar_types = (bool, int, float, str, type(None))
    for scenario in data["scenarios"]:
        for entry in _acceptance_entries(scenario):
            if isinstance(entry["expected"], scalar_types) and isinstance(
                entry["observed"], scalar_types
            ):
                expected_result = "pass" if entry["observed"] == entry["expected"] else "fail"
                assert entry["result"] == expected_result


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_match_verification_strength_when_requirement_declares_unambiguous_strength(
    evidence_path: Path | None,
):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    strengths = {
        strength
        for rid in data["requirement_ids"]
        for strength in [_requirement_strength(rid)]
        if strength is not None
    }
    if len(strengths) == 1:
        assert data["verification_type"] == next(iter(strengths))
    elif strengths:
        assert data["verification_type"] in strengths


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_cover_evidence_requirements_from_referenced_tifs(evidence_path: Path | None):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    if not data["tif_ids"]:
        return
    tif_text = "\n".join(
        (_TIF_SPEC_DIR / f"{tif_id}.md").read_text(encoding="utf-8") for tif_id in data["tif_ids"]
    )
    missing = [rid for rid in data["requirement_ids"] if rid not in tif_text]
    assert not missing, f"{evidence_path.name}: requirements missing from TIF specs: {missing}"


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_should_require_tif_ids_for_behavioral_evidence_when_not_exempt(evidence_path: Path | None):
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")
    data = _load_json(evidence_path)
    if data["verification_type"] not in _BEHAVIORAL_VERIFICATION_TYPES:
        return
    if all(_has_valid_tif_exemption(rid) for rid in data["requirement_ids"]):
        return
    assert data["tif_ids"], f"{evidence_path.name}: behavioral evidence requires tif_ids"


def test_should_list_every_active_tif_spec_in_tif_readme():
    readme = _TIF_README.read_text(encoding="utf-8")
    missing = [path.name for path in _TIF_SPEC_DIR.glob("CE-TIF-*.md") if path.name not in readme]
    assert not missing, f"TIF README missing active specs: {missing}"


def test_should_have_raw_evidence_for_every_active_tif_spec():
    evidence_text = "\n".join(path.read_text(encoding="utf-8") for path in _evidence_files())
    missing = [tif_id for tif_id in _tif_spec_index() if tif_id not in evidence_text]
    assert not missing, f"Active TIF specs without raw evidence or rationale: {missing}"


@pytest.mark.skipif(
    os.environ.get("CE_REQUIRE_CURRENT_EVIDENCE") != "1",
    reason="historical evidence is allowed unless CE_REQUIRE_CURRENT_EVIDENCE=1",
)
def test_should_match_current_head_when_current_evidence_required():
    import subprocess

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    stale = [path.name for path in _evidence_files() if _load_json(path)["commit_sha"] != head]
    assert not stale, f"Evidence not generated at current HEAD: {stale}"
