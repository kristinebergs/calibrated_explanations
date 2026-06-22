"""Evidence reference validator — cross-reference integrity for raw evidence JSON.

Validates every JSON file in reports/verification/ against the claim, requirement,
and TIF spec files that exist on disk. Prevents the chain from silently referencing
artifacts that do not exist, making the evidence structurally invalid while appearing
sound.

Rules enforced:
  1. Each evidence file is valid JSON with all required top-level fields.
  2. Every entry in claim_ids references an existing CE-CAP-*.yaml file.
  3. Every entry in requirement_ids references an existing CE-REQ-*.md file.
  4. Every entry in tif_ids references an existing CE-TIF-*.md spec file.
  5. verification_type is one of the documented allowed values.
  6. result is "pass" or "fail".
  7. commit_sha is present and not the literal string "unknown".

Skips gracefully if reports/verification/ is empty or absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_DIR = _REPO_ROOT / "reports" / "verification"
_CLAIM_DIR = _REPO_ROOT / "development" / "capabilities" / "claims"
_REQ_DIR = _REPO_ROOT / "development" / "capabilities" / "requirements"
_TIF_SPEC_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"

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

_VALID_RESULTS = {"pass", "fail"}


def _evidence_files() -> list[Path]:
    if not _EVIDENCE_DIR.exists():
        return []
    return sorted(_EVIDENCE_DIR.glob("*.json"))


def _claim_index() -> set[str]:
    return {p.stem for p in _CLAIM_DIR.glob("CE-CAP-*.yaml")}


def _req_index() -> set[str]:
    return {p.stem for p in _REQ_DIR.glob("CE-REQ-*.md")}


def _tif_spec_index() -> set[str]:
    return {p.stem for p in _TIF_SPEC_DIR.glob("CE-TIF-*.md")}


@pytest.fixture(scope="module")
def _indexes():
    return _claim_index(), _req_index(), _tif_spec_index()


def _parametrize_evidence():
    files = _evidence_files()
    if not files:
        return [pytest.param(None, id="no-evidence-files")]
    return [pytest.param(p, id=p.name) for p in files]


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_required_fields_present(evidence_path: Path | None):
    """Every evidence JSON must contain all required top-level fields."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    missing = _REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"{evidence_path.name}: missing required fields: {sorted(missing)}"


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_claim_ids_exist(evidence_path: Path | None, _indexes):
    """claim_ids in evidence must reference existing CE-CAP-*.yaml files."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")

    claims, _, _ = _indexes
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    missing = [cid for cid in data.get("claim_ids", []) if cid not in claims]
    assert not missing, (
        f"{evidence_path.name}: claim_ids reference non-existent claims: {missing}. "
        f"Expected files under development/capabilities/claims/."
    )


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_requirement_ids_exist(evidence_path: Path | None, _indexes):
    """requirement_ids in evidence must reference existing CE-REQ-*.md files."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")

    _, reqs, _ = _indexes
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    missing = [rid for rid in data.get("requirement_ids", []) if rid not in reqs]
    assert not missing, (
        f"{evidence_path.name}: requirement_ids reference non-existent requirements: {missing}. "
        f"Expected files under development/capabilities/requirements/."
    )


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_tif_ids_exist(evidence_path: Path | None, _indexes):
    """tif_ids in evidence (when non-empty) must reference existing CE-TIF-*.md spec files."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")

    _, _, tif_specs = _indexes
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    tif_ids = data.get("tif_ids", [])
    if not tif_ids:
        return  # empty tif_ids is permitted for TIF-exempt requirements
    missing = [tid for tid in tif_ids if tid not in tif_specs]
    assert not missing, (
        f"{evidence_path.name}: tif_ids reference non-existent TIF spec files: {missing}. "
        f"Expected CE-TIF-*.md files under development/capabilities/verification/tif/."
    )


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_verification_type_is_valid(evidence_path: Path | None):
    """verification_type must be one of the documented allowed values."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    vtype = data.get("verification_type", "")
    assert vtype in _VALID_VERIFICATION_TYPES, (
        f"{evidence_path.name}: verification_type '{vtype}' is not a recognised value. "
        f"Allowed: {sorted(_VALID_VERIFICATION_TYPES)}"
    )


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_result_is_valid(evidence_path: Path | None):
    """result field must be 'pass' or 'fail'."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    result = data.get("result", "")
    assert (
        result in _VALID_RESULTS
    ), f"{evidence_path.name}: result '{result}' is not valid. Must be 'pass' or 'fail'."


@pytest.mark.parametrize("evidence_path", _parametrize_evidence())
def test_evidence_commit_sha_is_not_placeholder(evidence_path: Path | None):
    """commit_sha must be present and not the fallback sentinel 'unknown'."""
    if evidence_path is None:
        pytest.skip("no evidence files in reports/verification/")

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    sha = data.get("commit_sha", "")
    assert sha and sha != "unknown", (
        f"{evidence_path.name}: commit_sha is '{sha}'. "
        "Evidence must record a real git SHA. "
        "Re-run scripts/generate_tif_evidence.py or scripts/generate_capability_evidence.py "
        "inside a git repository to capture the current commit."
    )
