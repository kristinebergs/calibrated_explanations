"""Tests for the capability-chain validator and raw-evidence structural validation.

These tests verify the validator logic using temporary fixture directories and files.
They do NOT execute TIF scenarios or mutate committed repository files.

Requirements exercised:
  All checks in scripts/quality/validate_capability_chain.py
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

# Add validator and evidence generator to path
_SCRIPTS_QUALITY = Path(__file__).parents[2] / "scripts" / "quality"
_SCRIPTS = Path(__file__).parents[2] / "scripts"
for _p in (_SCRIPTS_QUALITY, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import validate_capability_chain as vcc  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def _minimal_claim(claim_id: str, req_ids: list[str], *, atomic_rationale: bool = False) -> str:
    reqs_yaml = "\n".join(f"  - {r}" for r in req_ids)
    atomic = (
        "\n    atomic_rationale: >\n      Single requirement because it is genuinely atomic."
        if atomic_rationale
        else ""
    )
    return f"""\
    claim_id: {claim_id}
    claim_type: capability
    owner: calibrated_explanations
    status: current{atomic}
    requirements:
    {reqs_yaml}
    """


def _minimal_req(
    req_id: str,
    claim_ref: str,
    obligation_type: str = "api_contract",
    vstatus: str = "verified",
    tif_refs: list[str] | None = None,
    tif_exemption: str = "",
) -> str:
    tif_line = ""
    if tif_refs:
        tif_line = f"| tif_refs | {', '.join(tif_refs)} |"
    elif tif_exemption:
        tif_line = f"| tif_exemption | {tif_exemption} |"
    return f"""\
        # {req_id}

        ## Metadata

        | Field | Value |
        |---|---|
        | requirement_id | {req_id} |
        | obligation_type | {obligation_type} |
        | claim_refs | {claim_ref} |
        | verification_status | {vstatus} |
        {tif_line}

        ## Acceptance criterion
        The method is callable.

        ## Verification method
        Automated pytest.

        ## Verification targets
        - pytest: tests/capabilities/test_explanation_contracts.py
    """


def _minimal_tif_py(
    tif_id: str, *, include_wce: bool = True, include_forbidden: bool = False
) -> str:
    wce_import = (
        "from calibrated_explanations.core.wrap_explainer import WrapCalibratedExplainer"
        if include_wce
        else "# no WCE import"
    )
    wce_usage = (
        "    explainer = WrapCalibratedExplainer(None)" if include_wce else "    # no explainer"
    )
    forbidden = (
        "from calibrated_explanations.core.calibrated_explainer import CalibratedExplainer"
        if include_forbidden
        else ""
    )
    return f'''\
    """TIF {tif_id}.

    Requirements served: CE-REQ-TEST-001
    """
    from __future__ import annotations
    {wce_import}
    {forbidden}

    def run_scenario():
        """Run the TIF scenario."""
{wce_usage}
        return {{"exception_raised": False}}
    '''


def _minimal_raw_evidence(
    evid_id: str,
    claim_ids: list[str],
    req_ids: list[str],
    tif_ids: list[str],
    vtype: str = "api_contract",
) -> dict:
    return {
        "evidence_id": evid_id,
        "claim_ids": claim_ids,
        "requirement_ids": req_ids,
        "adr_refs": [],
        "standard_refs": [],
        "tif_ids": tif_ids,
        "verification_type": vtype,
        "result": "pass",
        "timestamp": "2026-06-22T00:00:00+00:00",
        "commit_sha": "a" * 40,
        "package_version": "v0.11.4",
        "python_version": "3.11",
        "platform": "linux",
        "dataset_id": "test",
        "random_seed": 42,
        "configuration": {},
        "scenarios": [
            {
                "scenario_id": "s1",
                "observations": {},
                "result": "pass",
                "acceptance": [
                    {
                        "criterion_ref": req_ids[0],
                        "field": "ok",
                        "expected": True,
                        "observed": True,
                        "result": "pass",
                    }
                ],
            }
        ],
        "artifacts": {"logs": None, "raw_output": None},
    }


# ---------------------------------------------------------------------------
# Fixtures for patching validator directories
# ---------------------------------------------------------------------------


@pytest.fixture()
def chain_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Create minimal chain directories and patch the validator globals."""
    dirs = {
        "claims": tmp_path / "claims",
        "reqs": tmp_path / "requirements",
        "tif": tmp_path / "tif",
        "evid": tmp_path / "evidence",
        "raw_evid": tmp_path / "raw_evidence",
    }
    for d in dirs.values():
        d.mkdir(parents=True)

    monkeypatch.setattr(vcc, "_CLAIMS_DIR", dirs["claims"])
    monkeypatch.setattr(vcc, "_REQ_DIR", dirs["reqs"])
    monkeypatch.setattr(vcc, "_TIF_DIR", dirs["tif"])
    monkeypatch.setattr(vcc, "_EVID_DIR", dirs["evid"])
    monkeypatch.setattr(vcc, "_RAW_EVID_DIR", dirs["raw_evid"])

    # Also patch the repo root so verification targets resolve (they won't exist,
    # but we want to avoid false failures on real repo files during unit tests).
    # We'll keep _REPO_ROOT as-is; verification target existence checks are based
    # on the real repo in integration mode and we skip them in these isolated tests
    # by using non-verified status or verified requirements whose targets exist.

    return dirs


# ---------------------------------------------------------------------------
# Test: valid chain passes
# ---------------------------------------------------------------------------


def test_should_pass_when_chain_is_structurally_valid(chain_dirs: dict[str, Path]) -> None:
    """A minimal but structurally valid chain passes with no errors."""
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    # Write claim
    _write(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-API-001"], atomic_rationale=True),
    )

    # Write requirement (TIF-exempt so no TIF spec needed)
    _write(
        reqs_dir / "CE-REQ-TEST-API-001.md",
        _minimal_req(
            "CE-REQ-TEST-API-001",
            "CE-CAP-TEST-001",
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )

    errors, warnings = vcc.run_checks()
    assert not errors, f"Unexpected errors: {errors}"


def test_should_fail_when_requirement_file_is_missing(chain_dirs: dict[str, Path]) -> None:
    """A claim referencing a non-existent requirement file produces an error."""
    claims_dir = chain_dirs["claims"]
    _write(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-MISSING-001"]),
    )

    errors, _ = vcc.run_checks()
    assert any("CE-REQ-MISSING-001" in e and "not found" in e for e in errors), errors


def test_should_fail_when_behavioral_requirement_has_no_tif_refs(
    chain_dirs: dict[str, Path],
) -> None:
    """A verified behavioral requirement with no tif_refs and no tif_exemption fails."""
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    _write(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-API-001"], atomic_rationale=True),
    )
    # api_contract + verified + no tif_refs + no tif_exemption -> ERROR
    _write(
        reqs_dir / "CE-REQ-TEST-API-001.md",
        _minimal_req(
            "CE-REQ-TEST-API-001",
            "CE-CAP-TEST-001",
            obligation_type="api_contract",
            vstatus="verified",
        ),
    )

    errors, _ = vcc.run_checks()
    assert any("neither tif_refs nor tif_exemption" in e for e in errors), errors


def test_should_fail_when_tif_executable_missing_wrapcalibratedexplainer(
    chain_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """A TIF executable that does not reference WrapCalibratedExplainer fails the guard."""
    tif_dir = chain_dirs["tif"]
    # Write a TIF py without WrapCalibratedExplainer
    _write(tif_dir / "tif_test.py", _minimal_tif_py("CE-TIF-TEST-001", include_wce=False))

    errors, _ = vcc.run_checks()
    assert any("WrapCalibratedExplainer not found" in e for e in errors), errors


def test_should_fail_when_tif_imports_core_calibrated_explainer(
    chain_dirs: dict[str, Path],
) -> None:
    """A TIF that imports from core.calibrated_explainer fails the guard."""
    tif_dir = chain_dirs["tif"]
    _write(
        tif_dir / "tif_test.py",
        _minimal_tif_py("CE-TIF-TEST-001", include_wce=True, include_forbidden=True),
    )

    errors, _ = vcc.run_checks()
    assert any("forbidden import" in e for e in errors), errors


def test_should_fail_when_raw_evidence_has_unknown_requirement_id(
    chain_dirs: dict[str, Path],
) -> None:
    """Raw evidence referencing a non-existent requirement ID fails."""
    raw_dir = chain_dirs["raw_evid"]
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    _write(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-API-001"], atomic_rationale=True),
    )
    _write(
        reqs_dir / "CE-REQ-TEST-API-001.md",
        _minimal_req(
            "CE-REQ-TEST-API-001",
            "CE-CAP-TEST-001",
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )

    evid = _minimal_raw_evidence(
        "CE-EVID-TEST-001-20260622",
        claim_ids=["CE-CAP-TEST-001"],
        req_ids=["CE-REQ-NONEXISTENT-001"],  # <- unknown
        tif_ids=["CE-TIF-TEST-001"],
    )
    (raw_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    errors, _ = vcc.run_checks()
    assert any("CE-REQ-NONEXISTENT-001" in e and "not found" in e for e in errors), errors


def test_should_fail_when_raw_evidence_id_mismatches_filename(chain_dirs: dict[str, Path]) -> None:
    """Raw evidence where evidence_id != filename stem fails."""
    raw_dir = chain_dirs["raw_evid"]
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    _write(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-API-001"], atomic_rationale=True),
    )
    _write(
        reqs_dir / "CE-REQ-TEST-API-001.md",
        _minimal_req(
            "CE-REQ-TEST-API-001",
            "CE-CAP-TEST-001",
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )

    evid = _minimal_raw_evidence(
        "CE-EVID-WRONG-ID-20260622",  # mismatches filename
        claim_ids=["CE-CAP-TEST-001"],
        req_ids=["CE-REQ-TEST-API-001"],
        tif_ids=["CE-TIF-TEST-001"],
    )
    (raw_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    errors, _ = vcc.run_checks()
    assert any("evidence_id" in e and "filename stem" in e for e in errors), errors


def test_should_pass_when_documentation_boundary_curated_evidence_uses_none_raw_ref(
    chain_dirs: dict[str, Path],
) -> None:
    """Curated evidence for TIF-exempt documentation-boundary requirements can use raw_evidence_ref: none."""
    evid_dir = chain_dirs["evid"]
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    _write(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-DOC-001"], atomic_rationale=True),
    )
    _write(
        reqs_dir / "CE-REQ-TEST-DOC-001.md",
        _minimal_req(
            "CE-REQ-TEST-DOC-001",
            "CE-CAP-TEST-001",
            obligation_type="documentation_boundary",
            vstatus="verified",
            tif_exemption="documentation_boundary",
        ),
    )

    _write(
        evid_dir / "evidence_test_doc_boundaries.md",
        """\
        # Doc boundary evidence

        | Field | Value |
        |---|---|
        | requirement_ids | CE-REQ-TEST-DOC-001 |
        | verification_strength | documentation_boundary |
        | result | PASS |

        raw_evidence_ref: none — TIF-exempt documentation-boundary review
    """,
    )

    errors, _ = vcc.run_checks()
    assert not errors, f"Unexpected errors: {errors}"


def test_should_fail_when_claim_has_single_req_without_atomic_rationale(
    chain_dirs: dict[str, Path],
) -> None:
    """A claim with exactly one requirement but no atomic_rationale fails."""
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    _write(
        claims_dir / "CE-CAP-TEST-001.yaml", _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"])
    )
    _write(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )

    errors, _ = vcc.run_checks()
    assert any("atomic_rationale" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Test: validate_existing evidence (generate_tif_evidence.py)
# ---------------------------------------------------------------------------


def test_should_pass_validate_existing_when_committed_evidence_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--validate-existing passes for structurally valid evidence files."""
    import generate_tif_evidence as gte

    out_dir = tmp_path / "verification"
    out_dir.mkdir()
    monkeypatch.setattr(gte, "_OUT_DIR", out_dir)

    evid = _minimal_raw_evidence(
        "CE-EVID-TEST-001-20260622", ["CE-CAP-TEST-001"], ["CE-REQ-TEST-001"], ["CE-TIF-TEST-001"]
    )
    (out_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    rc = gte.main(validate_existing=True)
    assert rc == 0


def test_should_fail_validate_existing_when_evidence_id_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--validate-existing fails when evidence_id mismatches the filename stem."""
    import generate_tif_evidence as gte

    out_dir = tmp_path / "verification"
    out_dir.mkdir()
    monkeypatch.setattr(gte, "_OUT_DIR", out_dir)

    evid = _minimal_raw_evidence(
        "CE-EVID-WRONG-20260622", ["CE-CAP-TEST-001"], ["CE-REQ-TEST-001"], ["CE-TIF-TEST-001"]
    )
    (out_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    rc = gte.main(validate_existing=True)
    assert rc != 0


def test_should_fail_validate_existing_when_behavioral_evidence_has_no_tif_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--validate-existing fails when behavioral evidence has empty tif_ids."""
    import generate_tif_evidence as gte

    out_dir = tmp_path / "verification"
    out_dir.mkdir()
    monkeypatch.setattr(gte, "_OUT_DIR", out_dir)

    evid = _minimal_raw_evidence(
        "CE-EVID-TEST-001-20260622",
        ["CE-CAP-TEST-001"],
        ["CE-REQ-TEST-001"],
        tif_ids=[],
        vtype="api_contract",
    )
    (out_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    rc = gte.main(validate_existing=True)
    assert rc != 0


# ---------------------------------------------------------------------------
# Helpers for TIF spec / README fixture creation
# ---------------------------------------------------------------------------


def _minimal_tif_spec(
    tif_id: str,
    executable: str,
    evidence_key: str,
    *,
    verification_type: str = "api_contract",
    status: str = "active",
    evidence_builder: str = "build_evidence_payload()",
) -> str:
    return textwrap.dedent(f"""\
        # {tif_id}

        ## Identity

        | Field | Value |
        |---|---|
        | tif_id | {tif_id} |
        | executable | `{executable}` |
        | evidence_builder | `{evidence_builder}` |
        | evidence_key | {evidence_key} |
        | verification_type | {verification_type} |
        | status | {status} |
    """)


def _readme_with_tif_table(rows: list[dict]) -> str:
    """Build a README snippet with a 'Current TIF Interfaces' table."""
    header = "| TIF ID | Executable | Evidence Key | Verification Type | Status |"
    sep = "|---|---|---|---|---|"
    data = "\n".join(
        f"| {r['tif_id']} | {r.get('executable', '')} | {r.get('evidence_key', '')} "
        f"| {r.get('verification_type', 'api_contract')} | {r.get('status', 'active')} |"
        for r in rows
    )
    return textwrap.dedent(f"""\
        # TIF README

        ## Current TIF Interfaces

        {header}
        {sep}
        {data}
    """)


# ---------------------------------------------------------------------------
# Tests: README bidirectional cross-check (validate_capability_chain)
# ---------------------------------------------------------------------------


def test_should_fail_when_active_spec_missing_from_readme(
    chain_dirs: dict[str, Path],
) -> None:
    """An active TIF spec not listed in the README table fails."""
    tif_dir = chain_dirs["tif"]
    exec_path = tif_dir / "tif_test.py"
    _write(exec_path, _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))
    # Add build_evidence_payload to the executable
    exec_path.write_text(
        exec_path.read_text(encoding="utf-8")
        + "\ndef build_evidence_payload(**_kw):\n    return {}\n",
        encoding="utf-8",
    )
    _write(
        tif_dir / "CE-TIF-TEST-001.md",
        _minimal_tif_spec(
            "CE-TIF-TEST-001",
            f"{tif_dir}/tif_test.py",
            "TEST-001",
        ),
    )
    # README exists but does NOT list CE-TIF-TEST-001
    _write(tif_dir / "README.md", _readme_with_tif_table([]))

    errors, _ = vcc.run_checks()
    assert any("CE-TIF-TEST-001" in e and "not listed" in e for e in errors), errors


def test_should_fail_when_readme_has_stale_extra_row(
    chain_dirs: dict[str, Path],
) -> None:
    """A README row whose TIF ID has no matching active spec fails."""
    tif_dir = chain_dirs["tif"]
    # No spec files — README has a row for a non-existent spec
    _write(
        tif_dir / "README.md",
        _readme_with_tif_table(
            [
                {
                    "tif_id": "CE-TIF-GHOST-001",
                    "executable": "tif_ghost.py",
                    "evidence_key": "GHOST-001",
                    "verification_type": "api_contract",
                    "status": "active",
                }
            ]
        ),
    )

    errors, _ = vcc.run_checks()
    assert any("CE-TIF-GHOST-001" in e and "no matching active spec" in e for e in errors), errors


def test_should_fail_when_readme_metadata_mismatches_spec(
    chain_dirs: dict[str, Path],
) -> None:
    """A README row whose verification_type differs from the spec fails."""
    tif_dir = chain_dirs["tif"]
    exec_path = tif_dir / "tif_test.py"
    _write(exec_path, _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))
    exec_path.write_text(
        exec_path.read_text(encoding="utf-8")
        + "\ndef build_evidence_payload(**_kw):\n    return {}\n",
        encoding="utf-8",
    )
    _write(
        tif_dir / "CE-TIF-TEST-001.md",
        _minimal_tif_spec(
            "CE-TIF-TEST-001",
            f"{tif_dir}/tif_test.py",
            "TEST-001",
            verification_type="behavioral_contract",
        ),
    )
    # README has wrong verification_type
    _write(
        tif_dir / "README.md",
        _readme_with_tif_table(
            [
                {
                    "tif_id": "CE-TIF-TEST-001",
                    "executable": "tif_test.py",
                    "evidence_key": "TEST-001",
                    "verification_type": "api_contract",  # mismatch
                    "status": "active",
                }
            ]
        ),
    )

    errors, _ = vcc.run_checks()
    assert any("CE-TIF-TEST-001" in e and "verification_type" in e for e in errors), errors


def test_should_fail_when_tif_executable_missing_build_evidence_payload(
    chain_dirs: dict[str, Path],
) -> None:
    """A tif_*.py that does not define build_evidence_payload() fails."""
    tif_dir = chain_dirs["tif"]
    # Write a valid TIF py with WrapCalibratedExplainer but no build_evidence_payload
    _write(tif_dir / "tif_test.py", _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))

    errors, _ = vcc.run_checks()
    assert any("build_evidence_payload" in e for e in errors), errors


def test_should_not_have_manual_registry_in_generate_capability_evidence() -> None:
    """generate_capability_evidence.py must be deleted or contain no _RUNNERS or TIF imports."""
    script = Path(__file__).parents[2] / "scripts" / "generate_capability_evidence.py"
    if not script.exists():
        return  # Deleted — compliant
    text = script.read_text(encoding="utf-8")
    assert (
        "_RUNNERS" not in text
    ), "generate_capability_evidence.py still contains a manual _RUNNERS registry"
    assert not re.search(
        r"^(?:import|from)\s+tif_", text, re.MULTILINE
    ), "generate_capability_evidence.py has direct TIF module imports"
