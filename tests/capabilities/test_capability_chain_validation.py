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
from tests.helpers.capability_utils import write_text_fixture

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
    tif_exemption_rationale: str = "",
) -> str:
    tif_line = ""
    if tif_refs:
        tif_line = f"| tif_refs | {', '.join(tif_refs)} |"
    elif tif_exemption:
        tif_line = f"| tif_exemption | {tif_exemption} |"
        if tif_exemption_rationale:
            tif_line += f"\n        | tif_exemption_rationale | {tif_exemption_rationale} |"
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

    return dirs


# ---------------------------------------------------------------------------
# Test: valid chain passes
# ---------------------------------------------------------------------------


def test_should_pass_when_chain_is_structurally_valid(chain_dirs: dict[str, Path]) -> None:
    """A minimal but structurally valid chain passes with no errors."""
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    # Write claim
    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-API-001"], atomic_rationale=True),
        dedent=True,
    )

    # Write requirement (TIF-exempt so no TIF spec needed)
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-API-001.md",
        _minimal_req(
            "CE-REQ-TEST-API-001",
            "CE-CAP-TEST-001",
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
        dedent=True,
    )

    errors, warnings = vcc.run_checks()
    assert not errors, f"Unexpected errors: {errors}"


def test_should_fail_when_requirement_file_is_missing(chain_dirs: dict[str, Path]) -> None:
    """A claim referencing a non-existent requirement file produces an error."""
    claims_dir = chain_dirs["claims"]
    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-MISSING-001"]),
        dedent=True,
    )

    errors, _ = vcc.run_checks()
    assert any("CE-REQ-MISSING-001" in e and "not found" in e for e in errors), errors


def test_should_fail_when_behavioral_requirement_has_no_tif_refs(
    chain_dirs: dict[str, Path],
) -> None:
    """A verified behavioral requirement with no tif_refs and no tif_exemption fails."""
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-API-001"], atomic_rationale=True),
        dedent=True,
    )
    # api_contract + verified + no tif_refs + no tif_exemption -> ERROR
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-API-001.md",
        _minimal_req(
            "CE-REQ-TEST-API-001",
            "CE-CAP-TEST-001",
            obligation_type="api_contract",
            vstatus="verified",
        ),
        dedent=True,
    )

    errors, _ = vcc.run_checks()
    assert any("neither tif_refs nor tif_exemption" in e for e in errors), errors


def test_should_fail_when_tif_executable_missing_wrapcalibratedexplainer(
    chain_dirs: dict[str, Path], tmp_path: Path
) -> None:
    """A TIF executable that does not reference WrapCalibratedExplainer fails the guard."""
    tif_dir = chain_dirs["tif"]
    # Write a TIF py without WrapCalibratedExplainer
    write_text_fixture(
        tif_dir / "tif_test.py",
        _minimal_tif_py("CE-TIF-TEST-001", include_wce=False),
        dedent=True,
    )

    errors, _ = vcc.run_checks()
    assert any("WrapCalibratedExplainer not found" in e for e in errors), errors


def test_should_fail_when_tif_imports_core_calibrated_explainer(
    chain_dirs: dict[str, Path],
) -> None:
    """A TIF that imports from core.calibrated_explainer fails the guard."""
    tif_dir = chain_dirs["tif"]
    write_text_fixture(
        tif_dir / "tif_test.py",
        _minimal_tif_py("CE-TIF-TEST-001", include_wce=True, include_forbidden=True),
        dedent=True,
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

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-API-001"], atomic_rationale=True),
    )
    write_text_fixture(
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

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-API-001"], atomic_rationale=True),
    )
    write_text_fixture(
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

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-DOC-001"], atomic_rationale=True),
    )
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-DOC-001.md",
        _minimal_req(
            "CE-REQ-TEST-DOC-001",
            "CE-CAP-TEST-001",
            obligation_type="documentation_boundary",
            vstatus="verified",
            tif_exemption="documentation_boundary",
        ),
    )

    write_text_fixture(
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

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml", _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"])
    )
    write_text_fixture(
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
    write_text_fixture(exec_path, _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))
    # Add build_evidence_payload to the executable
    exec_path.write_text(
        exec_path.read_text(encoding="utf-8")
        + "\ndef build_evidence_payload(**_kw):\n    return {}\n",
        encoding="utf-8",
    )
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _minimal_tif_spec(
            "CE-TIF-TEST-001",
            f"{tif_dir}/tif_test.py",
            "TEST-001",
        ),
    )
    # README exists but does NOT list CE-TIF-TEST-001
    write_text_fixture(tif_dir / "README.md", _readme_with_tif_table([]))

    errors, _ = vcc.run_checks()
    assert any("CE-TIF-TEST-001" in e and "not listed" in e for e in errors), errors


def test_should_fail_when_readme_has_stale_extra_row(
    chain_dirs: dict[str, Path],
) -> None:
    """A README row whose TIF ID has no matching active spec fails."""
    tif_dir = chain_dirs["tif"]
    # No spec files — README has a row for a non-existent spec
    write_text_fixture(
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
    write_text_fixture(exec_path, _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))
    exec_path.write_text(
        exec_path.read_text(encoding="utf-8")
        + "\ndef build_evidence_payload(**_kw):\n    return {}\n",
        encoding="utf-8",
    )
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _minimal_tif_spec(
            "CE-TIF-TEST-001",
            f"{tif_dir}/tif_test.py",
            "TEST-001",
            verification_type="behavioral_contract",
        ),
    )
    # README has wrong verification_type
    write_text_fixture(
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
    write_text_fixture(
        tif_dir / "tif_test.py", _minimal_tif_py("CE-TIF-TEST-001", include_wce=True)
    )

    errors, _ = vcc.run_checks()
    assert any("build_evidence_payload" in e for e in errors), errors


def test_should_not_have_manual_registry_in_generate_capability_evidence() -> None:
    """generate_capability_evidence.py must be deleted or contain no _RUNNERS or TIF imports."""
    script = Path(__file__).parents[2] / "scripts" / "generate_capability_evidence.py"
    if not script.exists():
        return  # Deleted — compliant
    text = script.read_text(encoding="utf-8")
    assert "_RUNNERS" not in text, (
        "generate_capability_evidence.py still contains a manual _RUNNERS registry"
    )
    assert not re.search(r"^(?:import|from)\s+tif_", text, re.MULTILINE), (
        "generate_capability_evidence.py has direct TIF module imports"
    )


# ---------------------------------------------------------------------------
# Helper: TIF spec with optional Requirements/Claims/ADR sections
# ---------------------------------------------------------------------------


def _tif_spec_with_sections(
    tif_id: str,
    executable: str,
    evidence_key: str,
    *,
    verification_type: str = "api_contract",
    status: str = "active",
    evidence_builder: str = "build_evidence_payload()",
    requirements_served: list[str] | None = None,
    claims_served: list[str] | None = None,
    adr_refs: list[str] | None = None,
) -> str:
    identity = textwrap.dedent(f"""\
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
    req_section = ""
    if requirements_served:
        rows = "\n".join(f"| {r} | - |" for r in requirements_served)
        req_section = (
            "\n## Requirements served\n\n"
            "| Requirement | Observation fields used |\n|---|---|\n"
            f"{rows}\n"
        )
    claims_section = ""
    if claims_served:
        items = "\n".join(f"- {c}" for c in claims_served)
        claims_section = f"\n## Claims served\n\n{items}\n"
    adr_section = ""
    if adr_refs:
        items = "\n".join(f"- {a}" for a in adr_refs)
        adr_section = f"\n## ADR refs\n\n{items}\n"
    return identity + req_section + claims_section + adr_section


def _readme_with_tif_table_full(rows: list[dict]) -> str:
    """Build a README snippet with all columns including requirements_served and claims_served."""
    header = (
        "| TIF ID | Executable | Evidence Key | Verification Type | Status "
        "| Requirements served | Claims served |"
    )
    sep = "|---|---|---|---|---|---|---|"
    data = "\n".join(
        f"| {r['tif_id']} | {r.get('executable', '')} | {r.get('evidence_key', '')} "
        f"| {r.get('verification_type', 'api_contract')} | {r.get('status', 'active')} "
        f"| {r.get('requirements_served', '')} | {r.get('claims_served', '')} |"
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
# Tests: committed evidence vs TIF spec cross-check
# ---------------------------------------------------------------------------


def test_should_fail_when_committed_evidence_has_wrong_claim_ids(
    chain_dirs: dict[str, Path],
) -> None:
    """Evidence referencing a TIF spec is missing a claim declared by that spec."""
    tif_dir = chain_dirs["tif"]
    raw_dir = chain_dirs["raw_evid"]

    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            claims_served=["CE-CAP-TEST-001"],
            requirements_served=["CE-REQ-TEST-001"],
        ),
    )
    evid = _minimal_raw_evidence(
        "CE-EVID-TEST-001-20260622",
        claim_ids=["CE-CAP-WRONG-001"],  # missing CE-CAP-TEST-001
        req_ids=["CE-REQ-TEST-001"],
        tif_ids=["CE-TIF-TEST-001"],
    )
    (raw_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    errors, _ = vcc.run_checks()
    assert any("CE-CAP-TEST-001" in e and "claim_ids" in e for e in errors), errors


def test_should_fail_when_committed_evidence_has_wrong_requirement_ids(
    chain_dirs: dict[str, Path],
) -> None:
    """Evidence referencing a TIF spec is missing a requirement declared by that spec."""
    tif_dir = chain_dirs["tif"]
    raw_dir = chain_dirs["raw_evid"]

    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            claims_served=["CE-CAP-TEST-001"],
            requirements_served=["CE-REQ-TEST-001"],
        ),
    )
    evid = _minimal_raw_evidence(
        "CE-EVID-TEST-001-20260622",
        claim_ids=["CE-CAP-TEST-001"],
        req_ids=["CE-REQ-WRONG-001"],  # missing CE-REQ-TEST-001
        tif_ids=["CE-TIF-TEST-001"],
    )
    (raw_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    errors, _ = vcc.run_checks()
    assert any("CE-REQ-TEST-001" in e and "requirement_ids" in e for e in errors), errors


def test_should_fail_when_committed_evidence_omits_declared_adr_refs(
    chain_dirs: dict[str, Path],
) -> None:
    """Evidence referencing a TIF spec is missing an ADR ref declared by that spec."""
    tif_dir = chain_dirs["tif"]
    raw_dir = chain_dirs["raw_evid"]

    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            claims_served=["CE-CAP-TEST-001"],
            requirements_served=["CE-REQ-TEST-001"],
            adr_refs=["ADR-001"],
        ),
    )
    evid = _minimal_raw_evidence(
        "CE-EVID-TEST-001-20260622",
        claim_ids=["CE-CAP-TEST-001"],
        req_ids=["CE-REQ-TEST-001"],
        tif_ids=["CE-TIF-TEST-001"],
    )
    evid["adr_refs"] = []  # spec declares ADR-001, evidence omits it
    (raw_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    errors, _ = vcc.run_checks()
    assert any("ADR-001" in e and "adr_refs" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Tests: README requirements_served / claims_served cross-check
# ---------------------------------------------------------------------------


def test_should_fail_when_readme_row_has_wrong_requirements_served(
    chain_dirs: dict[str, Path],
) -> None:
    """A README row whose requirements_served differs from the spec fails."""
    tif_dir = chain_dirs["tif"]
    exec_path = tif_dir / "tif_test.py"
    write_text_fixture(exec_path, _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))
    exec_path.write_text(
        exec_path.read_text(encoding="utf-8")
        + "\ndef build_evidence_payload(**_kw):\n    return {}\n",
        encoding="utf-8",
    )
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            requirements_served=["CE-REQ-TEST-001"],
            claims_served=["CE-CAP-TEST-001"],
        ),
    )
    write_text_fixture(
        tif_dir / "README.md",
        _readme_with_tif_table_full(
            [
                {
                    "tif_id": "CE-TIF-TEST-001",
                    "executable": "tif_test.py",
                    "evidence_key": "TEST-001",
                    "verification_type": "api_contract",
                    "status": "active",
                    "requirements_served": "CE-REQ-WRONG-001",  # mismatch
                    "claims_served": "CE-CAP-TEST-001",
                }
            ]
        ),
    )

    errors, _ = vcc.run_checks()
    assert any("CE-TIF-TEST-001" in e and "requirements_served" in e for e in errors), errors


def test_should_fail_when_readme_row_has_wrong_claims_served(
    chain_dirs: dict[str, Path],
) -> None:
    """A README row whose claims_served differs from the spec fails."""
    tif_dir = chain_dirs["tif"]
    exec_path = tif_dir / "tif_test.py"
    write_text_fixture(exec_path, _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))
    exec_path.write_text(
        exec_path.read_text(encoding="utf-8")
        + "\ndef build_evidence_payload(**_kw):\n    return {}\n",
        encoding="utf-8",
    )
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            requirements_served=["CE-REQ-TEST-001"],
            claims_served=["CE-CAP-TEST-001"],
        ),
    )
    write_text_fixture(
        tif_dir / "README.md",
        _readme_with_tif_table_full(
            [
                {
                    "tif_id": "CE-TIF-TEST-001",
                    "executable": "tif_test.py",
                    "evidence_key": "TEST-001",
                    "verification_type": "api_contract",
                    "status": "active",
                    "requirements_served": "CE-REQ-TEST-001",
                    "claims_served": "CE-CAP-WRONG-001",  # mismatch
                }
            ]
        ),
    )

    errors, _ = vcc.run_checks()
    assert any("CE-TIF-TEST-001" in e and "claims_served" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Tests: active TIF must have committed evidence (_check_active_tifs_have_evidence)
# ---------------------------------------------------------------------------


def test_should_fail_when_active_tif_spec_has_no_committed_evidence(
    chain_dirs: dict[str, Path],
) -> None:
    """An active TIF spec with no matching CE-EVID-*.json file fails."""
    tif_dir = chain_dirs["tif"]
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
        ),
    )
    # chain_dirs["raw_evid"] is empty — no evidence written

    errors, _ = vcc.run_checks()
    assert any("CE-TIF-TEST-001" in e and "no committed raw evidence" in e for e in errors), errors


def test_should_pass_when_active_tif_spec_has_committed_evidence(
    chain_dirs: dict[str, Path],
) -> None:
    """An active TIF with matching evidence does not produce an evidence-missing error."""
    tif_dir = chain_dirs["tif"]
    raw_dir = chain_dirs["raw_evid"]
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
        ),
    )
    evid = _minimal_raw_evidence(
        "CE-EVID-TEST-001-20260622",
        claim_ids=["CE-CAP-TEST-001"],
        req_ids=["CE-REQ-TEST-001"],
        tif_ids=["CE-TIF-TEST-001"],
    )
    (raw_dir / "CE-EVID-TEST-001-20260622.json").write_text(json.dumps(evid), encoding="utf-8")

    errors, _ = vcc.run_checks()
    assert not any("CE-TIF-TEST-001" in e and "no committed raw evidence" in e for e in errors), (
        errors
    )


# ---------------------------------------------------------------------------
# Tests: bidirectional claim↔requirement reciprocity
# ---------------------------------------------------------------------------


def test_should_fail_when_claim_to_req_link_not_reciprocal(
    chain_dirs: dict[str, Path],
) -> None:
    """Claim lists a requirement whose claim_refs does not include the claim."""
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    # Requirement points to a DIFFERENT claim — not back to CE-CAP-TEST-001
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-OTHER-001",  # wrong claim_ref
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )

    errors, _ = vcc.run_checks()
    assert any(
        "CE-CAP-TEST-001" in e and "CE-REQ-TEST-001" in e and "does not link back" in e
        for e in errors
    ), errors


def test_should_fail_when_req_to_claim_link_not_reciprocal(
    chain_dirs: dict[str, Path],
) -> None:
    """Requirement's claim_ref points to a claim that does not list this requirement."""
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    # Claim does NOT list CE-REQ-TEST-001 in its requirements
    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-002"], atomic_rationale=True),
    )
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-002.md",
        _minimal_req(
            "CE-REQ-TEST-002",
            "CE-CAP-TEST-001",
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )
    # This requirement references CE-CAP-TEST-001 but the claim does not list it
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",  # references the claim
            vstatus="verified",
            tif_exemption="documentation_boundary",
        ),
    )

    errors, _ = vcc.run_checks()
    assert any(
        "CE-REQ-TEST-001" in e and "CE-CAP-TEST-001" in e and "does not list this requirement" in e
        for e in errors
    ), errors


# ---------------------------------------------------------------------------
# Tests: bidirectional TIF↔requirement reciprocity
# ---------------------------------------------------------------------------


def test_should_fail_when_req_tif_ref_not_in_tif_requirements_served(
    chain_dirs: dict[str, Path],
) -> None:
    """Requirement has a tif_ref to T, but T's requirements_served does not include the req."""
    tif_dir = chain_dirs["tif"]
    reqs_dir = chain_dirs["reqs"]
    claims_dir = chain_dirs["claims"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    # TIF spec with empty requirements_served — doesn't list CE-REQ-TEST-001
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            requirements_served=[],  # intentionally empty
            claims_served=["CE-CAP-TEST-001"],
        ),
    )
    # Requirement references the TIF
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",
            obligation_type="api_contract",
            vstatus="verified",
            tif_refs=["CE-TIF-TEST-001"],
        ),
    )

    errors, _ = vcc.run_checks()
    assert any(
        "CE-REQ-TEST-001" in e and "CE-TIF-TEST-001" in e and "does not list this requirement" in e
        for e in errors
    ), errors


def test_should_fail_when_tif_requirements_served_not_in_req_tif_refs(
    chain_dirs: dict[str, Path],
) -> None:
    """TIF spec lists a requirement in requirements_served, but that req's tif_refs omits the TIF."""
    tif_dir = chain_dirs["tif"]
    reqs_dir = chain_dirs["reqs"]
    claims_dir = chain_dirs["claims"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    # TIF spec serves CE-REQ-TEST-001
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            requirements_served=["CE-REQ-TEST-001"],
            claims_served=["CE-CAP-TEST-001"],
        ),
    )
    # Requirement does NOT reference the TIF back (uses tif_exemption instead)
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",
            obligation_type="api_contract",
            vstatus="verified",
            tif_exemption="documentation_boundary",  # no tif_refs
        ),
    )

    errors, _ = vcc.run_checks()
    assert any(
        "CE-TIF-TEST-001" in e and "CE-REQ-TEST-001" in e and "does not reference" in e
        for e in errors
    ), errors


# ---------------------------------------------------------------------------
# Tests: TIF→claim reachability
# ---------------------------------------------------------------------------


def test_should_fail_when_tif_served_claim_not_reachable_through_served_requirements(
    chain_dirs: dict[str, Path],
) -> None:
    """TIF serves a claim, but none of its requirements_served have that claim in claim_refs."""
    tif_dir = chain_dirs["tif"]
    reqs_dir = chain_dirs["reqs"]
    claims_dir = chain_dirs["claims"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    write_text_fixture(
        claims_dir / "CE-CAP-ORPHAN-001.yaml",
        _minimal_claim("CE-CAP-ORPHAN-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",  # only references CE-CAP-TEST-001, not CE-CAP-ORPHAN-001
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )
    # TIF claims to serve CE-CAP-ORPHAN-001, but its served requirements only link to CE-CAP-TEST-001
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            requirements_served=["CE-REQ-TEST-001"],
            claims_served=["CE-CAP-ORPHAN-001"],  # not reachable
        ),
    )

    errors, _ = vcc.run_checks()
    assert any(
        "CE-TIF-TEST-001" in e and "CE-CAP-ORPHAN-001" in e and "not reachable" in e for e in errors
    ), errors


def test_should_pass_when_tif_served_claim_is_reachable_through_served_requirements(
    chain_dirs: dict[str, Path],
) -> None:
    """TIF serving a claim whose requirement links back passes reachability check."""
    tif_dir = chain_dirs["tif"]
    reqs_dir = chain_dirs["reqs"]
    claims_dir = chain_dirs["claims"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",
            vstatus="not_implemented",
            tif_exemption="documentation_boundary",
        ),
    )
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        _tif_spec_with_sections(
            "CE-TIF-TEST-001",
            str(tif_dir / "tif_test.py"),
            "TEST-001",
            requirements_served=["CE-REQ-TEST-001"],
            claims_served=["CE-CAP-TEST-001"],
        ),
    )

    errors, _ = vcc.run_checks()
    assert not any("not reachable" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Tests: hardened curated evidence raw_evidence_ref checks
# ---------------------------------------------------------------------------


def test_should_fail_when_curated_evidence_has_unresolved_raw_evidence_ref(
    chain_dirs: dict[str, Path],
) -> None:
    """A curated evidence file referencing a non-existent CE-EVID-* ID fails with a hard error."""
    evid_dir = chain_dirs["evid"]
    write_text_fixture(
        evid_dir / "evidence_test_001.md",
        """\
        # Test Evidence

        | Field | Value |
        |---|---|
        | requirement_ids | CE-REQ-TEST-001 |

        raw_evidence_ref: CE-EVID-NONEXISTENT-001-20260622
        """,
    )

    errors, _ = vcc.run_checks()
    assert any("CE-EVID-NONEXISTENT-001-20260622" in e and "exact match" in e for e in errors), (
        errors
    )


def test_should_pass_when_curated_tif_exempt_evidence_uses_none_raw_ref(
    chain_dirs: dict[str, Path],
) -> None:
    """Curated evidence with raw_evidence_ref: none passes when requirements are TIF-exempt."""
    evid_dir = chain_dirs["evid"]
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-SCHEMA-001"], atomic_rationale=True),
    )
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-SCHEMA-001.md",
        _minimal_req(
            "CE-REQ-TEST-SCHEMA-001",
            "CE-CAP-TEST-001",
            obligation_type="schema_validation",
            vstatus="verified",
            tif_exemption="schema_validation",
        ),
    )
    write_text_fixture(
        evid_dir / "evidence_test_schema.md",
        """\
        # Schema validation evidence

        | Field | Value |
        |---|---|
        | requirement_ids | CE-REQ-TEST-SCHEMA-001 |

        raw_evidence_ref: none — TIF-exempt schema validation check
        """,
    )

    errors, _ = vcc.run_checks()
    assert not any("raw_evidence_ref" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Test: generate_tif_evidence.py has no manual registry
# ---------------------------------------------------------------------------


def test_should_not_have_manual_registry_in_generate_tif_evidence() -> None:
    """generate_tif_evidence.py must not contain a manual runner registry or direct TIF imports."""
    script = Path(__file__).parents[2] / "scripts" / "generate_tif_evidence.py"
    assert script.exists(), "generate_tif_evidence.py not found"
    text = script.read_text(encoding="utf-8")
    assert "_RUNNERS" not in text, "generate_tif_evidence.py contains a manual _RUNNERS registry"
    assert not re.search(r"^(?:import|from)\s+tif_", text, re.MULTILINE), (
        "generate_tif_evidence.py has direct TIF module imports"
    )


# ---------------------------------------------------------------------------
# Tests: entry_functions validation
# ---------------------------------------------------------------------------


def test_should_fail_when_active_tif_spec_declares_missing_entry_function(
    chain_dirs: dict[str, Path],
) -> None:
    """An active TIF spec that declares an entry function not defined in the executable fails."""
    tif_dir = chain_dirs["tif"]
    exec_path = tif_dir / "tif_test.py"
    # Executable has WrapCalibratedExplainer and build_evidence_payload but NOT the declared fn
    write_text_fixture(exec_path, _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))
    exec_path.write_text(
        exec_path.read_text(encoding="utf-8")
        + "\ndef build_evidence_payload(**_kw):\n    return {}\n",
        encoding="utf-8",
    )
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        textwrap.dedent(f"""\
            # CE-TIF-TEST-001

            ## Identity

            | Field | Value |
            |---|---|
            | tif_id | CE-TIF-TEST-001 |
            | executable | `{exec_path}` |
            | entry_functions | `run_nonexistent_scenario()` |
            | evidence_builder | `build_evidence_payload()` |
            | evidence_key | TEST-001 |
            | verification_type | api_contract |
            | status | active |
        """),
    )

    errors, _ = vcc.run_checks()
    assert any("run_nonexistent_scenario" in e and "not found" in e for e in errors), (
        f"Expected entry-function-not-found error, got: {errors}"
    )


def test_should_pass_when_active_tif_spec_entry_functions_exist(
    chain_dirs: dict[str, Path],
) -> None:
    """An active TIF spec whose declared entry functions exist in the executable passes."""
    tif_dir = chain_dirs["tif"]
    exec_path = tif_dir / "tif_test.py"
    write_text_fixture(exec_path, _minimal_tif_py("CE-TIF-TEST-001", include_wce=True))
    exec_path.write_text(
        exec_path.read_text(encoding="utf-8")
        + "\ndef run_scenario():\n    pass\n\ndef build_evidence_payload(**_kw):\n    return {}\n",
        encoding="utf-8",
    )
    write_text_fixture(
        tif_dir / "CE-TIF-TEST-001.md",
        textwrap.dedent(f"""\
            # CE-TIF-TEST-001

            ## Identity

            | Field | Value |
            |---|---|
            | tif_id | CE-TIF-TEST-001 |
            | executable | `{exec_path}` |
            | entry_functions | `run_scenario()` |
            | evidence_builder | `build_evidence_payload()` |
            | evidence_key | TEST-001 |
            | verification_type | api_contract |
            | status | active |
        """),
    )

    errors, _ = vcc.run_checks()
    assert not any("run_scenario" in e and "not found" in e for e in errors), (
        f"Unexpected entry-function errors: {errors}"
    )


# ---------------------------------------------------------------------------
# Tests: TIF spec metadata injection into build_evidence_payload
# ---------------------------------------------------------------------------


def test_should_not_have_hardcoded_envelope_in_tif_executables() -> None:
    """TIF executables must use spec-injected kwargs, not hardcode claim/req/tif IDs.

    Regression guard: build_evidence_payload() must accept spec_* kwargs.
    This test scans each active TIF executable's build_evidence_payload signature.
    """
    import ast

    tif_dir = Path(__file__).parents[2] / "development" / "capabilities" / "verification" / "tif"
    executables = [p for p in tif_dir.glob("tif_*.py") if not p.stem.endswith("_helpers")]
    for exec_path in executables:
        source = exec_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "build_evidence_payload"):
                continue
            arg_names = {a.arg for a in node.args.kwonlyargs}
            assert "spec_claim_ids" in arg_names, (
                f"{exec_path.name}: build_evidence_payload() missing 'spec_claim_ids' kwarg — "
                "envelope metadata must be injected from the TIF spec, not hardcoded"
            )
            assert "spec_requirement_ids" in arg_names, (
                f"{exec_path.name}: build_evidence_payload() missing 'spec_requirement_ids' kwarg"
            )
            assert "spec_tif_id" in arg_names, (
                f"{exec_path.name}: build_evidence_payload() missing 'spec_tif_id' kwarg"
            )


# ---------------------------------------------------------------------------
# Tests: no manifest/sidecar/registry files added
# ---------------------------------------------------------------------------


def test_should_not_have_manifest_or_registry_sidecar_files() -> None:
    """No manifest, generated registry, or sidecar mapping file should exist in the repo.

    Forbidden patterns: files named *_manifest.*, *_registry.*, *_catalog.*,
    *_inventory.*, or tif_map.* anywhere under development/capabilities/ or scripts/.
    """
    repo_root = Path(__file__).parents[2]
    forbidden_patterns = [
        "*_manifest.*",
        "*_registry.*",
        "*_catalog.*",
        "*_inventory.*",
        "tif_map.*",
        "claim_map.*",
        "req_map.*",
    ]
    search_dirs = [
        repo_root / "development" / "capabilities",
        repo_root / "scripts",
    ]
    found: list[str] = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in forbidden_patterns:
            for match in search_dir.rglob(pattern):
                # Allow Python cache dirs
                if "__pycache__" not in str(match):
                    found.append(str(match.relative_to(repo_root)))
    assert not found, (
        "Manifest/registry/sidecar file(s) detected — these are forbidden "
        f"(active inventories must not be added as sidecar files): {found}"
    )


# ---------------------------------------------------------------------------
# Tests: tif_exemption_rationale required for behavioral types
# ---------------------------------------------------------------------------


def test_should_warn_when_behavioral_requirement_exempted_without_rationale(
    chain_dirs: dict[str, Path],
) -> None:
    """A behavioral requirement with tif_exemption but no tif_exemption_rationale warns.

    repository_policy exemptions on runtime_behavior/api_contract/serialization_contract etc.
    must document WHY WrapCalibratedExplainer-based TIF is not appropriate.
    """
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",
            obligation_type="runtime_behavior",
            vstatus="verified",
            tif_exemption="repository_policy",
            # tif_exemption_rationale intentionally omitted
        ),
    )

    _, warnings = vcc.run_checks()
    assert any("tif_exemption_rationale" in w and "CE-REQ-TEST-001" in w for w in warnings), (
        warnings
    )


def test_should_not_warn_when_behavioral_requirement_exempted_with_rationale(
    chain_dirs: dict[str, Path],
) -> None:
    """A behavioral requirement with tif_exemption AND tif_exemption_rationale does not warn about rationale."""
    claims_dir = chain_dirs["claims"]
    reqs_dir = chain_dirs["reqs"]

    write_text_fixture(
        claims_dir / "CE-CAP-TEST-001.yaml",
        _minimal_claim("CE-CAP-TEST-001", ["CE-REQ-TEST-001"], atomic_rationale=True),
    )
    write_text_fixture(
        reqs_dir / "CE-REQ-TEST-001.md",
        _minimal_req(
            "CE-REQ-TEST-001",
            "CE-CAP-TEST-001",
            obligation_type="runtime_behavior",
            vstatus="verified",
            tif_exemption="repository_policy",
            tif_exemption_rationale="Verified by unit tests targeting internals not observable through WrapCalibratedExplainer.",
        ),
    )

    _, warnings = vcc.run_checks()
    assert not any("tif_exemption_rationale" in w and "CE-REQ-TEST-001" in w for w in warnings), (
        warnings
    )
