"""Capability metadata checks for ADR evidence-bearing links."""

from __future__ import annotations

import re
from pathlib import Path


_REQUIRED_SEMANTIC_CLAIMS = {
    "ADR-027": ['CE-CAP-EXPL-FAST-FILTER-001'],
    "ADR-032": ['CE-CAP-GUARD-MEDIAN-PROBE-001', 'CE-CAP-GUARD-AUDIT-001', 'CE-CAP-GUARD-CALIBRATION-ALIGNMENT-001', 'CE-CAP-GUARD-CONJUNCTION-001', 'CE-CAP-GUARD-PLUGIN-SUPPORT-001', 'CE-CAP-GUARD-NO-FAST-001', 'CE-CAP-ALT-TARGET-CONFIDENCE-001'],
    "ADR-029": ['CE-CAP-REJECT-DEFAULT-OFF-001', 'CE-CAP-REJECT-POLICY-ENUM-001', 'CE-CAP-REJECT-RESULT-ENVELOPE-001', 'CE-CAP-REJECT-STRATEGY-REGISTRY-001', 'CE-CAP-REJECT-NO-VIZ-001'],
    "ADR-013": ['CE-CAP-INTERVAL-PLUGIN-PROTOCOL-001', 'CE-CAP-INTERVAL-CONTEXT-IMMUTABILITY-001', 'CE-CAP-INTERVAL-PLUGIN-FALLBACK-001', 'CE-CAP-FAST-INTERVAL-PLUGIN-001', 'CE-CAP-INTERVAL-PLUGIN-VALIDATION-001'],
    "ADR-036": ['CE-CAP-PLOTSPEC-CANONICAL-DATACLASS-001', 'CE-CAP-PLOTSPEC-BOUNDARY-SERIALIZATION-001', 'CE-CAP-PLOTSPEC-BUILDER-VALIDATION-001', 'CE-CAP-PLOTSPEC-BACKEND-NEUTRALITY-001'],
    "ADR-037": ['CE-CAP-VIZ-EXTENSION-METADATA-001', 'CE-CAP-VIZ-PLOT-KIND-GOVERNANCE-001', 'CE-CAP-VIZ-DEFAULT-PLOTSPEC-PATH-001'],
    "ADR-034": ['CE-CAP-CONFIG-AUTHORITY-001', 'CE-CAP-CONFIG-PRECEDENCE-SNAPSHOT-001', 'CE-CAP-CONFIG-STRICT-VALIDATION-001', 'CE-CAP-CONFIG-DIAGNOSTIC-EXPORT-001', 'CE-CAP-CONFIG-CI-ENFORCEMENT-001'],
    "ADR-028": ['CE-CAP-LOG-DOMAIN-TAXONOMY-001', 'CE-CAP-GOVERNANCE-LOG-SEPARATION-001', 'CE-CAP-LOG-CONTEXT-PROPAGATION-001', 'CE-CAP-LOG-STRUCTURED-COMPATIBILITY-001', 'CE-CAP-LOG-NO-GLOBAL-HANDLERS-001', 'CE-CAP-LOG-DATA-MINIMISATION-001'],
    "ADR-031": ['CE-CAP-CALIBRATOR-PRIMITIVE-SCHEMA-001', 'CE-CAP-EXPLAINER-STATE-PERSISTENCE-001', 'CE-CAP-SERIAL-FAIL-FAST-VERSIONING-001', 'CE-CAP-SERIAL-ROUNDTRIP-INVARIANTS-001'],
    "ADR-005": ['CE-CAP-SCHEMA-PAYLOAD-V1-001', 'CE-CAP-SCHEMA-VALIDATION-001', 'CE-CAP-SCHEMA-EXTENSION-SURFACE-001'],
    "ADR-008": ['CE-CAP-DOMAIN-MODEL-001', 'CE-CAP-DOMAIN-LEGACY-ADAPTER-001', 'CE-CAP-EXPL-PAPER-SEMANTICS-001'],
    "ADR-006": ['CE-CAP-PLUGIN-TRUST-POLICY-001', 'CE-CAP-PLUGIN-DISCOVERY-REPORT-001', 'CE-CAP-PLUGIN-AUDIT-EVENTS-001'],
    "ADR-033": ['CE-CAP-MODALITY-METADATA-001', 'CE-CAP-MODALITY-RESOLUTION-001', 'CE-CAP-MODALITY-SHIMS-001', 'CE-CAP-MODALITY-TABULAR-INVARIANT-001'],
}


def test_should_validate_curated_semantic_claim_presence() -> None:
    """Verifies the minimum semantic ADR claim split required by the hardening pass."""
    adr_dir = _repo_root() / "development" / "adrs"
    claim_dir = _repo_root() / "development" / "capabilities" / "claims"
    errors: list[str] = []

    for adr_id, required_claims in _REQUIRED_SEMANTIC_CLAIMS.items():
        adr_matches = list(adr_dir.glob(f"{adr_id}-*.md"))
        if not adr_matches:
            errors.append(f"missing ADR file for {adr_id}")
            continue
        adr_text = adr_matches[0].read_text(encoding="utf-8")
        for claim_id in required_claims:
            claim_path = claim_dir / f"{claim_id}.yaml"
            if not claim_path.exists():
                errors.append(f"{claim_id} is required for {adr_id} but has no claim file")
                continue
            if f"`{claim_id}`" not in adr_text:
                errors.append(f"{adr_id} does not list required semantic claim {claim_id}")
            if adr_id not in _yaml_list(claim_path.read_text(encoding="utf-8"), "adr_links"):
                errors.append(f"{claim_id} does not link back to {adr_id}")

    adr027_text = next(adr_dir.glob("ADR-027-*.md")).read_text(encoding="utf-8")
    assert "`CE-CAP-EXPL-FILTER-001`" not in adr027_text
    assert not errors, "\n".join(errors)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _yaml_list(text: str, key: str) -> list[str]:
    match = re.search(rf"^{key}:\n((?:  - .+\n)+)", text, re.MULTILINE)
    if not match:
        return []
    return [line.split("-", 1)[1].strip() for line in match.group(1).splitlines()]


def _metadata_value(text: str, field: str) -> str:
    match = re.search(rf"^\| {re.escape(field)} \| ([^|]+) \|", text, re.MULTILINE)
    return match.group(1).strip() if match else ""




_BEHAVIORAL_TYPES = {
    "api_contract",
    "runtime_behavior",
    "serialization_contract",
    "payload_schema",
    "static_policy",
    "quality_gate",
}

_METADATA_ONLY_TESTS = {
    "test_should_validate_adr_claim_requirement_link_metadata",
    "test_should_validate_curated_semantic_claim_presence",
}


def _requirement_files() -> list[Path]:
    return sorted((_repo_root() / "development" / "capabilities" / "requirements").glob("CE-REQ-*.md"))


def _requirement_id(path: Path) -> str:
    return path.stem


def test_should_require_behavioral_requirements_to_name_concrete_evidence() -> None:
    """Verifies behavioral requirements do not rely only on metadata tests."""
    errors: list[str] = []

    for path in _requirement_files():
        text = path.read_text(encoding="utf-8")
        obligation_type = _metadata_value(text, "obligation_type")
        if obligation_type not in _BEHAVIORAL_TYPES:
            continue
        if "## Verification targets" in text:
            target_block = text.split("## Verification targets", 1)[1].split("\n## ", 1)[0]
        elif "Automated pytest" in text and "tests/capabilities/" in text:
            # Existing capability requirements predate the explicit Verification targets
            # section and name concrete pytest files/functions in their Verification method.
            target_block = text
        else:
            errors.append(f"{_requirement_id(path)} has no concrete verification targets")
            continue
        has_non_metadata_target = any(
            marker in target_block
            for marker in (
                "- source:",
                "- quality_script:",
                "- docs:",
                "- schema:",
                "- standard:",
                "- manual_review:",
                "- pytest:",
                "tests/capabilities/",
                "Test ID:",
                "Test IDs:",
            )
        )
        if not has_non_metadata_target:
            errors.append(f"{_requirement_id(path)} only names metadata tests as evidence")
        for metadata_test in _METADATA_ONLY_TESTS:
            if metadata_test in target_block and not has_non_metadata_target:
                errors.append(f"{_requirement_id(path)} treats {metadata_test} as sole evidence")

    assert not errors, "\n".join(errors)


def test_should_prevent_runtime_obligations_from_using_documentation_boundary_type() -> None:
    """Verifies runtime MUST language is not mislabeled as documentation-only."""
    runtime_terms = (
        "MUST raise",
        "MUST reject",
        "MUST return",
        "MUST fail",
        "MUST preserve",
        "MUST validate",
        "MUST apply",
        "MUST expose",
        "MUST read",
        "MUST serialize",
        "MUST restore",
    )
    errors: list[str] = []

    for path in _requirement_files():
        text = path.read_text(encoding="utf-8")
        obligation_type = _metadata_value(text, "obligation_type")
        if obligation_type != "documentation_boundary":
            continue
        observable = text.split("## Observable behavior", 1)[1].split("\n## ", 1)[0]
        if any(term in observable for term in runtime_terms):
            errors.append(f"{_requirement_id(path)} has runtime MUST language but documentation_boundary type")

    assert not errors, "\n".join(errors)


def test_should_require_semantic_claims_to_have_non_metadata_requirements() -> None:
    """Verifies curated semantic claims are backed by non-documentation requirements."""
    req_dir = _repo_root() / "development" / "capabilities" / "requirements"
    claim_dir = _repo_root() / "development" / "capabilities" / "claims"
    errors: list[str] = []

    for required_claims in _REQUIRED_SEMANTIC_CLAIMS.values():
        for claim_id in required_claims:
            claim_text = (claim_dir / f"{claim_id}.yaml").read_text(encoding="utf-8")
            requirement_ids = _yaml_list(claim_text, "requirements")
            if not requirement_ids:
                errors.append(f"{claim_id} has no requirement links")
                continue
            concrete = False
            for requirement_id in requirement_ids:
                req_path = req_dir / f"{requirement_id}.md"
                if not req_path.exists():
                    errors.append(f"{claim_id} references missing requirement {requirement_id}")
                    continue
                req_text = req_path.read_text(encoding="utf-8")
                if _metadata_value(req_text, "obligation_type") != "documentation_boundary":
                    concrete = True
            if not concrete:
                errors.append(f"{claim_id} has only documentation-boundary requirements")

    assert not errors, "\n".join(errors)


def test_should_validate_adr_claim_requirement_link_metadata() -> None:
    """Verifies ADR-governed CE-CAP/CE-REQ link consistency."""
    root = _repo_root()
    adr_dir = root / "development" / "adrs"
    claim_dir = root / "development" / "capabilities" / "claims"
    req_dir = root / "development" / "capabilities" / "requirements"

    claim_files = {path.stem: path for path in claim_dir.glob("CE-CAP-*.yaml")}
    requirement_files = {path.stem: path for path in req_dir.glob("CE-REQ-*.md")}
    active_adr_files = [path for path in adr_dir.glob("ADR-*.md")]

    errors: list[str] = []

    for adr_path in active_adr_files:
        adr_id = re.search(r"ADR-\d+", adr_path.name).group(0)
        adr_text = adr_path.read_text(encoding="utf-8")
        governed_claims = re.findall(r"^- `(CE-CAP-[A-Z0-9-]+)`", adr_text, re.MULTILINE)
        if not governed_claims:
            errors.append(f"{adr_id} does not list governed claims")
            continue
        for claim_id in governed_claims:
            claim_path = claim_files.get(claim_id)
            if claim_path is None:
                errors.append(f"{adr_id} references missing claim {claim_id}")
                continue
            claim_text = claim_path.read_text(encoding="utf-8")
            if adr_id not in _yaml_list(claim_text, "adr_links"):
                errors.append(f"{claim_id} does not link back to {adr_id}")

    for claim_id, claim_path in claim_files.items():
        claim_text = claim_path.read_text(encoding="utf-8")
        adr_links = _yaml_list(claim_text, "adr_links")
        requirements = _yaml_list(claim_text, "requirements")
        if not adr_links:
            errors.append(f"{claim_id} has no adr_links")
        if not requirements:
            errors.append(f"{claim_id} has no requirements")
        for adr_id in adr_links:
            matches = list(adr_dir.glob(f"{adr_id}-*.md"))
            if not matches:
                errors.append(f"{claim_id} references missing ADR {adr_id}")
                continue
            if f"`{claim_id}`" not in matches[0].read_text(encoding="utf-8"):
                errors.append(f"{adr_id} does not list {claim_id}")
        for req_id in requirements:
            req_path = requirement_files.get(req_id)
            if req_path is None:
                errors.append(f"{claim_id} references missing requirement {req_id}")
                continue
            req_text = req_path.read_text(encoding="utf-8")
            claim_refs = _metadata_value(req_text, "claim_refs")
            if claim_id not in claim_refs:
                errors.append(f"{req_id} does not link back to {claim_id}")

    assert not errors, "\n".join(errors)
