"""Requirements-as-code checks for ADR, claim, requirement, and evidence links."""

from __future__ import annotations

import ast
import re
from pathlib import Path


BEHAVIORAL_OBLIGATION_TYPES = {
    "api_contract",
    "runtime_behavior",
    "serialization_contract",
    "payload_schema",
    "plugin_behavior",
    "visualization_behavior",
    "static_policy",
    "quality_gate",
    "numerical_behavior",
    "statistical_method_alignment",
    "empirical_smoke",
}
GAP_STATUSES = {"adr_gap_open", "not_implemented"}
MANUAL_EVIDENCE_TERMS = {
    "manual_review",
    "manual_review_required",
    "human verification",
    "human_review",
    "manual verification",
}
METADATA_ONLY_TESTS = {
    "tests/capabilities/test_adr_capability_links.py::test_should_validate_adr_claim_requirement_link_metadata",
}
PYTEST_TARGET_RE = re.compile(
    r"pytest:\s+(tests/[^`\s]+\.py)::([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)"
)
QUALITY_TARGET_RE = re.compile(r"(?:quality-gate|quality_gate|ci-gate|ci_gate):\s+([^`\n]+)")
TRACEABILITY_ACCEPTANCE_PHRASE = "capability traceability validation test passes"


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


def _section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    return match.group(1).strip() if match else ""


def _test_index(root: Path) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for path in (root / "tests").rglob("test_*.py"):
        rel_path = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
        node_ids: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                node_ids.add(node.name)
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(
                        item, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and item.name.startswith("test_"):
                        node_ids.add(f"{node.name}::{item.name}")
        index[rel_path] = node_ids
    return index


def _referenced_gap_exists(
    appendix_text: str, requirement_id: str, adr_refs: str, gap_ref: str
) -> bool:
    if gap_ref and gap_ref in appendix_text:
        return requirement_id in appendix_text
    return requirement_id in appendix_text and any(
        adr.strip() in appendix_text for adr in adr_refs.split(",")
    )


def test_should_validate_adr_claim_requirement_link_metadata() -> None:
    """Verifies ADR-governed CE-CAP/CE-REQ link consistency."""
    root = _repo_root()
    adr_dir = root / "development" / "adrs"
    claim_dir = root / "development" / "capabilities" / "claims"
    req_dir = root / "development" / "capabilities" / "requirements"

    claim_files = {path.stem: path for path in claim_dir.glob("CE-CAP-*.yaml")}
    requirement_files = {path.stem: path for path in req_dir.glob("CE-REQ-*.md")}
    active_adr_files = list(adr_dir.glob("ADR-*.md"))

    errors: list[str] = []

    for adr_path in active_adr_files:
        adr_id = re.search(r"ADR-\d+", adr_path.name).group(0)
        adr_text = adr_path.read_text(encoding="utf-8")
        if "## Governed claims" not in adr_text:
            errors.append(f"{adr_id} has no ## Governed claims section")
            continue
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
            adr_refs = _metadata_value(req_text, "adr_refs")
            if claim_id not in claim_refs:
                errors.append(f"{req_id} does not link back to {claim_id}")
            if not adr_refs:
                errors.append(f"{req_id} has no adr_refs")
            for adr_id in [ref.strip() for ref in adr_refs.split(",") if ref.strip()]:
                if not list(adr_dir.glob(f"{adr_id}-*.md")):
                    errors.append(f"{req_id} references missing ADR {adr_id}")

    assert not errors, "\n".join(errors)


def test_should_require_executable_evidence_when_behavioral_requirement_is_implemented() -> None:
    """Rejects prose, metadata-only, or manual evidence for implemented behavior."""
    root = _repo_root()
    req_dir = root / "development" / "capabilities" / "requirements"
    appendix_text = (
        root / "development" / "current-work" / "RELEASE_PLAN_status_appendix.md"
    ).read_text(encoding="utf-8")
    errors: list[str] = []

    for req_path in sorted(req_dir.glob("CE-REQ-*.md")):
        req_text = req_path.read_text(encoding="utf-8")
        requirement_id = _metadata_value(req_text, "requirement_id") or req_path.stem
        obligation_type = _metadata_value(req_text, "obligation_type")
        verification_status = _metadata_value(req_text, "verification_status")
        adr_refs = _metadata_value(req_text, "adr_refs")
        gap_ref = _metadata_value(req_text, "gap_ref") or _metadata_value(req_text, "adr_gap_ref")
        evidence_text = "\n".join(
            [
                _section(req_text, "Verification method"),
                _section(req_text, "Verification targets"),
                _section(req_text, "Evidence required"),
            ]
        )
        acceptance_text = _section(req_text, "Acceptance criterion")
        contains_manual_evidence = any(
            term in evidence_text.lower() for term in MANUAL_EVIDENCE_TERMS
        )
        is_gap = verification_status in GAP_STATUSES
        if (
            TRACEABILITY_ACCEPTANCE_PHRASE in acceptance_text.lower()
            and obligation_type != "governance_traceability"
        ):
            errors.append(
                f"{requirement_id} uses metadata-linkage acceptance for a non-traceability requirement"
            )
        if contains_manual_evidence and not is_gap:
            errors.append(f"{requirement_id} uses manual/human verification without ADR gap status")
        if is_gap:
            if not gap_ref:
                errors.append(
                    f"{requirement_id} is {verification_status} but has no gap_ref/adr_gap_ref"
                )
            if not _referenced_gap_exists(appendix_text, requirement_id, adr_refs, gap_ref):
                errors.append(
                    f"{requirement_id} gap reference is absent from RELEASE_PLAN_status_appendix.md"
                )
            continue
        if obligation_type in BEHAVIORAL_OBLIGATION_TYPES:
            pytest_targets = [
                f"{path}::{func}" for path, func in PYTEST_TARGET_RE.findall(evidence_text)
            ]
            quality_targets = QUALITY_TARGET_RE.findall(evidence_text)
            executable_targets = [
                target for target in pytest_targets if target not in METADATA_ONLY_TESTS
            ] + quality_targets
            if not executable_targets:
                errors.append(f"{requirement_id} has no executable behavioral evidence target")
        metadata_targets = [
            f"{path}::{node_id}"
            for path, node_id in PYTEST_TARGET_RE.findall(evidence_text)
            if f"{path}::{node_id}" in METADATA_ONLY_TESTS
        ]
        if metadata_targets and obligation_type != "governance_traceability":
            errors.append(
                f"{requirement_id} cites metadata-only evidence for a non-traceability requirement"
            )

    assert not errors, "\n".join(errors)


def test_should_reference_existing_pytest_targets_when_requirements_cite_tests() -> None:
    """Builds an AST test index and rejects missing cited pytest evidence."""
    root = _repo_root()
    test_index = _test_index(root)
    errors: list[str] = []

    for req_path in sorted(
        (root / "development" / "capabilities" / "requirements").glob("CE-REQ-*.md")
    ):
        req_text = req_path.read_text(encoding="utf-8")
        requirement_id = _metadata_value(req_text, "requirement_id") or req_path.stem
        for test_file, test_node_id in PYTEST_TARGET_RE.findall(req_text):
            if test_file not in test_index:
                errors.append(f"{requirement_id} cites missing test file {test_file}")
            elif test_node_id not in test_index[test_file]:
                errors.append(
                    f"{requirement_id} cites missing pytest target {test_file}::{test_node_id}"
                )

    assert not errors, "\n".join(errors)
