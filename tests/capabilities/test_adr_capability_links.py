"""Capability metadata checks for ADR evidence-bearing links."""

from __future__ import annotations

import re
from pathlib import Path


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


def test_should_validate_adr_capability_links_when_metadata_changes() -> None:
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
