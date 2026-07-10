"""Validate the CE capability verification chain.

Checks structural consistency across claims, requirements, TIF specs,
TIF executables, raw evidence, and curated evidence. Does not execute TIF
scenarios or mutate any files.

Usage:
    python scripts/quality/validate_capability_chain.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_REPO_ROOT = Path(__file__).parents[2]
_CLAIMS_DIR = _REPO_ROOT / "development" / "capabilities" / "claims"
_REQ_DIR = _REPO_ROOT / "development" / "capabilities" / "requirements"
_TIF_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"
_EVID_DIR = _REPO_ROOT / "development" / "capabilities" / "evidence"
_RAW_EVID_DIR = _REPO_ROOT / "reports" / "verification"

# Obligation types for which a verified requirement must have tif_refs (or tif_exemption).
_TIF_REQUIRED_TYPES: set[str] = {
    "api_contract",
    "behavioral_contract",
    "numerical_behavior",
    "empirical_smoke",
    "visualization_behavior",
    "runtime_behavior",
    "serialization_contract",
}

# Obligation types where using tif_exemption instead of tif_refs is a hard error
# because the requirement describes observable CE public-API behavior.
_TIF_MANDATORY_NO_EXEMPTION: set[str] = {
    "behavioral_contract",
}

# Recognised valid tif_exemption type strings.
_VALID_TIF_EXEMPTION_TYPES: set[str] = {
    "documentation_boundary",
    "schema_validation",
    "repository_policy",
    "metadata_linkage",
    "static_importability_check",
}

# Recognised verification_status values.
_VALID_VERIFICATION_STATUS: set[str] = {
    "verified",
    "not_implemented",
    "adr_gap_open",
    "deferred",
    "superseded",
}

# Obligation types whose verified requirements should have behavioral raw evidence.
_BEHAVIORAL_EVIDENCE_TYPES: set[str] = {
    "api_contract",
    "behavioral_contract",
    "numerical_behavior",
    "statistical_method_alignment",
    "empirical_smoke",
    "visualization_structure",
    "visualization_behavior",
}

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# TIF executable patterns that are forbidden.
_TIF_FORBIDDEN_IMPORT_RE = re.compile(
    r"from\s+calibrated_explanations\.core\.calibrated_explainer\b"
)
_TIF_DIRECT_CONSTRUCT_RE = re.compile(r"\b(FactualExplanation|AlternativeExplanation)\s*\(")
# Private member access: ._ preceded by an identifier (but not in comments or strings — best effort).
_TIF_PRIVATE_ACCESS_RE = re.compile(r"\w\._\w")


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _parse_md_table(text: str) -> dict[str, str]:
    """Extract key-value pairs from all Markdown metadata table rows."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        key_raw = cells[0].strip()
        val_raw = cells[1].strip()
        if not key_raw or key_raw.lower() == "field" or re.fullmatch(r"[-:\s]+", key_raw):
            continue
        result[key_raw.lower().replace(" ", "_")] = val_raw
    return result


def _split_refs(value: str) -> list[str]:
    """Split a comma/space-separated reference list into individual IDs."""
    parts = re.split(r"[,\s]+", value.strip())
    return [p for p in parts if p and not re.fullmatch(r"[-]+", p)]


def _extract_section_text(text: str, section_name: str) -> str:
    """Return the body of a `## <section_name>` Markdown section."""
    pattern = rf"(?m)^##\s+{re.escape(section_name)}\s*$\n(.*?)(?=^##\s|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_entry_functions(entry_fn_raw: str) -> list[str]:
    """Parse function names from an entry_functions metadata value."""
    names: list[str] = []
    for part in entry_fn_raw.split(","):
        fn_match = re.match(r"(\w+)\s*\(", part.strip().strip("`"))
        if fn_match:
            names.append(fn_match.group(1))
    return names


def _parse_tif_spec_metadata(spec_path: Path) -> dict[str, Any]:
    """Parse Identity table fields and served refs from a CE-TIF-*.md spec file."""
    text = spec_path.read_text(encoding="utf-8")

    def table_value(field: str) -> str:
        match = re.search(rf"\|\s*{re.escape(field)}\s*\|\s*([^|]+?)\s*\|", text)
        return match.group(1).strip().strip("`") if match else ""

    reqs_section = _extract_section_text(text, "Requirements served")
    requirements_served: list[str] = re.findall(r"\|\s*(CE-REQ-[\w-]+)\s*\|", reqs_section)

    claims_section = _extract_section_text(text, "Claims served")
    claims_served: list[str] = re.findall(
        r"^\s*[-*]\s+(CE-CAP-[\w-]+)", claims_section, re.MULTILINE
    )

    adr_section = _extract_section_text(text, "ADR refs")
    adr_refs: list[str] = re.findall(r"^\s*[-*]\s+(ADR-\d+)", adr_section, re.MULTILINE)

    entry_fn_raw = table_value("entry_functions")
    entry_functions = _parse_entry_functions(entry_fn_raw) if entry_fn_raw else []

    return {
        "tif_id": table_value("tif_id"),
        "status": table_value("status"),
        "executable": table_value("executable"),
        "entry_functions": entry_functions,
        "evidence_builder": table_value("evidence_builder"),
        "evidence_key": table_value("evidence_key"),
        "verification_type": table_value("verification_type"),
        "requirements_served": requirements_served,
        "claims_served": claims_served,
        "adr_refs": adr_refs,
    }


def _parse_tif_readme_table(text: str) -> list[dict[str, str]]:
    """Parse the 'Current TIF Interfaces' Markdown table into a list of row dicts."""
    section = _extract_section_text(text, "Current TIF Interfaces")
    headers: list[str] = []
    rows: list[dict[str, str]] = []

    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not headers:
            headers = [h.strip().lower().replace(" ", "_") for h in cells]
            continue
        if all(re.fullmatch(r"[-:\s]+", c) for c in cells if c):
            continue
        if not cells or not any(c for c in cells):
            continue
        row: dict[str, str] = {}
        for i, h in enumerate(headers):
            val = cells[i].strip() if i < len(cells) else ""
            val = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", val)
            row[h] = val
        if any(row.values()):
            rows.append(row)

    return rows


def _parse_verification_targets(text: str) -> list[str]:
    """Extract file paths from the Verification targets section."""
    section = _extract_section_text(text, "Verification targets")
    paths: list[str] = []
    for line in section.splitlines():
        stripped = re.sub(r"^[-*\s`]+", "", line.strip()).strip("`")
        if stripped.startswith("pytest:"):
            raw = stripped[len("pytest:") :].strip()
            file_part = raw.split("::")[0].strip()
            if file_part:
                paths.append(file_part)
        elif stripped.startswith("quality-gate:"):
            cmd = stripped[len("quality-gate:") :].strip()
            for tok in cmd.split():
                if tok.endswith(".py") or tok.startswith("scripts/") or tok.startswith("tests/"):
                    paths.append(tok)
                    break
    return paths


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not _YAML_AVAILABLE:
        return None
    try:
        data = _yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chain checks
# ---------------------------------------------------------------------------


def _check_claims(errors: list[str], warnings: list[str]) -> dict[str, dict[str, Any]]:
    """Load and validate all claim YAML files. Returns {claim_id: data}."""
    claims: dict[str, dict[str, Any]] = {}
    if not _YAML_AVAILABLE:
        warnings.append("PyYAML not available; skipping YAML-based claim validation")
        for path in _CLAIMS_DIR.glob("CE-CAP-*.yaml"):
            claims[path.stem] = {}
        return claims

    for path in sorted(_CLAIMS_DIR.glob("CE-CAP-*.yaml")):
        expected_id = path.stem
        data = _load_yaml(path)
        if data is None:
            errors.append(f"claim {path.name}: YAML parse error")
            continue

        claim_id = data.get("claim_id", "")
        if not claim_id:
            errors.append(f"claim {path.name}: claim_id is missing")
        elif claim_id != expected_id:
            errors.append(
                f"claim {path.name}: claim_id '{claim_id}' != filename stem '{expected_id}'"
            )

        if not data.get("owner"):
            errors.append(f"claim {path.name}: owner is missing or empty")
        if not data.get("status"):
            errors.append(f"claim {path.name}: status is missing or empty")

        reqs: list[str] = data.get("requirements", [])
        if not reqs:
            errors.append(f"claim {path.name}: requirements list is empty")
        for req_id in reqs:
            if not (_REQ_DIR / f"{req_id}.md").exists():
                errors.append(f"claim {path.name}: requirement '{req_id}' file not found")

        if len(reqs) == 1 and not data.get("atomic_rationale"):
            errors.append(
                f"claim {path.name}: single requirement '{reqs[0]}' but atomic_rationale is missing"
            )

        claims[expected_id] = data
    return claims


def _check_requirements(
    errors: list[str],
    warnings: list[str],
    claims: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load and validate all requirement Markdown files. Returns {req_id: data}."""
    reqs: dict[str, dict[str, Any]] = {}

    for path in sorted(_REQ_DIR.glob("CE-REQ-*.md")):
        expected_id = path.stem
        text = path.read_text(encoding="utf-8")
        table = _parse_md_table(text)

        req_id = table.get("requirement_id", "")
        if not req_id:
            errors.append(f"req {path.name}: requirement_id not found in metadata table")
        elif req_id != expected_id:
            errors.append(
                f"req {path.name}: requirement_id '{req_id}' != filename stem '{expected_id}'"
            )

        obligation_type = table.get("obligation_type", "")
        vstatus = table.get("verification_status", "")

        if vstatus and vstatus not in _VALID_VERIFICATION_STATUS:
            warnings.append(f"req {path.name}: unrecognised verification_status '{vstatus}'")

        # Check claim_refs
        claim_refs = _split_refs(table.get("claim_refs", ""))
        if not claim_refs:
            errors.append(f"req {path.name}: claim_refs is missing or empty")
        for cref in claim_refs:
            if cref and cref not in claims and not (_CLAIMS_DIR / f"{cref}.yaml").exists():
                errors.append(f"req {path.name}: claim_ref '{cref}' file not found")

        # Check TIF refs
        tif_refs = _split_refs(table.get("tif_refs", ""))
        tif_exemption = table.get("tif_exemption", "")

        if obligation_type in _TIF_REQUIRED_TYPES and vstatus == "verified":
            if not tif_refs and not tif_exemption:
                errors.append(
                    f"req {path.name}: obligation_type '{obligation_type}' is verified "
                    f"but has neither tif_refs nor tif_exemption"
                )
            if tif_exemption:
                if obligation_type in _TIF_MANDATORY_NO_EXEMPTION:
                    errors.append(
                        f"req {path.name}: obligation_type '{obligation_type}' is behavioral "
                        f"and must not use tif_exemption — use tif_refs instead"
                    )
                elif tif_exemption not in _VALID_TIF_EXEMPTION_TYPES:
                    errors.append(
                        f"req {path.name}: tif_exemption '{tif_exemption}' is not a recognised type"
                    )
                else:
                    warnings.append(
                        f"req {path.name}: obligation_type '{obligation_type}' uses "
                        f"tif_exemption '{tif_exemption}'; confirm this is a non-runtime check"
                    )
                    if not table.get("tif_exemption_rationale", ""):
                        warnings.append(
                            f"req {path.name}: tif_exemption '{tif_exemption}' is set but "
                            f"tif_exemption_rationale is missing — add this field to document "
                            f"why WrapCalibratedExplainer-based TIF coverage is not appropriate"
                        )

        # Check each TIF spec file exists
        for tif_id in tif_refs:
            spec = _TIF_DIR / f"{tif_id}.md"
            if not spec.exists():
                errors.append(
                    f"req {path.name}: tif_ref '{tif_id}' spec not found at {spec.relative_to(_REPO_ROOT)}"
                )

        # Check verification targets exist on disk
        if vstatus == "verified":
            targets = _parse_verification_targets(text)
            for target in targets:
                full = _REPO_ROOT / target
                if not full.exists():
                    errors.append(f"req {path.name}: verification_target not found: {target}")

        # Documentation-boundary not_implemented must have gap_ref
        if obligation_type == "documentation_boundary" and vstatus == "not_implemented":
            if not table.get("gap_ref"):
                warnings.append(
                    f"req {path.name}: documentation_boundary with not_implemented "
                    f"should have gap_ref and intended closure path"
                )

        reqs[expected_id] = {
            "obligation_type": obligation_type,
            "verification_status": vstatus,
            "tif_refs": tif_refs,
            "tif_exemption": tif_exemption,
            "claim_refs": claim_refs,
        }

    return reqs


def _check_tif_entry_functions(
    errors: list[str],
    active_tif_specs: dict[str, dict[str, Any]],
) -> None:
    """For each active TIF spec, verify declared entry functions exist in the executable."""
    for tif_id, meta in sorted(active_tif_specs.items()):
        exec_str = meta.get("executable", "")
        entry_functions = meta.get("entry_functions", [])
        if not exec_str or not entry_functions:
            continue
        exec_path = _REPO_ROOT / exec_str
        if not exec_path.exists():
            continue  # missing executable already reported elsewhere
        try:
            text = exec_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for fn_name in entry_functions:
            if not re.search(rf"^def {re.escape(fn_name)}\b", text, re.MULTILINE):
                errors.append(
                    f"tif {tif_id}: declared entry function '{fn_name}' "
                    f"not found in executable '{exec_str}'"
                )


def _check_tif_files(
    errors: list[str],
    warnings: list[str],
    reqs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Check TIF spec/executable presence and README inventory.

    Returns {tif_id: spec_metadata} for all active TIF specs discovered from
    CE-TIF-*.md files (status=active).
    """
    # Collect all TIF IDs referenced by requirements and check each has a spec file
    req_tif_ids: set[str] = set()
    for req_data in reqs.values():
        for tif_id in req_data.get("tif_refs", []):
            req_tif_ids.add(tif_id)

    for tif_id in sorted(req_tif_ids):
        spec = _TIF_DIR / f"{tif_id}.md"
        if not spec.exists():
            errors.append(f"tif {tif_id}: spec file {spec.relative_to(_REPO_ROOT)} not found")

    # Check all tif_*.py scenario executables for CE-first compliance.
    # Skip *_helpers.py files — they are shared utilities, not TIF scenario executables.
    for exec_path in sorted(_TIF_DIR.glob("tif_*.py")):
        if exec_path.stem.endswith("_helpers"):
            continue
        _check_tif_executable(errors, warnings, exec_path)

    # Discover active TIF specs and cross-check against README inventory
    active_tif_specs = _check_readme_tif_inventory(errors, warnings)

    # Check that each active spec's declared entry functions exist in the executable
    _check_tif_entry_functions(errors, active_tif_specs)

    return active_tif_specs


def _check_readme_tif_inventory(
    errors: list[str], warnings: list[str]
) -> dict[str, dict[str, Any]]:
    """Discover active TIF specs and bidirectionally cross-check against README table.

    Returns {tif_id: spec_metadata} for all active specs discovered from CE-TIF-*.md.
    """
    readme = _TIF_DIR / "README.md"

    # Discover active specs from CE-TIF-*.md
    active_specs: dict[str, dict[str, Any]] = {}
    for spec_path in sorted(_TIF_DIR.glob("CE-TIF-*.md")):
        meta = _parse_tif_spec_metadata(spec_path)
        tif_id = meta.get("tif_id", "")
        if not tif_id:
            errors.append(f"tif {spec_path.name}: tif_id missing from Identity table")
            tif_id = spec_path.stem
        if tif_id != spec_path.stem:
            errors.append(
                f"tif {spec_path.name}: tif_id '{tif_id}' does not match filename stem '{spec_path.stem}'"
            )
        if meta.get("status", "") != "active":
            continue
        for field in ("executable", "evidence_builder", "evidence_key", "verification_type"):
            if not meta.get(field):
                errors.append(f"tif {spec_path.name}: active spec missing required field '{field}'")
        active_specs[tif_id] = meta

    if not readme.exists():
        warnings.append("tif README.md not found; skipping inventory check")
        return active_specs

    readme_text = readme.read_text(encoding="utf-8")

    # Only skip if the section is entirely absent; an empty-data table still requires cross-check.
    if not _extract_section_text(readme_text, "Current TIF Interfaces"):
        warnings.append(
            "tif README.md: 'Current TIF Interfaces' section not found; skipping inventory check"
        )
        return active_specs

    rows = _parse_tif_readme_table(readme_text)
    readme_tif_ids: set[str] = {row.get("tif_id", "") for row in rows if row.get("tif_id")}

    # active spec missing from README
    for tif_id in sorted(active_specs):
        if tif_id not in readme_tif_ids:
            errors.append(
                f"tif README: active spec '{tif_id}' is not listed in TIF README inventory"
            )

    # README row with no matching active spec (stale)
    for row in rows:
        tif_id = row.get("tif_id", "")
        if not tif_id:
            continue
        if tif_id not in active_specs:
            errors.append(
                f"tif README: row '{tif_id}' has no matching active spec (stale or inactive)"
            )
            continue

        spec_meta = active_specs[tif_id]

        readme_exec = row.get("executable", "")
        spec_exec_name = Path(spec_meta.get("executable", "")).name
        if readme_exec and readme_exec != spec_exec_name:
            errors.append(
                f"tif README: '{tif_id}' executable '{readme_exec}' != spec '{spec_exec_name}'"
            )

        readme_evid_key = row.get("evidence_key", "")
        spec_evid_key = spec_meta.get("evidence_key", "")
        if readme_evid_key and readme_evid_key != spec_evid_key:
            errors.append(
                f"tif README: '{tif_id}' evidence_key '{readme_evid_key}' != spec '{spec_evid_key}'"
            )

        readme_vtype = row.get("verification_type", "")
        spec_vtype = spec_meta.get("verification_type", "")
        if readme_vtype and readme_vtype != spec_vtype:
            errors.append(
                f"tif README: '{tif_id}' verification_type '{readme_vtype}' != spec '{spec_vtype}'"
            )

        readme_status = row.get("status", "")
        spec_status = spec_meta.get("status", "")
        if readme_status and readme_status != spec_status:
            errors.append(
                f"tif README: '{tif_id}' status '{readme_status}' != spec '{spec_status}'"
            )

        readme_reqs_cell = row.get("requirements_served", "")
        if readme_reqs_cell:
            readme_reqs = set(_split_refs(readme_reqs_cell))
            spec_reqs = set(spec_meta.get("requirements_served", []))
            if readme_reqs != spec_reqs:
                errors.append(
                    f"tif README: '{tif_id}' requirements_served {sorted(readme_reqs)} "
                    f"!= spec {sorted(spec_reqs)}"
                )

        readme_claims_cell = row.get("claims_served", "")
        if readme_claims_cell:
            readme_claims = set(_split_refs(readme_claims_cell))
            spec_claims = set(spec_meta.get("claims_served", []))
            if readme_claims != spec_claims:
                errors.append(
                    f"tif README: '{tif_id}' claims_served {sorted(readme_claims)} "
                    f"!= spec {sorted(spec_claims)}"
                )

    return active_specs


def _check_tif_executable(errors: list[str], warnings: list[str], path: Path) -> None:
    """Guard: check that a TIF executable uses WrapCalibratedExplainer and avoids forbidden patterns."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        errors.append(f"tif {path.name}: cannot read file")
        return

    if "WrapCalibratedExplainer" not in text:
        errors.append(
            f"tif {path.name}: WrapCalibratedExplainer not found — "
            f"TIF must enter CE through WrapCalibratedExplainer"
        )

    if not re.search(r"^def build_evidence_payload\b", text, re.MULTILINE):
        errors.append(
            f"tif {path.name}: build_evidence_payload() not defined — "
            f"every TIF executable must expose build_evidence_payload()"
        )

    if any(
        _TIF_FORBIDDEN_IMPORT_RE.search(line)
        for line in text.splitlines()
        if line.lstrip().startswith("from ")
    ):
        errors.append(
            f"tif {path.name}: forbidden import from calibrated_explanations.core.calibrated_explainer — "
            f"use the public entry point only"
        )

    for m in _TIF_DIRECT_CONSTRUCT_RE.finditer(text):
        name = m.group(1)
        # Exclude lines that are imports or type hints
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start : text.find("\n", m.start())].strip()
        if line.startswith(("from ", "import ", "#")):
            continue
        if ":" in line and not line.strip().startswith(name):
            # Likely a type annotation
            continue
        errors.append(
            f"tif {path.name}: direct construction of '{name}(...)' found — "
            f"TIF must not construct explanation objects directly"
        )

    # Check for private member access (best-effort, may have false negatives for strings/comments)
    for m in _TIF_PRIVATE_ACCESS_RE.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start : text.find("\n", m.start())].strip()
        if line.startswith("#"):
            continue
        # Allow docstring lines (heuristic: inside triple-quoted blocks is hard to detect)
        warnings.append(
            f"tif {path.name}: possible private-member access '{m.group()}' at char {m.start()} — "
            f"verify this is not accessing internal CE state"
        )
        break  # one warning per file is enough


def _check_claim_req_reciprocity(
    errors: list[str],
    claims: dict[str, dict[str, Any]],
    reqs: dict[str, dict[str, Any]],
) -> None:
    """Validate that claim↔requirement links are reciprocal.

    - For each requirement R in claim C.requirements: R.claim_refs must include C.
    - For each claim_ref C in requirement R: C.requirements must include R.
    Superseded requirements are excluded from the second direction.
    """
    claim_reqs_map: dict[str, list[str]] = {
        cid: (data.get("requirements", []) if isinstance(data, dict) else [])
        for cid, data in claims.items()
    }

    for claim_id, req_ids in claim_reqs_map.items():
        for req_id in req_ids:
            if req_id not in reqs:
                continue  # missing file already reported by _check_claims
            req_claim_refs = reqs[req_id].get("claim_refs", [])
            if claim_id not in req_claim_refs:
                errors.append(
                    f"claim {claim_id}: requirement '{req_id}' does not link back "
                    f"to this claim in its claim_refs (found: {req_claim_refs})"
                )

    for req_id, req_data in reqs.items():
        vstatus = req_data.get("verification_status", "")
        if vstatus in ("superseded", "deferred"):
            continue
        for claim_id in req_data.get("claim_refs", []):
            if claim_id not in claims:
                continue  # missing file already reported by _check_requirements
            if req_id not in claim_reqs_map.get(claim_id, []):
                errors.append(
                    f"req {req_id}: claim_ref '{claim_id}' does not list this requirement "
                    f"in its requirements (claim requirements: {claim_reqs_map.get(claim_id, [])})"
                )


def _check_tif_req_reciprocity(
    errors: list[str],
    reqs: dict[str, dict[str, Any]],
    active_tif_specs: dict[str, dict[str, Any]],
) -> None:
    """Validate that TIF↔requirement links are reciprocal.

    - For each tif_ref T in requirement R: T.requirements_served must include R.
    - For each served requirement R in TIF T: R.tif_refs must include T.
    """
    for req_id, req_data in reqs.items():
        for tif_id in req_data.get("tif_refs", []):
            if tif_id not in active_tif_specs:
                continue  # missing spec already reported by _check_tif_files
            spec_reqs = active_tif_specs[tif_id].get("requirements_served", [])
            if req_id not in spec_reqs:
                errors.append(
                    f"req {req_id}: tif_ref '{tif_id}' spec does not list this requirement "
                    f"in Requirements served (spec has: {spec_reqs})"
                )

    for tif_id, spec_meta in active_tif_specs.items():
        for req_id in spec_meta.get("requirements_served", []):
            if req_id not in reqs:
                errors.append(
                    f"tif {tif_id}: requirements_served '{req_id}' not found in requirements"
                )
                continue
            req_tif_refs = reqs[req_id].get("tif_refs", [])
            if tif_id not in req_tif_refs:
                errors.append(
                    f"tif {tif_id}: requirements_served '{req_id}' does not reference "
                    f"this TIF in its tif_refs (req tif_refs: {req_tif_refs})"
                )


def _check_tif_claim_reachability(
    errors: list[str],
    claims: dict[str, dict[str, Any]],
    reqs: dict[str, dict[str, Any]],
    active_tif_specs: dict[str, dict[str, Any]],
) -> None:
    """Validate that each TIF's served claims are reachable through its served requirements.

    A served claim C is reachable from TIF T when at least one requirement R in
    T.requirements_served has C in its claim_refs.
    """
    for tif_id, spec_meta in active_tif_specs.items():
        served_reqs = spec_meta.get("requirements_served", [])
        for claim_id in spec_meta.get("claims_served", []):
            if claim_id not in claims:
                errors.append(f"tif {tif_id}: claims_served '{claim_id}' not found in claims")
                continue
            reachable = any(
                claim_id in reqs[r].get("claim_refs", []) for r in served_reqs if r in reqs
            )
            if not reachable:
                errors.append(
                    f"tif {tif_id}: served claim '{claim_id}' is not reachable through "
                    f"its served requirements {sorted(served_reqs)} — no requirement in "
                    f"that set has '{claim_id}' in its claim_refs"
                )


def _check_active_tifs_have_evidence(
    errors: list[str],
    warnings: list[str],
    active_tif_specs: dict[str, dict[str, Any]],
) -> None:
    """Verify that every active TIF spec has at least one committed raw evidence record."""
    # Build set of tif_ids referenced in committed evidence
    evidence_tif_ids: set[str] = set()
    for path in sorted(_RAW_EVID_DIR.glob("CE-EVID-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            evidence_tif_ids.update(data.get("tif_ids", []))
        except (json.JSONDecodeError, OSError):
            pass

    for tif_id in sorted(active_tif_specs):
        if tif_id not in evidence_tif_ids:
            errors.append(
                f"tif {tif_id}: active TIF spec has no committed raw evidence "
                f"(no CE-EVID-*.json references tif_id='{tif_id}')"
            )


def _check_raw_evidence(
    errors: list[str],
    warnings: list[str],
    claims: dict[str, dict[str, Any]],
    reqs: dict[str, dict[str, Any]],
) -> set[str]:
    """Validate committed raw evidence JSON files. Returns set of evidence_ids found."""
    evidence_ids: set[str] = set()
    claim_ids = set(claims.keys())
    req_ids = set(reqs.keys())

    for path in sorted(_RAW_EVID_DIR.glob("CE-EVID-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append(f"evidence {path.name}: JSON parse error")
            continue

        evid_id = data.get("evidence_id", "")
        if evid_id != path.stem:
            errors.append(
                f"evidence {path.name}: evidence_id '{evid_id}' != filename stem '{path.stem}'"
            )
        evidence_ids.add(path.stem)

        # claim_ids
        ev_claims: list[str] = data.get("claim_ids", [])
        if not ev_claims:
            errors.append(f"evidence {path.name}: claim_ids is empty")
        for cid in ev_claims:
            if cid not in claim_ids:
                errors.append(f"evidence {path.name}: claim_id '{cid}' not found in claims")

        # requirement_ids
        ev_reqs: list[str] = data.get("requirement_ids", [])
        if not ev_reqs:
            errors.append(f"evidence {path.name}: requirement_ids is empty")
        for rid in ev_reqs:
            if rid not in req_ids:
                errors.append(
                    f"evidence {path.name}: requirement_id '{rid}' not found in requirements"
                )

        # tif_ids for behavioral evidence
        vtype = data.get("verification_type", "")
        if vtype in _BEHAVIORAL_EVIDENCE_TYPES:
            if not data.get("tif_ids"):
                errors.append(
                    f"evidence {path.name}: behavioral verification_type '{vtype}' but tif_ids is empty"
                )

        # commit_sha format
        sha = data.get("commit_sha", "")
        if sha and sha != "unknown" and not _SHA_RE.fullmatch(sha):
            errors.append(f"evidence {path.name}: commit_sha '{sha}' is not a 40-character hex SHA")

        # result consistency with scenarios
        top_result = data.get("result", "")
        scenarios: list[dict[str, Any]] = data.get("scenarios", [])
        if not scenarios:
            errors.append(f"evidence {path.name}: scenarios list is empty")
        else:
            all_pass = all(s.get("result") == "pass" for s in scenarios)
            expected_top = "pass" if all_pass else "fail"
            if top_result and top_result != expected_top:
                errors.append(
                    f"evidence {path.name}: top-level result '{top_result}' "
                    f"disagrees with scenario results (expected '{expected_top}')"
                )

        # criterion_ref consistency
        ev_req_set = set(ev_reqs)
        for scenario in scenarios:
            for acc in scenario.get("acceptance", []):
                cref = acc.get("criterion_ref", "")
                if cref and cref not in ev_req_set:
                    errors.append(
                        f"evidence {path.name}: scenario '{scenario.get('scenario_id')}': "
                        f"criterion_ref '{cref}' not in requirement_ids"
                    )

    return evidence_ids


def _check_raw_evidence_vs_tif_specs(
    errors: list[str],
    warnings: list[str],
) -> None:
    """Cross-check committed raw evidence against its referenced TIF specs."""
    active_specs: dict[str, dict[str, Any]] = {}
    for spec_path in sorted(_TIF_DIR.glob("CE-TIF-*.md")):
        meta = _parse_tif_spec_metadata(spec_path)
        if meta.get("status") == "active":
            active_specs[meta.get("tif_id", spec_path.stem)] = meta

    for path in sorted(_RAW_EVID_DIR.glob("CE-EVID-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # already reported by _check_raw_evidence

        ev_claims = set(data.get("claim_ids", []))
        ev_reqs = set(data.get("requirement_ids", []))
        ev_adrs = set(data.get("adr_refs", []))

        for tif_id in data.get("tif_ids", []):
            spec = active_specs.get(tif_id)
            if not spec:
                continue  # no active spec for this tif_id; skip (other checks catch this)

            spec_claims = spec.get("claims_served", [])
            missing_claims = [c for c in spec_claims if c not in ev_claims]
            if missing_claims:
                errors.append(
                    f"evidence {path.name}: tif '{tif_id}' spec declares claims "
                    f"{missing_claims} not present in evidence claim_ids"
                )

            spec_reqs = spec.get("requirements_served", [])
            missing_reqs = [r for r in spec_reqs if r not in ev_reqs]
            if missing_reqs:
                errors.append(
                    f"evidence {path.name}: tif '{tif_id}' spec declares requirements "
                    f"{missing_reqs} not present in evidence requirement_ids"
                )

            spec_adrs = spec.get("adr_refs", [])
            if spec_adrs:
                missing_adrs = [a for a in spec_adrs if a not in ev_adrs]
                if missing_adrs:
                    errors.append(
                        f"evidence {path.name}: tif '{tif_id}' spec declares ADR refs "
                        f"{missing_adrs} not present in evidence adr_refs"
                    )


def _check_curated_evidence(
    errors: list[str],
    warnings: list[str],
    reqs: dict[str, dict[str, Any]],
    raw_evidence_ids: set[str],
) -> None:
    """Validate curated evidence Markdown files."""
    req_ids = set(reqs.keys())

    for path in sorted(_EVID_DIR.glob("evidence_*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")

        # Look for requirement_ids in the file (table or inline)
        req_id_matches = re.findall(r"\bCE-REQ-[\w-]+\b", text)
        for rid in set(req_id_matches):
            if rid not in req_ids:
                warnings.append(
                    f"curated_evidence {path.name}: requirement_id '{rid}' not found in current requirements "
                    f"(may be a historical reference or shorthand notation)"
                )

        # Check raw_evidence_ref for behavioral evidence
        raw_ref_match = re.search(r"raw_evidence_ref\s*[:|]\s*([^\n]+)", text, re.IGNORECASE)
        if raw_ref_match:
            raw_ref = raw_ref_match.group(1).strip().rstrip(",")
            is_none_ref = raw_ref.lower().startswith("none")
            if is_none_ref:
                # Valid only when all referenced requirements are TIF-exempt
                table_req_match = re.search(
                    r"\|\s*requirement_ids\s*\|\s*([^|]+)\|", text, re.IGNORECASE
                )
                if table_req_match:
                    curated_req_ids = [
                        r.strip()
                        for r in re.split(r"[,\s]+", table_req_match.group(1))
                        if r.strip().startswith("CE-REQ-")
                    ]
                    non_exempt = [
                        r for r in curated_req_ids if r in reqs and not reqs[r].get("tif_exemption")
                    ]
                    if non_exempt:
                        errors.append(
                            f"curated_evidence {path.name}: raw_evidence_ref is 'none' but "
                            f"requirements {non_exempt} are not TIF-exempt"
                        )
            else:
                # CE-EVID-* IDs must resolve exactly
                ref_ids = re.findall(r"\bCE-EVID-[\w-]+\b", raw_ref)
                for ref_id in ref_ids:
                    if ref_id not in raw_evidence_ids:
                        errors.append(
                            f"curated_evidence {path.name}: raw_evidence_ref '{ref_id}' "
                            f"not found in committed raw evidence files (exact match required)"
                        )


def _check_verified_behavioral_have_raw_evidence(
    errors: list[str],
    warnings: list[str],
    reqs: dict[str, dict[str, Any]],
    raw_evidence_ids: set[str],
) -> None:
    """Verified behavioral requirements should have corresponding raw evidence."""
    # Build a set of requirement_ids that appear in raw evidence
    covered_req_ids: set[str] = set()
    for path in sorted(_RAW_EVID_DIR.glob("CE-EVID-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            covered_req_ids.update(data.get("requirement_ids", []))
        except (json.JSONDecodeError, OSError):
            pass

    for req_id, req_data in reqs.items():
        obl = req_data.get("obligation_type", "")
        vstatus = req_data.get("verification_status", "")
        tif_exemption = req_data.get("tif_exemption", "")
        tif_refs = req_data.get("tif_refs", [])

        if (
            obl in _BEHAVIORAL_EVIDENCE_TYPES
            and vstatus == "verified"
            and tif_refs  # has TIF refs — so behavioral TIF is the verification path
            and req_id not in covered_req_ids
        ):
            warnings.append(
                f"req {req_id}: verified behavioral requirement with tif_refs "
                f"but not found in any raw evidence file"
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_checks() -> tuple[list[str], list[str]]:
    """Run all chain checks. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    claims = _check_claims(errors, warnings)
    reqs = _check_requirements(errors, warnings, claims)
    _check_claim_req_reciprocity(errors, claims, reqs)
    active_tif_specs = _check_tif_files(errors, warnings, reqs)
    _check_tif_req_reciprocity(errors, reqs, active_tif_specs)
    _check_tif_claim_reachability(errors, claims, reqs, active_tif_specs)
    _check_active_tifs_have_evidence(errors, warnings, active_tif_specs)
    raw_evidence_ids = _check_raw_evidence(errors, warnings, claims, reqs)
    _check_raw_evidence_vs_tif_specs(errors, warnings)
    _check_curated_evidence(errors, warnings, reqs, raw_evidence_ids)
    _check_verified_behavioral_have_raw_evidence(errors, warnings, reqs, raw_evidence_ids)

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run all chain checks. Exit 1 if any hard errors are found.",
    )
    args = parser.parse_args()

    if not args.check:
        parser.print_help()
        return 0

    print("CE capability-chain validator")
    print(f"  claims    : {_CLAIMS_DIR.relative_to(_REPO_ROOT)}")
    print(f"  requirements: {_REQ_DIR.relative_to(_REPO_ROOT)}")
    print(f"  TIF       : {_TIF_DIR.relative_to(_REPO_ROOT)}")
    print(f"  raw evidence: {_RAW_EVID_DIR.relative_to(_REPO_ROOT)}")
    print(f"  curated evidence: {_EVID_DIR.relative_to(_REPO_ROOT)}")
    print()

    errors, warnings = run_checks()

    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")

    total = len(errors) + len(warnings)
    if not total:
        print("All checks passed.")
        return 0

    print(f"\n{len(warnings)} warning(s), {len(errors)} error(s).")
    if errors:
        return 1
    print("No hard errors (warnings only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
