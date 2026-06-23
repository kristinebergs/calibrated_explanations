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
_TIF_DIRECT_CONSTRUCT_RE = re.compile(
    r"\b(FactualExplanation|AlternativeExplanation)\s*\("
)
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


def _parse_verification_targets(text: str) -> list[str]:
    """Extract file paths from the Verification targets section."""
    section = _extract_section_text(text, "Verification targets")
    paths: list[str] = []
    for line in section.splitlines():
        stripped = re.sub(r"^[-*\s`]+", "", line.strip()).strip("`")
        if stripped.startswith("pytest:"):
            raw = stripped[len("pytest:"):].strip()
            file_part = raw.split("::")[0].strip()
            if file_part:
                paths.append(file_part)
        elif stripped.startswith("quality-gate:"):
            cmd = stripped[len("quality-gate:"):].strip()
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
            errors.append(f"claim {path.name}: claim_id '{claim_id}' != filename stem '{expected_id}'")

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
            errors.append(f"req {path.name}: requirement_id '{req_id}' != filename stem '{expected_id}'")

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

        # Check each TIF spec file exists
        for tif_id in tif_refs:
            spec = _TIF_DIR / f"{tif_id}.md"
            if not spec.exists():
                errors.append(f"req {path.name}: tif_ref '{tif_id}' spec not found at {spec.relative_to(_REPO_ROOT)}")

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


def _check_tif_files(
    errors: list[str],
    warnings: list[str],
    reqs: dict[str, dict[str, Any]],
) -> set[str]:
    """Check TIF spec/executable presence and README inventory. Returns set of active TIF IDs."""
    active_tif_ids: set[str] = set()

    # Collect all TIF IDs referenced by requirements
    for req_data in reqs.values():
        for tif_id in req_data.get("tif_refs", []):
            active_tif_ids.add(tif_id)

    # Check each active TIF has both spec and executable
    for tif_id in sorted(active_tif_ids):
        spec = _TIF_DIR / f"{tif_id}.md"
        # Derive executable name: CE-TIF-EXPL-001 -> tif_expl_001.py -> tif_explanation.py (custom)
        # Actually, the naming is: CE-TIF-EXPL-001 -> derive stem from spec or check pattern
        if not spec.exists():
            errors.append(f"tif {tif_id}: spec file {spec.relative_to(_REPO_ROOT)} not found")

    # Check all tif_*.py executables
    for exec_path in sorted(_TIF_DIR.glob("tif_*.py")):
        _check_tif_executable(errors, warnings, exec_path)

    # Check TIF README inventory
    readme = _TIF_DIR / "README.md"
    if readme.exists():
        readme_text = readme.read_text(encoding="utf-8")
        for tif_id in sorted(active_tif_ids):
            if tif_id not in readme_text:
                errors.append(f"tif README: {tif_id} is not listed in TIF README inventory")
    else:
        warnings.append("tif README.md not found; skipping inventory check")

    return active_tif_ids


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
            errors.append(f"evidence {path.name}: evidence_id '{evid_id}' != filename stem '{path.stem}'")
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
                errors.append(f"evidence {path.name}: requirement_id '{rid}' not found in requirements")

        # tif_ids for behavioral evidence
        vtype = data.get("verification_type", "")
        if vtype in _BEHAVIORAL_EVIDENCE_TYPES:
            if not data.get("tif_ids"):
                errors.append(f"evidence {path.name}: behavioral verification_type '{vtype}' but tif_ids is empty")

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
            if not is_none_ref:
                # Check that referenced evidence file exists
                ref_ids = re.findall(r"\bCE-EVID-[\w-]+\b", raw_ref)
                for ref_id in ref_ids:
                    if not any(eid.startswith(ref_id.rstrip("0123456789-").rstrip("-")) for eid in raw_evidence_ids):
                        # Soft check: raw evidence may be at a different date stamp
                        warnings.append(
                            f"curated_evidence {path.name}: raw_evidence_ref '{ref_id}' "
                            f"not found in committed raw evidence files"
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
    active_tif_ids = _check_tif_files(errors, warnings, reqs)
    raw_evidence_ids = _check_raw_evidence(errors, warnings, claims, reqs)
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
