"""Generate raw evidence records by executing active TIF scenarios.

Discovers active TIFs dynamically from CE-TIF-*.md spec files.
No manually maintained runner registry — what to run is driven entirely
by the 'status' and 'executable' fields in each TIF spec.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parents[1]
_TIF_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"
_OUT_DIR = _REPO_ROOT / "reports" / "verification"

if str(_TIF_DIR) not in sys.path:
    sys.path.insert(0, str(_TIF_DIR))

import calibrated_explanations

_BEHAVIORAL_TYPES = {
    "api_contract",
    "behavioral_contract",
    "numerical_behavior",
    "statistical_method_alignment",
    "empirical_smoke",
    "visualization_structure",
}
_REQUIRED_PAYLOAD_FIELDS = (
    "evidence_id",
    "claim_ids",
    "requirement_ids",
    "verification_type",
    "result",
    "timestamp",
    "commit_sha",
    "scenarios",
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_NOW = datetime.now(timezone.utc)


def _extract_section_text(text: str, section_name: str) -> str:
    """Return the body of a `## <section_name>` Markdown section."""
    pattern = rf"(?m)^##\s+{re.escape(section_name)}\s*$\n(.*?)(?=^##\s|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""
_DATE_SUFFIX = _NOW.strftime("%Y%m%d")
_TIMESTAMP = _NOW.isoformat()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


_COMMIT_SHA = _git_sha()
_PACKAGE_VERSION = calibrated_explanations.__version__


def _validate_payload(payload: dict[str, Any], output_stem: str) -> None:
    """Validate a raw evidence payload structure. Raises ValueError on any violation."""
    missing = [f for f in _REQUIRED_PAYLOAD_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"{output_stem}: missing required fields: {', '.join(missing)}")

    if payload["evidence_id"] != output_stem:
        raise ValueError(f"{output_stem}: evidence_id does not match filename stem")
    if not payload["claim_ids"]:
        raise ValueError(f"{output_stem}: claim_ids must be non-empty")
    if not payload["requirement_ids"]:
        raise ValueError(f"{output_stem}: requirement_ids must be non-empty")
    if payload["verification_type"] in _BEHAVIORAL_TYPES and not payload.get("tif_ids"):
        raise ValueError(f"{output_stem}: behavioral evidence must include tif_ids")
    if not payload["scenarios"]:
        raise ValueError(f"{output_stem}: scenarios must be non-empty")
    if payload["result"] not in ("pass", "fail"):
        raise ValueError(f"{output_stem}: result must be 'pass' or 'fail'")
    scenario_results = [s["result"] for s in payload["scenarios"]]
    expected_result = "pass" if all(r == "pass" for r in scenario_results) else "fail"
    if payload["result"] != expected_result:
        raise ValueError(f"{output_stem}: top-level result disagrees with scenarios")
    if not _SHA_RE.fullmatch(payload["commit_sha"]):
        raise ValueError(f"{output_stem}: commit_sha must be a full 40-character git SHA")
    requirement_ids = set(payload["requirement_ids"])
    for scenario in payload["scenarios"]:
        for acceptance in scenario.get("acceptance", []):
            if acceptance["criterion_ref"] not in requirement_ids:
                raise ValueError(
                    f"{output_stem}: criterion_ref {acceptance['criterion_ref']!r} is not listed"
                )


def _parse_tif_spec(spec_path: Path) -> dict[str, Any]:
    """Parse Identity table metadata and served refs from a CE-TIF-*.md spec file."""
    text = spec_path.read_text(encoding="utf-8")

    def table_value(field: str) -> str:
        match = re.search(rf"\|\s*{re.escape(field)}\s*\|\s*([^|]+?)\s*\|", text)
        return match.group(1).strip().strip("`") if match else ""

    reqs_section = _extract_section_text(text, "Requirements served")
    requirements_served: list[str] = re.findall(r"\|\s*(CE-REQ-[\w-]+)\s*\|", reqs_section)

    claims_section = _extract_section_text(text, "Claims served")
    claims_served: list[str] = re.findall(r"^\s*[-*]\s+(CE-CAP-[\w-]+)", claims_section, re.MULTILINE)

    adr_section = _extract_section_text(text, "ADR refs")
    adr_refs: list[str] = re.findall(r"^\s*[-*]\s+(ADR-\d+)", adr_section, re.MULTILINE)

    return {
        "tif_id": table_value("tif_id"),
        "status": table_value("status"),
        "executable": table_value("executable"),
        "evidence_builder": table_value("evidence_builder"),
        "evidence_key": table_value("evidence_key"),
        "verification_type": table_value("verification_type"),
        "requirements_served": requirements_served,
        "claims_served": claims_served,
        "adr_refs": adr_refs,
    }


def _discover_active_tifs() -> list[tuple[str, Any, dict[str, str]]]:
    """Discover active TIF specs and return (tif_id, module, meta) triples.

    Globs CE-TIF-*.md, skips non-active specs, validates all required metadata,
    imports each declared executable, and verifies the module exposes
    build_evidence_payload().

    Raises RuntimeError for any active TIF that cannot be discovered, validated,
    imported, or does not expose the required entry point.
    """
    runners: list[tuple[str, Any, dict[str, str]]] = []
    for spec_path in sorted(_TIF_DIR.glob("CE-TIF-*.md")):
        meta = _parse_tif_spec(spec_path)
        if meta.get("status", "") != "active":
            continue

        tif_id = meta.get("tif_id") or spec_path.stem

        if tif_id != spec_path.stem:
            raise RuntimeError(
                f"{spec_path.name}: tif_id '{tif_id}' does not match filename stem '{spec_path.stem}'"
            )

        for field in ("executable", "evidence_builder", "evidence_key", "verification_type"):
            if not meta.get(field):
                raise RuntimeError(
                    f"{spec_path.name}: active TIF spec missing required field '{field}'"
                )

        executable = meta["executable"]
        exec_path = _REPO_ROOT / executable
        if not exec_path.exists():
            raise RuntimeError(
                f"{spec_path.name}: declared executable '{executable}' does not exist"
            )
        try:
            exec_path.relative_to(_TIF_DIR)
        except ValueError:
            raise RuntimeError(
                f"{spec_path.name}: executable '{executable}' is not under the TIF directory"
            ) from None

        if meta["evidence_builder"] != "build_evidence_payload()":
            raise RuntimeError(
                f"{spec_path.name}: evidence_builder must be 'build_evidence_payload()' "
                f"but got '{meta['evidence_builder']}'"
            )

        module_name = Path(executable).stem
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"{spec_path.name}: cannot import declared executable '{module_name}': {exc}"
            ) from exc

        if not hasattr(module, "build_evidence_payload"):
            raise RuntimeError(
                f"{spec_path.name}: executable '{module_name}' has no build_evidence_payload() "
                "function — every active TIF executable must expose build_evidence_payload()"
            )

        runners.append((tif_id, module, meta))

    if not runners:
        raise RuntimeError(
            f"No active TIF specs found in {_TIF_DIR} — "
            "expected at least one CE-TIF-*.md with status=active"
        )

    return runners


def _validate_payload_against_spec(
    payload: dict[str, Any], tif_id: str, meta: dict[str, Any]
) -> None:
    """Check that the generated payload is consistent with its TIF spec."""
    if tif_id not in payload.get("tif_ids", []):
        raise ValueError(
            f"payload tif_ids {payload.get('tif_ids')} does not include spec tif_id '{tif_id}'"
        )
    spec_vtype = meta["verification_type"]
    if payload.get("verification_type") != spec_vtype:
        raise ValueError(
            f"payload verification_type '{payload.get('verification_type')}' "
            f"does not match spec '{spec_vtype}'"
        )
    evidence_key = meta["evidence_key"]
    expected_prefix = f"CE-EVID-{evidence_key}-"
    if not payload.get("evidence_id", "").startswith(expected_prefix):
        raise ValueError(
            f"payload evidence_id '{payload.get('evidence_id')}' "
            f"does not start with expected prefix '{expected_prefix}'"
        )

    spec_claims = meta.get("claims_served", [])
    if spec_claims:
        payload_claims = set(payload.get("claim_ids", []))
        missing = [c for c in spec_claims if c not in payload_claims]
        if missing:
            raise ValueError(
                f"payload claim_ids missing spec-declared claims: {missing}"
            )

    spec_reqs = meta.get("requirements_served", [])
    if spec_reqs:
        payload_reqs = set(payload.get("requirement_ids", []))
        missing = [r for r in spec_reqs if r not in payload_reqs]
        if missing:
            raise ValueError(
                f"payload requirement_ids missing spec-declared requirements: {missing}"
            )

    spec_adr_refs = meta.get("adr_refs", [])
    if spec_adr_refs:
        payload_adrs = set(payload.get("adr_refs", []))
        missing = [a for a in spec_adr_refs if a not in payload_adrs]
        if missing:
            raise ValueError(
                f"payload adr_refs missing spec-declared ADR refs: {missing}"
            )

    if spec_reqs:
        criteria_refs: set[str] = set()
        for scenario in payload.get("scenarios", []):
            for acc in scenario.get("acceptance", []):
                criteria_refs.add(acc.get("criterion_ref", ""))
        uncovered = [r for r in spec_reqs if r not in criteria_refs]
        if uncovered:
            raise ValueError(
                f"payload scenarios have no acceptance criterion for spec requirements: {uncovered}"
            )


def _validate_existing_evidence() -> int:
    """Validate committed CE-EVID-*.json files without executing any TIF scenarios."""
    print(f"Validating committed evidence files in {_OUT_DIR}")
    paths = sorted(_OUT_DIR.glob("CE-EVID-*.json"))
    if not paths:
        print("  no CE-EVID-*.json files found")
        return 0

    failed: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _validate_payload(payload, path.stem)
            print(f"  OK: {path.name}")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"  FAIL: {path.name}: {exc}")
            failed.append(path.name)

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(paths)} evidence file(s) validated.")
    return 0


def _write(payload: dict[str, Any]) -> None:
    out = _OUT_DIR / f"{payload['evidence_id']}.json"
    _validate_payload(payload, out.stem)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out.name}")


def main(*, check_current: bool = False, validate_existing: bool = False) -> int:
    if validate_existing:
        return _validate_existing_evidence()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating TIF evidence records -> {_OUT_DIR}")
    print(f"  package: {_PACKAGE_VERSION}  commit: {_COMMIT_SHA}")

    try:
        runners = _discover_active_tifs()
    except RuntimeError as exc:
        print(f"DISCOVERY ERROR: {exc}")
        return 1

    print(f"  discovered {len(runners)} active TIF(s)")

    build_kwargs = {
        "commit_sha": _COMMIT_SHA,
        "timestamp": _TIMESTAMP,
        "date_suffix": _DATE_SUFFIX,
        "package_version": _PACKAGE_VERSION,
        "python_version": sys.version.split()[0],
        "platform_str": sys.platform,
    }

    failed: list[str] = []
    written: list[dict[str, Any]] = []
    for tif_id, module, meta in runners:
        print(f"Running {tif_id}...")
        try:
            payload = module.build_evidence_payload(**build_kwargs)
            _validate_payload_against_spec(payload, tif_id, meta)
            _write(payload)
            written.append(payload)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append(tif_id)

    if check_current:
        current_sha = _git_sha()
        stale = [p["evidence_id"] for p in written if p["commit_sha"] != current_sha]
        failing = [p["evidence_id"] for p in written if p["result"] == "fail"]
        if stale:
            print(f"  ERROR: generated evidence is not at current HEAD: {', '.join(stale)}")
            failed.extend(stale)
        if failing:
            print(f"  ERROR: generated evidence has failing result: {', '.join(failing)}")
            failed.extend(failing)

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(runners)} TIF evidence records written.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-current",
        action="store_true",
        help="After generating, assert all evidence matches the current HEAD SHA.",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help=(
            "Validate committed CE-EVID-*.json files without executing TIF scenarios. "
            "Non-mutating: reads only, writes nothing."
        ),
    )
    args = parser.parse_args()
    sys.exit(main(check_current=args.check_current, validate_existing=args.validate_existing))
