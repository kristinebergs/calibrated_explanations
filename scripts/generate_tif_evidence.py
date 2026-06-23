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
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_NOW = datetime.now(timezone.utc)
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
    if payload["evidence_id"] != output_stem:
        raise ValueError(f"{output_stem}: evidence_id does not match filename stem")
    if not payload["claim_ids"]:
        raise ValueError(f"{output_stem}: claim_ids must be non-empty")
    if not payload["requirement_ids"]:
        raise ValueError(f"{output_stem}: requirement_ids must be non-empty")
    if payload["verification_type"] in _BEHAVIORAL_TYPES and not payload["tif_ids"]:
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


def _parse_tif_spec(spec_path: Path) -> dict[str, str]:
    """Parse Identity table metadata from a CE-TIF-*.md spec file."""
    text = spec_path.read_text(encoding="utf-8")

    def table_value(field: str) -> str:
        match = re.search(rf"\|\s*{re.escape(field)}\s*\|\s*([^|]+?)\s*\|", text)
        return match.group(1).strip() if match else ""

    return {
        "tif_id": table_value("tif_id"),
        "status": table_value("status"),
        "executable": table_value("executable").strip("`"),
    }


def _discover_active_tifs() -> list[tuple[str, Any]]:
    """Discover active TIF specs and return (tif_id, module) pairs.

    Globs CE-TIF-*.md, skips non-active specs, imports each declared executable,
    and verifies the module exposes build_evidence_payload().

    Raises RuntimeError for any active TIF that cannot be discovered, imported,
    or validated.
    """
    runners: list[tuple[str, Any]] = []
    for spec_path in sorted(_TIF_DIR.glob("CE-TIF-*.md")):
        meta = _parse_tif_spec(spec_path)
        status = meta.get("status", "")
        if status != "active":
            continue

        tif_id = meta.get("tif_id") or spec_path.stem
        executable = meta.get("executable", "")
        if not executable:
            raise RuntimeError(
                f"{spec_path.name}: active TIF spec has no 'executable' declared in Identity table"
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

        runners.append((tif_id, module))

    if not runners:
        raise RuntimeError(
            f"No active TIF specs found in {_TIF_DIR} — "
            "expected at least one CE-TIF-*.md with status=active"
        )

    return runners


def _write(payload: dict[str, Any]) -> None:
    out = _OUT_DIR / f"{payload['evidence_id']}.json"
    _validate_payload(payload, out.stem)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out.name}")


def main(*, check_current: bool = False) -> int:
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
    for tif_id, module in runners:
        print(f"Running {tif_id}...")
        try:
            payload = module.build_evidence_payload(**build_kwargs)
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
    parser.add_argument("--check-current", action="store_true")
    args = parser.parse_args()
    sys.exit(main(check_current=args.check_current))
