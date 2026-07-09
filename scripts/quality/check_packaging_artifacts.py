"""Validate release artifacts include required packaging metadata and files."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from email import message_from_bytes
from pathlib import Path


PACKAGE_NAME = "calibrated_explanations"


@dataclass(frozen=True)
class ArtifactCheckResult:
    """Structured packaging-artifact inspection result."""

    wheel_path: str
    sdist_path: str
    wheel_license_members: list[str]
    sdist_license_members: list[str]
    wheel_license_metadata: list[str]
    status: str
    errors: list[str]


def _latest_artifact(dist_dir: Path, suffix: str) -> Path:
    matches = sorted(
        dist_dir.glob(f"{PACKAGE_NAME}-*{suffix}"),
        key=lambda candidate: (candidate.stat().st_mtime, candidate.name),
    )
    if not matches:
        raise FileNotFoundError(f"No {suffix} artifact found under {dist_dir}")
    return matches[-1]


def _is_license_name(member_name: str) -> bool:
    return Path(member_name).name.upper() == "LICENSE"


def _inspect_wheel(wheel_path: Path) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(wheel_path) as archive:
        members = archive.namelist()
        license_members = [name for name in members if _is_license_name(name)]
        metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
        metadata = message_from_bytes(archive.read(metadata_name))
    return license_members, metadata.get_all("License-File") or []


def _inspect_sdist(sdist_path: Path) -> list[str]:
    with tarfile.open(sdist_path, "r:gz") as archive:
        return [name for name in archive.getnames() if _is_license_name(name)]


def inspect_packaging_artifacts(dist_dir: Path) -> ArtifactCheckResult:
    """Inspect the newest wheel and sdist artifacts in ``dist_dir``."""
    wheel_path = _latest_artifact(dist_dir, ".whl")
    sdist_path = _latest_artifact(dist_dir, ".tar.gz")
    wheel_license_members, wheel_license_metadata = _inspect_wheel(wheel_path)
    sdist_license_members = _inspect_sdist(sdist_path)
    errors: list[str] = []

    if not wheel_license_members:
        errors.append("Wheel artifact does not contain a LICENSE file.")
    if not sdist_license_members:
        errors.append("Sdist artifact does not contain a LICENSE file.")
    if not any(_is_license_name(value) for value in wheel_license_metadata):
        errors.append("Wheel METADATA does not expose License-File: LICENSE.")

    return ArtifactCheckResult(
        wheel_path=str(wheel_path),
        sdist_path=str(sdist_path),
        wheel_license_members=wheel_license_members,
        sdist_license_members=sdist_license_members,
        wheel_license_metadata=wheel_license_metadata,
        status="pass" if not errors else "fail",
        errors=errors,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that built wheel/sdist artifacts include the LICENSE text."
    )
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Directory containing built wheel and sdist artifacts.",
    )
    parser.add_argument(
        "--report",
        default="reports/packaging/license_artifact_check.json",
        help="JSON output path for the inspection result.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = inspect_packaging_artifacts(Path(args.dist_dir))
    except (FileNotFoundError, StopIteration, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}")
        return 1

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")

    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Wheel artifact: {result.wheel_path}")
    print(f"Sdist artifact: {result.sdist_path}")
    print(f"Wheel LICENSE entries: {', '.join(result.wheel_license_members)}")
    print(f"Sdist LICENSE entries: {', '.join(result.sdist_license_members)}")
    print(f"Wheel License-File metadata: {', '.join(result.wheel_license_metadata)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
