"""Validate release artifacts include required packaging metadata and files."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tomllib
import zipfile
from dataclasses import asdict, dataclass
from email import message_from_bytes
from pathlib import Path


PACKAGE_NAME = "calibrated_explanations"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArtifactCheckResult:
    """Structured packaging-artifact inspection result."""

    wheel_path: str
    sdist_path: str
    wheel_license_members: list[str]
    sdist_license_members: list[str]
    wheel_license_metadata: list[str]
    wheel_required_members: list[str]
    sdist_required_members: list[str]
    wheel_requires_python: str | None
    expected_requires_python: str | None
    wheel_top_level_packages: list[str]
    unexpected_top_level_packages: list[str]
    stale_artifacts_removed: list[str]
    status: str
    errors: list[str]


def _required_package_members() -> list[str]:
    package_root = REPO_ROOT / "src" / PACKAGE_NAME
    required_members = [
        f"{PACKAGE_NAME}/py.typed",
        f"{PACKAGE_NAME}/templates/explain_template.yaml",
        f"{PACKAGE_NAME}/utils/configurations/plot_config.ini",
    ]
    schemas_dir = package_root / "schemas"
    required_members.extend(
        sorted(
            f"{PACKAGE_NAME}/{path.relative_to(package_root).as_posix()}"
            for path in schemas_dir.rglob("*.json")
        )
    )
    return sorted(required_members)


REQUIRED_PACKAGE_MEMBERS = _required_package_members()


def _latest_artifact(dist_dir: Path, suffix: str) -> Path:
    matches = sorted(
        dist_dir.glob(f"{PACKAGE_NAME}-*{suffix}"),
        key=lambda candidate: (candidate.stat().st_mtime, candidate.name),
    )
    if not matches:
        raise FileNotFoundError(f"No {suffix} artifact found under {dist_dir}")
    return matches[-1]


def _all_artifacts(dist_dir: Path) -> list[Path]:
    artifacts = list(dist_dir.glob(f"{PACKAGE_NAME}-*.whl"))
    artifacts.extend(dist_dir.glob(f"{PACKAGE_NAME}-*.tar.gz"))
    return sorted(artifacts, key=lambda candidate: candidate.name)


def _is_license_name(member_name: str) -> bool:
    return Path(member_name).name.upper() == "LICENSE"


def _top_level_packages(members: list[str]) -> tuple[list[str], list[str]]:
    wheel_packages: set[str] = set()
    unexpected_packages: set[str] = set()
    for member in members:
        root = member.split("/", 1)[0]
        if root == PACKAGE_NAME:
            wheel_packages.add(root)
            continue
        if root.endswith(".dist-info") or root.endswith(".data"):
            continue
        if "/" in member:
            unexpected_packages.add(root)
    return sorted(wheel_packages), sorted(unexpected_packages)


def _inspect_wheel(
    wheel_path: Path,
) -> tuple[list[str], list[str], list[str], str | None, list[str], list[str]]:
    with zipfile.ZipFile(wheel_path) as archive:
        members = archive.namelist()
        license_members = [name for name in members if _is_license_name(name)]
        metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
        metadata = message_from_bytes(archive.read(metadata_name))
    wheel_packages, unexpected_packages = _top_level_packages(members)
    required_members = [name for name in REQUIRED_PACKAGE_MEMBERS if name in members]
    return (
        license_members,
        metadata.get_all("License-File") or [],
        required_members,
        metadata.get("Requires-Python"),
        wheel_packages,
        unexpected_packages,
    )


def _inspect_sdist(sdist_path: Path) -> tuple[list[str], list[str]]:
    with tarfile.open(sdist_path, "r:gz") as archive:
        members = archive.getnames()
    root_prefix = f"{PACKAGE_NAME}-"
    normalized_members = [
        member.split("/", 1)[1]
        for member in members
        if member.startswith(root_prefix) and "/" in member
    ]
    required_members = [
        member for member in normalized_members if member.startswith(f"src/{PACKAGE_NAME}/")
    ]
    return [name for name in members if _is_license_name(name)], required_members


def _expected_requires_python(pyproject_path: Path) -> str | None:
    if not pyproject_path.is_file():
        return None
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = payload.get("project", {})
    if not isinstance(project, dict):
        return None
    value = project.get("requires-python")
    return value if isinstance(value, str) else None


def _remove_stale_artifacts(dist_dir: Path, keep: set[Path]) -> list[str]:
    removed: list[str] = []
    for artifact in _all_artifacts(dist_dir):
        if artifact in keep:
            continue
        artifact.unlink()
        removed.append(artifact.name)
    return removed


def inspect_packaging_artifacts(
    dist_dir: Path,
    *,
    expected_requires_python: str | None = None,
    clean_dist: bool = False,
) -> ArtifactCheckResult:
    """Inspect the newest wheel and sdist artifacts in ``dist_dir``."""
    wheel_path = _latest_artifact(dist_dir, ".whl")
    sdist_path = _latest_artifact(dist_dir, ".tar.gz")
    keep = {wheel_path, sdist_path}
    stale_artifacts_removed = _remove_stale_artifacts(dist_dir, keep) if clean_dist else []
    (
        wheel_license_members,
        wheel_license_metadata,
        wheel_required_members,
        wheel_requires_python,
        wheel_packages,
        unexpected_top_level_packages,
    ) = _inspect_wheel(wheel_path)
    sdist_license_members, sdist_required_members = _inspect_sdist(sdist_path)
    errors: list[str] = []
    if not clean_dist:
        stale_artifacts = [
            artifact.name for artifact in _all_artifacts(dist_dir) if artifact not in keep
        ]
        if stale_artifacts:
            errors.append(
                "dist contains stale artifacts; rerun with --clean-dist or remove them before upload: "
                + ", ".join(stale_artifacts)
            )

    if not wheel_license_members:
        errors.append("Wheel artifact does not contain a LICENSE file.")
    if not sdist_license_members:
        errors.append("Sdist artifact does not contain a LICENSE file.")
    if not any(_is_license_name(value) for value in wheel_license_metadata):
        errors.append("Wheel METADATA does not expose License-File: LICENSE.")
    if expected_requires_python is not None and wheel_requires_python != expected_requires_python:
        errors.append(
            "Wheel METADATA Requires-Python does not match pyproject.toml: "
            f"expected {expected_requires_python!r}, got {wheel_requires_python!r}."
        )
    if not wheel_packages:
        errors.append(
            f"Wheel artifact does not contain the expected top-level package: {PACKAGE_NAME}."
        )
    if unexpected_top_level_packages:
        errors.append(
            "Wheel artifact contains unexpected top-level packages: "
            + ", ".join(unexpected_top_level_packages)
        )
    missing_wheel_members = [
        name for name in REQUIRED_PACKAGE_MEMBERS if name not in wheel_required_members
    ]
    if missing_wheel_members:
        errors.append(
            "Wheel artifact is missing required package data: "
            + ", ".join(sorted(missing_wheel_members))
        )
    missing_sdist_members = [
        name for name in REQUIRED_PACKAGE_MEMBERS if f"src/{name}" not in sdist_required_members
    ]
    if missing_sdist_members:
        errors.append(
            "Sdist artifact is missing required package data: "
            + ", ".join(sorted(missing_sdist_members))
        )

    return ArtifactCheckResult(
        wheel_path=str(wheel_path),
        sdist_path=str(sdist_path),
        wheel_license_members=wheel_license_members,
        sdist_license_members=sdist_license_members,
        wheel_license_metadata=wheel_license_metadata,
        wheel_required_members=wheel_required_members,
        sdist_required_members=sdist_required_members,
        wheel_requires_python=wheel_requires_python,
        expected_requires_python=expected_requires_python,
        wheel_top_level_packages=wheel_packages,
        unexpected_top_level_packages=unexpected_top_level_packages,
        stale_artifacts_removed=stale_artifacts_removed,
        status="pass" if not errors else "fail",
        errors=errors,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check built wheel/sdist artifacts for packaging metadata, package data, and stale dist hazards."
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
    parser.add_argument(
        "--clean-dist",
        action="store_true",
        help="Remove stale calibrated_explanations artifacts from dist/ before reporting.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = inspect_packaging_artifacts(
            Path(args.dist_dir),
            expected_requires_python=_expected_requires_python(REPO_ROOT / "pyproject.toml"),
            clean_dist=args.clean_dist,
        )
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
    print(f"Wheel Requires-Python: {result.wheel_requires_python}")
    print(f"Wheel required package data: {', '.join(result.wheel_required_members)}")
    print(f"Sdist required package data: {', '.join(result.sdist_required_members)}")
    print(f"Wheel top-level packages: {', '.join(result.wheel_top_level_packages)}")
    if result.stale_artifacts_removed:
        print(f"Removed stale dist artifacts: {', '.join(result.stale_artifacts_removed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
