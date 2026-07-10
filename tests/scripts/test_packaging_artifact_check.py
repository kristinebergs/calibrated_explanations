"""Tests for the packaging artifact license smoke checker."""

from __future__ import annotations

import json
import tarfile
import zipfile
from email.message import Message
from pathlib import Path

from scripts.quality.check_packaging_artifacts import (
    REQUIRED_PACKAGE_MEMBERS,
    inspect_packaging_artifacts,
    main,
)


def _write_wheel(
    dist_dir: Path,
    *,
    include_license_member: bool,
    license_metadata: list[str] | None,
    requires_python: str | None = ">=3.10",
    package_members: list[str] | None = None,
) -> Path:
    wheel_path = dist_dir / "calibrated_explanations-0.11.6.dev0-py3-none-any.whl"
    metadata = Message()
    metadata["Metadata-Version"] = "2.4"
    metadata["Name"] = "calibrated_explanations"
    metadata["Version"] = "0.11.6.dev0"
    metadata["License-Expression"] = "BSD-3-Clause"
    if requires_python is not None:
        metadata["Requires-Python"] = requires_python
    for value in license_metadata or []:
        metadata["License-File"] = value

    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr(
            "calibrated_explanations-0.11.6.dev0.dist-info/METADATA",
            metadata.as_string(),
        )
        archive.writestr(
            "calibrated_explanations-0.11.6.dev0.dist-info/WHEEL", "Wheel-Version: 1.0\n"
        )
        if include_license_member:
            archive.writestr(
                "calibrated_explanations-0.11.6.dev0.dist-info/licenses/LICENSE",
                "BSD 3-Clause License\n",
            )
        for member in package_members or []:
            archive.writestr(member, "placeholder\n")
    return wheel_path


def _write_sdist(
    dist_dir: Path,
    *,
    include_license_member: bool,
    package_members: list[str] | None = None,
) -> Path:
    sdist_path = dist_dir / "calibrated_explanations-0.11.6.dev0.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as archive:
        package_dir = dist_dir / "package"
        package_dir.mkdir()
        readme_path = package_dir / "README.md"
        readme_path.write_text("placeholder\n", encoding="utf-8")
        archive.add(readme_path, arcname="calibrated_explanations-0.11.6.dev0/README.md")
        if include_license_member:
            license_path = package_dir / "LICENSE"
            license_path.write_text("BSD 3-Clause License\n", encoding="utf-8")
            archive.add(license_path, arcname="calibrated_explanations-0.11.6.dev0/LICENSE")
        for member in package_members or []:
            member_path = package_dir / member
            member_path.parent.mkdir(parents=True, exist_ok=True)
            member_path.write_text("placeholder\n", encoding="utf-8")
            archive.add(
                member_path,
                arcname=f"calibrated_explanations-0.11.6.dev0/src/{member}",
            )
    return sdist_path


def test_should_report_pass_when_wheel_and_sdist_include_license(tmp_path: Path) -> None:
    """The checker should pass when both artifacts carry the license text and metadata."""
    required_members = REQUIRED_PACKAGE_MEMBERS
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir,
        include_license_member=True,
        license_metadata=["LICENSE"],
        package_members=required_members,
    )
    _write_sdist(
        dist_dir,
        include_license_member=True,
        package_members=required_members,
    )

    result = inspect_packaging_artifacts(dist_dir, expected_requires_python=">=3.10")

    assert result.status == "pass"
    assert result.errors == []
    assert result.wheel_license_members
    assert result.sdist_license_members
    assert result.wheel_license_metadata == ["LICENSE"]
    assert sorted(result.wheel_required_members) == sorted(required_members)
    assert len(result.sdist_required_members) == len(required_members)
    assert result.wheel_requires_python == ">=3.10"
    assert result.unexpected_top_level_packages == []
    assert result.stale_artifacts_removed == []


def test_should_require_preprocessing_package_marker(tmp_path: Path) -> None:
    """The checker should fail when the preprocessing package lacks ``__init__.py``."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    missing_marker_members = [
        member
        for member in REQUIRED_PACKAGE_MEMBERS
        if member != "calibrated_explanations/preprocessing/__init__.py"
    ]
    _write_wheel(
        dist_dir,
        include_license_member=True,
        license_metadata=["LICENSE"],
        package_members=missing_marker_members,
    )
    _write_sdist(
        dist_dir,
        include_license_member=True,
        package_members=missing_marker_members,
    )

    result = inspect_packaging_artifacts(dist_dir, expected_requires_python=">=3.10")

    assert result.status == "fail"
    assert any(
        "calibrated_explanations/preprocessing/__init__.py" in error
        for error in result.errors
    )


def test_should_report_fail_when_artifacts_omit_license(tmp_path: Path) -> None:
    """The checker should fail when either artifact or metadata omits the license."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(dist_dir, include_license_member=False, license_metadata=[])
    _write_sdist(dist_dir, include_license_member=False)

    result = inspect_packaging_artifacts(dist_dir, expected_requires_python=">=3.10")

    assert result.status == "fail"
    assert any("Wheel artifact" in error for error in result.errors)
    assert any("Sdist artifact" in error for error in result.errors)
    assert any("Wheel METADATA" in error for error in result.errors)
    assert any("required package data" in error for error in result.errors)


def test_should_fail_when_wheel_metadata_or_top_level_packages_are_wrong(tmp_path: Path) -> None:
    """The checker should reject mismatched Requires-Python metadata and stray packages."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    package_members = [*REQUIRED_PACKAGE_MEMBERS, "tests/__init__.py"]
    _write_wheel(
        dist_dir,
        include_license_member=True,
        license_metadata=["LICENSE"],
        requires_python=">=3.9",
        package_members=package_members,
    )
    _write_sdist(dist_dir, include_license_member=True, package_members=REQUIRED_PACKAGE_MEMBERS)

    result = inspect_packaging_artifacts(dist_dir, expected_requires_python=">=3.10")

    assert result.status == "fail"
    assert any("Requires-Python" in error for error in result.errors)
    assert any("unexpected top-level packages" in error for error in result.errors)
    assert result.unexpected_top_level_packages == ["tests"]


def test_should_remove_stale_artifacts_when_clean_dist_is_enabled(tmp_path: Path) -> None:
    """The checker should remove older matching artifacts when asked to clean dist."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "calibrated_explanations-0.11.2.dev0-py3-none-any.whl").write_text(
        "stale\n", encoding="utf-8"
    )
    (dist_dir / "calibrated_explanations-0.11.2.dev0.tar.gz").write_text(
        "stale\n", encoding="utf-8"
    )
    _write_wheel(
        dist_dir,
        include_license_member=True,
        license_metadata=["LICENSE"],
        package_members=REQUIRED_PACKAGE_MEMBERS,
    )
    _write_sdist(dist_dir, include_license_member=True, package_members=REQUIRED_PACKAGE_MEMBERS)

    result = inspect_packaging_artifacts(
        dist_dir,
        expected_requires_python=">=3.10",
        clean_dist=True,
    )

    assert result.status == "pass"
    assert sorted(result.stale_artifacts_removed) == [
        "calibrated_explanations-0.11.2.dev0-py3-none-any.whl",
        "calibrated_explanations-0.11.2.dev0.tar.gz",
    ]
    assert not (dist_dir / "calibrated_explanations-0.11.2.dev0-py3-none-any.whl").exists()
    assert not (dist_dir / "calibrated_explanations-0.11.2.dev0.tar.gz").exists()


def test_should_write_report_and_exit_zero_when_check_passes(tmp_path: Path, monkeypatch) -> None:
    """CLI mode should persist the inspection result report."""
    dist_dir = tmp_path / "dist"
    report_path = tmp_path / "reports" / "packaging.json"
    dist_dir.mkdir()
    required_members = REQUIRED_PACKAGE_MEMBERS
    _write_wheel(
        dist_dir,
        include_license_member=True,
        license_metadata=["LICENSE"],
        package_members=required_members,
    )
    _write_sdist(
        dist_dir,
        include_license_member=True,
        package_members=required_members,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_packaging_artifacts.py",
            "--dist-dir",
            str(dist_dir),
            "--report",
            str(report_path),
        ],
    )

    rc = main()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == "pass"
    assert payload["wheel_license_metadata"] == ["LICENSE"]
    assert payload["wheel_requires_python"] == ">=3.10"
    assert sorted(payload["wheel_required_members"]) == sorted(required_members)
