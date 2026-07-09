"""Tests for the packaging artifact license smoke checker."""

from __future__ import annotations

import json
import tarfile
import zipfile
from email.message import Message
from pathlib import Path

from scripts.quality.check_packaging_artifacts import inspect_packaging_artifacts, main


def _write_wheel(
    dist_dir: Path,
    *,
    include_license_member: bool,
    license_metadata: list[str] | None,
) -> Path:
    wheel_path = dist_dir / "calibrated_explanations-0.11.6.dev0-py3-none-any.whl"
    metadata = Message()
    metadata["Metadata-Version"] = "2.4"
    metadata["Name"] = "calibrated_explanations"
    metadata["Version"] = "0.11.6.dev0"
    metadata["License-Expression"] = "BSD-3-Clause"
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
    return wheel_path


def _write_sdist(dist_dir: Path, *, include_license_member: bool) -> Path:
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
    return sdist_path


def test_should_report_pass_when_wheel_and_sdist_include_license(tmp_path: Path) -> None:
    """The checker should pass when both artifacts carry the license text and metadata."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(dist_dir, include_license_member=True, license_metadata=["LICENSE"])
    _write_sdist(dist_dir, include_license_member=True)

    result = inspect_packaging_artifacts(dist_dir)

    assert result.status == "pass"
    assert result.errors == []
    assert result.wheel_license_members
    assert result.sdist_license_members
    assert result.wheel_license_metadata == ["LICENSE"]


def test_should_report_fail_when_artifacts_omit_license(tmp_path: Path) -> None:
    """The checker should fail when either artifact or metadata omits the license."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(dist_dir, include_license_member=False, license_metadata=[])
    _write_sdist(dist_dir, include_license_member=False)

    result = inspect_packaging_artifacts(dist_dir)

    assert result.status == "fail"
    assert any("Wheel artifact" in error for error in result.errors)
    assert any("Sdist artifact" in error for error in result.errors)
    assert any("Wheel METADATA" in error for error in result.errors)


def test_should_write_report_and_exit_zero_when_check_passes(tmp_path: Path, monkeypatch) -> None:
    """CLI mode should persist the inspection result report."""
    dist_dir = tmp_path / "dist"
    report_path = tmp_path / "reports" / "packaging.json"
    dist_dir.mkdir()
    _write_wheel(dist_dir, include_license_member=True, license_metadata=["LICENSE"])
    _write_sdist(dist_dir, include_license_member=True)
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
