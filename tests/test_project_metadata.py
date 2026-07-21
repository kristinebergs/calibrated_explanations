"""Project metadata governance tests."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def test_pyproject_development_status_classifier_matches_release_phase() -> None:
    """Ensure release-governed maturity metadata matches package release phase.

    Per the packaging metadata maturity control policy, the classifier moves
    from Beta to Production/Stable exactly once, at the v1.0.0 GA transition,
    and does not revert to Beta for later development/patch builds of the
    stable v1.x line (e.g. ``1.0.1-dev``). Only versions before v1.0.0 (the
    ``0.x`` line) or v1.0.0 prereleases/dev-builds themselves (e.g.
    ``1.0.0rc1``, ``1.0.0-rc-dev``) are pre-GA and expect Beta.
    """
    from packaging.version import Version  # noqa: PLC0415

    project_root = Path(__file__).resolve().parents[1]
    pyproject_path = project_root / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    classifiers = project.get("classifiers", [])

    version = project.get("version", "")
    development_status_classifiers = [
        classifier for classifier in classifiers if classifier.startswith("Development Status ::")
    ]

    assert "Development Status :: 3 - Alpha" not in development_status_classifiers

    parsed_version = Version(version)
    ga_release = (1, 0, 0)
    is_pre_ga = parsed_version.release < ga_release or (
        parsed_version.release == ga_release and parsed_version.is_prerelease
    )
    if is_pre_ga:
        expected = "Development Status :: 4 - Beta"
    else:
        expected = "Development Status :: 5 - Production/Stable"

    assert development_status_classifiers == [expected]


def test_runtime_version_matches_installed_package_metadata() -> None:
    """Task 13: runtime ``__version__`` should match installed package metadata."""
    import calibrated_explanations as ce

    try:
        metadata_version = importlib_metadata.version("calibrated_explanations")
    except importlib_metadata.PackageNotFoundError:
        metadata_version = importlib_metadata.version("calibrated-explanations")

    assert ce.__version__ == metadata_version
