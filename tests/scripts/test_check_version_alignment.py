"""Tests for the version-source alignment checker."""

from __future__ import annotations

import scripts.quality.check_version_alignment as check_version_alignment


def test_should_report_no_errors_when_versions_follow_task_36_policy(monkeypatch) -> None:
    """The checker should accept normalized docs/runtime alignment plus base-version metadata."""
    monkeypatch.setattr(check_version_alignment, "_pyproject_version", lambda: "0.11.6-dev")
    monkeypatch.setattr(check_version_alignment, "_runtime_version", lambda: "0.11.6.dev0")
    monkeypatch.setattr(check_version_alignment, "_metadata_version", lambda: "0.11.6.dev0")
    monkeypatch.setattr(check_version_alignment, "_docs_release", lambda: "0.11.6.dev0")
    monkeypatch.setattr(check_version_alignment, "_docs_version", lambda: "0.11")
    monkeypatch.setattr(check_version_alignment, "_citation_version", lambda: "v0.11.6")
    monkeypatch.setattr(check_version_alignment, "_metadata_json_version", lambda: "0.11.6")

    observed, errors = check_version_alignment.evaluate_alignment(allow_normalized=True)

    assert errors == []
    assert observed["pyproject_base_version"] == "0.11.6"


def test_should_report_errors_when_release_metadata_does_not_match_pyproject_base(
    monkeypatch,
) -> None:
    """The checker should fail when CITATION.cff or METADATA.json lag the target release."""
    monkeypatch.setattr(check_version_alignment, "_pyproject_version", lambda: "0.11.6-dev")
    monkeypatch.setattr(check_version_alignment, "_runtime_version", lambda: "0.11.6.dev0")
    monkeypatch.setattr(check_version_alignment, "_metadata_version", lambda: "0.11.6.dev0")
    monkeypatch.setattr(check_version_alignment, "_docs_release", lambda: "0.11.6.dev0")
    monkeypatch.setattr(check_version_alignment, "_docs_version", lambda: "0.11")
    monkeypatch.setattr(check_version_alignment, "_citation_version", lambda: "v0.11.5")
    monkeypatch.setattr(check_version_alignment, "_metadata_json_version", lambda: "0.11.5")

    _, errors = check_version_alignment.evaluate_alignment(allow_normalized=True)

    assert any("CITATION.cff version" in error for error in errors)
    assert any("METADATA.json version" in error for error in errors)


def test_should_require_allow_normalized_for_pep440_equivalent_runtime_versions(monkeypatch) -> None:
    """The checker should only accept equivalent dev spellings when normalization is enabled."""
    monkeypatch.setattr(check_version_alignment, "_pyproject_version", lambda: "0.11.6-dev")
    monkeypatch.setattr(check_version_alignment, "_runtime_version", lambda: "0.11.6-dev")
    monkeypatch.setattr(check_version_alignment, "_metadata_version", lambda: "0.11.6.dev0")
    monkeypatch.setattr(check_version_alignment, "_docs_release", lambda: "0.11.6.dev0")
    monkeypatch.setattr(check_version_alignment, "_docs_version", lambda: "0.11")
    monkeypatch.setattr(check_version_alignment, "_citation_version", lambda: "v0.11.6")
    monkeypatch.setattr(check_version_alignment, "_metadata_json_version", lambda: "0.11.6")

    _, strict_errors = check_version_alignment.evaluate_alignment(allow_normalized=False)
    _, normalized_errors = check_version_alignment.evaluate_alignment(allow_normalized=True)

    assert any("Runtime __version__" in error for error in strict_errors)
    assert normalized_errors == []
