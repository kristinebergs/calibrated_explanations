"""Behavior tests for the release-profile meta-test guard.

These tests verify the durable detection behavior of
``scripts/quality/check_no_release_profile_meta_tests.py`` against synthetic
fixtures. The offending tokens below live inside tmp-dir fixture content; the
guard allowlists this file for exactly that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.helpers.capability_utils import write_text_fixture

import scripts.quality.check_no_release_profile_meta_tests as guard

OFFENDING_TEST = """\
import scripts.local_checks as local_checks


def test_task_profile_wiring() -> None:
    plan = local_checks.build_profile_plan("task", task=9)
    assert "Task 9 local-check profile tests" in [s.name for s in plan.steps]
    assert any(g["gate"] == "coverage" for g in plan.skipped_heavy_gates)
"""

DURABLE_TEST = """\
def test_should_fail_when_fixture_wheel_omits_license(tmp_path) -> None:
    result = run_checker(tmp_path / "broken.whl")
    assert result.exit_code == 1
"""




def test_should_flag_profile_meta_test_patterns_when_present_in_fixture(
    tmp_path: Path,
) -> None:
    """Each prohibited marker in a test file should produce a finding."""
    # Arrange
    write_text_fixture(tmp_path / "tests" / "test_offender.py", OFFENDING_TEST)

    # Act
    findings = guard.scan_tests(tmp_path / "tests", repo_root=tmp_path)

    # Assert
    patterns = {finding["pattern"] for finding in findings}
    assert any("build_profile_plan" in pattern for pattern in patterns)
    assert any("step-name assertion" in pattern for pattern in patterns)
    assert any("skipped-heavy-gate" in pattern for pattern in patterns)
    assert all(finding["path"] == "tests/test_offender.py" for finding in findings)


def test_should_pass_when_tests_contain_only_durable_checker_assertions(
    tmp_path: Path,
) -> None:
    """Behavior-level checker tests must not trip the guard."""
    # Arrange
    write_text_fixture(tmp_path / "tests" / "test_durable.py", DURABLE_TEST)

    # Act
    findings = guard.scan_tests(tmp_path / "tests", repo_root=tmp_path)

    # Assert
    assert findings == []


def test_should_skip_allowlisted_paths_when_scanning(tmp_path: Path) -> None:
    """Files on the documented allowlist are exempt from scanning."""
    # Arrange
    allowlisted = next(iter(guard.ALLOWLIST))
    write_text_fixture(tmp_path / allowlisted, OFFENDING_TEST)

    # Act
    findings = guard.scan_tests(tmp_path / "tests", repo_root=tmp_path)

    # Assert
    assert findings == []


def test_should_return_failure_exit_code_and_report_when_offender_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI should exit 1 and emit a machine-readable failure report."""
    # Arrange
    write_text_fixture(tmp_path / "tests" / "test_offender.py", OFFENDING_TEST)
    monkeypatch.chdir(tmp_path)

    # Act
    rc = guard.main(["--report", "reports/quality/meta_test_guard.json"])

    # Assert
    payload = json.loads(Path("reports/quality/meta_test_guard.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["status"] == "fail"
    assert payload["findings_count"] == len(payload["findings"]) > 0


def test_should_return_success_exit_code_when_tests_are_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI should exit 0 on a clean test tree."""
    # Arrange
    write_text_fixture(tmp_path / "tests" / "test_durable.py", DURABLE_TEST)
    monkeypatch.chdir(tmp_path)

    # Act
    rc = guard.main([])

    # Assert
    assert rc == 0
