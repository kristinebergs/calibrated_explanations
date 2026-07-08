"""Tests for local-check profile planning and task reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.local_checks as local_checks


def _step_names(plan: local_checks.ProfilePlan) -> list[str]:
    """Return step names for a profile plan."""
    return [step.name for step in plan.steps]


def test_should_exclude_heavy_and_pre_commit_steps_from_quick_profile() -> None:
    """The quick profile should stay inner-loop focused."""
    plan = local_checks.build_profile_plan(
        "quick",
        task=None,
        mypy_targets=["src/calibrated_explanations/core/exceptions.py"],
        lint_targets=["scripts/local_checks.py"],
        pre_commit_available=True,
    )

    step_names = _step_names(plan)
    assert plan.profile == "quick"
    assert "Ruff check" in step_names
    assert "Ruff format check" in step_names
    assert "Mypy (Phase 1B scope)" in step_names
    assert "ADR-001 boundary check" in step_names
    assert "ADR-002 compliance check" in step_names
    assert "Unit tests (fast/no viz/no slow/no cov)" in step_names
    assert "Pre-commit" not in step_names
    assert "Docstring coverage" not in step_names
    assert "All non-viz tests (no coverage)" not in step_names
    assert any(item["gate"] == "coverage" for item in plan.skipped_heavy_gates)
    assert any(item["gate"] == "dependency_audit" for item in plan.skipped_heavy_gates)


def test_should_require_task_id_for_task_profile() -> None:
    """The task profile should fail fast when no task is supplied."""
    with pytest.raises(ValueError, match="requires --task"):
        local_checks.build_profile_plan(
            "task",
            task=None,
            mypy_targets=["src/calibrated_explanations/core/exceptions.py"],
            lint_targets=["scripts/local_checks.py"],
            pre_commit_available=False,
        )


def test_should_include_quick_steps_and_task_mapping_for_task_profile() -> None:
    """The task profile should layer focused task verification on top of quick checks."""
    plan = local_checks.build_profile_plan(
        "task",
        task=9,
        mypy_targets=["src/calibrated_explanations/core/exceptions.py"],
        lint_targets=["scripts/local_checks.py"],
        pre_commit_available=False,
    )

    step_names = _step_names(plan)
    assert plan.task == 9
    assert step_names[:5] == [
        "Ruff check",
        "Ruff format check",
        "Mypy (Phase 1B scope)",
        "ADR-001 boundary check",
        "ADR-002 compliance check",
    ]
    assert "Task 9 local-check profile tests" in step_names
    assert "Task 9 ADR-030 ratification" in step_names
    assert "Task 9 instruction consistency" in step_names
    assert "Pre-commit" not in step_names
    assert any(item["gate"] == "release_bookkeeping" for item in plan.skipped_heavy_gates)


def test_should_scope_task_7_lint_targets_to_instruction_verification_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 7 should route narrow lint targets through the public task-profile entrypoint."""
    captured: dict[str, object] = {}

    def fake_build_profile_plan(
        profile: str,
        *,
        task: int | None,
        mypy_targets: list[str],
        lint_targets: list[str],
        pre_commit_available: bool,
    ) -> local_checks.ProfilePlan:
        captured["profile"] = profile
        captured["task"] = task
        captured["lint_targets"] = lint_targets
        return local_checks.ProfilePlan(
            profile=profile, task=task, steps=[], skipped_heavy_gates=[]
        )

    monkeypatch.setattr(local_checks, "build_profile_plan", fake_build_profile_plan)
    monkeypatch.setattr(local_checks, "run_profile_plan", lambda *args, **kwargs: 0)
    monkeypatch.setattr(local_checks, "_pytest_supports_no_cov", lambda: True)
    monkeypatch.setattr(local_checks.shutil, "which", lambda name: "tool")
    monkeypatch.setattr(
        local_checks.subprocess,
        "call",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        local_checks.sys,
        "argv",
        ["local_checks.py", "--profile", "task", "--task", "7"],
    )

    rc = local_checks.main()

    assert rc == 0
    assert captured["profile"] == "task"
    assert captured["task"] == 7
    assert captured["lint_targets"] == [
        "scripts/local_checks.py",
        "scripts/quality/check_agent_instruction_consistency.py",
        "tests/scripts/test_local_checks_profiles.py",
    ]


def test_should_include_pr_gates_without_heavy_optional_work() -> None:
    """The PR profile should keep blocking PR checks and skip heavy release-only work."""
    plan = local_checks.build_profile_plan(
        "pr",
        task=None,
        mypy_targets=["src/calibrated_explanations/core/exceptions.py"],
        lint_targets=["scripts/local_checks.py"],
        pre_commit_available=True,
    )

    step_names = _step_names(plan)
    assert "Pre-commit" in step_names
    assert step_names.count("Pre-commit") == 1
    assert "Docstring coverage" in step_names
    assert "STD-005 logger domain enforcement" in step_names
    assert "STD-001 nomenclature enforcement" in step_names
    assert "Capability-chain validator" in step_names
    assert "Raw evidence structural validation" in step_names
    assert "Private-member scan" in step_names
    assert "ADR-030 anti-pattern detector" in step_names
    assert "ADR-006 trust-mutation primitive guard" in step_names
    assert "ADR-034 ConfigManager usage guard" in step_names
    assert "Generated report local-path guard" in step_names
    assert "All non-viz tests (no coverage)" in step_names
    assert "Docs build (HTML)" not in step_names
    assert "Dependency audit" not in step_names
    assert "Capability evidence refresh" not in step_names


def test_should_include_main_style_gates_in_full_profile() -> None:
    """The full profile should extend PR scope with heavy local validation."""
    plan = local_checks.build_profile_plan(
        "full",
        task=None,
        mypy_targets=["src/calibrated_explanations/core/exceptions.py"],
        lint_targets=["scripts/local_checks.py"],
        pre_commit_available=False,
    )

    step_names = _step_names(plan)
    assert "Docs build (HTML)" in step_names
    assert "Core tests with coverage" in step_names
    assert "Per-module coverage gates" in step_names
    assert "Micro benchmark" in step_names
    assert "Perf thresholds" in step_names
    assert "Over-testing coverage contexts" in step_names
    assert "Over-testing report" in step_names
    assert "Redundant tests report" in step_names
    assert plan.skipped_heavy_gates == []


def test_should_include_release_only_gates_in_release_profile() -> None:
    """The release profile should add release-boundary checks on top of full validation."""
    plan = local_checks.build_profile_plan(
        "release",
        task=None,
        mypy_targets=["src/calibrated_explanations/core/exceptions.py"],
        lint_targets=["scripts/local_checks.py"],
        pre_commit_available=False,
    )

    step_names = _step_names(plan)
    assert "Dependency audit" in step_names
    assert "Notebook audit" in step_names
    assert "Notebook execution report" in step_names
    assert "Strict docs build" in step_names
    assert "Docs linkcheck" in step_names
    assert "Warning policy" in step_names
    assert "Deprecation closure" in step_names
    assert "Capability evidence refresh" in step_names
    assert "Public API snapshot" in step_names


def test_should_write_machine_readable_task_report(monkeypatch, tmp_path: Path) -> None:
    """Task execution should emit the requested machine-readable report."""
    plan = local_checks.build_profile_plan(
        "task",
        task=9,
        mypy_targets=[],
        lint_targets=["scripts/local_checks.py"],
        pre_commit_available=False,
    )
    executed_steps: list[str] = []

    def fake_run_step(step: local_checks.Step) -> int:
        executed_steps.append(step.name)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-07-08T00:00:00+00:00")

    rc = local_checks.run_profile_plan(
        plan,
        task_report_path=Path("reports/local_checks/task_profile_report.json"),
        requested_paths=["scripts/local_checks.py", "Makefile"],
    )

    report = json.loads(
        Path("reports/local_checks/task_profile_report.json").read_text(encoding="utf-8")
    )
    assert rc == 0
    assert executed_steps
    assert report["selected_profile"] == "task"
    assert report["task_id"] == 9
    assert report["exit_status"] == 0
    assert len(report["requested_paths"]) == 2
    assert "scripts/local_checks.py" in report["requested_paths"]
    assert "Makefile" in report["requested_paths"]
    assert report["commands_run"]
    assert (
        report["commands_run"][-1]
        == "python scripts/quality/check_agent_instruction_consistency.py"
    )
    assert any(item["gate"] == "coverage" for item in report["skipped_heavy_gates"])
