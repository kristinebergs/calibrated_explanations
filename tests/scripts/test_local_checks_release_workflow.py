"""Tests for the strict release-preflight and release-finalize guards.

Why a new test file?
--------------------
The release workflow guard is a standalone local-checks script surface with its
own CLI/reporting contract and git/plan readiness behavior. Existing
``test_local_checks_deprecation_closure.py`` and
``test_local_checks_adr030_ratification.py`` cover other focused lanes; adding
release-handoff tests there would mix unrelated concerns.
"""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import scripts.local_checks as local_checks


def write_release_plan(
    path: Path,
    *,
    unchecked_tasks: set[int] | None = None,
) -> None:
    """Write a minimal release plan with task verification checklists."""
    unchecked = unchecked_tasks or set()
    sections: list[str] = ["# Minimal release plan", ""]
    for task_id in range(1, 45):
        mark = " " if task_id in unchecked else "x"
        sections.extend(
            [
                f"## {task_id}) Task {task_id}",
                "",
                f"### {task_id}.1 Verification checklist",
                "",
                f"- [{mark}] Task {task_id} closure evidence captured",
                "",
            ]
        )
    sections.extend(
        [
            "## Release gate summary",
            "",
            "| Gate criterion | Task | Status |",
            "|---|---|---|",
            "| Placeholder | 45 | Not started |",
        ]
    )
    path.write_text(
        "\n".join(sections),
        encoding="utf-8",
        newline="\n",
    )


def test_should_fail_release_readiness_when_any_release_gate_task_is_pending(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release readiness should stop when any task checklist has unchecked items."""
    # Arrange
    plan_path = tmp_path / "v0.11.6_plan.md"
    write_release_plan(plan_path, unchecked_tasks={41})
    monotonic_values = count(10)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(local_checks, "_current_git_status_porcelain", lambda: "")
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "0.11.6")
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-07-10T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: next(monotonic_values) / 10)

    # Act
    rc = local_checks.run_release_preflight(plan_path=plan_path)

    # Assert
    assert rc == 1
    payload = json.loads(local_checks.RELEASE_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    assert payload["branch"] == "main"
    assert payload["pyproject_version"] == "0.11.6"
    assert payload["task_checklist_state"]["41"]["all_items_checked"] is False
    assert payload["task_checklist_state"]["41"]["checked_items"] == 0
    assert payload["task_checklist_state"]["41"]["total_items"] == 1
    assert payload["steps"][0]["name"] == "Release readiness guard"
    assert payload["steps"][0]["exit_code"] == 1


def test_should_write_release_preflight_report_when_release_gate_passes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release preflight should emit a reusable handoff report on success."""
    # Arrange
    plan_path = tmp_path / "v0.11.6_plan.md"
    monotonic_values = count(100)
    commands_seen: list[str] = []
    write_release_plan(plan_path)

    def fake_run_step(step: local_checks.Step) -> int:
        commands_seen.append(step.name)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(local_checks, "_current_git_status_porcelain", lambda: "M CHANGELOG.md")
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "0.11.6")
    monkeypatch.setattr(local_checks, "_release_notebook_steps", lambda: [local_checks.Step("Release notebooks", ["python", "-m", "fake"] )])
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_run_release_twine_check", lambda: 0)
    monkeypatch.setattr(local_checks, "_run_release_wheel_smoke", lambda: 0)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-07-10T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: next(monotonic_values) / 10)

    # Act
    rc = local_checks.run_release_preflight(plan_path=plan_path)

    # Assert
    payload = json.loads(local_checks.RELEASE_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["preflight_passed"] is True
    assert payload["exit_status"] == 0
    assert payload["branch"] == "main"
    assert payload["manual_release_steps"] == [11, 12, 13, 14, 15, 16, 17]
    assert [step["name"] for step in payload["steps"]] == [
        "Release readiness guard",
        "Full pytest suite",
        "Editable install (release tree)",
        "Editable install version smoke",
        "Release notebooks",
        "Release profile",
        "Release artifact validation",
        "Release wheel smoke",
    ]
    assert commands_seen == [
        "Full pytest suite",
        "Editable install (release tree)",
        "Editable install version smoke",
        "Release notebooks",
        "Release profile",
    ]


def test_should_fail_release_finalize_when_preflight_report_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release finalize must refuse to continue without a successful preflight report."""
    # Arrange
    plan_path = tmp_path / "v0.11.6_plan.md"
    write_release_plan(plan_path)
    monkeypatch.chdir(tmp_path)

    # Act
    rc = local_checks.run_release_finalize(plan_path=plan_path)

    # Assert
    assert rc == 1


def test_should_fail_release_finalize_when_worktree_changed_since_preflight(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release finalize must invalidate stale preflight snapshots."""
    # Arrange
    plan_path = tmp_path / "v0.11.6_plan.md"
    write_release_plan(plan_path)
    report_path = tmp_path / local_checks.RELEASE_PREFLIGHT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "exit_status": 0,
                "preflight_passed": True,
                "branch": "main",
                "git_status_porcelain": "M CHANGELOG.md",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(local_checks, "_current_git_status_porcelain", lambda: "")
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "0.11.6")

    # Act
    rc = local_checks.run_release_finalize(plan_path=plan_path)

    # Assert
    assert rc == 1


def test_should_pass_release_finalize_when_snapshot_and_plan_still_match(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release finalize should unlock the manual phase only for a matching green snapshot."""
    # Arrange
    plan_path = tmp_path / "v0.11.6_plan.md"
    write_release_plan(plan_path)
    report_path = tmp_path / local_checks.RELEASE_PREFLIGHT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "exit_status": 0,
                "preflight_passed": True,
                "branch": "main",
                "git_status_porcelain": "M CHANGELOG.md",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(local_checks, "_current_git_status_porcelain", lambda: "M CHANGELOG.md")
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "0.11.6")

    # Act
    rc = local_checks.run_release_finalize(plan_path=plan_path)

    # Assert
    assert rc == 0
