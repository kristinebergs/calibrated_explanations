"""Tests for the focused ADR-030 local ratification lane."""

from __future__ import annotations

import json
import re
from itertools import count
from pathlib import Path

import scripts.local_checks as local_checks


def test_should_define_adr030_ratification_steps_in_expected_order() -> None:
    """The ADR-030 lane should mirror the ratification gate sequence."""
    # Arrange
    # Act
    steps = local_checks.adr030_ratification_steps()

    # Assert
    assert [step.name for step in steps] == [
        "Private-member scan",
        "ADR-030 anti-pattern detector",
        "ADR-030 test-helper export guard",
        "ADR-030 marker hygiene",
        "Generated report local-path guard",
    ]
    assert steps[0].command[0] == "python"
    assert steps[0].command[1] == "scripts/anti-pattern-analysis/scan_private_usage.py"
    assert steps[0].command[2:] == ["tests", "--check"]
    assert steps[1].command[1] == "scripts/anti-pattern-analysis/detect_test_anti_patterns.py"
    assert steps[1].command[2:5] == ["--tests-dir", "tests", "--check"]
    assert steps[2].command[1] == "scripts/quality/check_no_test_helper_exports.py"
    assert steps[2].command[2] == "--root"
    assert steps[2].command[3] == "src/calibrated_explanations"
    assert steps[3].command[1] == "scripts/quality/check_marker_hygiene.py"
    assert steps[3].command[2] == "--check"
    assert steps[4].command[1] == "scripts/quality/check_no_local_paths_in_reports.py"
    assert steps[4].command[2] == "--check"


def test_should_write_timing_report_when_adr030_ratification_lane_passes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The ADR-030 lane should write deterministic timing evidence."""
    # Arrange
    monotonic_values = count(100)

    def fake_run_step(step: local_checks.Step) -> int:
        for report_path in local_checks.adr030_expected_reports():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-05-12T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: next(monotonic_values) / 10)

    # Act
    rc = local_checks.run_adr030_ratification()

    # Assert
    timing_report = Path("reports/anti-pattern-analysis/adr030_ratification_timing.json")
    payload = json.loads(timing_report.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["schema_version"] == 1
    assert payload["generated_at"] == "2026-05-12T00:00:00+00:00"
    assert payload["python_version"]
    assert payload["platform"]
    assert [step["name"] for step in payload["steps"]] == [
        "Private-member scan",
        "ADR-030 anti-pattern detector",
        "ADR-030 test-helper export guard",
        "ADR-030 marker hygiene",
        "Generated report local-path guard",
    ]
    assert [step["exit_code"] for step in payload["steps"]] == [0, 0, 0, 0, 0]
    assert all(step["elapsed_seconds"] >= 0 for step in payload["steps"])
    assert payload["total_elapsed_seconds"] >= 0


def test_should_stop_and_return_failure_when_adr030_step_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The ADR-030 lane should stop immediately on a failing gate."""
    # Arrange
    calls: list[str] = []
    monotonic_values = count(200)

    def fake_run_step(step: local_checks.Step) -> int:
        calls.append(step.name)
        if step.name == "ADR-030 anti-pattern detector":
            return 7
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-05-12T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: next(monotonic_values) / 10)

    # Act
    rc = local_checks.run_adr030_ratification()

    # Assert
    assert rc == 7
    assert calls == ["Private-member scan", "ADR-030 anti-pattern detector"]


def test_should_return_failure_when_adr030_reports_are_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The ADR-030 lane should fail when gate reports are absent."""
    # Arrange
    monotonic_values = count(300)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", lambda step: 0)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-05-12T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: next(monotonic_values) / 10)

    # Act
    rc = local_checks.run_adr030_ratification()

    # Assert
    assert rc == 1


def test_should_not_include_absolute_paths_in_timing_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The ADR-030 timing report should use repo-relative command strings."""
    # Arrange
    monotonic_values = count(400)
    windows_drive_pattern = re.compile(r"^[A-Za-z]:[\\/]")

    def fake_run_step(step: local_checks.Step) -> int:
        for report_path in local_checks.adr030_expected_reports():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-05-12T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: next(monotonic_values) / 10)

    # Act
    rc = local_checks.run_adr030_ratification()

    # Assert
    timing_report = Path("reports/anti-pattern-analysis/adr030_ratification_timing.json")
    payload = json.loads(timing_report.read_text(encoding="utf-8"))
    commands = [step["command"] for step in payload["steps"]]
    assert rc == 0
    assert all(not Path(command.split()[0]).is_absolute() for command in commands)
    assert all(windows_drive_pattern.match(command) is None for command in commands)


def test_should_preserve_explicit_empty_task_lint_targets(monkeypatch, tmp_path: Path) -> None:
    """An explicit empty task lint target list should disable task lint preflight."""
    # Arrange
    plan_path = tmp_path / "plan.md"
    plan_path.write_text(
        """
```toml ce-task-verification
schema_version = 1

[task.38]
lint_targets = []

[[task.38.steps]]
name = "Instruction consistency"
command = "python scripts/quality/check_agent_instruction_consistency.py"
```
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(local_checks, "_changed_python_targets", lambda: ["src", "tests", "scripts"])

    # Act
    lint_targets = local_checks.task_specific_lint_targets(38, plan_path=plan_path)

    # Assert
    assert lint_targets == []


def test_should_skip_ruff_commands_in_task_run_when_lint_targets_are_empty(
    monkeypatch, tmp_path: Path
) -> None:
    """A task run with explicit empty lint targets should skip Ruff preflight."""
    # Arrange
    plan_path = tmp_path / "plan.md"
    plan_path.write_text(
        """
```toml ce-task-verification
schema_version = 1

[task.38]
lint_targets = []

[[task.38.steps]]
name = "Instruction consistency"
command = "python scripts/quality/check_agent_instruction_consistency.py"
```
""".strip(),
        encoding="utf-8",
    )
    commands_seen: list[str] = []

    def fake_run_step(step: local_checks.Step) -> int:
        commands_seen.append(" ".join(step.command))
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_mypy_targets", lambda: [])
    monkeypatch.setattr(local_checks, "_pytest_supports_no_cov", lambda: True)
    monkeypatch.setattr(local_checks.shutil, "which", lambda name: "tool" if name in {"mypy", "pre-commit"} else None)

    # Act
    monkeypatch.setattr(
        local_checks.sys,
        "argv",
        [
            "local_checks.py",
            "--profile",
            "task",
            "--task",
            "38",
            "--plan",
            str(plan_path),
        ],
    )
    rc = local_checks.main()

    # Assert
    assert rc == 0
    assert all("ruff" not in command_text for command_text in commands_seen)
    assert any("check_import_graph.py" in command_text for command_text in commands_seen)
    assert any("check_adr002_compliance.py" in command_text for command_text in commands_seen)


def test_should_enable_ruff_preview_for_markdown_task_lint_targets(
    monkeypatch, tmp_path: Path
) -> None:
    """Task lint preflight should use Ruff preview mode when formatting Markdown."""
    # Arrange
    plan_path = tmp_path / "plan.md"
    plan_path.write_text(
        """
```toml ce-task-verification
schema_version = 1

[task.52]
lint_targets = ["docs/migration/deprecations.md", "tests/integration/test_doc_examples_smoke.py"]

[[task.52.steps]]
name = "Doc smoke"
command = "python -m pytest tests/integration/test_doc_examples_smoke.py -q"
```
""".strip(),
        encoding="utf-8",
    )
    commands_seen: list[list[str]] = []

    def fake_run_step(step: local_checks.Step) -> int:
        commands_seen.append(step.command)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_mypy_targets", lambda: [])
    monkeypatch.setattr(local_checks, "_pytest_supports_no_cov", lambda: True)
    monkeypatch.setattr(local_checks.shutil, "which", lambda name: "tool" if name in {"mypy", "pre-commit"} else None)

    # Act
    monkeypatch.setattr(
        local_checks.sys,
        "argv",
        [
            "local_checks.py",
            "--profile",
            "task",
            "--task",
            "52",
            "--plan",
            str(plan_path),
        ],
    )
    rc = local_checks.main()

    # Assert
    assert rc == 0
    format_commands = [command for command in commands_seen if command[2:4] == ["ruff", "format"]]
    assert len(format_commands) == 1
    assert "--check" in format_commands[0]
    assert "--preview" in format_commands[0]
