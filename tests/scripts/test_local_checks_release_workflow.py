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
import os
from pathlib import Path
import zipfile

import pytest

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


def write_release_file_fixture(
    root: Path,
    *,
    release_version: str,
    development_version: str,
    previous_version: str,
) -> None:
    """Write the release.md file set with deliberately stale metadata."""
    (root / "pyproject.toml").write_text(
        f'[project]\nversion = "{development_version}"\n',
        encoding="utf-8",
    )
    init_dir = root / "src/calibrated_explanations"
    init_dir.mkdir(parents=True)
    (init_dir / "__init__.py").write_text(
        f'def _resolve_package_version():\n    return "{development_version}"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        f"version: v{previous_version}\ndate-released: '2020-01-02'\n",
        encoding="utf-8",
    )
    docs_dir = root / "docs"
    docs_dir.mkdir()
    (docs_dir / "conf.py").write_text("release = 'metadata-derived'\n", encoding="utf-8")
    (docs_dir / "citing.md").write_text(
        "Earlier paper:\n\n```bibtex\n\tmonth = \t{September},\n"
        "\tyear = \t{2020}\n```\n\nTo cite this software, use:\n\n```bibtex\n"
        "\tversion = \t{v0.0.1},\n\tmonth = \t{January},\n\tyear = \t{2020}\n```\n",
        encoding="utf-8",
    )
    (root / "METADATA.json").write_text(
        json.dumps({"name": "calibrated-explanations", "version": previous_version}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        f"[Full changelog](https://example.test/compare/v{previous_version}...main)\n\n"
        "### Changed\n\n- Release automation is complete.\n\n"
        f"## [v{release_version}](https://example.test/tag/v{release_version}) - 2020-02-03\n\n"
        "### Changed\n\n- Existing release note.\n\n"
        f"## [v{previous_version}](https://example.test/tag/v{previous_version}) - 2020-01-02\n",
        encoding="utf-8",
    )
    master_dir = root / "development/current-work"
    master_dir.mkdir(parents=True, exist_ok=True)
    (master_dir / "RELEASE_PLAN_v1.md").write_text(
        f"# Release Plan\n\n## Current released version: v{previous_version}\n\n"
        f"> Status: v{previous_version} shipped previously.\n\n"
        "### Control snapshot\n\n"
        f"- **Current released version:** v{previous_version}\n"
        f"- **Active detailed milestone:** v{previous_version}\n"
        "- **Next milestone:** v9.9.9\n",
        encoding="utf-8",
    )


def test_should_fail_release_readiness_when_any_release_gate_task_is_pending(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release readiness should stop when any task checklist has unchecked items."""
    # Arrange
    plan_path = tmp_path / "release_readiness_fixture.md"
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
    assert payload["preflight_passed"] is False


def test_should_write_release_preflight_report_when_release_gate_passes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release preflight should emit a reusable handoff report on success."""
    # Arrange
    plan_path = tmp_path / "release_readiness_fixture.md"
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
    monkeypatch.setattr(
        local_checks,
        "_release_notebook_steps",
        lambda: [local_checks.Step("Release notebooks", ["python", "-m", "fake"])],
    )
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_prepare_release_files", lambda *args, **kwargs: [])
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
    assert payload["automated_release_steps"] == list(range(1, 11))
    assert payload["manual_release_steps"] == [11, 12, 13]
    assert payload["postcommit_release_steps"] == [14, 15, 16, 17]
    assert payload["prepared_release_files"] == list(local_checks.RELEASE_PREPARED_FILES)
    assert payload["changed_release_files"] == []
    verified_release_files = payload["verified_release_files"]
    assert len(verified_release_files) == 2
    assert verified_release_files[0] == "docs/conf.py"
    assert verified_release_files[1] == plan_path.as_posix()
    assert [step["name"] for step in payload["steps"]] == [
        "Release readiness guard",
        "Release file preparation",
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


def test_should_not_publish_green_preflight_report_before_all_steps_complete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """An interrupted preflight must not leave a reusable green snapshot."""
    plan_path = tmp_path / "release_readiness_fixture.md"
    write_release_plan(plan_path)

    def interrupt(_step: local_checks.Step) -> int:
        raise KeyboardInterrupt

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(local_checks, "_current_git_status_porcelain", lambda: "M release files")
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "7.8.9")
    monkeypatch.setattr(local_checks, "_prepare_release_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(local_checks, "_release_notebook_steps", lambda: [])
    monkeypatch.setattr(local_checks, "_run_step", interrupt)

    with pytest.raises(KeyboardInterrupt):
        local_checks.run_release_preflight(plan_path=plan_path)

    report = json.loads(local_checks.RELEASE_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    assert report["exit_status"] is None
    assert report["preflight_passed"] is False
    assert report["prepared_release_files"] == list(local_checks.RELEASE_PREPARED_FILES)


@pytest.mark.parametrize(
    ("release_version", "previous_version"),
    [("0.12.3", "0.12.2"), ("2.4.0", "2.3.9"), ("1.0.0rc2", "1.0.0rc1")],
)
def test_should_prepare_every_release_file_for_any_version(
    monkeypatch,
    tmp_path: Path,
    release_version: str,
    previous_version: str,
) -> None:
    """Preflight should own release metadata and changelog preparation for every version."""
    plan_path = tmp_path / f"development/current-work/v{release_version}_plan.md"
    plan_path.parent.mkdir(parents=True)
    write_release_plan(plan_path)
    write_release_file_fixture(
        tmp_path,
        release_version=release_version,
        development_version=f"{release_version}.dev0",
        previous_version=previous_version,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(local_checks, "_current_git_status_porcelain", lambda: "M release files")
    monkeypatch.setattr(
        local_checks,
        "_release_notebook_steps",
        lambda: [local_checks.Step("Release notebooks", ["python", "-m", "fake"])],
    )
    monkeypatch.setattr(local_checks, "_run_step", lambda step: 0)
    monkeypatch.setattr(local_checks, "_run_release_twine_check", lambda: 0)
    monkeypatch.setattr(local_checks, "_run_release_wheel_smoke", lambda: 0)

    rc = local_checks.run_release_preflight(
        release_date="2031-08-09",
    )

    assert rc == 0
    assert f'version = "{release_version}"' in (tmp_path / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert f'return "{release_version}"' in (
        tmp_path / "src/calibrated_explanations/__init__.py"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "CITATION.cff").read_text(encoding="utf-8") == (
        f"version: v{release_version}\ndate-released: '2031-08-09'\n"
    )
    citing = (tmp_path / "docs/citing.md").read_text(encoding="utf-8")
    assert f"{{v{release_version}}}" in citing
    assert "{August}" in citing
    assert "{2031}" in citing
    assert "{September}" in citing
    assert citing.count("{2020}") == 1
    assert (
        json.loads((tmp_path / "METADATA.json").read_text(encoding="utf-8"))["version"]
        == release_version
    )
    changelog = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    release_heading = f"## [v{release_version}]"
    assert release_heading in changelog
    assert f"compare/v{previous_version}...v{release_version}" in changelog
    assert "Release automation is complete." not in changelog.split(release_heading, 1)[0]
    assert "Release automation is complete." in changelog.split(release_heading, 1)[1]
    assert "Existing release note." in changelog.split(release_heading, 1)[1]
    assert changelog.split(release_heading, 1)[1].count("### Changed") == 1
    master = (tmp_path / "development/current-work/RELEASE_PLAN_v1.md").read_text(encoding="utf-8")
    assert f"## Current released version: v{release_version}" in master
    assert f"**Current released version:** v{release_version}" in master
    report = json.loads(local_checks.RELEASE_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    assert report["release_version"] == release_version
    assert report["prepared_release_files"] == list(local_checks.RELEASE_PREPARED_FILES)
    assert report["changed_release_files"] == list(local_checks.RELEASE_PREPARED_FILES)
    verified_release_files = report["verified_release_files"]
    assert len(verified_release_files) == 2
    assert verified_release_files[0] == "docs/conf.py"
    assert verified_release_files[1] == plan_path.relative_to(tmp_path).as_posix()

    rc_again = local_checks.run_release_preflight(
        release_date="2031-08-09",
    )
    report_again = json.loads(local_checks.RELEASE_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    assert rc_again == 0
    assert report_again["prepared_release_files"] == list(local_checks.RELEASE_PREPARED_FILES)
    assert report_again["changed_release_files"] == []
    assert (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8").count(
        "Release automation is complete."
    ) == 1


def test_should_remove_stale_async_release_logs_before_preflight(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release preflight should purge stale async capture logs before validation."""
    plan_path = tmp_path / "release_readiness_fixture.md"
    write_release_plan(plan_path)
    monotonic_values = count(200)
    async_log = tmp_path / "reports/local_checks/release_preflight_async.log"
    async_err_log = tmp_path / "reports/local_checks/release_preflight_async.err.log"
    async_log.parent.mkdir(parents=True, exist_ok=True)
    async_log.write_text("captured stdout", encoding="utf-8")
    async_err_log.write_text("captured stderr", encoding="utf-8")

    def fake_run_step(step: local_checks.Step) -> int:
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(local_checks, "_current_git_status_porcelain", lambda: "")
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "0.11.6")
    monkeypatch.setattr(
        local_checks,
        "_release_notebook_steps",
        lambda: [local_checks.Step("Release notebooks", ["python", "-m", "fake"])],
    )
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks, "_prepare_release_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(local_checks, "_run_release_twine_check", lambda: 0)
    monkeypatch.setattr(local_checks, "_run_release_wheel_smoke", lambda: 0)
    monkeypatch.setattr(local_checks, "_utc_now_iso", lambda: "2026-07-10T00:00:00+00:00")
    monkeypatch.setattr(local_checks.time, "monotonic", lambda: next(monotonic_values) / 10)

    rc = local_checks.run_release_preflight(plan_path=plan_path)

    assert rc == 0
    assert not async_log.exists()
    assert not async_err_log.exists()


def test_should_reject_invalid_release_date_before_mutating_release_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A malformed override must fail before any deterministic file is changed."""
    release_version = "6.7.8"
    plan_path = tmp_path / f"development/current-work/v{release_version}_plan.md"
    plan_path.parent.mkdir(parents=True)
    write_release_plan(plan_path)
    write_release_file_fixture(
        tmp_path,
        release_version=release_version,
        development_version=f"{release_version}.dev0",
        previous_version="6.7.7",
    )
    pyproject = tmp_path / "pyproject.toml"
    before = pyproject.read_text(encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_current_git_branch", lambda: "main")
    monkeypatch.setattr(local_checks, "_current_git_status_porcelain", lambda: "")

    rc = local_checks.run_release_preflight(release_date="not-a-date")

    assert rc == 1
    assert pyproject.read_text(encoding="utf-8") == before


def test_should_install_latest_local_wheel_for_release_smoke(monkeypatch, tmp_path: Path) -> None:
    """Release wheel smoke should install the freshest local artifact directly."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    older_wheel = dist_dir / "calibrated_explanations-0.11.5-py3-none-any.whl"
    newer_wheel = dist_dir / "calibrated_explanations-0.11.6.dev0-py3-none-any.whl"
    for wheel in (older_wheel, newer_wheel):
        with zipfile.ZipFile(wheel, "w"):
            pass
    os.utime(older_wheel, (1, 1))
    os.utime(newer_wheel, (2, 2))

    steps_seen: list[local_checks.Step] = []

    def fake_run_step(step: local_checks.Step) -> int:
        steps_seen.append(step)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_step", fake_run_step)
    monkeypatch.setattr(local_checks.venv.EnvBuilder, "create", lambda self, path: None)
    monkeypatch.setattr(
        local_checks, "_venv_python", lambda _: Path("venv-wheel/Scripts/python.exe")
    )

    wheel_smoke_name = "_" + "run_release_wheel_smoke"
    rc = local_checks.__dict__[wheel_smoke_name]()

    assert rc == 0
    install_step = steps_seen[1]
    assert install_step.name == "Wheel smoke: install release artifact"
    assert install_step.command[-1] == str(newer_wheel.resolve())
    assert install_step.command[-1].endswith("0.11.6.dev0-py3-none-any.whl")


def test_should_fail_release_finalize_when_preflight_report_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release finalize must refuse to continue without a successful preflight report."""
    # Arrange
    plan_path = tmp_path / "release_readiness_fixture.md"
    write_release_plan(plan_path)
    monkeypatch.chdir(tmp_path)

    # Act
    rc = local_checks.run_release_finalize(plan_path=plan_path)

    # Assert
    assert rc == 1


@pytest.mark.parametrize("exit_status", [None, 0])
def test_should_explain_incomplete_preflight_snapshot_when_finalize_runs(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    exit_status: int | None,
) -> None:
    """Finalize should distinguish an incomplete checkpoint from a failed preflight."""
    plan_path = tmp_path / "release_readiness_fixture.md"
    write_release_plan(plan_path)
    report_path = tmp_path / local_checks.RELEASE_PREFLIGHT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "exit_status": exit_status,
                "preflight_passed": False,
                "steps": [
                    {
                        "name": "Release file preparation",
                        "exit_code": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    rc = local_checks.run_release_finalize(plan_path=plan_path)

    assert rc == 1
    output = capsys.readouterr().out
    assert "release-preflight is incomplete" in output
    assert "Release file preparation" in output
    assert "does not mean the complete preflight passed" in output


def test_should_fail_release_finalize_when_worktree_changed_since_preflight(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release finalize must invalidate stale preflight snapshots."""
    # Arrange
    plan_path = tmp_path / "release_readiness_fixture.md"
    write_release_plan(plan_path)
    report_path = tmp_path / local_checks.RELEASE_PREFLIGHT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "exit_status": 0,
                "preflight_passed": True,
                "release_version": "0.11.6",
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
    plan_path = tmp_path / "release_readiness_fixture.md"
    write_release_plan(plan_path)
    report_path = tmp_path / local_checks.RELEASE_PREFLIGHT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "exit_status": 0,
                "preflight_passed": True,
                "release_version": "0.11.6",
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


def test_should_fail_release_finalize_when_report_version_is_stale(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Finalize must not unlock publication using another version's green report."""
    plan_path = tmp_path / "release_readiness_fixture.md"
    write_release_plan(plan_path)
    report_path = tmp_path / local_checks.RELEASE_PREFLIGHT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "exit_status": 0,
                "preflight_passed": True,
                "release_version": "8.0.0",
                "branch": "main",
                "git_status_porcelain": "",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "8.0.1")

    rc = local_checks.run_release_finalize(plan_path=plan_path)

    assert rc == 1


def test_should_refuse_release_postcommit_when_pyproject_still_dev(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Postcommit must refuse to run before the release version has actually shipped."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "0.11.7-dev")

    rc = local_checks.run_release_postcommit()

    assert rc == 1


@pytest.mark.parametrize("released_version", ["0.12.3", "2.4.9"])
def test_should_run_postcommit_smoke_scaffold_and_bump_for_any_release_version(
    monkeypatch,
    tmp_path: Path,
    released_version: str,
) -> None:
    """Postcommit should smoke-test the release, scaffold the next plan, and bump dev version."""
    # Arrange
    major, minor, patch = (int(part) for part in released_version.split("."))
    next_version = f"{major}.{minor}.{patch + 1}"
    plan_path = tmp_path / f"development/current-work/v{released_version}_plan.md"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(f"# v{released_version} release plan\n", encoding="utf-8")

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nversion = "{released_version}"\n', encoding="utf-8")

    init_dir = tmp_path / "src/calibrated_explanations"
    init_dir.mkdir(parents=True)
    init_path = init_dir / "__init__.py"
    init_path.write_text(f'    return "{released_version}"\n', encoding="utf-8")

    smoke_calls: list[str] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: released_version)
    monkeypatch.setattr(local_checks, "_run_release_pypi_page_check", lambda version: 0)
    monkeypatch.setattr(
        local_checks,
        "_run_release_pypi_install_smoke",
        lambda version: smoke_calls.append(version) or 0,
    )

    # Act
    rc = local_checks.run_release_postcommit(plan_path=plan_path)

    # Assert
    assert rc == 0
    assert smoke_calls == [released_version]
    assert pyproject.read_text(encoding="utf-8") == (f'[project]\nversion = "{next_version}-dev"\n')
    assert init_path.read_text(encoding="utf-8") == (f'    return "{next_version}.dev0"\n')
    scaffolded = tmp_path / f"development/current-work/v{next_version}_plan.md"
    assert scaffolded.exists()
    assert f"v{next_version} Release Task Implementation Plan" in scaffolded.read_text(
        encoding="utf-8"
    )
    assert not plan_path.exists()
    assert (tmp_path / f"development/finished-work/v{released_version}_plan.md").exists()


def test_should_follow_master_next_milestone_for_release_candidate_handoff(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Postcommit should use maintained next-plan metadata instead of assuming a patch release."""
    current_plan = tmp_path / "development/current-work/v3.1.4_plan.md"
    current_plan.parent.mkdir(parents=True)
    current_plan.write_text("# v3.1.4 release plan\n", encoding="utf-8")
    next_plan = current_plan.parent / "v4.0.0-rc_plan.md"
    next_plan.write_text(
        "# v4.0.0-rc Release Task Implementation Plan\n\n"
        "> **Release version:** `4.0.0rc1`\n"
        "> **Development version:** `4.0.0-rc-dev`\n",
        encoding="utf-8",
    )
    master = current_plan.parent / "RELEASE_PLAN_v1.md"
    master.write_text(
        "# Release Plan\n\n## Current released version: v3.1.3\n\n"
        "> Status: v3.1.3 shipped.\n\n"
        "- **Current released version:** v3.1.3\n"
        "- **Active detailed milestone:** v3.1.4\n"
        "- **Next milestone:** v4.0.0-rc\n\n"
        "### v4.0.0-rc (release candidate)\n\n### v4.0.0 (general availability)\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "3.1.4"\n', encoding="utf-8")
    init_dir = tmp_path / "src/calibrated_explanations"
    init_dir.mkdir(parents=True)
    init_path = init_dir / "__init__.py"
    init_path.write_text('    return "3.1.4"\n', encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_release_pypi_page_check", lambda version: 0)
    monkeypatch.setattr(local_checks, "_run_release_pypi_install_smoke", lambda version: 0)

    rc = local_checks.run_release_postcommit(
        plan_path=current_plan,
        release_date="2032-09-10",
    )

    assert rc == 0
    assert 'version = "4.0.0-rc-dev"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'return "4.0.0rc0.dev0"' in init_path.read_text(encoding="utf-8")
    assert next_plan.exists()
    updated_master = master.read_text(encoding="utf-8")
    assert "**Active detailed milestone:** v4.0.0-rc" in updated_master
    assert "**Next milestone:** v4.0.0" in updated_master
    assert "v3.1.4 shipped on 2032-09-10" in updated_master


def test_should_stop_postcommit_when_pypi_smoke_fails(monkeypatch, tmp_path: Path) -> None:
    """A failed clean-venv install smoke test must block the version bump and scaffold."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.11.6"\n', encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_pyproject_release_version", lambda: "0.11.6")
    monkeypatch.setattr(local_checks, "_run_release_pypi_page_check", lambda version: 0)
    monkeypatch.setattr(local_checks, "_run_release_pypi_install_smoke", lambda version: 1)

    rc = local_checks.run_release_postcommit(plan_path=tmp_path / "unused_plan.md")

    assert rc == 1
    # pyproject.toml must remain untouched when the smoke test fails.
    assert pyproject.read_text(encoding="utf-8") == '[project]\nversion = "0.11.6"\n'


def test_should_stop_postcommit_on_release_plan_archive_collision(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Postcommit must not leave duplicate current and archived release plans."""
    released_version = "5.2.1"
    current_plan = tmp_path / f"development/current-work/v{released_version}_plan.md"
    archived_plan = tmp_path / f"development/finished-work/v{released_version}_plan.md"
    current_plan.parent.mkdir(parents=True)
    archived_plan.parent.mkdir(parents=True)
    current_plan.write_text("current plan\n", encoding="utf-8")
    archived_plan.write_text("archived plan\n", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nversion = "{released_version}"\n', encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_checks, "_run_release_pypi_page_check", lambda version: 0)
    monkeypatch.setattr(local_checks, "_run_release_pypi_install_smoke", lambda version: 0)

    rc = local_checks.run_release_postcommit(plan_path=current_plan)

    assert rc == 1
    assert current_plan.exists()
    assert archived_plan.read_text(encoding="utf-8") == "archived plan\n"
    assert pyproject.read_text(encoding="utf-8") == (f'[project]\nversion = "{released_version}"\n')
