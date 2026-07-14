"""Tests for the ADR-035 full-inventory CI-policy validator (v0.11.6 Task 60).

The validator is blocking and validates the *complete* workflow inventory, so
these tests cover the accepted repository inventory plus representative
forbidden cases built from a minimal compliant fixture.
"""

from __future__ import annotations

from pathlib import Path

from scripts.quality.validate_ci_policy import (
    APPROVED_WORKFLOWS,
    FULL_INVENTORY_COMMAND,
    validate_full_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

CHECKOUT_SHA = "df4cb1c069e1874edd31b4311f1884172cec0e10"

CI_FIXTURE = f"""# Responsibility: PR and main validation (fixture).
name: CI
on:
  pull_request:
    branches: [ main ]
permissions:
  contents: read
concurrency:
  group: ci-fixture
  cancel-in-progress: true
jobs:
  policy:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@{CHECKOUT_SHA} # v6
      - run: {FULL_INVENTORY_COMMAND}
  required:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - run: echo ok
"""

SCHEDULED_FIXTURE = """# Responsibility: weekly heavy assurance (fixture).
name: Scheduled assurance
on:
  schedule:
    - cron: '0 3 * * 1'
  workflow_dispatch: {}
permissions:
  contents: read
jobs:
  audit:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: pip install pip-audit -c constraints.txt
"""

MAINTENANCE_FIXTURE = """# Responsibility: manual maintenance with write scopes (fixture).
name: Maintenance
on:
  workflow_dispatch:
    inputs:
      reason:
        required: true
        type: string
permissions:
  contents: read
jobs:
  update-baseline:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: write
      pull-requests: write
    steps:
      - run: echo baseline
"""


def write_minimal_inventory(tmp_path: Path) -> None:
    """Write a compliant three-workflow inventory plus guard-wiring stubs."""
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(CI_FIXTURE, encoding="utf-8")
    (workflows / "scheduled.yml").write_text(SCHEDULED_FIXTURE, encoding="utf-8")
    (workflows / "maintenance.yml").write_text(MAINTENANCE_FIXTURE, encoding="utf-8")

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "local_checks.py").write_text(
        'STEP = ["scripts/quality/validate_ci_policy.py", "--full-inventory"]\n',
        encoding="utf-8",
    )
    (tmp_path / ".pre-commit-config.yaml").write_text(
        "repos:\n"
        "  - repo: local\n"
        "    hooks:\n"
        "      - id: ci-policy-full-inventory\n"
        f"        entry: {FULL_INVENTORY_COMMAND}\n"
        "        language: system\n"
        "        files: ^\\.github/\n"
        "        pass_filenames: false\n",
        encoding="utf-8",
    )
    (tmp_path / "Makefile").write_text(
        "local-checks-pr:\n\tpython scripts/local_checks.py --profile pr\n",
        encoding="utf-8",
    )


def test_should_accept_the_real_repository_inventory() -> None:
    result = validate_full_inventory(REPO_ROOT)

    assert result.errors == []


def test_should_accept_the_minimal_compliant_fixture(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)

    result = validate_full_inventory(tmp_path)

    assert result.errors == []


def test_should_reject_unapproved_workflow_file(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    (tmp_path / ".github" / "workflows" / "extra.yml").write_text(
        "# Responsibility: sneaky.\nname: Extra\non: [push]\npermissions:\n  contents: read\n",
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("extra.yml: not in the approved inventory" in error for error in result.errors)


def test_should_reject_missing_approved_workflow(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    (tmp_path / ".github" / "workflows" / "scheduled.yml").unlink()

    result = validate_full_inventory(tmp_path)

    assert any("scheduled.yml: approved workflow is missing" in error for error in result.errors)


def test_should_reject_private_skill_repository_reference(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "scheduled.yml"
    path.write_text(
        SCHEDULED_FIXTURE + "      - run: git clone org/generic-skill-library\n",
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("private skill repository" in error for error in result.errors)


def test_should_reject_automated_skill_directory_write(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "scheduled.yml"
    path.write_text(
        SCHEDULED_FIXTURE + "      - run: cp -r built/ .claude/skills/\n",
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("agent skill directory" in error for error in result.errors)


def test_should_reject_write_permissions_outside_maintenance(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "ci.yml"
    path.write_text(
        CI_FIXTURE.replace("  contents: read\n", "  contents: write\n", 1),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("write permissions are only allowed" in error for error in result.errors)


def test_should_reject_unpinned_external_action(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "ci.yml"
    path.write_text(
        CI_FIXTURE.replace(f"actions/checkout@{CHECKOUT_SHA} # v6", "actions/checkout@v6"),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("full 40-character commit SHA" in error for error in result.errors)


def test_should_reject_pip_install_without_constraints(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "scheduled.yml"
    path.write_text(
        SCHEDULED_FIXTURE.replace(
            "pip install pip-audit -c constraints.txt", "pip install pip-audit"
        ),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("pip install must include -c constraints.txt" in error for error in result.errors)


def test_should_reject_publishing_mechanics(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "scheduled.yml"
    path.write_text(
        SCHEDULED_FIXTURE + "      - run: twine upload dist/*\n",
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("package publication" in error for error in result.errors)


def test_should_reject_pr_opening_mechanism_outside_maintenance(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "ci.yml"
    path.write_text(
        CI_FIXTURE + "  open-pr:\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 5\n"
        "    steps:\n"
        f"      - uses: peter-evans/create-pull-request@{CHECKOUT_SHA} # v7\n",
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("PR-opening write mechanism" in error for error in result.errors)


def test_should_reject_continue_on_error(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "scheduled.yml"
    path.write_text(
        SCHEDULED_FIXTURE.replace(
            "    timeout-minutes: 10\n",
            "    timeout-minutes: 10\n    continue-on-error: true\n",
        ),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("continue-on-error is forbidden" in error for error in result.errors)


def test_should_reject_job_without_timeout(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "scheduled.yml"
    path.write_text(
        SCHEDULED_FIXTURE.replace("    timeout-minutes: 10\n", ""),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("needs timeout-minutes" in error for error in result.errors)


def test_should_reject_guard_wiring_drift_in_ci_workflow(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "ci.yml"
    path.write_text(
        CI_FIXTURE.replace(f"      - run: {FULL_INVENTORY_COMMAND}\n", "      - run: echo skip\n"),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("policy job must execute" in error for error in result.errors)


def test_should_reject_guard_wiring_drift_in_pre_commit_and_local_checks(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
    (tmp_path / "scripts" / "local_checks.py").write_text("STEP = []\n", encoding="utf-8")

    result = validate_full_inventory(tmp_path)

    assert any(".pre-commit-config.yaml" in error for error in result.errors)
    assert any("scripts/local_checks.py" in error for error in result.errors)


def test_should_reject_duplicate_workflow_names(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "scheduled.yml"
    path.write_text(
        SCHEDULED_FIXTURE.replace("name: Scheduled assurance", "name: CI"),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("duplicates" in error for error in result.errors)


def test_should_reject_unknown_local_profile_reference(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "ci.yml"
    path.write_text(
        CI_FIXTURE + "  gate:\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 5\n"
        "    steps:\n"
        "      - run: make local-checks-nope\n",
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("local profile 'nope'" in error for error in result.errors)


def test_should_reject_maintenance_with_non_manual_trigger(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "maintenance.yml"
    path.write_text(
        MAINTENANCE_FIXTURE.replace("on:\n", "on:\n  push:\n    branches: [ main ]\n", 1),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("workflow_dispatch only" in error for error in result.errors)


def test_should_reject_scheduled_workflow_running_on_pr(tmp_path: Path) -> None:
    write_minimal_inventory(tmp_path)
    path = tmp_path / ".github" / "workflows" / "scheduled.yml"
    path.write_text(
        SCHEDULED_FIXTURE.replace("on:\n", "on:\n  pull_request:\n    branches: [ main ]\n", 1),
        encoding="utf-8",
    )

    result = validate_full_inventory(tmp_path)

    assert any("must not trigger on PR/push" in error for error in result.errors)


def test_should_keep_approved_inventory_to_three_workflows() -> None:
    assert sorted(APPROVED_WORKFLOWS) == ["ci.yml", "maintenance.yml", "scheduled.yml"]


def test_should_cover_scripts_local_checks_path_in_codeowners() -> None:
    codeowners_text = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert "/scripts/local_checks.py @tuvelofstrom" in codeowners_text
