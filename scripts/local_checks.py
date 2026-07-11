"""Run local verification profiles in the current environment.

This runner preserves focused standalone lanes (ADR-030 ratification,
deprecation closure, workflow run-block smoke) and adds explicit local
verification profiles for quick, task, PR, full, and release validation.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import importlib.util
import json
import os
import platform
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
import venv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Step:
    """A single local check step."""

    name: str
    command: list[str]
    optional: bool = False


@dataclass(frozen=True)
class ProfilePlan:
    """A concrete local-check execution plan for one profile."""

    profile: str
    steps: list[Step]
    skipped_heavy_gates: list[dict[str, str]]
    task: int | None = None


def _python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def _dependency_audit_command() -> list[str]:
    """Return the preferred dependency-audit command for this environment."""
    if shutil.which("pip-audit") is not None:
        return [
            "pip-audit",
            "-r",
            "requirements.txt",
            "-r",
            "docs/requirements-doc.txt",
            "--ignore-vuln",
            "GHSA-xm59-rqc7-hhvf",
        ]
    if importlib.util.find_spec("pip_audit") is not None:
        return _python_cmd(
            "-m",
            "pip_audit",
            "-r",
            "requirements.txt",
            "-r",
            "docs/requirements-doc.txt",
            "--ignore-vuln",
            "GHSA-xm59-rqc7-hhvf",
        )
    return [
        "pip-audit",
        "-r",
        "requirements.txt",
        "-r",
        "docs/requirements-doc.txt",
        "--ignore-vuln",
        "GHSA-xm59-rqc7-hhvf",
    ]


def _is_python_build_module_step(command: list[str]) -> bool:
    """Return True when a step invokes the PyPA ``build`` module."""
    return len(command) >= 3 and command[0] == sys.executable and command[1:3] == ["-m", "build"]


def _is_pre_commit_step(step: Step) -> bool:
    if not step.command:
        return False
    head = Path(step.command[0]).name.lower()
    return head in {"pre-commit", "pre-commit.exe"}


def _mypy_targets() -> list[str]:
    candidates = [
        "src/calibrated_explanations/core/exceptions.py",
        "src/calibrated_explanations/core/validation.py",
        "src/calibrated_explanations/api/params.py",
    ]
    return [path for path in candidates if Path(path).is_file()]


def _run_step(step: Step) -> int:
    repo_root = Path.cwd()
    command = list(step.command)
    run_cwd: Path | None = None
    if _is_python_build_module_step(command):
        # Avoid importing the repository's top-level ``build/`` directory as a namespace
        # package instead of the installed PyPA ``build`` tool.
        if len(command) == 3:
            command.append(str(repo_root))
        run_cwd = Path(tempfile.mkdtemp(prefix="ce-local-check-build-"))
    cmd_text = " ".join(command)
    print(f"\n[{step.name}]")
    print(f"$ {cmd_text}")
    env = dict(os.environ)
    env.setdefault("PRE_COMMIT_HOME", str(Path(".cache/pre-commit").resolve()))
    if "coverage" in step.name.lower():
        env.setdefault("COVERAGE_FILE", f".coverage.local_checks.{os.getpid()}")
    if step.name in {
        "Core tests (no viz/no cov)",
        "Core tests with coverage",
        "Unit tests (fast/no viz/no slow/no cov)",
        "All non-viz tests (no coverage)",
    }:
        env.pop("CE_DEPRECATIONS", None)
    try:
        if _is_pre_commit_step(step):
            result = subprocess.run(  # noqa: S603
                command,
                check=False,
                env=env,
                cwd=run_cwd,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        else:
            result = subprocess.run(command, check=False, env=env, cwd=run_cwd)  # noqa: S603
    except FileNotFoundError as exc:
        if step.optional:
            print(f"Step skipped (optional command unavailable): {step.name} ({exc})")
            return 0
        print(f"ERROR: required command unavailable for step '{step.name}': {exc}")
        return 127
    if result.returncode == 0:
        return 0
    if _is_pre_commit_step(step):
        combined = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
        if _is_network_fetch_failure(combined):
            print(
                "Pre-commit could not fetch hook repos (offline/network-restricted). Continuing with stacked checks."
            )
            return 0
    if step.optional:
        print(f"Step failed but is advisory/optional: {step.name} (rc={result.returncode})")
        return 0
    return result.returncode


def _venv_python(venv_path: Path) -> Path:
    """Return the Python executable path for a virtual environment."""
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _run_timed_command(command: list[str]) -> tuple[int, int]:
    """Run a command and return ``(returncode, elapsed_seconds)``."""
    start = time.monotonic()
    result = subprocess.run(command, check=False)  # noqa: S603
    elapsed = int(round(time.monotonic() - start))
    return result.returncode, elapsed


def _run_uv_install_smoke() -> int:
    """Reproduce the CI uv install smoke and timing lane locally."""
    uv_binary = shutil.which("uv")
    if uv_binary is None:
        print("ERROR: uv not found. Install uv before running the optional uv install smoke.")
        return 127

    run_dir = Path(tempfile.mkdtemp(prefix="ce-uv-install-smoke-")).resolve()
    pip_venv = run_dir / "venv-pip"
    uv_venv = run_dir / "venv-uv"
    timing_report = Path("reports/ci/uv_install_timing.txt")
    smoke_python = "3.11"

    print("\n[uv install smoke and timing]")
    print(f"Working directory: {run_dir}")
    print(f"Provisioning smoke envs with Python {smoke_python} via uv")
    timing_report.parent.mkdir(parents=True, exist_ok=True)

    for venv_path in (pip_venv, uv_venv):
        rc = subprocess.run(  # noqa: S603
            [uv_binary, "venv", "--python", smoke_python, "--seed", str(venv_path)],
            check=False,
        ).returncode
        if rc != 0:
            return rc

    pip_python = _venv_python(pip_venv)
    uv_python = _venv_python(uv_venv)
    binary_only = "--only-binary=numpy,scipy,scikit-learn"

    pip_rc, pip_seconds = _run_timed_command(
        [
            str(pip_python),
            "-m",
            "pip",
            "install",
            binary_only,
            "-e",
            ".[dev]",
            "-c",
            "constraints.txt",
        ]
    )
    if pip_rc != 0:
        return pip_rc

    uv_rc, uv_seconds = _run_timed_command(
        [
            uv_binary,
            "pip",
            "install",
            "--python",
            str(uv_python),
            binary_only,
            "-e",
            ".[dev]",
            "-c",
            "constraints.txt",
        ]
    )
    if uv_rc != 0:
        return uv_rc

    smoke = subprocess.run(  # noqa: S603
        [
            str(uv_python),
            "-c",
            (
                "from importlib.metadata import version; "
                "import calibrated_explanations; "
                "from calibrated_explanations import WrapCalibratedExplainer; "
                "print(calibrated_explanations.__name__); "
                "print(WrapCalibratedExplainer.__name__); "
                "print(version('calibrated-explanations'))"
            ),
        ],
        check=False,
    )
    if smoke.returncode != 0:
        return smoke.returncode

    timing_report.write_text(
        f"pip_install_seconds={pip_seconds}\nuv_install_seconds={uv_seconds}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(timing_report.read_text(encoding="utf-8"), end="")
    return 0


def adr030_ratification_steps() -> list[Step]:
    """Return the focused ADR-030 ratification gate sequence."""
    return [
        Step(
            "Private-member scan",
            ["python", "scripts/anti-pattern-analysis/scan_private_usage.py", "tests", "--check"],
        ),
        Step(
            "ADR-030 anti-pattern detector",
            [
                "python",
                "scripts/anti-pattern-analysis/detect_test_anti_patterns.py",
                "--tests-dir",
                "tests",
                "--check",
                "--output",
                "reports/anti-pattern-analysis/test_anti_pattern_report.csv",
                "--report",
                "reports/anti-pattern-analysis/test_quality_report.json",
                "--baseline",
                ".github/test-quality-baseline.json",
            ],
        ),
        Step(
            "ADR-030 test-helper export guard",
            [
                "python",
                "scripts/quality/check_no_test_helper_exports.py",
                "--root",
                "src/calibrated_explanations",
                "--report",
                "reports/anti-pattern-analysis/test_helper_wrapper_report.json",
            ],
        ),
        Step(
            "ADR-030 marker hygiene",
            [
                "python",
                "scripts/quality/check_marker_hygiene.py",
                "--check",
                "--report",
                "reports/marker-hygiene/marker_hygiene_report.json",
                "--baseline",
                ".github/marker-hygiene-baseline.json",
            ],
        ),
        Step(
            "Generated report local-path guard",
            [
                "python",
                "scripts/quality/check_no_local_paths_in_reports.py",
                "--check",
                "--report",
                "reports/quality/no_local_paths_report.json",
            ],
        ),
        Step(
            "Class-surface allowlist gate",
            [
                "python",
                "scripts/quality/check_class_surface_allowlist.py",
                "--check",
                "--report",
                "reports/quality/class_surface_allowlist_report.json",
            ],
        ),
    ]


def adr030_expected_reports() -> list[Path]:
    """Return reports that the ADR-030 ratification lane must produce."""
    return [
        Path("reports/anti-pattern-analysis/private_usage_scan.csv"),
        Path("reports/anti-pattern-analysis/test_anti_pattern_report.csv"),
        Path("reports/anti-pattern-analysis/test_quality_report.json"),
        Path("reports/anti-pattern-analysis/test_helper_wrapper_report.json"),
        Path("reports/marker-hygiene/marker_hygiene_report.json"),
        Path("reports/quality/no_local_paths_report.json"),
        Path("reports/quality/class_surface_allowlist_report.json"),
    ]


def _utc_now_iso() -> str:
    """Return the current UTC timestamp for generated local-check reports."""
    return datetime.now(timezone.utc).isoformat()


def _command_text(command: list[str]) -> str:
    """Return a repo-relative command string for timing reports."""
    display_command = list(command)
    if display_command and Path(display_command[0]) == Path(sys.executable):
        display_command[0] = "python"
    return " ".join(display_command)


def _write_adr030_timing_report(
    records: list[dict[str, object]], started_at: float, output_path: Path
) -> None:
    """Write the focused ADR-030 ratification timing report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "steps": records,
        "total_elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _active_deprecation_rows(ledger_path: Path) -> list[dict[str, str]]:
    """Return data rows from the Active deprecations table."""
    text = ledger_path.read_text(encoding="utf-8")
    try:
        active_section = text.split("### Active deprecations", 1)[1].split(
            "### Removed deprecations (history)",
            1,
        )[0]
    except IndexError:
        raise RuntimeError("Could not locate Active deprecations ledger section") from None

    rows: list[dict[str, str]] = []
    for line in active_section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|---") or "Deprecated symbol" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 5:
            continue
        rows.append(
            {
                "deprecated_symbol": cells[0],
                "replacement": cells[1],
                "deprecated_since": cells[2],
                "removal_eta": cells[3],
                "notes": cells[4],
            }
        )
    return rows


def _write_active_deprecations_report(rows: list[dict[str, str]], output_path: Path) -> int:
    """Write the active-deprecation ledger artifact and return its gate code.

    Any row present in the Active deprecations section is blocking for 1.0.0rc1.
    The Active deprecations section must be empty before the RC gate passes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "status": "pass" if not rows else "fail",
        "active_rows_count": len(rows),
        "blocking_rows_count": len(rows),
        "blocking_symbols": [r["deprecated_symbol"] for r in rows],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    if rows:
        print("ERROR: Active deprecations remain in docs/migration/deprecations.md:")
        for row in rows:
            print(f"  {row['deprecated_symbol']} (ETA: {row['removal_eta']})")
        return 1
    return 0


def _write_deprecation_closure_timing_report(
    records: list[dict[str, object]],
    started_at: float,
    output_path: Path,
) -> None:
    """Write the pre-v1.0 deprecation-closure timing report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "steps": records,
        "total_elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def deprecation_closure_steps() -> list[Step]:
    """Return the pre-v1.0 deprecation-closure validation sequence."""
    return [
        Step(
            "Focused deprecation closure tests",
            _python_cmd(
                "-m",
                "pytest",
                "tests/",
                "-k",
                "deprecat or lime or shap or reject or plugin or parallel or calibration",
                "-v",
                "--no-cov",
            ),
        ),
        Step(
            "ADR-030 ratification lane",
            _python_cmd("scripts/local_checks.py", "--adr030-ratification"),
        ),
    ]


def run_deprecation_closure() -> int:
    """Run the pre-v1.0 deprecation-closure lane and emit timing evidence."""
    ledger_report = Path("reports/deprecations/active_deprecations_check.json")
    timing_report = Path("reports/deprecations/deprecation_closure_timing.json")
    records: list[dict[str, object]] = []
    started_at = time.monotonic()

    step_started_at = time.monotonic()
    try:
        rows = _active_deprecation_rows(Path("docs/migration/deprecations.md"))
        rc = _write_active_deprecations_report(rows, ledger_report)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        rc = 1
    records.append(
        {
            "name": "Active deprecations ledger check",
            "command": "parse docs/migration/deprecations.md",
            "exit_code": rc,
            "elapsed_seconds": round(time.monotonic() - step_started_at, 3),
        }
    )
    _write_deprecation_closure_timing_report(records, started_at, timing_report)
    if rc != 0:
        return rc

    for step in deprecation_closure_steps():
        step_started_at = time.monotonic()
        rc = _run_step(step)
        records.append(
            {
                "name": step.name,
                "command": _command_text(step.command),
                "exit_code": rc,
                "elapsed_seconds": round(time.monotonic() - step_started_at, 3),
            }
        )
        _write_deprecation_closure_timing_report(records, started_at, timing_report)
        if rc != 0:
            return rc

    return 0


def _validate_adr030_ratification_outputs(timing_report: Path) -> int:
    """Validate required ADR-030 ratification artifacts exist and are parseable."""
    missing_reports = [path.as_posix() for path in adr030_expected_reports() if not path.exists()]
    if missing_reports:
        print("ERROR: ADR-030 ratification lane did not produce expected reports:")
        for path in missing_reports:
            print(f"  {path}")
        return 1
    try:
        json.loads(timing_report.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: ADR-030 timing report is not valid JSON: {exc}")
        return 1
    return 0


def run_adr030_ratification() -> int:
    """Run the focused ADR-030 ratification lane and emit timing evidence."""
    timing_report = Path("reports/anti-pattern-analysis/adr030_ratification_timing.json")
    records: list[dict[str, object]] = []
    started_at = time.monotonic()

    for step in adr030_ratification_steps():
        step_started_at = time.monotonic()
        rc = _run_step(step)
        records.append(
            {
                "name": step.name,
                "command": _command_text(step.command),
                "exit_code": rc,
                "elapsed_seconds": round(time.monotonic() - step_started_at, 3),
            }
        )
        _write_adr030_timing_report(records, started_at, timing_report)
        if rc != 0:
            return rc

    return _validate_adr030_ratification_outputs(timing_report)


def _is_network_fetch_failure(stderr: str) -> bool:
    text = (stderr or "").lower()
    if "unable to access 'https://github.com" in text:
        return True
    if "failed to connect to github.com" in text:
        return True
    return "could not connect to server" in text


def _pytest_supports_no_cov() -> bool:
    """Return True if pytest supports the --no-cov option."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--help"], check=False, capture_output=True, text=True
    )
    return "--no-cov" in (result.stdout or "")


def _changed_python_targets() -> list[str]:
    """Return changed Python files for fast lint/format checks.

    Falls back to broad repo paths when git metadata is unavailable or when no
    changed Python files are detected.
    """
    default_targets = ["src", "tests", "scripts"]
    git = shutil.which("git")
    if git is None:
        return default_targets

    commands = [
        [git, "diff", "--name-only", "--relative", "HEAD", "--", "src", "tests", "scripts"],
        [git, "ls-files", "--others", "--exclude-standard", "--", "src", "tests", "scripts"],
    ]
    changed: list[str] = []
    for command in commands:
        result = subprocess.run(command, check=False, capture_output=True, text=True)  # noqa: S603
        if result.returncode != 0:
            return default_targets
        for line in (result.stdout or "").splitlines():
            path = line.strip()
            if path.endswith(".py") and path not in changed:
                changed.append(path)
    return changed or default_targets


def _ruff_check_command(*paths: str) -> list[str]:
    """Return the Ruff lint command for the provided paths."""
    return _python_cmd("-m", "ruff", "check", *paths)


def _ruff_format_check_command(*paths: str) -> list[str]:
    """Return the Ruff format-check command for the provided paths."""
    return _python_cmd("-m", "ruff", "format", "--check", *paths)


def _pytest_no_cov_command(*args: str) -> list[str]:
    """Return a pytest command and add ``--no-cov`` when supported."""
    command = [sys.executable, "-m", "pytest", *args]
    if _pytest_supports_no_cov():
        command.append("--no-cov")
    return command


def _quick_pytest_command() -> list[str]:
    """Return the inner-loop pytest command."""
    return _pytest_no_cov_command(
        "-q",
        "tests/unit",
        "-m",
        "not viz and not slow",
        "--maxfail=1",
        "-o",
        "addopts=",
    )


def _all_non_viz_pytest_command() -> list[str]:
    """Return the PR-scope pytest command."""
    return _pytest_no_cov_command("-q", "-m", "not viz", "-o", "addopts=")


def _quick_steps(mypy_targets: list[str], lint_targets: list[str]) -> list[Step]:
    """Return the inner-loop verification steps."""
    steps: list[Step] = []
    if lint_targets:
        steps.extend(
            [
                Step("Ruff check", _ruff_check_command(*lint_targets)),
                Step("Ruff format check", _ruff_format_check_command(*lint_targets)),
            ]
        )
    if mypy_targets:
        steps.append(
            Step(
                "Mypy (Phase 1B scope)",
                _python_cmd("-m", "mypy", *mypy_targets, "--config-file", "pyproject.toml"),
            )
        )
    steps.extend(
        [
            Step("ADR-001 boundary check", _python_cmd("scripts/quality/check_import_graph.py")),
            Step(
                "ADR-002 compliance check",
                _python_cmd("scripts/quality/check_adr002_compliance.py"),
            ),
            Step("Unit tests (fast/no viz/no slow/no cov)", _quick_pytest_command()),
        ]
    )
    return steps


def _pr_steps(
    mypy_targets: list[str], lint_targets: list[str], *, pre_commit_available: bool
) -> list[Step]:
    """Return the PR-scope verification steps."""
    steps = list(_quick_steps(mypy_targets, lint_targets))
    steps.extend(
        [
            Step(
                "Docstring coverage",
                _python_cmd("scripts/quality/check_docstring_coverage.py", "--fail-under", "94.0"),
            ),
            Step(
                "STD-005 logger domain enforcement",
                _python_cmd(
                    "scripts/quality/check_logging_domains.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--report",
                    "reports/quality/logging_domain_report.json",
                ),
            ),
            Step(
                "STD-001 nomenclature enforcement",
                _python_cmd(
                    "scripts/quality/check_std001_nomenclature.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--report",
                    "reports/nomenclature_violation_inventory.json",
                    "--check",
                ),
            ),
            Step(
                "Parameter naming CI guard (removed aliases)",
                _python_cmd(
                    "scripts/quality/check_parameter_naming.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--check",
                ),
            ),
            Step(
                "Capability-chain validator",
                _python_cmd("scripts/quality/validate_capability_chain.py", "--check"),
            ),
            Step(
                "Raw evidence structural validation",
                _python_cmd("scripts/generate_tif_evidence.py", "--validate-existing"),
            ),
            Step(
                "Private-member scan",
                _python_cmd(
                    "scripts/anti-pattern-analysis/scan_private_usage.py", "tests", "--check"
                ),
            ),
            Step(
                "ADR-030 anti-pattern detector",
                _python_cmd(
                    "scripts/anti-pattern-analysis/detect_test_anti_patterns.py",
                    "--tests-dir",
                    "tests",
                    "--check",
                    "--output",
                    "reports/anti-pattern-analysis/test_anti_pattern_report.csv",
                    "--report",
                    "reports/anti-pattern-analysis/test_quality_report.json",
                    "--baseline",
                    ".github/test-quality-baseline.json",
                ),
            ),
            Step(
                "ADR-030 test-helper export guard",
                _python_cmd(
                    "scripts/quality/check_no_test_helper_exports.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--report",
                    "reports/anti-pattern-analysis/test_helper_wrapper_report.json",
                ),
            ),
            Step(
                "Class-surface allowlist gate",
                _python_cmd(
                    "scripts/quality/check_class_surface_allowlist.py",
                    "--check",
                    "--report",
                    "reports/quality/class_surface_allowlist_report.json",
                ),
            ),
            Step(
                "ADR-006 trust-mutation primitive guard",
                _python_cmd(
                    "scripts/quality/check_trust_mutation_primitive.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--report",
                    "reports/trust_mutation_inventory.json",
                    "--check",
                ),
            ),
            Step(
                "ADR-034 ConfigManager usage guard",
                _python_cmd(
                    "scripts/quality/check_config_manager_usage.py",
                    "--root",
                    "src/calibrated_explanations",
                    "--scope",
                    "runtime",
                    "--report",
                    "reports/quality/config_manager_usage_report.json",
                    "--check",
                ),
            ),
            Step(
                "ADR-030 marker hygiene",
                _python_cmd(
                    "scripts/quality/check_marker_hygiene.py",
                    "--check",
                    "--report",
                    "reports/marker-hygiene/marker_hygiene_report.json",
                    "--baseline",
                    ".github/marker-hygiene-baseline.json",
                ),
            ),
            Step(
                "Generated report local-path guard",
                _python_cmd(
                    "scripts/quality/check_no_local_paths_in_reports.py",
                    "--check",
                    "--report",
                    "reports/quality/no_local_paths_report.json",
                ),
            ),
            Step(
                "Forbidden doc-pattern gates",
                _python_cmd(
                    "scripts/quality/check_forbidden_doc_patterns.py",
                    "--all-checks",
                ),
            ),
            Step(
                "Release-profile meta-test guard",
                _python_cmd(
                    "scripts/quality/check_no_release_profile_meta_tests.py",
                    "--report",
                    "reports/quality/meta_test_guard.json",
                ),
            ),
            Step("All non-viz tests (no coverage)", _all_non_viz_pytest_command()),
        ]
    )
    if pre_commit_available:
        steps.append(Step("Pre-commit", ["pre-commit", "run", "--all-files"]))
    return steps


def _full_steps(
    mypy_targets: list[str], lint_targets: list[str], *, pre_commit_available: bool
) -> list[Step]:
    """Return the heavier main/full verification steps."""
    steps = list(_pr_steps(mypy_targets, lint_targets, pre_commit_available=pre_commit_available))
    steps.extend(
        [
            Step(
                "Docs build (HTML)",
                [sys.executable, "-m", "sphinx", "-b", "html", "docs", "docs/_build/html"],
            ),
            Step(
                "Core tests with coverage",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    "--cov=src/calibrated_explanations",
                    "--cov-config=pyproject.toml",
                    "--cov-report=xml:coverage.xml",
                    "--cov-context=test",
                    "--cov-fail-under=90",
                ],
            ),
            Step(
                "Per-module coverage gates",
                _python_cmd("scripts/quality/check_coverage_gates.py", "coverage.xml"),
            ),
            Step("Micro benchmark", _python_cmd("scripts/perf/micro_bench_perf.py")),
            Step(
                "Perf thresholds",
                _python_cmd(
                    "scripts/perf/check_perf_micro.py",
                    "tests/benchmarks/micro_current.json",
                    "tests/benchmarks/perf_thresholds.json",
                ),
            ),
            Step(
                "Over-testing coverage contexts",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    "--cov=src/calibrated_explanations",
                    "--cov-config=pyproject.toml",
                    "--cov-context=test",
                    "--no-cov-on-fail",
                ],
            ),
            Step(
                "Over-testing report",
                _python_cmd(
                    "scripts/over_testing/over_testing_report.py",
                    "--require-multiple-contexts",
                    "--output-lines",
                    "reports/over_testing/line_coverage_counts.csv",
                    "--output-blocks",
                    "reports/over_testing/block_coverage_counts.csv",
                    "--output-summary",
                    "reports/over_testing/summary.json",
                    "--output-metadata",
                    "reports/over_testing/metadata.json",
                ),
            ),
            Step(
                "Redundant tests report",
                _python_cmd("scripts/over_testing/detect_redundant_tests.py"),
            ),
        ]
    )
    return steps


def _release_steps(
    mypy_targets: list[str], lint_targets: list[str], *, pre_commit_available: bool
) -> list[Step]:
    """Return the release-oriented verification steps."""
    steps = list(_full_steps(mypy_targets, lint_targets, pre_commit_available=pre_commit_available))
    steps.extend(
        [
            Step(
                "Dependency audit",
                _dependency_audit_command(),
            ),
            Step(
                "Notebook audit",
                _python_cmd(
                    "scripts/quality/audit_notebook_api.py",
                    "notebooks",
                    "--json",
                    "artifacts/notebook_audit.json",
                ),
            ),
            Step(
                "Notebook execution report",
                _python_cmd(
                    "scripts/docs/run_notebooks.py",
                    "--mode",
                    "advisory",
                    "--output",
                    "reports/docs/notebook_execution_report.json",
                ),
            ),
            Step(
                "Strict docs build",
                [sys.executable, "-m", "sphinx", "-W", "--keep-going", "docs", "docs/_build/html"],
            ),
            Step(
                "Docs linkcheck",
                [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "-b",
                    "linkcheck",
                    "-D",
                    "nbsphinx_execute=never",
                    "docs",
                    "docs/_build/linkcheck",
                ],
            ),
            Step(
                "Warning policy",
                _python_cmd(
                    "scripts/quality/check_warning_policy.py",
                    "--check",
                    "--report",
                    "reports/quality/warning_policy.json",
                ),
            ),
            Step(
                "Deprecation closure",
                _python_cmd("scripts/local_checks.py", "--deprecation-closure"),
            ),
            Step(
                "Version alignment",
                _python_cmd(
                    "scripts/quality/check_version_alignment.py",
                    "--check",
                    "--allow-normalized",
                ),
            ),
            Step("Release packaging build", _python_cmd("-m", "build")),
            Step(
                "Release packaging artifact smoke",
                _python_cmd(
                    "scripts/quality/check_packaging_artifacts.py",
                    "--clean-dist",
                    "--report",
                    "reports/packaging/license_artifact_check.json",
                ),
            ),
            Step(
                "Capability evidence refresh",
                _python_cmd("scripts/generate_tif_evidence.py", "--check-current"),
            ),
            Step("Public API snapshot", _python_cmd("scripts/quality/snapshot_public_api.py")),
        ]
    )
    return steps


RELEASE_PREFLIGHT_REPORT = Path("reports/local_checks/release_preflight_report.json")
RELEASE_MANUAL_STEP_RANGE = tuple(range(11, 18))
RELEASE_STANDARD_NOTEBOOK_PATTERNS: tuple[str, ...] = (
    "notebooks/quickstart.ipynb",
    "notebooks/quickstart_guarded.ipynb",
    "notebooks/quickstart_tiny.ipynb",
    "notebooks/core_demos/*.ipynb",
    "notebooks/miscellaneous/*.ipynb",
    "notebooks/paper_based/*.ipynb",
    "notebooks/advanced/demo_conditional.ipynb",
    "notebooks/advanced/demo_config_management.ipynb",
    "notebooks/advanced/demo_narrative_explanations.ipynb",
    "notebooks/advanced/demo_plugin_wiring.ipynb",
    "notebooks/advanced/demo_reject.ipynb",
    "notebooks/advanced/demo_under_the_hood.ipynb",
)
RELEASE_SLOW_NOTEBOOK_PATTERNS: tuple[str, ...] = (
    "notebooks/advanced/fast_feature_filtering_demo.ipynb",
)


def _resolve_release_notebook_paths(patterns: tuple[str, ...]) -> list[str]:
    """Resolve release notebook globs into deterministic repo-relative paths."""
    resolved_paths: list[str] = []
    missing_patterns: list[str] = []
    for pattern in patterns:
        if any(token in pattern for token in "*?["):
            matches = sorted(Path().glob(pattern))
        else:
            candidate = Path(pattern)
            matches = [candidate] if candidate.exists() else []
        if not matches:
            missing_patterns.append(pattern)
            continue
        for match in matches:
            resolved_paths.append(match.as_posix())
    if missing_patterns:
        joined = ", ".join(sorted(missing_patterns))
        raise RuntimeError(f"Release notebook patterns did not resolve to files: {joined}")
    return resolved_paths


def _release_notebook_steps() -> list[Step]:
    """Return the strict in-place notebook execution steps for releases."""
    standard_paths = _resolve_release_notebook_paths(RELEASE_STANDARD_NOTEBOOK_PATTERNS)
    slow_paths = _resolve_release_notebook_paths(RELEASE_SLOW_NOTEBOOK_PATTERNS)
    steps = [
        Step(
            "Release notebooks (saved in-place)",
            _python_cmd(
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                *standard_paths,
            ),
        )
    ]
    if slow_paths:
        steps.append(
            Step(
                "Release slow notebook (saved in-place)",
                _python_cmd(
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=5400",
                    *slow_paths,
                ),
            )
        )
    return steps


def _git_text(*args: str) -> str | None:
    """Return trimmed stdout from a git command, or ``None`` when unavailable."""
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(  # noqa: S603
        [git, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _current_git_branch() -> str | None:
    """Return the current git branch, if available."""
    return _git_text("branch", "--show-current")


def _current_git_status_porcelain() -> str | None:
    """Return the current git status snapshot used by release-finalize."""
    return _git_text("status", "--short")


def _pyproject_release_version() -> str:
    """Return the raw project version from pyproject.toml."""
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def _release_task_checklist_state(
    plan_path: Path,
) -> tuple[dict[int, dict[str, int | bool]], list[str]]:
    """Parse per-task verification checklist completion state from the release plan."""
    text = plan_path.read_text(encoding="utf-8")

    header_pattern = re.compile(r"^##\s+(\d+)\)\s+", re.MULTILINE)
    checklist_header_pattern = re.compile(
        r"^###\s+\d+\.\d+\s+Verification checklist\s*$", re.MULTILINE
    )
    checklist_item_pattern = re.compile(r"^\s*[-*]\s*\[([ xX])\]\s+", re.MULTILINE)

    task_sections = list(header_pattern.finditer(text))
    if not task_sections:
        raise ValueError(f"Could not locate any numbered task sections in {plan_path.as_posix()}.")

    states: dict[int, dict[str, int | bool]] = {}
    parse_errors: list[str] = []

    for index, match in enumerate(task_sections):
        task_id = int(match.group(1))
        start = match.start()
        end = task_sections[index + 1].start() if index + 1 < len(task_sections) else len(text)
        section = text[start:end]

        checklist_header = checklist_header_pattern.search(section)
        if checklist_header is None:
            parse_errors.append(
                f"Task {task_id} is missing a 'Verification checklist' subsection in {plan_path.as_posix()}."
            )
            continue

        checklist_text = section[checklist_header.end() :]
        checklist_marks = checklist_item_pattern.findall(checklist_text)
        if not checklist_marks:
            parse_errors.append(
                f"Task {task_id} verification checklist has no checkbox items in {plan_path.as_posix()}."
            )
            continue

        total_items = len(checklist_marks)
        checked_items = sum(1 for mark in checklist_marks if mark.lower() == "x")
        states[task_id] = {
            "all_items_checked": checked_items == total_items,
            "checked_items": checked_items,
            "total_items": total_items,
        }

    return states, parse_errors


def _evaluate_release_plan_readiness(plan_path: Path) -> tuple[dict[str, object], list[str]]:
    """Evaluate whether Task 45 prerequisites are satisfied for release handoff."""
    task_checklist_state, parse_errors = _release_task_checklist_state(plan_path)
    branch = _current_git_branch()
    pyproject_version = _pyproject_release_version()
    observed: dict[str, object] = {
        "plan_path": plan_path.as_posix(),
        "branch": branch,
        "pyproject_version": pyproject_version,
        "task_checklist_state": task_checklist_state,
        "parse_errors": parse_errors,
    }

    errors = list(parse_errors)
    if branch is None:
        errors.append("Git is unavailable; release-preflight requires a git checkout.")
    elif branch != "main":
        errors.append(
            f"Release-preflight must run from the main branch; current branch is {branch!r}."
        )

    if "dev" in pyproject_version.lower():
        errors.append(
            "pyproject.toml still carries a development version. Bump to the release version before preflight."
        )

    for task_id in range(1, 45):
        state = task_checklist_state.get(task_id)
        if state is None:
            errors.append(
                f"Task {task_id} verification checklist state is unavailable for release handoff."
            )
            continue
        if not bool(state["all_items_checked"]):
            open_items = int(state["total_items"]) - int(state["checked_items"])
            errors.append(
                "Task "
                f"{task_id} verification checklist is not closed for release handoff: "
                f"{open_items}/{state['total_items']} items unchecked."
            )
    return observed, errors


def _run_release_readiness_guard(plan_path: Path) -> tuple[int, dict[str, object]]:
    """Run the Task 45 readiness guard and return the observed state."""
    observed, errors = _evaluate_release_plan_readiness(plan_path)
    print("\n[Release readiness guard]")
    print(f"Active release plan: {plan_path.as_posix()}")
    if errors:
        print("ERROR: release handoff prerequisites are not satisfied:")
        for error in errors:
            print(f"- {error}")
        return 1, observed
    print("PASS: release plan summary, branch, and version state are ready for Task 45.")
    return 0, observed


def _write_release_preflight_report(
    records: list[dict[str, object]],
    started_at: float,
    output_path: Path,
    *,
    observed: dict[str, object],
    exit_status: int,
    git_status_porcelain: str | None,
) -> None:
    """Write the strict release-preflight report consumed by release-finalize."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "plan_path": observed.get("plan_path"),
        "branch": observed.get("branch"),
        "pyproject_version": observed.get("pyproject_version"),
        "task_checklist_state": observed.get("task_checklist_state", {}),
        "parse_errors": observed.get("parse_errors", []),
        "steps": records,
        "exit_status": exit_status,
        "preflight_passed": exit_status == 0,
        "git_status_porcelain": git_status_porcelain,
        "manual_release_steps": list(RELEASE_MANUAL_STEP_RANGE),
        "total_elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _run_release_twine_check() -> int:
    """Validate built release artifacts with twine."""
    artifacts = sorted(path.as_posix() for path in Path("dist").glob("*"))
    if not artifacts:
        print("ERROR: No build artifacts found under dist/ for twine validation.")
        return 1
    return _run_step(
        Step("Release artifact validation", _python_cmd("-m", "twine", "check", *artifacts))
    )


def _run_release_wheel_smoke() -> int:
    """Smoke-test the built wheel in an isolated temporary virtual environment."""
    if not any(Path("dist").glob("*.whl")):
        print("ERROR: No wheel artifacts found under dist/ for the release smoke test.")
        return 1

    run_dir = Path(tempfile.mkdtemp(prefix="ce-release-wheel-smoke-")).resolve()
    venv_dir = run_dir / "venv-wheel"
    builder = venv.EnvBuilder(with_pip=True, clear=True)
    builder.create(venv_dir)
    venv_python = _venv_python(venv_dir)
    smoke_steps = [
        Step(
            "Wheel smoke: upgrade pip",
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        ),
        Step(
            "Wheel smoke: install release artifact",
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "calibrated_explanations",
                "--find-links",
                str(Path("dist").resolve()),
            ],
        ),
        Step(
            "Wheel smoke: import release artifact",
            [
                str(venv_python),
                "-c",
                "import calibrated_explanations as ce; print(ce.__version__)",
            ],
        ),
    ]
    for step in smoke_steps:
        rc = _run_step(step)
        if rc != 0:
            return rc
    return 0


def run_release_preflight(*, plan_path: Path | None = None) -> int:
    """Run the strict pre-step-11 release gate and persist a handoff snapshot."""
    resolved_plan = _resolved_plan_path(plan_path)
    records: list[dict[str, object]] = []
    started_at = time.monotonic()

    def record_step(name: str, command: str, rc: int, step_started_at: float) -> None:
        records.append(
            {
                "name": name,
                "command": command,
                "exit_code": rc,
                "elapsed_seconds": round(time.monotonic() - step_started_at, 3),
            }
        )

    readiness_started_at = time.monotonic()
    readiness_rc, observed = _run_release_readiness_guard(resolved_plan)
    record_step(
        "Release readiness guard",
        f"parse {resolved_plan.as_posix()} release gate summary",
        readiness_rc,
        readiness_started_at,
    )
    git_status_porcelain = _current_git_status_porcelain()
    _write_release_preflight_report(
        records,
        started_at,
        RELEASE_PREFLIGHT_REPORT,
        observed=observed,
        exit_status=readiness_rc,
        git_status_porcelain=git_status_porcelain,
    )
    if readiness_rc != 0:
        return readiness_rc

    steps = [
        Step("Full pytest suite", _python_cmd("-m", "pytest", "-q")),
        Step(
            "Editable install (release tree)", _python_cmd("-m", "pip", "install", "-e", ".[dev]")
        ),
        Step(
            "Editable install version smoke",
            _python_cmd("-c", "import calibrated_explanations as ce; print(ce.__version__)"),
        ),
        *_release_notebook_steps(),
        Step("Release profile", _python_cmd("scripts/local_checks.py", "--profile", "release")),
    ]
    for step in steps:
        step_started_at = time.monotonic()
        rc = _run_step(step)
        record_step(step.name, _command_text(step.command), rc, step_started_at)
        git_status_porcelain = _current_git_status_porcelain()
        _write_release_preflight_report(
            records,
            started_at,
            RELEASE_PREFLIGHT_REPORT,
            observed=observed,
            exit_status=rc,
            git_status_porcelain=git_status_porcelain,
        )
        if rc != 0:
            return rc

    twine_started_at = time.monotonic()
    twine_rc = _run_release_twine_check()
    record_step(
        "Release artifact validation",
        "python -m twine check dist/*",
        twine_rc,
        twine_started_at,
    )
    git_status_porcelain = _current_git_status_porcelain()
    _write_release_preflight_report(
        records,
        started_at,
        RELEASE_PREFLIGHT_REPORT,
        observed=observed,
        exit_status=twine_rc,
        git_status_porcelain=git_status_porcelain,
    )
    if twine_rc != 0:
        return twine_rc

    wheel_started_at = time.monotonic()
    wheel_rc = _run_release_wheel_smoke()
    record_step(
        "Release wheel smoke",
        "create temp venv; install calibrated_explanations from dist; import version",
        wheel_rc,
        wheel_started_at,
    )
    git_status_porcelain = _current_git_status_porcelain()
    _write_release_preflight_report(
        records,
        started_at,
        RELEASE_PREFLIGHT_REPORT,
        observed=observed,
        exit_status=wheel_rc,
        git_status_porcelain=git_status_porcelain,
    )
    return wheel_rc


def run_release_finalize(*, plan_path: Path | None = None) -> int:
    """Verify that release-preflight is still valid before manual release steps."""
    resolved_plan = _resolved_plan_path(plan_path)
    print("\n[Release finalize guard]")
    if not RELEASE_PREFLIGHT_REPORT.exists():
        print(
            "ERROR: No release-preflight report found. Run `make release-preflight` before the manual release phase."
        )
        return 1

    try:
        payload = json.loads(RELEASE_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: release-preflight report is not valid JSON: {exc}")
        return 1

    if payload.get("exit_status") != 0 or not payload.get("preflight_passed"):
        print(
            "ERROR: The latest release-preflight run did not pass. Re-run `make release-preflight`."
        )
        return 1

    readiness_rc, _ = _run_release_readiness_guard(resolved_plan)
    if readiness_rc != 0:
        return readiness_rc

    current_branch = _current_git_branch()
    if current_branch != payload.get("branch"):
        print(
            "ERROR: The current branch no longer matches the successful preflight snapshot. "
            f"Current branch: {current_branch!r}; preflight branch: {payload.get('branch')!r}."
        )
        return 1

    current_status = _current_git_status_porcelain()
    if current_status != payload.get("git_status_porcelain"):
        print(
            "ERROR: The worktree changed after release-preflight. Re-run `make release-preflight` before step 11."
        )
        return 1

    print(
        "PASS: release-preflight is still valid. Continue with the manual release phase "
        f"(steps {RELEASE_MANUAL_STEP_RANGE[0]}-{RELEASE_MANUAL_STEP_RANGE[-1]} in the private runbook)."
    )
    return 0


def _common_skipped_heavy_gates() -> list[dict[str, str]]:
    """Return the heavy gates excluded from quick/task/pr by default."""
    return [
        {"gate": "coverage", "reason": "Reserved for full/release validation."},
        {"gate": "docs_html", "reason": "Reserved for full/release validation."},
        {"gate": "docs_linkcheck", "reason": "Reserved for release validation."},
        {"gate": "notebook_audit", "reason": "Reserved for release validation."},
        {"gate": "notebook_execution", "reason": "Reserved for release validation."},
        {"gate": "dependency_audit", "reason": "Reserved for release validation."},
        {"gate": "performance", "reason": "Reserved for full/release validation."},
        {"gate": "capability_evidence_refresh", "reason": "Reserved for release validation."},
    ]


ACTIVE_RELEASE_PLAN = Path("development/current-work/v0.11.6_plan.md")
_TASK_VERIFICATION_BLOCK_RE = re.compile(
    r"```toml ce-task-verification\r?\n(?P<body>.*?)```",
    re.DOTALL,
)


def _load_task_verification_config(plan_path: Path) -> dict[str, object]:
    """Parse the task-verification TOML block from a release plan."""
    plan_text = plan_path.read_text(encoding="utf-8")
    match = _TASK_VERIFICATION_BLOCK_RE.search(plan_text)
    if match is None:
        raise ValueError(
            f"Could not find a ```toml ce-task-verification``` block in {plan_path.as_posix()}."
        )
    config = tomllib.loads(match.group("body"))
    if config.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported ce-task-verification schema in {plan_path.as_posix()}: "
            f"{config.get('schema_version')!r}."
        )
    task_table = config.get("task")
    if not isinstance(task_table, dict) or not task_table:
        raise ValueError(f"No task mappings found in {plan_path.as_posix()}.")
    return config


def _resolved_plan_path(plan_path: Path | None) -> Path:
    """Return the active plan path or an explicit override."""
    return ACTIVE_RELEASE_PLAN if plan_path is None else plan_path


def _task_verification_tasks(plan_path: Path | None = None) -> dict[int, dict[str, object]]:
    """Return normalized task-verification entries keyed by task id."""
    config = _load_task_verification_config(_resolved_plan_path(plan_path))
    raw_tasks = config["task"]
    assert isinstance(raw_tasks, dict)
    tasks: dict[int, dict[str, object]] = {}
    for raw_id, raw_payload in raw_tasks.items():
        if not isinstance(raw_payload, dict):
            raise ValueError(f"Task mapping for {raw_id!r} must be a table.")
        task_id = int(raw_id)
        tasks[task_id] = raw_payload
    return tasks


def supported_task_ids(*, plan_path: Path | None = None) -> frozenset[int]:
    """Return task ids mapped in the selected release plan."""
    return frozenset(_task_verification_tasks(plan_path).keys())


def _unsupported_task_mapping_error(task: int, plan_path: Path | None = None) -> ValueError:
    supported = sorted(supported_task_ids(plan_path=plan_path))
    return ValueError(f"Unsupported task profile mapping: {task}. Supported task ids: {supported}.")


def _command_from_plan(command: str) -> list[str]:
    """Parse a task-plan command string into an executable argument vector."""
    parts = shlex.split(command, posix=True)
    if not parts:
        raise ValueError("Task verification command entries must not be empty.")
    if parts[0] == "python":
        return _python_cmd(*parts[1:])
    return parts


def _step_from_plan(step_config: dict[str, object]) -> Step:
    """Build a local-check step from one plan step table."""
    name = step_config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Each task-verification step must define a non-empty 'name'.")
    if "pytest" in step_config:
        pytest_paths = step_config["pytest"]
        pytest_args = step_config.get("pytest_args", [])
        if not isinstance(pytest_paths, list) or not all(
            isinstance(path, str) and path for path in pytest_paths
        ):
            raise ValueError(f"{name}: pytest steps must define a non-empty string list.")
        if not isinstance(pytest_args, list) or not all(
            isinstance(arg, str) for arg in pytest_args
        ):
            raise ValueError(f"{name}: pytest_args must be a string list when present.")
        return Step(
            name,
            _pytest_no_cov_command("-q", *pytest_paths, *pytest_args, "-o", "addopts="),
        )
    command = step_config.get("command")
    if isinstance(command, str) and command:
        return Step(name, _command_from_plan(command))
    raise ValueError(f"{name}: task-verification steps must define either pytest or command.")


def _task_specific_steps(task: int, *, plan_path: Path | None = None) -> list[Step]:
    """Return the focused validation steps for a mapped release-plan task."""
    tasks = _task_verification_tasks(plan_path)
    if task not in tasks:
        raise _unsupported_task_mapping_error(task, plan_path)
    raw_steps = tasks[task].get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"Task {task} has no step definitions in the verification block.")
    return [
        _step_from_plan(step_config) for step_config in raw_steps if isinstance(step_config, dict)
    ]


def _task_specific_lint_targets(task: int, *, plan_path: Path | None = None) -> list[str]:
    """Return lint/format targets scoped to one mapped release-plan task."""
    tasks = _task_verification_tasks(plan_path)
    if task not in tasks:
        raise _unsupported_task_mapping_error(task, plan_path)
    task_config = tasks[task]
    if "lint_targets" not in task_config:
        return _changed_python_targets()
    lint_targets = task_config["lint_targets"]
    if not isinstance(lint_targets, list) or not all(
        isinstance(target, str) and target for target in lint_targets
    ):
        raise ValueError(f"Task {task} lint_targets must be a string list when present.")
    return lint_targets


def task_specific_steps(task: int, *, plan_path: Path | None = None) -> list[Step]:
    """Public accessor for the focused validation steps of one task."""
    return _task_specific_steps(task, plan_path=plan_path)


def task_specific_lint_targets(task: int, *, plan_path: Path | None = None) -> list[str]:
    """Public accessor for the lint/format targets of one task."""
    return _task_specific_lint_targets(task, plan_path=plan_path)


def _step_runs_docs_html(step: Step) -> bool:
    """Return True when a task step already runs a Sphinx HTML build."""
    command = step.command
    if len(command) < 2:
        return False
    if command[0] != sys.executable:
        return False
    if "-m" not in command:
        return False
    try:
        module_name = command[command.index("-m") + 1]
    except IndexError:
        return False
    return module_name == "sphinx" and "docs/_build/html" in command


def build_profile_plan(
    profile: str,
    *,
    task: int | None,
    mypy_targets: list[str],
    lint_targets: list[str],
    pre_commit_available: bool,
    plan_path: Path | None = None,
) -> ProfilePlan:
    """Build the local-check step plan for the requested profile."""
    skipped_heavy = _common_skipped_heavy_gates()
    if profile == "quick":
        return ProfilePlan(
            profile="quick",
            steps=_quick_steps(mypy_targets, lint_targets),
            skipped_heavy_gates=skipped_heavy,
        )
    if profile == "task":
        if task is None:
            raise ValueError("The task profile requires --task / TASK=<n>.")
        task_steps = _task_specific_steps(task, plan_path=plan_path)
        task_skips = [
            entry
            for entry in skipped_heavy
            if not (
                entry["gate"] == "docs_html"
                and any(_step_runs_docs_html(step) for step in task_steps)
            )
        ]
        return ProfilePlan(
            profile="task",
            task=task,
            steps=[*_quick_steps(mypy_targets, lint_targets), *task_steps],
            skipped_heavy_gates=task_skips,
        )
    if profile == "pr":
        return ProfilePlan(
            profile="pr",
            steps=_pr_steps(mypy_targets, lint_targets, pre_commit_available=pre_commit_available),
            skipped_heavy_gates=skipped_heavy,
        )
    if profile == "full":
        return ProfilePlan(
            profile="full",
            steps=_full_steps(
                mypy_targets, lint_targets, pre_commit_available=pre_commit_available
            ),
            skipped_heavy_gates=[],
        )
    if profile == "release":
        return ProfilePlan(
            profile="release",
            steps=_release_steps(
                mypy_targets, lint_targets, pre_commit_available=pre_commit_available
            ),
            skipped_heavy_gates=[],
        )
    raise ValueError(f"Unknown profile: {profile}")


def _write_task_profile_report(
    plan: ProfilePlan,
    *,
    commands_run: list[str],
    exit_status: int,
    output_path: Path,
    requested_paths: list[str] | None = None,
) -> None:
    """Write the machine-readable task-profile execution report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "selected_profile": plan.profile,
        "task_id": plan.task,
        "commands_run": commands_run,
        "skipped_heavy_gates": plan.skipped_heavy_gates,
        "requested_paths": requested_paths or [],
        "exit_status": exit_status,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def run_profile_plan(
    plan: ProfilePlan,
    *,
    task_report_path: Path | None = None,
    requested_paths: list[str] | None = None,
) -> int:
    """Run a planned profile and optionally persist a task-profile report."""
    commands_run: list[str] = []
    exit_status = 0
    for step in plan.steps:
        commands_run.append(_command_text(step.command))
        rc = _run_step(step)
        if rc != 0:
            exit_status = rc
            break
    if task_report_path is not None:
        _write_task_profile_report(
            plan,
            commands_run=commands_run,
            exit_status=exit_status,
            output_path=task_report_path,
            requested_paths=requested_paths,
        )
    return exit_status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local verification profiles in the current environment."
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "task", "pr", "full", "release"),
        help="Verification profile to run. Defaults to 'full' unless --skip-main is used.",
    )
    parser.add_argument("--task", type=int, help="Focused v0.11.6 task id for --profile task.")
    parser.add_argument(
        "--plan",
        type=Path,
        help="Optional release-plan override containing the ce-task-verification block.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Optional changed-path hints for task-focused reporting/routing.",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help="Compatibility alias for --profile pr.",
    )
    parser.add_argument(
        "--ci-parity",
        action="store_true",
        help="Run the local workflow run-block smoke runner (not an exact GitHub Actions emulator).",
    )
    parser.add_argument(
        "--uv-install-smoke",
        action="store_true",
        help="Run the optional uv install smoke and pip-vs-uv install timing lane.",
    )
    parser.add_argument(
        "--adr030-ratification",
        action="store_true",
        help="Run the focused ADR-030 ratification lane and timing report.",
    )
    parser.add_argument(
        "--deprecation-closure",
        action="store_true",
        help="Run the pre-v1.0 deprecation-closure lane and timing report.",
    )
    parser.add_argument(
        "--release-preflight",
        action="store_true",
        help="Run the strict pre-step-11 release gate and write the handoff report.",
    )
    parser.add_argument(
        "--release-finalize",
        action="store_true",
        help="Validate the latest release-preflight snapshot before manual release steps.",
    )
    args = parser.parse_args()

    if args.uv_install_smoke:
        return _run_uv_install_smoke()
    if args.adr030_ratification:
        return run_adr030_ratification()
    if args.deprecation_closure:
        return run_deprecation_closure()
    if args.release_preflight:
        return run_release_preflight(plan_path=args.plan)
    if args.release_finalize:
        return run_release_finalize(plan_path=args.plan)

    if args.ci_parity:
        print("Run-block smoke mode: delegating to scripts/run_ci_locally.py")
        shell_arg = "bash"
        if os.name == "nt":
            shell_arg = "bash"
        with suppress(Exception):
            for cov in Path(".").glob(".coverage*"):
                with suppress(Exception):
                    cov.unlink()
        return subprocess.call([sys.executable, "scripts/run_ci_locally.py", "--shell", shell_arg])  # noqa: S603

    selected_profile = args.profile or ("pr" if args.skip_main else "full")
    mypy_available = (
        shutil.which("mypy") is not None
        or subprocess.call(
            [sys.executable, "-m", "mypy", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )
    if not mypy_available:
        print("ERROR: mypy not found in current environment.")
        return 2

    pre_commit_available = shutil.which("pre-commit") is not None
    if selected_profile in {"pr", "full", "release"} and not pre_commit_available:
        print("WARNING: pre-commit not found in current environment; skipping the pre-commit step.")

    if not _pytest_supports_no_cov():
        print("WARNING: pytest-cov/--no-cov unavailable; running pytest commands without --no-cov.")

    mypy_targets = _mypy_targets()
    if not mypy_targets:
        print("No mypy target files found; skipping mypy step.")
    try:
        lint_targets = (
            _task_specific_lint_targets(args.task, plan_path=args.plan)
            if selected_profile == "task" and args.task is not None
            else _changed_python_targets()
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        build_kwargs = {
            "task": args.task,
            "mypy_targets": mypy_targets,
            "lint_targets": lint_targets,
            "pre_commit_available": pre_commit_available,
        }
        if args.plan is not None:
            build_kwargs["plan_path"] = args.plan
        plan = build_profile_plan(
            selected_profile,
            **build_kwargs,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    task_report_path = (
        Path("reports/local_checks/task_profile_report.json")
        if selected_profile == "task"
        else None
    )
    rc = run_profile_plan(plan, task_report_path=task_report_path, requested_paths=args.paths)
    if rc != 0:
        return rc

    print(f"\nLocal checks completed ({selected_profile} profile).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
