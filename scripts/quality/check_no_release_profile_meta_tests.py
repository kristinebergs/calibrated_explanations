"""Guard against release-specific local-check profile meta-tests (v0.11.6 Task 29).

Tests must protect stable product behavior, stable public developer tooling
contracts, or durable quality gates. They must not assert the internal
composition of transient release-plan verification wiring: local-check profile
step names, task-profile mappings, skipped-heavy-gate lists, release-plan task
ids, or machine-readable task-profile report internals. That meta-testing
fossilizes one release plan's temporary orchestration details and forces
unrelated test rewrites on every plan edit.

This guard scans ``tests/`` for the obvious markers of that anti-pattern and
fails when any non-allowlisted test file references them. Durable
checker-behavior tests (a quality script failing on a malformed fixture, a
packaging checker detecting missing package data, and so on) do not need these
markers and are unaffected.

Allowlist policy: entries must be explicit repo-relative paths, kept narrow,
and each must carry a reason. Do not allowlist a file to preserve profile or
task-routing assertions; that is the exact pattern this guard prohibits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Repo-relative paths exempt from scanning, with the reason they are exempt.
ALLOWLIST: dict[str, str] = {
    # The guard's own behavior tests must embed prohibited tokens inside
    # tmp-dir fixture content to prove the guard detects them.
    "tests/scripts/test_check_no_release_profile_meta_tests.py": (
        "fixture content for this guard's own detection tests"
    ),
    # Durable capability-governance contract, not release-task wiring: gap-status
    # requirements must reference an entry in the archived status appendix
    # (development/finished-work/RELEASE_PLAN_status_appendix.md), mirroring
    # validate_capability_chain.py.
    # It asserts no profile steps, task ids, or milestone-plan mechanics.
    "tests/capabilities/test_adr_capability_links.py": (
        "capability evidence-chain contract references the status appendix"
    ),
    # Durable release-file I/O contract test: run_release_preflight and
    # run_release_postcommit read/write real plan-file paths as their stable
    # public behavior. It asserts no profile steps, task ids, or transient
    # plan-orchestration internals.
    "tests/scripts/test_local_checks_release_workflow.py": (
        "release-file I/O contract for run_release_preflight/"
        "run_release_postcommit; asserts stable path-handling behavior, not"
        " transient plan task-id/profile-step wiring"
    ),
}

PROHIBITED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("profile-planner internals (build_profile_plan)", re.compile(r"\bbuild_profile_plan\b")),
    ("profile-planner internals (ProfilePlan)", re.compile(r"\bProfilePlan\b")),
    ("skipped-heavy-gate internals", re.compile(r"\bskipped_heavy_gates\b")),
    ("task-profile report internals", re.compile(r"\btask_profile_report\b")),
    ("release task-profile invocation (TASK=<n>)", re.compile(r"\bTASK=\d+\b")),
    ("release-task step-name assertion", re.compile(r"""["']Task \d+ """)),
    (
        "release-plan file reference",
        re.compile(r"\bACTIVE_RELEASE_PLAN\b|\w*_plan\.md|\bRELEASE_PLAN\w*\.md"),
    ),
)


def scan_file(path: Path, *, repo_root: Path) -> list[dict[str, object]]:
    """Return prohibited-pattern findings for one test file."""
    rel = path.relative_to(repo_root).as_posix()
    if rel in ALLOWLIST:
        return []
    findings: list[dict[str, object]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line_number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in PROHIBITED_PATTERNS:
            if pattern.search(line):
                findings.append(
                    {
                        "path": rel,
                        "line": line_number,
                        "pattern": name,
                        "text": line.strip(),
                    }
                )
    return findings


def scan_tests(tests_dir: Path, *, repo_root: Path) -> list[dict[str, object]]:
    """Scan every Python file under ``tests_dir`` for prohibited patterns."""
    findings: list[dict[str, object]] = []
    for path in sorted(tests_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        findings.extend(scan_file(path, repo_root=repo_root))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the guard and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tests-dir", default="tests", help="test tree to scan")
    parser.add_argument("--report", default=None, help="optional JSON report path")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    tests_dir = repo_root / args.tests_dir
    if not tests_dir.is_dir():
        print(f"error: tests directory not found: {tests_dir}", file=sys.stderr)
        return 2

    findings = scan_tests(tests_dir, repo_root=repo_root)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "fail" if findings else "pass",
                    "findings_count": len(findings),
                    "findings": findings,
                    "allowlist": ALLOWLIST,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if findings:
        print("Release-specific local-check meta-test patterns found in tests/:")
        for finding in findings:
            print(f"  {finding['path']}:{finding['line']}: {finding['pattern']}")
            print(f"    {finding['text']}")
        print(
            "\nThese assertions couple tests to transient release-plan verification "
            "wiring. Test durable checker behavior instead (see v0.11.6 plan Task 29)."
        )
        return 1

    print("No release-specific local-check meta-test patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
