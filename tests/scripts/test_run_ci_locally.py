from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_ci_locally import collect_all_runs, find_workflow_files, summary_cwd

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_summary_cwd_returns_relative_path() -> None:
    assert summary_cwd(".") == "."


def test_summary_cwd_does_not_force_absolute_path_for_external_value() -> None:
    assert summary_cwd("reports") == "reports"


def test_should_discover_exactly_the_approved_workflow_inventory(monkeypatch) -> None:
    """Local workflow discovery must match the ADR-035 v1 inventory (Task 60)."""
    pytest.importorskip("yaml")
    monkeypatch.chdir(REPO_ROOT)

    discovered = collect_all_runs(find_workflow_files())

    assert set(discovered) == {"ci", "scheduled", "maintenance"}
