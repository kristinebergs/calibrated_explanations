"""Tests for the Task 15 wheel-only docs smoke checker."""

from __future__ import annotations

import scripts.quality.check_docs_wheel_smoke as docs_wheel_smoke


def test_should_return_one_when_wheel_build_fails(monkeypatch) -> None:
    """CLI mode should report a failed smoke when wheel building errors out."""
    monkeypatch.setattr(
        docs_wheel_smoke,
        "run_docs_wheel_smoke",
        lambda: (_ for _ in ()).throw(FileNotFoundError("No wheel built under dist")),
    )
    monkeypatch.setattr("sys.argv", ["check_docs_wheel_smoke.py"])

    rc = docs_wheel_smoke.main()
    assert rc == 1
