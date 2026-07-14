"""Verify the fallback-visibility pytest warning filter actually fires.

``pyproject.toml`` declares ``error:.*fall.*back.*:UserWarning`` so that
fallback warnings fail the suite, but pytest gives later ``filterwarnings``
entries precedence, and a later ``ignore::UserWarning`` can silently disable
the policy (pre-v2-gaps finding P6, v0.11.6 Task 25). This check runs two
probe tests under the repository pytest configuration:

1. a test emitting ``UserWarning("... falling back ...")`` — must FAIL;
2. a benign test emitting an unrelated ``UserWarning`` — must PASS.

Usage
-----
    python scripts/quality/check_fallback_filter_live.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO_ROOT / ".tmp_fallback_filter_probe"

FALLBACK_PROBE = """\
import warnings


def test_fallback_warning_probe():
    warnings.warn("probe: falling back to default template", UserWarning)
"""

BENIGN_PROBE = """\
import warnings


def test_benign_warning_probe():
    warnings.warn("probe: unrelated informational warning", UserWarning)
"""


def _run_probe(filename: str, content: str) -> int:
    probe_path = PROBE_DIR / filename
    probe_path.write_text(content, encoding="utf-8")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(probe_path),
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode


def main() -> int:
    PROBE_DIR.mkdir(exist_ok=True)
    try:
        fallback_rc = _run_probe("test_fallback_probe.py", FALLBACK_PROBE)
        benign_rc = _run_probe("test_benign_probe.py", BENIGN_PROBE)
    finally:
        shutil.rmtree(PROBE_DIR, ignore_errors=True)

    if benign_rc != 0:
        print(
            "FAIL: the benign UserWarning probe did not pass (rc="
            f"{benign_rc}); the filter configuration is broken in the other direction."
        )
        return 1
    if fallback_rc == 0:
        print(
            "FAIL: a test emitting a 'falling back' UserWarning passed the suite. "
            "The error:.*fall.*back.*:UserWarning filter in pyproject.toml is dead "
            "(shadowed by a later ignore::UserWarning entry); reorder the "
            "filterwarnings list so the fallback error filter takes precedence."
        )
        return 1
    print("PASS: fallback warnings fail the suite; benign warnings do not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
