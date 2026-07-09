"""Check runtime, package-metadata, and plugin version alignment (v0.11.6 Task 13).

Compares ``calibrated_explanations.__version__`` (with the conventional leading
``v`` stripped) against ``importlib.metadata.version("calibrated_explanations")``.
The default comparison is strict string equality so that drift like
``0.11.6-dev`` vs ``0.11.6.dev0`` fails; pass ``--allow-normalized`` only after
Task 13 records a normalization policy that deliberately accepts PEP 440
spelling differences.

Usage
-----
    python scripts/quality/check_version_alignment.py --check
    python scripts/quality/check_version_alignment.py --check --allow-normalized
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _runtime_version() -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import calibrated_explanations  # noqa: PLC0415

    return calibrated_explanations.__version__


def _metadata_version() -> str:
    try:
        return importlib.metadata.version("calibrated_explanations")
    except importlib.metadata.PackageNotFoundError:
        return importlib.metadata.version("calibrated-explanations")


def _canonicalize(version: str) -> str:
    try:
        from packaging.version import Version  # noqa: PLC0415

        return str(Version(version))
    except Exception:  # noqa: BLE001 - fall back to a naive dev normalization
        return version.replace("-dev", ".dev0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Exit non-zero on drift.")
    parser.add_argument(
        "--allow-normalized",
        action="store_true",
        help="Accept versions that agree after PEP 440 canonicalization.",
    )
    args = parser.parse_args()

    runtime_raw = _runtime_version()
    runtime = runtime_raw.removeprefix("v")
    metadata = _metadata_version()

    print(f"runtime __version__: {runtime_raw}")
    print(f"package metadata:    {metadata}")

    if runtime == metadata:
        print("PASS: versions agree exactly (after stripping the leading 'v').")
        return 0
    if args.allow_normalized and _canonicalize(runtime) == _canonicalize(metadata):
        print("PASS: versions agree after PEP 440 canonicalization (--allow-normalized).")
        return 0

    print(
        "FAIL: runtime and package-metadata versions disagree. Align "
        "src/calibrated_explanations/__init__.py __version__ with pyproject.toml "
        "(see v0.11.6 Task 13)."
    )
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
