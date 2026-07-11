"""Check version-source alignment across runtime, docs, and release metadata.

Task 13 aligned runtime ``__version__`` with installed package metadata.
Task 36 extends that gate to cover:

- ``docs/conf.py`` derived ``release`` / ``version`` values
- ``CITATION.cff`` version metadata
- ``METADATA.json`` version metadata

Task 55 extends the gate further to cover prose Python-floor claims in
user-facing docs (``README.md``, compliance playbooks) against
``pyproject.toml``'s ``requires-python``.

Policy
------
- Runtime, installed package metadata, and docs ``release`` must agree after
  optional PEP 440 normalization.
- ``CITATION.cff`` and ``METADATA.json`` are release-facing metadata and must
  track the base release version derived from ``pyproject.toml``. They may omit
  the development suffix during the pre-tag window.
- Every "Python >= X.Y" / "Python ≥ X.Y" prose claim in the tracked
  user-facing docs listed in ``PYTHON_FLOOR_DOC_TARGETS`` must state the same
  floor as ``requires-python``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
import sys
import tomllib
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
    except Exception:  # noqa: BLE001 - fallback keeps the checker runnable
        return version.replace("-dev", ".dev0")


def _base_release_version(version: str) -> str:
    try:
        from packaging.version import Version  # noqa: PLC0415

        return Version(version).base_version
    except Exception:  # noqa: BLE001 - fallback keeps the checker runnable
        normalized = _canonicalize(version)
        return normalized.split(".dev", 1)[0]


def _strip_leading_v(version: str) -> str:
    return version.removeprefix("v")


def _pyproject_version() -> str:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def _pyproject_requires_python_floor() -> str:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    requires_python = str(payload["project"]["requires-python"])
    match = re.search(r">=\s*(?P<floor>\d+\.\d+)", requires_python)
    if match is None:
        raise RuntimeError(
            f"Could not parse a '>=X.Y' floor from requires-python={requires_python!r}."
        )
    return match.group("floor")


PYTHON_FLOOR_DOC_TARGETS = (
    "README.md",
    "docs/practitioner/playbooks/eu-ai-act-compliance.md",
)

PYTHON_FLOOR_CLAIM_PATTERN = re.compile(r"Python\s*(?:>=|≥)\s*(?P<floor>\d+\.\d+)")


def _doc_python_floor_claims(*, requires_python_floor: str) -> list[str]:
    """Return violation strings for stale Python-floor prose claims in docs."""
    errors: list[str] = []
    for relative_path in PYTHON_FLOOR_DOC_TARGETS:
        doc_path = REPO_ROOT / relative_path
        if not doc_path.exists():
            errors.append(f"{relative_path} is a declared Python-floor doc target but is missing.")
            continue
        text = doc_path.read_text(encoding="utf-8")
        matches = list(PYTHON_FLOOR_CLAIM_PATTERN.finditer(text))
        if not matches:
            errors.append(f"{relative_path} no longer states a 'Python >= X.Y' floor claim.")
            continue
        for match in matches:
            claimed = match.group("floor")
            if claimed != requires_python_floor:
                errors.append(
                    f"{relative_path} claims Python >= {claimed}, but pyproject.toml "
                    f"requires-python floor is {requires_python_floor}."
                )
    return errors


def _citation_version() -> str:
    citation_path = REPO_ROOT / "CITATION.cff"
    text = citation_path.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(?P<version>\S+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not find a version field in CITATION.cff.")
    return match.group("version")


def _metadata_json_version() -> str:
    metadata_path = REPO_ROOT / "METADATA.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(payload["version"])


def _docs_conf_module():
    docs_conf_path = REPO_ROOT / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("ce_docs_conf", docs_conf_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load docs config from {docs_conf_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _docs_release() -> str:
    module = _docs_conf_module()
    release = getattr(module, "release", None)
    if not isinstance(release, str) or not release:
        raise RuntimeError("docs/conf.py did not resolve a non-empty string release value.")
    return release


def _docs_version() -> str:
    module = _docs_conf_module()
    version = getattr(module, "version", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("docs/conf.py did not resolve a non-empty string version value.")
    return version


def _expected_docs_version(release: str) -> str:
    numeric_parts = release.split(".")
    if len(numeric_parts) >= 2:
        return ".".join(numeric_parts[:2])
    return release


def _evaluate_alignment(*, allow_normalized: bool) -> tuple[dict[str, str], list[str]]:
    pyproject_raw = _pyproject_version()
    pyproject_base = _base_release_version(pyproject_raw)
    runtime_raw = _runtime_version()
    metadata_raw = _metadata_version()
    docs_release = _docs_release()
    docs_version = _docs_version()
    citation_raw = _citation_version()
    metadata_json_raw = _metadata_json_version()
    requires_python_floor = _pyproject_requires_python_floor()

    observed = {
        "pyproject_version": pyproject_raw,
        "pyproject_base_version": pyproject_base,
        "runtime_version": runtime_raw,
        "package_metadata_version": metadata_raw,
        "docs_release": docs_release,
        "docs_version": docs_version,
        "citation_version": citation_raw,
        "metadata_json_version": metadata_json_raw,
        "requires_python_floor": requires_python_floor,
    }

    errors: list[str] = []
    runtime = _strip_leading_v(runtime_raw)
    metadata = _strip_leading_v(metadata_raw)
    docs = _strip_leading_v(docs_release)
    versions_match = runtime == metadata == docs
    if not versions_match and allow_normalized:
        versions_match = (
            len(
                {
                    _canonicalize(runtime),
                    _canonicalize(metadata),
                    _canonicalize(docs),
                }
            )
            == 1
        )
    if not versions_match:
        errors.append(
            "Runtime __version__, installed package metadata, and docs/conf.py release disagree."
        )

    expected_docs_version = _expected_docs_version(docs_release)
    if docs_version != expected_docs_version:
        errors.append(
            "docs/conf.py version should be derived from release as the short X.Y string."
        )

    citation = _strip_leading_v(citation_raw)
    if citation != pyproject_base:
        errors.append(
            "CITATION.cff version must match the pyproject base release version "
            "(dev suffix omitted by policy)."
        )

    if metadata_json_raw != pyproject_base:
        errors.append(
            "METADATA.json version must match the pyproject base release version "
            "(dev suffix omitted by policy)."
        )

    errors.extend(_doc_python_floor_claims(requires_python_floor=requires_python_floor))

    return observed, errors


def evaluate_alignment(*, allow_normalized: bool) -> tuple[dict[str, str], list[str]]:
    """Return observed version sources plus any policy violations."""
    return _evaluate_alignment(allow_normalized=allow_normalized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero on drift.",
    )
    parser.add_argument(
        "--allow-normalized",
        action="store_true",
        help="Accept runtime/metadata/docs release versions after PEP 440 canonicalization.",
    )
    args = parser.parse_args()

    observed, errors = evaluate_alignment(allow_normalized=args.allow_normalized)

    print(f"pyproject version:         {observed['pyproject_version']}")
    print(f"pyproject base version:    {observed['pyproject_base_version']}")
    print(f"runtime __version__:       {observed['runtime_version']}")
    print(f"package metadata:          {observed['package_metadata_version']}")
    print(f"docs/conf.py release:      {observed['docs_release']}")
    print(f"docs/conf.py version:      {observed['docs_version']}")
    print(f"CITATION.cff version:      {observed['citation_version']}")
    print(f"METADATA.json version:     {observed['metadata_json_version']}")
    print(f"requires-python floor:     {observed['requires_python_floor']}")

    if not errors:
        print("PASS: version sources align with the documented Task 36 policy.")
        return 0

    print("FAIL: version-source alignment drift detected:")
    for error in errors:
        print(f"- {error}")
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
