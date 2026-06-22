"""TIF policy tests — WrapCalibratedExplainer enforcement scanner.

These tests scan TIF files under development/capabilities/verification/tif/
for forbidden patterns that would indicate a TIF scenario is bypassing
WrapCalibratedExplainer or using private/internal CE APIs.

Requirements verified:
  TIF policy: all behavioral CE TIF scenarios must use WrapCalibratedExplainer.
  TIF policy: TIF scenarios must not use private/internal CE APIs.

Governance source:
  development/capabilities/verification/README.md — Non-negotiable TIF rule
  development/capabilities/verification/tif/README.md — TIF constraints

Intentionally lightweight: these checks flag obvious violations without being
brittle to legitimate future patterns. If current repository patterns make a
rule unsafe, prefer a focused guard over a noisy one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_TIF_PY_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"


def _get_tif_py_files() -> list[Path]:
    """Return all .py files under the TIF directory (excluding __init__.py)."""
    return [p for p in _TIF_PY_DIR.glob("*.py") if p.name != "__init__.py"]


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Rule 1: TIF files must import WrapCalibratedExplainer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tif_file", _get_tif_py_files(), ids=lambda p: p.name)
def test_tif_should_import_wrap_calibrated_explainer(tif_file: Path):
    """Every TIF .py file must import WrapCalibratedExplainer.

    A TIF file that does not import WrapCalibratedExplainer cannot be using
    the required public workflow as its CE entry point.
    """
    source = _read_source(tif_file)
    assert "WrapCalibratedExplainer" in source, (
        f"TIF policy violation in {tif_file.name}: "
        "WrapCalibratedExplainer is not imported. "
        "All behavioral CE TIF scenarios must use WrapCalibratedExplainer as the "
        "CE entry point (development/capabilities/verification/README.md)."
    )


# ---------------------------------------------------------------------------
# Rule 2: TIF files must not import CalibratedExplainer from core directly
# ---------------------------------------------------------------------------

_FORBIDDEN_INTERNAL_IMPORT = re.compile(
    r"from\s+calibrated_explanations\.core\.calibrated_explainer\s+import",
    re.MULTILINE,
)


@pytest.mark.parametrize("tif_file", _get_tif_py_files(), ids=lambda p: p.name)
def test_tif_should_not_import_calibrated_explainer_directly(tif_file: Path):
    """TIF files must not import CalibratedExplainer from core directly.

    Importing CalibratedExplainer from the internal module bypasses the
    WrapCalibratedExplainer entry point and exposes internal implementation.
    """
    source = _read_source(tif_file)
    assert not _FORBIDDEN_INTERNAL_IMPORT.search(source), (
        f"TIF policy violation in {tif_file.name}: "
        "direct import from calibrated_explanations.core.calibrated_explainer detected. "
        "TIF scenarios must use WrapCalibratedExplainer from core.wrap_explainer, "
        "not CalibratedExplainer from the internal module."
    )


# ---------------------------------------------------------------------------
# Rule 3: TIF files must not construct FactualExplanation or AlternativeExplanation directly
# ---------------------------------------------------------------------------

_FORBIDDEN_DIRECT_CONSTRUCTION = re.compile(
    r"\b(FactualExplanation|AlternativeExplanation)\s*\(",
    re.MULTILINE,
)


@pytest.mark.parametrize("tif_file", _get_tif_py_files(), ids=lambda p: p.name)
def test_tif_should_not_construct_explanation_objects_directly(tif_file: Path):
    """TIF files must not construct FactualExplanation or AlternativeExplanation directly.

    Direct construction bypasses the WrapCalibratedExplainer workflow and produces
    explanation objects that are not produced through the supported public lifecycle.
    """
    source = _read_source(tif_file)
    match = _FORBIDDEN_DIRECT_CONSTRUCTION.search(source)
    assert not match, (
        f"TIF policy violation in {tif_file.name}: "
        f"direct construction of explanation object detected at position {match.start() if match else 'N/A'}. "
        "TIF scenarios must obtain explanation objects through WrapCalibratedExplainer, "
        "not by constructing FactualExplanation or AlternativeExplanation directly."
    )


# ---------------------------------------------------------------------------
# Rule 4: TIF files must not access private members of CE objects
# ---------------------------------------------------------------------------

_PRIVATE_MEMBER_PATTERN = re.compile(
    r"\b(explainer|collection|explanation|factual|alternatives|result)\._\w+",
    re.MULTILINE,
)


@pytest.mark.parametrize("tif_file", _get_tif_py_files(), ids=lambda p: p.name)
def test_tif_should_not_access_private_members(tif_file: Path):
    """TIF files must not access private members of CE objects.

    Private member access (._something) on CE objects indicates the TIF is
    relying on internal implementation details rather than the public API.

    This check uses a heuristic pattern. It looks for ._<attr> access on
    common CE object variable names. It may miss indirect private access via
    intermediate variables but provides a focused guard against obvious violations.
    """
    source = _read_source(tif_file)
    match = _PRIVATE_MEMBER_PATTERN.search(source)
    assert not match, (
        f"TIF policy violation in {tif_file.name}: "
        f"private member access detected: '{match.group(0) if match else ''}'. "
        "TIF scenarios must use only public CE API members."
    )


# ---------------------------------------------------------------------------
# Rule 5: TIF directory must have a README
# ---------------------------------------------------------------------------


def test_tif_directory_should_have_readme():
    """The TIF directory must contain a README.md documenting TIF rules."""
    readme = _TIF_PY_DIR / "README.md"
    assert readme.exists(), (
        f"TIF directory {_TIF_PY_DIR} is missing README.md. "
        "TIF directory must document TIF definition and constraints."
    )


# ---------------------------------------------------------------------------
# Rule 6: Each TIF .py file should have a corresponding .md spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tif_file", _get_tif_py_files(), ids=lambda p: p.name)
def test_tif_python_file_should_have_corresponding_spec(tif_file: Path):
    """Each TIF executable should be covered by a CE-TIF-*.md specification.

    This is a documentation completeness check, not a behavioral check.
    It ensures TIF executables are not written without a governing specification.
    """
    stem = tif_file.stem  # e.g. "tif_conjunction"
    # Look for any CE-TIF-*.md that references this executable
    md_files = list(_TIF_PY_DIR.glob("CE-TIF-*.md"))
    area = stem.replace("tif_", "").upper()  # e.g. "CONJUNCTION"

    found = any(
        area in md.stem.upper() or _read_source(md).find(tif_file.name) != -1 for md in md_files
    )
    assert found, (
        f"TIF policy: no CE-TIF-*.md specification found for {tif_file.name}. "
        "Each TIF executable must have a corresponding specification file in the "
        "same directory."
    )


# ---------------------------------------------------------------------------
# Rule 7: Requirements citing tests/capabilities/ must have tif_refs or tif_exemption
# ---------------------------------------------------------------------------

_CAPABILITY_TEST_RE = re.compile(r"pytest:\s+tests/capabilities/")
_TIF_REFS_RE = re.compile(r"^\| tif_refs \|", re.MULTILINE)
_TIF_EXEMPTION_RE = re.compile(r"tif_exemption")


def test_capability_requirements_should_declare_tif_refs_or_exemption():
    """Requirements that cite tests/capabilities/ must declare tif_refs or tif_exemption.

    Any requirement whose verification targets include a test in tests/capabilities/
    is a CE capability-facing requirement and must either:
    - Declare tif_refs pointing to one or more CE-TIF-*.md interfaces, OR
    - Declare tif_exemption with a rationale (for non-WrapCalibratedExplainer checks).

    Governance requirements verified through tests/unit/ are excluded from this rule.
    """
    repo_root = _REPO_ROOT
    req_dir = repo_root / "development" / "capabilities" / "requirements"
    errors: list[str] = []

    for req_path in sorted(req_dir.glob("CE-REQ-*.md")):
        text = req_path.read_text(encoding="utf-8")

        if not _CAPABILITY_TEST_RE.search(text):
            continue

        has_tif_refs = bool(_TIF_REFS_RE.search(text))
        has_tif_exemption = bool(_TIF_EXEMPTION_RE.search(text))

        if not has_tif_refs and not has_tif_exemption:
            req_id = re.search(r"^\| requirement_id \| (\S+)", text, re.MULTILINE)
            errors.append(
                f"{req_id.group(1) if req_id else req_path.stem}: cites tests/capabilities/ "
                "but has neither tif_refs nor tif_exemption — add one or the other"
            )

    assert not errors, (
        "TIF architecture policy: every requirement that cites a test in tests/capabilities/ "
        "must declare tif_refs (pointing to a CE-TIF-*.md interface) or tif_exemption "
        "(with rationale). Missing:\n" + "\n".join(f"  {e}" for e in errors)
    )
