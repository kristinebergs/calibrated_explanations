"""Fail on runtime deprecation emitters not covered by the deprecation ledger.

The ``make deprecation-closure`` gate parses only
``docs/migration/deprecations.md``; it cannot see live ``DeprecationWarning``
emitters in ``src/``. This check closes that blind spot (pre-v2-gaps finding
P2, v0.11.6 Task 25): it AST-scans the package for calls to the central
``deprecate``/``deprecate_alias`` helpers and for direct ``warnings.warn``
calls with a ``DeprecationWarning``/``PendingDeprecationWarning`` category,
excluding the helper modules themselves. Any hit means an active runtime
deprecation cycle exists and must either be removed (fail fast) or recorded in
the Active deprecations ledger with the gate expected to fail.

Usage
-----
    python scripts/quality/check_runtime_deprecations.py --check
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "calibrated_explanations"

EXCLUDED_MODULES = {
    "utils/deprecation.py",
    "utils/deprecations.py",
}

DEPRECATE_HELPERS = {"deprecate", "deprecate_alias", "deprecate_public_api_symbol"}
DEPRECATION_CATEGORIES = {"DeprecationWarning", "PendingDeprecationWarning"}


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _mentions_deprecation_category(node: ast.Call) -> bool:
    candidates: list[ast.expr] = list(node.args)
    candidates.extend(kw.value for kw in node.keywords if kw.arg in {None, "category"})
    for arg in candidates:
        if isinstance(arg, ast.Name) and arg.id in DEPRECATION_CATEGORIES:
            return True
        if isinstance(arg, ast.Attribute) and arg.attr in DEPRECATION_CATEGORIES:
            return True
    return False


def _scan_module(path: Path) -> list[str]:
    rel = path.relative_to(SRC_ROOT).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{rel}:{exc.lineno}: unparsable module ({exc.msg})"]

    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in DEPRECATE_HELPERS:
            findings.append(f"{rel}:{node.lineno}: call to deprecation helper '{name}'")
        elif name == "warn" and _mentions_deprecation_category(node):
            findings.append(f"{rel}:{node.lineno}: warnings.warn with DeprecationWarning category")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Exit non-zero on findings.")
    args = parser.parse_args()

    findings: list[str] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.relative_to(SRC_ROOT).as_posix() in EXCLUDED_MODULES:
            continue
        findings.extend(_scan_module(path))

    if findings:
        print("FAIL: live runtime deprecation emitters found outside the helper modules:")
        for finding in findings:
            print(f"  src/calibrated_explanations/{finding}")
        print(
            "Resolve each emitter: remove the deprecated path (fail fast, preferred "
            "for v0.11.6) or record it in docs/migration/deprecations.md Active "
            "deprecations so the closure gate reports it."
        )
        return 1 if args.check else 0

    print("PASS: no runtime deprecation emitters outside utils/deprecation*.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
