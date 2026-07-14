"""Guard against undocumented public-surface growth on core CE classes.

`pre-v4.md` finding S4-H1 (v0.11.6 Task 49) found that ADR-030 private-member
remediation had quietly converted internal `CalibratedExplainer`/
`WrapCalibratedExplainer` state into public, mostly-mutable aliases whose only
purpose was letting tests bypass the private-member gate. This script freezes
the current public class surface into a checked-in snapshot
(``.github/class_surface_allowlist.json``) tiered as ``stable``,
``experimental``, ``compatibility``, or ``internal``, so a future PR cannot
quietly reopen a test-only production alias: any new non-underscore,
non-dunder class member must be added to the allowlist deliberately, with a
disposition tier recorded, before the gate passes.

This complements (does not replace) ``scan_private_usage.py`` (which guards
private-member *access from tests*) and ``check_no_test_helper_exports.py``
(which guards module-level ``__all__`` exports) — this script is the
class-*attribute*-surface counterpart the other two do not cover.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_TIERS = {"stable", "experimental", "compatibility", "internal"}

# Fully qualified class names tracked by this gate.
TRACKED_CLASSES = [
    "calibrated_explanations.core.calibrated_explainer.CalibratedExplainer",
    "calibrated_explanations.core.wrap_explainer.WrapCalibratedExplainer",
]


@dataclass(frozen=True)
class Violation:
    """A single class-surface allowlist violation."""

    cls: str
    symbol: str
    reason: str

    def to_record(self) -> dict[str, str]:
        """Return a JSON-safe record."""
        return {"class": self.cls, "symbol": self.symbol, "reason": self.reason}


def _import_class(dotted_path: str) -> type:
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _public_own_members(cls: type) -> set[str]:
    """Return non-underscore, non-dunder names defined directly on *cls*.

    Only names present in ``cls.__dict__`` are considered — inherited object/
    mixin members are out of scope for this gate, since the concern is names
    this class itself introduces.
    """
    names = set()
    for name in vars(cls):
        if name.startswith("_"):
            continue
        names.add(name)
    return names


def load_allowlist(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Load the class-surface allowlist JSON file."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("classes", {})


def scan(tracked_classes: list[str]) -> dict[str, set[str]]:
    """Return {class_dotted_path: {public_own_member_names}}."""
    result: dict[str, set[str]] = {}
    for dotted_path in tracked_classes:
        cls = _import_class(dotted_path)
        result[dotted_path] = _public_own_members(cls)
    return result


def find_violations(
    current: dict[str, set[str]],
    allowlist: dict[str, dict[str, dict[str, str]]],
) -> list[Violation]:
    """Return unlisted-new-symbol and stale-listed-symbol violations."""
    violations: list[Violation] = []
    for cls, members in current.items():
        allowed_entries = allowlist.get(cls, {})
        for entry_name, entry in allowed_entries.items():
            tier = entry.get("tier")
            if tier not in VALID_TIERS:
                violations.append(
                    Violation(
                        cls=cls,
                        symbol=entry_name,
                        reason=f"allowlist entry has invalid tier {tier!r}",
                    )
                )
        allowed_names = set(allowed_entries)
        for name in sorted(members - allowed_names):
            violations.append(
                Violation(
                    cls=cls,
                    symbol=name,
                    reason=(
                        "new public class member is not in the class-surface "
                        "allowlist; add a deliberate stable/experimental/"
                        "compatibility/internal disposition instead of leaving "
                        "it undocumented (pre-v4 S4-H1 / v0.11.6 Task 49)"
                    ),
                )
            )
        for name in sorted(allowed_names - members):
            violations.append(
                Violation(
                    cls=cls,
                    symbol=name,
                    reason="allowlist entry no longer exists on the class; remove the stale entry",
                )
            )
    return violations


def write_report(report_path: Path, violations: list[Violation]) -> None:
    """Write a deterministic JSON report."""
    payload: dict[str, Any] = {
        "total_violations": len(violations),
        "violations": [v.to_record() for v in violations],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the class-surface allowlist gate."""
    parser = argparse.ArgumentParser(
        description="Guard core CE class public-surface growth against an allowlist."
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(".github/class_surface_allowlist.json"),
        help="Path to the class-surface allowlist JSON file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/quality/class_surface_allowlist_report.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if any violation is found.",
    )
    args = parser.parse_args(argv)

    allowlist = load_allowlist(args.allowlist)
    current = scan(TRACKED_CLASSES)
    violations = find_violations(current, allowlist)
    write_report(args.report, violations)

    if violations:
        print("Class-surface allowlist violations:")
        for v in violations:
            print(f"- {v.cls}.{v.symbol}: {v.reason}")
        print(f"Report written to {args.report}")
        if args.check:
            return 1
        return 0

    print("No class-surface allowlist violations found.")
    print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
