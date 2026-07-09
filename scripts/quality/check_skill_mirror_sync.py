"""Check agent skill mirrors for structural drift (v0.11.6 Task 21).

Two checks across the four skill trees (``.github/skills``, ``.agents/skills``,
``.codex/skills``, ``.claude/skills``):

1. **Referenced files exist (always fatal).** Every ``references/...`` path
   mentioned in a ``SKILL.md`` must exist inside that skill's directory. This
   catches drift like ``.codex/skills/ce-test-audit`` referencing
   ``references/adr-030-test-quality.md`` without shipping it.
2. **Skill-name set drift (fatal with ``--strict``).** The trees are expected
   to be mirrors; skills present in one tree but missing in another are
   reported. Intentional per-platform differences must be recorded in
   ``INTENTIONAL_DIFFERENCES`` with a reason.

Usage
-----
    python scripts/quality/check_skill_mirror_sync.py
    python scripts/quality/check_skill_mirror_sync.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SKILL_TREES = (
    Path(".github/skills"),
    Path(".agents/skills"),
    Path(".codex/skills"),
    Path(".claude/skills"),
)

# skill name -> reason it may legitimately be absent from some trees.
INTENTIONAL_DIFFERENCES: dict[str, str] = {}

REFERENCE_PATTERN = re.compile(r"references/[\w\-./]+")


def _skill_dirs(tree: Path) -> dict[str, Path]:
    root = REPO_ROOT / tree
    if not root.is_dir():
        return {}
    return {child.name: child for child in sorted(root.iterdir()) if child.is_dir()}


def _missing_references(skill_dir: Path) -> list[str]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return [f"{skill_dir.as_posix()}: missing SKILL.md"]
    text = skill_md.read_text(encoding="utf-8", errors="ignore")
    missing: list[str] = []
    for match in sorted(set(REFERENCE_PATTERN.findall(text))):
        candidate = skill_dir / match.rstrip(".,;:)")
        if not candidate.exists():
            rel = skill_md.relative_to(REPO_ROOT).as_posix()
            missing.append(f"{rel}: referenced file not found: {match}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on skill-name set drift between mirror trees.",
    )
    args = parser.parse_args()

    trees = {tree: _skill_dirs(tree) for tree in SKILL_TREES}
    exit_code = 0

    reference_failures: list[str] = []
    for skills in trees.values():
        for skill_dir in skills.values():
            reference_failures.extend(_missing_references(skill_dir))
    if reference_failures:
        exit_code = 1
        print("FAIL: skill files reference materials that do not exist:")
        for failure in reference_failures:
            print(f"  {failure}")

    all_names = {name for skills in trees.values() for name in skills}
    drift_lines: list[str] = []
    for name in sorted(all_names):
        if name in INTENTIONAL_DIFFERENCES:
            continue
        present = [tree.as_posix() for tree, skills in trees.items() if name in skills]
        if len(present) != len(SKILL_TREES):
            missing_from = [
                tree.as_posix() for tree in SKILL_TREES if tree.as_posix() not in present
            ]
            drift_lines.append(f"  {name}: missing from {', '.join(missing_from)}")
    if drift_lines:
        header = "FAIL" if args.strict else "WARN"
        print(f"{header}: skill-name drift across mirror trees ({len(drift_lines)} skills):")
        for line in drift_lines:
            print(line)
        if args.strict:
            exit_code = 1

    if exit_code == 0 and not drift_lines:
        print("PASS: skill mirrors are structurally synchronized.")
    elif exit_code == 0:
        print("PASS (non-strict): no missing referenced files; drift reported above.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
