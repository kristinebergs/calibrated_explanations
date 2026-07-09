"""Check canonical skill coverage and platform-tree drift (v0.11.6 Task 21).

The canonical shared skill catalog lives under ``.claude/skills``.
Other skill trees (``.github/skills``, ``.agents/skills``, ``.codex/skills``)
may expose compatibility copies or platform-only extras, but they must not
become independent sources of truth.

This check enforces three rules:

1. **Canonical references exist (always fatal).** Every ``references/...`` path
   mentioned in a canonical ``.claude/skills/*/SKILL.md`` must exist inside
   that canonical skill directory.
2. **Canonical coverage / intentional extras (fatal with ``--strict``).**
   Every canonical skill must exist in each shadow tree. Extra shadow-tree
   skills are allowed only when recorded in ``INTENTIONAL_DIFFERENCES`` with a
   reason.
3. **Shadow shared skills stay as shims (always fatal).** When a shadow tree
   contains a canonical shared skill, its local copy must be a thin reference
   shim that points back to ``.claude/skills/<name>/SKILL.md`` and must not ship
   its own ``assets/``, ``references/``, or ``scripts/`` directories.

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
    Path(".claude/skills"),
    Path(".agents/skills"),
    Path(".github/skills"),
    Path(".codex/skills"),
)
CANONICAL_TREE = Path(".claude/skills")
SHADOW_TREES = tuple(tree for tree in SKILL_TREES if tree != CANONICAL_TREE)

# skill name -> reason it may legitimately exist only in selected non-canonical trees.
INTENTIONAL_DIFFERENCES: dict[str, str] = {
    "ai-adoption-briefing": "Platform-local general AI advisory skill; not part of the canonical CE catalog.",
    "ai-readiness-assessor": "Platform-local general AI advisory skill; not part of the canonical CE catalog.",
    "ai-sprint-facilitator": "Platform-local general AI advisory skill; not part of the canonical CE catalog.",
    "ai-systems-architect": "Platform-local general AI advisory skill; not part of the canonical CE catalog.",
    "ce-doc-navigator": "Legacy platform convenience skill kept outside the canonical CE catalog.",
    "conformal-methods-reviewer": "Platform-local general research skill; not part of the canonical CE catalog.",
    "decision-memo-drafter": "Platform-local general documentation skill; not part of the canonical CE catalog.",
    "experiment-result-interpreter": "Platform-local general research skill; not part of the canonical CE catalog.",
    "lecture-designer": "Platform-local education skill; not part of the canonical CE catalog.",
    "notion-snapshot-refresh": "Codex-only workspace utility skill; not part of the canonical CE catalog.",
    "paper-distiller": "Platform-local general research skill; not part of the canonical CE catalog.",
    "peer-review-writer": "Platform-local general research skill; not part of the canonical CE catalog.",
    "process-automation-designer": "Platform-local general AI advisory skill; not part of the canonical CE catalog.",
    "red-team-my-idea": "Platform-local general AI advisory skill; not part of the canonical CE catalog.",
    "requirements-analyst": "Platform-local general analysis skill; not part of the canonical CE catalog.",
    "rigorous-technical-writer": "Platform-local general writing skill; not part of the canonical CE catalog.",
    "student-feedback-writer": "Platform-local education skill; not part of the canonical CE catalog.",
    "training-material-designer": "Platform-local education skill; not part of the canonical CE catalog.",
    "use-case-evaluator": "Platform-local general AI advisory skill; not part of the canonical CE catalog.",
    "workspace-setup": "Codex-only workspace utility skill; not part of the canonical CE catalog.",
}

REFERENCE_PATTERN = re.compile(r"references/[\w\-./]+")
SHIM_MARKER = "Compatibility shim for the canonical shared skill definition."


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


def _shadow_shim_failures(skill_name: str, skill_dir: Path) -> list[str]:
    failures: list[str] = []
    skill_md = skill_dir / "SKILL.md"
    rel_skill_md = skill_md.relative_to(REPO_ROOT).as_posix()
    canonical_path = f".claude/skills/{skill_name}/SKILL.md"
    if not skill_md.is_file():
        return [f"{rel_skill_md}: missing SKILL.md"]

    text = skill_md.read_text(encoding="utf-8", errors="ignore")
    if SHIM_MARKER not in text:
        failures.append(f"{rel_skill_md}: missing shadow-skill shim marker")
    if canonical_path not in text:
        failures.append(f"{rel_skill_md}: missing canonical path reference {canonical_path}")

    for child in sorted(skill_dir.iterdir()):
        if child.name == "SKILL.md":
            continue
        failures.append(
            f"{skill_dir.relative_to(REPO_ROOT).as_posix()}: unexpected duplicated support path "
            f"{child.name} for canonical skill {skill_name}"
        )
    return failures


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
    for skill_dir in trees[CANONICAL_TREE].values():
        reference_failures.extend(_missing_references(skill_dir))
    if reference_failures:
        exit_code = 1
        print("FAIL: canonical skill files reference materials that do not exist:")
        for failure in reference_failures:
            print(f"  {failure}")

    shim_failures: list[str] = []
    for tree in SHADOW_TREES:
        for skill_name in sorted(set(trees[CANONICAL_TREE]).intersection(trees[tree])):
            shim_failures.extend(_shadow_shim_failures(skill_name, trees[tree][skill_name]))
    if shim_failures:
        exit_code = 1
        print("FAIL: shadow-tree copies of canonical skills are not clean shims:")
        for failure in shim_failures:
            print(f"  {failure}")

    canonical_names = set(trees[CANONICAL_TREE])
    drift_lines: list[str] = []
    for tree in SHADOW_TREES:
        shadow_names = set(trees[tree])
        for missing_name in sorted(canonical_names - shadow_names):
            drift_lines.append(f"  {missing_name}: canonical skill missing from {tree.as_posix()}")
        for extra_name in sorted(shadow_names - canonical_names):
            reason = INTENTIONAL_DIFFERENCES.get(extra_name)
            if reason is None:
                drift_lines.append(
                    f"  {extra_name}: extra non-canonical skill in {tree.as_posix()} "
                    "without an INTENTIONAL_DIFFERENCES reason"
                )
    if drift_lines:
        header = "FAIL" if args.strict else "WARN"
        print(
            f"{header}: skill-tree drift relative to canonical .claude/skills ({len(drift_lines)} issues):"
        )
        for line in drift_lines:
            print(line)
        if args.strict:
            exit_code = 1

    if exit_code == 0 and not drift_lines:
        print("PASS: canonical skills and platform skill trees are structurally synchronized.")
    elif exit_code == 0:
        print("PASS (non-strict): canonical references are valid; drift reported above.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
