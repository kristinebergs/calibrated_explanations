"""Block generated reports and artifacts that contain local absolute paths."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

WINDOWS_DRIVE_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/][^\s\"'<>\]|]+)")
UNC_RE = re.compile(r"(\\\\[^\\/\s]+[\\/][^\s\"'<>\]|]+)")
UNIX_ABS_RE = re.compile(
    r"(?<![A-Za-z0-9+.\-:])"
    r"("
    r"/(?:Users|home|tmp|var|etc|opt|srv|mnt|private|root|proc|run|Volumes|Library|"
    r"Applications|System|usr|bin|sbin|dev|github|workspace)"
    r"[^\s\"'<>\]|]*"
    r")"
)

DEFAULT_SCAN_ROOTS = ("reports",)
DEFAULT_EXTRA_ARTIFACTS = (".pytest_matplotlib_debug.json",)

BINARY_SKIP_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".svg",
        ".pdf",
        ".eps",
        ".bin",
        ".pkl",
        ".pickle",
        ".npz",
        ".npy",
        ".mp4",
        ".avi",
        ".mov",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
    }
)


@dataclass(frozen=True)
class Violation:
    """A local-path violation discovered in a generated artifact."""

    artifact: str
    category: str
    match: str
    reason: str
    line: int | None = None
    json_path: str | None = None

    def sort_key(self) -> tuple[str, int, str, str, str]:
        """Return a deterministic sort key."""
        return (
            self.artifact,
            -1 if self.line is None else self.line,
            "" if self.json_path is None else self.json_path,
            self.category,
            self.match,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        payload: dict[str, object] = {
            "artifact": self.artifact,
            "category": self.category,
            "match_redacted": _redact_match(self.category),
            "reason": self.reason,
        }
        if self.line is not None:
            payload["line"] = self.line
        if self.json_path is not None:
            payload["json_path"] = self.json_path
        return payload


def _normalize_rel(path: Path, repo_root: Path) -> str:
    """Return a forward-slash path relative to *repo_root*."""
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _iter_default_targets(repo_root: Path) -> Iterable[Path]:
    """Yield the default report and artifact targets to scan."""
    for rel in DEFAULT_SCAN_ROOTS:
        root = repo_root / rel
        if root.is_dir():
            yield from sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() not in BINARY_SKIP_SUFFIXES
            )
        elif root.is_file() and root.suffix.lower() not in BINARY_SKIP_SUFFIXES:
            yield root

    for rel in DEFAULT_EXTRA_ARTIFACTS:
        path = repo_root / rel
        if path.is_file():
            yield path


def _resolve_targets(repo_root: Path, paths: list[str]) -> list[Path]:
    """Return the concrete paths that should be scanned."""
    if not paths:
        return list(dict.fromkeys(_iter_default_targets(repo_root)))

    resolved: list[Path] = []
    for raw in paths:
        candidate = (repo_root / raw).resolve()
        if candidate.is_dir():
            resolved.extend(
                sorted(
                    path
                    for path in candidate.rglob("*")
                    if path.is_file() and path.suffix.lower() not in BINARY_SKIP_SUFFIXES
                )
            )
        elif candidate.is_file() and candidate.suffix.lower() not in BINARY_SKIP_SUFFIXES:
            resolved.append(candidate)
    return list(dict.fromkeys(resolved))


def _find_matches(value: str) -> list[tuple[str, str, str]]:
    """Return local absolute-path matches in *value*."""
    findings: list[tuple[str, str, str]] = []
    for pattern, category, reason in (
        (WINDOWS_DRIVE_RE, "windows_drive", "local Windows drive paths are not allowed"),
        (UNC_RE, "unc_path", "local UNC paths are not allowed"),
        (UNIX_ABS_RE, "unix_absolute", "local Unix absolute paths are not allowed"),
    ):
        for match in pattern.finditer(value):
            findings.append((category, match.group(1), reason))
    return findings


def _json_path_for(key: object) -> str:
    """Return a compact JSON path token for *key*."""
    if isinstance(key, int):
        return f"[{key}]"
    escaped = str(key).replace("\\", "\\\\").replace('"', '\\"')
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", escaped):
        return f".{escaped}"
    return f'["{escaped}"]'


def _scan_json_value(
    value: object,
    *,
    artifact: str,
    json_path: str,
    out: list[Violation],
) -> None:
    """Recursively scan parsed JSON values."""
    if isinstance(value, str):
        for category, match, reason in _find_matches(value):
            out.append(
                Violation(
                    artifact=artifact,
                    category=category,
                    match=match,
                    reason=reason,
                    json_path=json_path or "$",
                )
            )
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_json_value(
                item,
                artifact=artifact,
                json_path=f"{json_path}{_json_path_for(index)}",
                out=out,
            )
        return

    if isinstance(value, dict):
        for key in sorted(value):
            _scan_json_value(
                value[key],
                artifact=artifact,
                json_path=f"{json_path}{_json_path_for(key)}",
                out=out,
            )


def _redact_match(category: str) -> str:
    """Return a safe placeholder for a redacted local-path match."""
    placeholders = {
        "windows_drive": "<redacted:windows-drive-path>",
        "unc_path": "<redacted:unc-path>",
        "unix_absolute": "<redacted:unix-absolute-path>",
    }
    return placeholders.get(category, "<redacted:local-path>")


def _location_for(violation: Violation) -> str:
    """Return a stable location label for a violation."""
    return violation.json_path if violation.json_path is not None else f"line {violation.line}"


def _build_summary(violations: list[Violation]) -> dict[str, object]:
    """Return per-artifact and per-category summaries for the JSON report."""
    artifact_counter = Counter(violation.artifact for violation in violations)
    category_counter = Counter(violation.category for violation in violations)

    artifact_summaries = []
    for artifact in sorted(artifact_counter):
        artifact_violations = [v for v in violations if v.artifact == artifact]
        artifact_categories = Counter(v.category for v in artifact_violations)
        sample_locations: list[str] = []
        for violation in artifact_violations:
            location = _location_for(violation)
            if location not in sample_locations:
                sample_locations.append(location)
            if len(sample_locations) == 5:
                break
        artifact_summaries.append(
            {
                "artifact": artifact,
                "violation_count": artifact_counter[artifact],
                "categories": dict(sorted(artifact_categories.items())),
                "sample_locations": sample_locations,
            }
        )

    return {
        "artifacts_with_violations": len(artifact_counter),
        "category_counts": dict(sorted(category_counter.items())),
        "artifacts": artifact_summaries,
    }


def scan_path(path: Path, repo_root: Path) -> list[Violation]:
    """Scan a single artifact for local absolute paths."""
    if path.suffix.lower() in BINARY_SKIP_SUFFIXES:
        return []
    artifact = _normalize_rel(path, repo_root)
    text = path.read_text(encoding="utf-8", errors="replace")
    violations: list[Violation] = []

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            _scan_json_value(payload, artifact=artifact, json_path="$", out=violations)
            return sorted(violations, key=lambda item: item.sort_key())

    for line_no, line in enumerate(text.splitlines(), 1):
        for category, match, reason in _find_matches(line):
            violations.append(
                Violation(
                    artifact=artifact,
                    line=line_no,
                    category=category,
                    match=match,
                    reason=reason,
                )
            )

    return sorted(violations, key=lambda item: item.sort_key())


def write_report(report_path: Path, violations: list[Violation]) -> None:
    """Write a deterministic JSON report."""
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "total_violations": len(violations),
        "summary": _build_summary(violations),
        "violations": [
            violation.to_dict()
            for violation in sorted(violations, key=lambda item: item.sort_key())
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the no-local-path report guard."""
    parser = argparse.ArgumentParser(
        description="Fail when generated reports or tracked debug artifacts contain local absolute paths.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional repo-relative file or directory paths to scan. Defaults to reports/ and known tracked debug artifacts.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used for relative-path normalization.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/quality/no_local_paths_report.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when local-path violations are found.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    report_path = (
        (repo_root / args.report).resolve()
        if not args.report.is_absolute()
        else args.report.resolve()
    )
    targets = [
        path for path in _resolve_targets(repo_root, args.paths) if path.resolve() != report_path
    ]
    violations: list[Violation] = []
    for target in targets:
        violations.extend(scan_path(target, repo_root))

    violations = sorted(violations, key=lambda item: item.sort_key())
    write_report(report_path, violations)

    if violations:
        print("Generated artifact local-path violations detected:")
        summary = _build_summary(violations)
        print(
            f"- {summary['artifacts_with_violations']} artifacts affected; "
            f"{len(violations)} violations total"
        )
        for category, count in summary["category_counts"].items():
            print(f"- category {category}: {count}")
        for artifact_summary in summary["artifacts"]:
            locations = ", ".join(artifact_summary["sample_locations"])
            print(
                f"- {artifact_summary['artifact']}: "
                f"{artifact_summary['violation_count']} violations "
                f"at {locations}"
            )
        print(f"Report written to {report_path}")
        return 1 if args.check else 0

    print("No local absolute paths detected in scanned reports/artifacts.")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
