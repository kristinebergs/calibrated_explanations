"""Shared filesystem and metadata helpers for capability-oriented tests."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path


def write_text_fixture(path: Path, content: str, *, dedent: bool = True) -> Path:
    """Write text fixture content, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if dedent:
        content = textwrap.dedent(content)
    path.write_text(content, encoding="utf-8")
    return path


def markdown_table_value(text: str, field: str) -> str | None:
    """Return the value from a two-column Markdown metadata table."""
    match = re.search(rf"\|\s*{re.escape(field)}\s*\|\s*([^|]+?)\s*\|", text)
    return match.group(1).strip() if match else None
