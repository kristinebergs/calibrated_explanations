"""Pytest configuration for capability contract tests.

Adds the TIF directory to sys.path so TIF scenario modules are importable
without src installation. TIF scenarios live under:
  development/capabilities/verification/tif/
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_TIF_DIR = _REPO_ROOT / "development" / "capabilities" / "verification" / "tif"

if str(_TIF_DIR) not in sys.path:
    sys.path.insert(0, str(_TIF_DIR))
