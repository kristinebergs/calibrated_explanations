"""RETIRED: EXPL-CONJ-only partial capability evidence generator.

This script is retired. It generated evidence only for the EXPL-CONJ
capability chain and cannot represent the full TIF suite.

Use scripts/generate_tif_evidence.py instead, which discovers and runs
all active TIF specs dynamically from CE-TIF-*.md files.

    python scripts/generate_tif_evidence.py
    python scripts/generate_tif_evidence.py --check-current

This script exits with code 1 unconditionally so it cannot be used
as a source of partial evidence in CI or automated pipelines.
"""

from __future__ import annotations

import sys

_MSG = """
ERROR: generate_capability_evidence.py is retired.

This script produced EXPL-CONJ-only partial evidence and is no longer
a valid source of truth for the full capability verification suite.

Use the dynamically-discovering full-suite generator instead:

    python scripts/generate_tif_evidence.py
    python scripts/generate_tif_evidence.py --check-current
"""


def main() -> int:
    print(_MSG.strip())
    return 1


if __name__ == "__main__":
    sys.exit(main())
