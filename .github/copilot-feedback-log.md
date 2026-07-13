# Copilot Feedback Log

Dated entries are added here by the `/refresh-ce-context` prompt whenever
`feedback=` is supplied. Once a pattern is reflected in an instruction file,
mark the entry ✅ and it can be removed in the next cleanup pass.

This file is the shared compatibility feedback log for all agent platforms
(Copilot, Codex, Claude Code, Gemini).

Format:
```
## YYYY-MM-DD – <short description>
**Feedback:** <what Copilot got wrong or missed>
**Root cause:** <why the miss happened>
**Durable fix:** <which instruction/test/script files were updated>
**Verification:** <command(s) that prove the fix>
**Status:** open | ✅ incorporated
```


<!-- entries will be appended below this line by /refresh-ce-context -->
## 2026-02-22 – Copilot optimization loop initialization
**Feedback:** Copilot/agents do not consistently learn from feedback or update canonical instructions.
**Root cause:** Feedback log not actively used; instruction files not updated after feedback.
**Durable fix:** Added feedback log entry template; will update copilot-instructions.md and CONTRIBUTOR_INSTRUCTIONS.md after each feedback.
**Verification:** Check that feedback log and instructions are updated after each PR/release.
**Status:** open

## 2026-02-23 – Test-quality method missed production test-helper wrappers
**Feedback:** Registry-level test-helper wrapper exports and trust-helper surface leakage were not treated as CI-blocking test-quality violations.
**Root cause:** ADR-030 enforcement focused on test-file anti-patterns and private-member scans, but lacked a hard source-surface guard for production `__all__` test-helper exports.
**Durable fix:** Added `scripts/quality/check_no_test_helper_exports.py`, wired it into `ci-pr.yml`, `ci-main.yml`, `test.yml`, and `scripts/local_checks.py`; updated ADR/test-quality docs and release task definitions; removed banned exports from `plugins/registry.py` and `plugins/__init__.py` `__all__`.
**Verification:** `pytest -q tests/scripts/test_check_no_test_helper_exports.py tests/scripts/test_detect_test_anti_patterns.py --no-cov`
**Status:** ✅ incorporated

## 2026-07-13 – Wheel packaging silently dropped `external_plugins` (reverts commit 554c3110)
**Feedback:** `tests/integration/test_doc_examples_smoke.py::test_wheel_install_supports_importable_fast_helper_but_not_python_m_execution` failed on a clean checkout with `ModuleNotFoundError: No module named 'external_plugins'` after installing the built wheel into a fresh venv. Locally the test appeared to pass, masking the bug.
**Root cause:** Commit `554c3110` ("Update verification checklist, enhance fallback warning handling, and improve test coverage", 2026-07-09) narrowed `[tool.setuptools.packages.find]` from `where = ["src", "external_plugins"]` to `where = ["src"]` plus an `include = ["calibrated_explanations", "calibrated_explanations.*"]` filter, as an incidental side effect of unrelated work — not a deliberate packaging decision. This silently excluded `src/external_plugins` from every wheel build from that commit onward. The regression was invisible locally because a stale, gitignored `calibrated_explanations.egg-info/` directory (predating the change) retained `external_plugins` in its cached `top_level.txt`/`SOURCES.txt`, so local `python -m build` runs kept bundling it by accident.
**Durable fix:** Reverted the inconsistent, premature narrowing in `pyproject.toml`: `include` now lists `external_plugins` and `external_plugins.*` alongside the `calibrated_explanations` entries, restoring the package to the wheel. This is a reversion of the 554c3110 packaging change, not a new decision.
**Verification:** Delete both `*.egg-info` dirs (`python -c "import shutil; shutil.rmtree('calibrated_explanations.egg-info', ignore_errors=True); shutil.rmtree('src/calibrated_explanations.egg-info', ignore_errors=True)"`) to force a clean build, then `pytest -q tests/integration/test_doc_examples_smoke.py::test_wheel_install_supports_importable_fast_helper_but_not_python_m_execution`.
**Status:** ✅ incorporated
