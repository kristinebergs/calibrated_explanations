# Pull Request Guide

Use the GitHub PR template provided in `.github/pull_request_template.md`.

Key items to verify before submitting:

1. Align with the sole active `development/current-work/vX.Y.Z_plan.md`
   (scoped per its GitHub milestone) and reference relevant ADRs.
2. Add or update tests for new behavior and ensure they pass locally.
3. Run mypy for changed modules and ruff/markdownlint for style.
4. Update README/docs when public behavior changes.
5. Note breaking changes with migration notes; update CHANGELOG when applicable.

This page is a convenience summary; the authoritative checklist lives in the PR template.
