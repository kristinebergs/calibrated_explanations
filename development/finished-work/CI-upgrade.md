> **Finished:** CI modular architecture migration is complete, and the modular
> inventory itself was superseded by the **v1 consolidation** (v0.11.6 Task 60,
> 2026-07-13) recorded at the end of this document. Ongoing CI governance
> policy is codified in ADR-035 (`development/adrs/ADR-035-ci-workflow-governance.md`),
> which references this file as its implementation appendix. Everything between
> this note and the "v1 consolidation record" section describes the
> pre-consolidation architecture and is retained as historical migration
> evidence only.

# CI Upgrade Operations Guide

## Scope and authority

- `kristinebergs/calibrated_explanations` CI is a **validation + artifact build verification** surface.
- `Moffran/calibrated_explanations` remains authoritative for versions, tags, GitHub releases, PyPI publication, changelog, security advisories, and documentation.
- Nothing in this document changes release authority allocation.

## Current architecture (state as of 2026-04-06)

Modular CI is the current architecture, not a future target.

- Fast PR lane: `.github/workflows/ci-pr.yml`
- Full PR viz/parity lane (path/manual): `.github/workflows/ci-full.yml`
- Main-branch safety and drift detection: `.github/workflows/ci-main.yml`
- Nightly/scheduled heavy jobs: `.github/workflows/ci-nightly.yml`
- Legacy duplicate wrappers have been decommissioned; the mapping table below is retained as migration evidence.

## Required PR path (critical lane)

The required PR lane is intentionally narrow and fast. Required checks are:

1. Lint
2. MyPy
3. Core tests
4. Private-member scan
5. Anti-pattern audit
6. Governance-event schema checks **only when governance paths are touched**

Heavy/manual/scheduled checks stay off the critical PR path unless explicitly promoted in planning docs:

- perf guard
- notebook execution
- full viz-focused checks
- over-testing density
- other heavy/manual/scheduled jobs

## Notebook execution policy

- Blocking on release branches only.
- Advisory/manual/non-blocking outside release-boundary contexts unless explicitly promoted by a milestone plan.
- `notebook-audit` strict mode (`--check`) is release-branch enforcement; non-release contexts remain advisory.

## Packaging verification policy (validation only)

For the development-mirror CI role, packaging verification means:

1. Build wheel and sdist.
2. Install from built artifacts in a clean environment.
3. Inspect built artifact contents.

This is a verification gate only and does not authorize publication.

## Local reproduction policy

Two-tier local reproduction is mandatory:

- Routine work: `make local-checks-pr`
- Milestone closure or branch-gate changes: `make local-checks`

CI/planning changes must keep local checks aligned (`scripts/local_checks.py` + Make targets).

## Branch protection policy

Branch protection should require:

- fast PR lane checks
- selected safety checks

Do **not** require nearly every non-nightly job. Keep required checks practical and stable.

## Action pinning policy

For supply-chain hardening, external GitHub Actions should be pinned to full commit SHAs rather than major tags.

- Pin external `owner/repo` actions to `@<40-char-sha>`.
- Keep local reusable workflows (`./.github/workflows/...`) and local composite actions (`./.github/actions/...`) unpinned because they are repository-relative.
- Record the source tag or release date in the PR description or changelog when rotating a SHA.
- Prefer automated refresh PRs for routine SHA updates so the maintenance cost stays bounded.
- Use the CI policy validator and local checks to catch noncompliant action references before merge.

## Workflow/check-name freeze policy during migration

- Workflow names and required check names are frozen during migration.
- Any rename must include, in the same change:
  1. branch-protection update,
  2. mapping-table update in this document,
  3. validation evidence that required checks still map 1:1.

## Legacy migration and decommission policy

1. Legacy workflows fall into two classes:
   - **Removal-eligible duplicates**: wrappers with complete replacement and no active parity purpose.
   - **Parity-retained legacy workflows**: intentionally retained for comparison/evidence.
2. Parity-retained legacy workflows **MUST NOT** be removed until their parity purpose is formally retired in planning docs.
3. CI migration readiness requires:
   - quantitative evidence, and
   - maintainer judgment.

Quantitative evidence is necessary but not sufficient on its own.

## Legacy → replacement mapping (mandatory control table)

| Legacy workflow file | Legacy required check name(s) | Replacement workflow file | Replacement required check name(s) | Retained for parity? | Artifact/check parity expectation | Validation status | Removal eligibility |
|---|---|---|---|---|---|---|---|
| `.github/workflows/test.yml` | `test (compat wrapper)`; `Anti-pattern Audit` | `.github/workflows/ci-pr.yml`, `.github/workflows/ci-main.yml`, `.github/workflows/ci-full.yml` | `CI — Pull Request checks / core-tests`; `CI — Pull Request checks / Anti-pattern Audit (ADR-030)`; `CI — Main branch gates / perf-guard`; `CI — Full PR checks (includes viz)` | No | Required PR + selected safety checks fully covered by modular workflows. | Completed | Removed |
| `.github/workflows/coverage.yml` | `Coverage (wrapper) / coverage` | `.github/workflows/ci-main.yml` | `CI — Main branch gates / core-with-coverage` and per-module coverage gate jobs | No | Coverage thresholds and reports matched branch-gate expectations. | Completed | Removed |
| `.github/workflows/examples.yml` | `Examples QA / examples` | `.github/workflows/ci-nightly.yml` | `CI — Nightly heavy jobs / examples-smoke` | No | Example smoke validity retained on nightly/manual schedule. | Completed | Removed |
| `.github/workflows/scan-private-members.yml` | `Scan Private Members in Tests / scan-private-members` | `.github/workflows/ci-pr.yml` | `CI — Pull Request checks / private-member-scan` | No | PR-path private-member enforcement parity confirmed. | Completed | Removed |
| `.github/workflows/notebook-audit.yml` | `notebook-audit / audit` | `.github/workflows/ci-nightly.yml` + release-branch notebook blocking policy | `CI — Nightly heavy jobs / notebook-audit` (advisory outside release boundary) | No | Release-branch blocking + non-release advisory split preserved. | Completed | Removed |
| `.github/workflows/docs.yml` | `docs / build` | `.github/workflows/ci-main.yml` and local milestone docs gates | `CI — Main branch gates / core-with-coverage` plus milestone docs gating policy | No | Docs gate retained without standalone duplicate wrapper. | Completed | Removed |
| `.github/workflows/lint.yml` | `Lint / lint` | `.github/workflows/ci-pr.yml` | `CI — Pull Request checks / lint` | No | Same lint stack and ADR checks covered in required PR lane. | Completed | Removed |
| `.github/workflows/mypy.yml` | `mypy / typecheck` | `.github/workflows/ci-pr.yml` | `CI — Pull Request checks / mypy` | No | Same typed-scope check retained in required PR lane. | Completed | Removed |
| `.github/workflows/dependency-audit.yml` | `dependency-audit / pip-audit` | `.github/workflows/ci-main.yml` local/milestone safety checks | Dependency audit retained as non-critical safety check outside required PR lane | No | Safety check retained without duplicate standalone wrapper. | Completed | Removed |

## Migration exit criteria (evidence-based)

A removal-eligible legacy wrapper may be removed only when all criteria below are met:

1. **Consecutive green runs:** at least 10 consecutive green replacement runs.
2. **Representative PR coverage:** evidence across at least 5 PRs spanning docs, runtime, tests, and workflow-touching changes.
3. **Check-name parity:** required check names are present and match branch protection expectations.
4. **Artifact/check parity:** expected artifacts and gate outcomes match legacy behavior for the same change set.
5. **No open parity defects:** any mismatch tickets are resolved or explicitly waived with dated rationale.
6. **Maintainer judgment recorded:** explicit go/no-go call recorded in the active milestone plan.

## Operational process

1. Keep mapping table current with every workflow/check change.
2. Do not reintroduce standalone duplicate wrappers once modular replacements exist.
3. Update branch-protection requirements only after replacement checks are proven and names are frozen.
4. Any new parity-retention exception must be explicitly approved in planning docs before adding a duplicate workflow.

## Local/CI parity maintenance

When workflow entrypoints change under `.github/workflows/`:

- update `scripts/local_checks.py` for equivalent local reproduction,
- keep `make local-checks-pr` and `make local-checks` behavior aligned,
- document changed commands in the milestone plan/checklist.

---

## v1 consolidation record (v0.11.6 Task 60, 2026-07-13)

The 14-file modular inventory described above accumulated overlapping
responsibilities and duplicate executions. It was replaced by a three-workflow
architecture on branch `ci/v1-consolidation`. ADR-035 sections 2–5 are the
authoritative statement of the resulting policy; this section is the migration
evidence.

### Original inventory and keep/fold/delete decisions

| Legacy file | Decision | Rationale / destination |
|---|---|---|
| `ci-pr.yml` | fold | PR gates now come from the repository-owned profile `make local-checks-pr` run by `ci.yml` `pr-gate` (Python 3.10) + `tests-newest` (3.13). The lint/audit steps that only existed in YAML (ruff `--select N`, nbqa notebooks, pydocstyle, agent-instruction consistency) moved *into* the PR profile so local and CI cannot drift. Governance-schema stays a path-sensitive `ci.yml` job. `uv-install-smoke` (pip-vs-uv timing) was deleted from CI; the local lane `make uv-install-smoke` remains. Feature-branch push runs (duplicates of PR runs) were dropped. The evaluation/ freeze guard moved into the PR profile. |
| `ci-main.yml` | fold | Coverage + Codecov + per-module gates moved to `ci.yml` `coverage` (main pushes). Duplicated audits (anti-pattern, marker hygiene, private-member, instruction consistency) were removed from the main path — the PR profile already gates them pre-merge. Perf guard, core-vs-extras parity, and over-testing density (a second near-full coverage-context run per main push) moved to weekly `scheduled.yml`. Docs build moved to `ci.yml` `docs-build` (strict). |
| `ci-full.yml` | fold | Viz tests → path-sensitive `ci.yml` `viz-tests` (one job instead of core re-run + viz). Manual parity run → `scheduled.yml` `parity-reference` (also `workflow_dispatch`). |
| `ci-nightly.yml` | fold | All jobs → `scheduled.yml`, daily → weekly. `continue-on-error` shields on examples/notebooks removed: the notebook driver is advisory-with-artifact by its own `--mode advisory` contract; examples smoke is now a real failing check. |
| `ci-policy.yml` + `.github/actions/ci-policy/` | fold | Replaced by the always-run `policy` job in `ci.yml` executing the blocking full-inventory validator directly (no composite indirection, no diff-based advisory mode). |
| `deprecation-check.yml` | delete | Ran the full unit suite a second time on every PR. v1 zero-active-deprecation enforcement is the ledger gate in the PR profile (`scripts/local_checks.py --deprecation-ledger`); the focused-test lane remains via `--deprecation-closure` (release profile). |
| `ci-release-docs.yml` | delete | This mirror does not cut release branches. Strict docs are enforced on every docs-touching PR and main push (`ci.yml` `docs-build`, `-W --keep-going`) and in `make local-checks-release` / `release-preflight`. |
| `dependency-submission.yml` | delete | Required `contents: write` on every main push, violating the single-write-workflow rule. GitHub's native dependency graph continues to index the Python manifests; if snapshot-based submission is ever needed again it must satisfy ADR-035's fourth-workflow bar. |
| `sync-skills.yml` | delete | OSS CI must not check out `generic-skill-library`, depend on private skill repositories, or auto-commit agent skill content. The validator now rejects any such reference (regression guard). Skill promotion happens in the private source repository or via reviewed PRs. |
| `update_baseline.yml` | delete | Duplicate of the maintenance baseline task; consolidated into `maintenance.yml` (which also had its broken `scripts/check_perf_micro.py` path fixed to `scripts/perf/check_perf_micro.py` and its no-op `regen-docs` option removed). |
| `reusable-python-test.yml` | delete | Inlined; Python setup + constrained installs centralized in the `setup-ce-python` composite action. |
| `reusable-build-docs.yml` | delete | Inlined into `ci.yml` `docs-build` and `scheduled.yml` `docs-linkcheck`. |
| `reusable-run-make.yml` | delete | No callers. |

### Final inventory

| Workflow | Name | Jobs | Triggers |
|---|---|---|---|
| `ci.yml` | `CI` | `policy`, `changes`, `pr-gate`, `tests-newest`, `viz-tests` (path), `docs-build` (path/main), `packaging` (path), `governance-schema` (path), `required` (aggregate), `coverage` (main), `package-validation` (main) | `pull_request` → main, `push` → main, `workflow_dispatch` |
| `scheduled.yml` | `Scheduled assurance` | `full-matrix-tests` (3.10–3.13), `viz-tests`, `parity-reference`, `parity-core-vs-extras`, `perf-regression`, `notebook-execution`, `examples-smoke`, `docs-linkcheck`, `over-testing-analysis`, `dependency-audit` | weekly cron (Mon 03:00 UTC), `workflow_dispatch` |
| `maintenance.yml` | `Maintenance` | `update-baseline` (only write-capable job; opens reviewable PR; unique branch; requires reason) | `workflow_dispatch` only |

Required branch-protection check: **`CI / required`**. Local reproduction:
`make local-checks-pr` (PR gate), `make test-viz` (viz job),
`python -m sphinx -W --keep-going -b html docs docs/_build/html` (docs job),
`make local-checks-full` / `make local-checks-release` (main/heavier assurance),
`make check-ci-policy` (policy job), `make uv-install-smoke` (removed CI lane).

### Before/after quantification

Counts derive from the workflow definitions themselves; per-run duration
history was not API-accessible from the implementation environment (no
authenticated `gh`), which is why no wall-clock speedup numbers are claimed.

| Metric | Before | After |
|---|---|---|
| Top-level workflow files | 14 | 3 |
| Jobs triggered by an ordinary source PR (src+tests, non-viz) | 11–12 (`ci-pr` lint, mypy, 4× core-tests, uv-smoke, private-member, anti-pattern, governance filter, evaluation-freeze; plus `deprecation-check`), duplicated again when a feature-branch push accompanied the PR | 5–6 (`policy`, `changes`, `pr-gate`, `tests-newest`, `required`; specialists only when paths match); no push duplication |
| Complete/near-complete pytest executions per ordinary PR | 5 (4 matrix versions + deprecation-check unit suite) | 2 (PR profile on 3.10, suite on 3.13) |
| Dependency installations per ordinary PR | ~10 | 3–4 |
| Complete/near-complete pytest executions per main push | 2 (coverage + over-testing contexts) plus parity/perf/audit jobs | 1 (coverage) |
| Python environments in routine paths | 3.10–3.13 on every PR | 3.10 + 3.13 on PRs; full matrix weekly |
| Duplicated static checks | audits run in both `ci-pr` and `ci-main`; policy in a separate workflow | once, in the PR profile |
| Scheduled heavy jobs | daily (`ci-nightly`, 4 jobs) | weekly (`scheduled.yml`, 10 jobs incl. work moved off PR/main paths) |
| Write-permission workflows | 3 (`maintenance`, `update_baseline`, `dependency-submission`) plus `sync-skills` (write) | 1 (`maintenance.yml`) |

### Remaining platform-admin actions

1. Branch protection: require exactly `CI / required`; remove the former
   required checks (`CI — Pull Request checks / *`, `ci-policy/validate-workflows`).
2. Optional: revisit Dependabot-alert expectations tied to the removed
   dependency-submission snapshots (native dependency graph remains).

### Deferred / unresolved risks

- `actionlint` is not installed in the implementation environment and is not
  yet a governed CI-development dependency; workflow YAML is validated by
  PyYAML parsing (`run_ci_locally.py`), the full-inventory validator, and
  GitHub's own schema validation on push. Adding a pinned actionlint step to
  the `policy` job is a candidate follow-up.
- External-action SHA pins were carried over verbatim from the
  pre-consolidation workflows (same SHAs, same upstreams); version comments
  reflect the tag family current when the pins were adopted.
- PR-time parity-harness gating on `requirements.txt`/`constraints.txt`
  changes (status-appendix risk #3) remains future work; the weekly
  `parity-reference` job is still the earliest automated signal.
