# CE Version Plan Reference (`vX.Y.Z_plan.md`)

Use this as the canonical scaffold for release implementation plans under
`development/current-work/`.

## Required front matter

Every plan declares:

1. `# v<plan-label> Release Task Implementation Plan`
2. A blockquote in the form `> **Release version:** X.Y.Z` (with the version
   rendered as inline code).
3. A blockquote in the form `> **Development version:** X.Y.Z-dev` (also inline code).
4. Scope, milestone type, and the authoritative milestone section in
   `development/current-work/RELEASE_PLAN_v1.md`.

For prereleases, the filename label and package version may differ. For example,
`v1.0.0-rc_plan.md` may declare release version `1.0.0rc1` and development
version `1.0.0-rc-dev`. The explicit fields are authoritative for automation.

## Mandatory sections

1. `## Source references reviewed`
2. `## Release tasks covered (from RELEASE_PLAN_v1.md)`
3. `## Global rules` when applicable
4. Numbered task sections matching the milestone tasks
5. `## N) Release preparation` as the final numbered task
6. `## Release gate summary`
7. `## Minimal new tests required`

## Task section contract

Each numbered task includes:

1. Goal
2. Status assessment: `Not started`, `Partial`, or `Implemented with evidence`
3. Relevant ADRs, standards, and source files
4. Current anchors in code/docs
5. Remaining gaps
6. Concrete ordered implementation steps
7. Verification checklist with commands and expected results

## Evidence rules

1. Mark a task complete only with verifiable code/doc/test evidence.
2. Do not rely on prior status prose alone.
3. When uncertain, use `Partial` and name the missing evidence.
4. Keep assumptions explicit.
5. A generated next-plan scaffold is not a completed plan; replace it through
   `ce-release-planner` before claiming step 16 closure.

## Release gate summary requirements

The summary maps every release criterion to evidence, names unresolved
blockers, and ends with `Ready to close` or `Not ready` plus reasons.

## Minimal tests requirements

List only tests/scripts strictly required for the remaining gaps, grouped by
task. Release workflow tests must use multiple synthetic versions, including a
non-patch transition when relevant; never assert only the current release
number.

## Suggested task skeleton

```md
# vX.Y.Z Release Task Implementation Plan

> **Release version:** `X.Y.Z`
> **Development version:** `X.Y.Z-dev`

## Source references reviewed

## Release tasks covered (from RELEASE_PLAN_v1.md)

## 1) <Task title>
### 1.0 Goal
### 1.1 Status assessment
### 1.2 Relevant references
### 1.3 Current anchors in code/docs
### 1.4 Gaps
### 1.5 Implementation steps
### 1.6 Verification checklist

## N) Release preparation
...

## Release gate summary

## Minimal new tests required
```

## Release preparation task template

Copy this as the final numbered task and substitute `N`, the plan label, exact
release version, and development/next milestone values. Keep the step numbers
aligned with `release.md`.

```md
## N) Release preparation

### N.0 Goal

Complete the full `release.md` sequence for vX.Y.Z. Repository automation owns
steps 1-10 and 14-17; the maintainer owns only the immutable/external steps
11-13. This task is always last and executes only after every prior task is
closed or explicitly deferred.

### N.1 Status assessment

Not started.

### N.2 Relevant references

- `release.md` — authoritative 17-step maintainer sequence
- `development/current-work/RELEASE_PLAN_v1.md` — milestone and next-version authority
- this version plan — exact release/development version declarations
- `Makefile` — `release-preflight`, `release-finalize`, `release-postcommit`
- `scripts/local_checks.py` — version-agnostic implementation and reports
- `.claude/skills/ce-release-planner/SKILL.md` — next-plan completion contract

### N.3 Current anchors in code/docs

- `make release-preflight` owns steps 1-10: readiness, release-file/changelog
  preparation, tests, notebooks, alignment, clean build, Twine/artifact checks,
  and clean-wheel smoke.
- `make release-finalize` proves the preflight snapshot is still current.
- Steps 11-13 remain human-gated: commit/tag/push, RTD publication, PyPI upload.
- `make release-postcommit` owns steps 14-17: PyPI page/metadata verification,
  exact published-install smoke, plan handoff/archive, and next-development bump.

### N.4 Gaps

All prior milestone tasks and release-file content must be complete before
preflight. A placeholder next plan produced by postcommit must be expanded with
`ce-release-planner`; no additional maintainer release action may be hidden
outside steps 11-13.

### N.5 Implementation steps

1. Confirm tasks 1-(N-1) are closed or have explicit maintainer-approved deferrals.
2. Confirm the plan's exact release and development version declarations match
   the milestone; use `VERSION=` only for a deliberate override.
3. Run `make release-preflight` (release.md steps 1-10). Confirm its report lists
   `automated_release_steps: [1..10]`, the updated release files, and a green result.
4. Run `make release-finalize` immediately before handoff.
5. Maintainer performs release.md steps 11-13 only: commit/tag/push, publish and
   verify RTD, upload the built artifacts to PyPI.
6. Run `make release-postcommit` (steps 14-17). Use `NEXT_VERSION=<milestone>`
   only if the master plan does not already name the intended next milestone.
7. If postcommit scaffolded the next plan, immediately replace the placeholder
   with a complete plan through `ce-release-planner`; verify master tracking,
   released-plan archive, and development version are correct.

### N.6 Verification checklist

- [ ] All earlier tasks closed or explicitly deferred.
- [ ] Exact release/development versions declared; no prior-release literal drives automation.
- [ ] `make release-preflight` exits 0 and reports steps 1-10 plus all release-file updates.
- [ ] `make release-finalize` exits 0 on the unchanged preflight snapshot.
- [ ] Maintainer confirms manual steps 11-13 completed successfully.
- [ ] `make release-postcommit` exits 0 and reports steps 14-17.
- [ ] PyPI page/metadata and exact clean-environment install verify the released version.
- [ ] Released plan archived; next maintained plan exists and is content-complete.
- [ ] Master release tracking names the released and next milestones correctly.
- [ ] `pyproject.toml` and runtime fallback use the declared next development version.
- [ ] Version-agnostic release workflow tests pass for more than one version line.
```
