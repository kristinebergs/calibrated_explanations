---
name: ce-release-planner
description: >
  Analyze RELEASE_PLAN_v1.md for an upcoming release version and produce a
  detailed vX.Y.Z_plan.md implementation plan with task breakdowns.
---

# CE Release Planner

You are creating a versioned implementation plan for an upcoming CE release.

## Required references

- `development/current-work/RELEASE_PLAN_v1.md` (master release plan with milestones,
  ADR gap appendix, and release gates)
- `release.md` (maintainer release sequence; steps 1-10 automated by preflight,
  11-13 manual, and 14-17 automated by postcommit)
- `scripts/local_checks.py` and `Makefile` (executable release workflow)
- `references/version_plan_reference.md` (canonical structure/template for
  `development/current-work/vX.Y.Z_plan.md` files)
- All ADR files referenced by the target milestone
- All STD files referenced by the target milestone
- Existing version plans for pattern reference:
  `development/finished-work/v0.11.1_plan.md`, `development/finished-work/v0.11.5_plan.md`

## Use this skill when

- Planning the next release version.
- Creating a new `vX.Y.Z_plan.md` from the master release plan.
- Reviewing which tasks from `RELEASE_PLAN_v1.md` apply to a specific version.

## Workflow

1. **Identify target version.**
   - Read `RELEASE_PLAN_v1.md` to find the current released version and the
     next planned milestone.
   - Confirm with the user which version to plan.
   - Record both the exact PEP 440 release version and its development
     placeholder. Do not infer `X.Y.(Z+1)` when the master plan names a minor,
     major, or prerelease milestone.

2. **Extract tasks from the master plan.**
   - Read the target milestone section in `RELEASE_PLAN_v1.md`.
   - For each task, identify:
     - governing ADRs and standards
     - current implementation status (check the appendix gap tables)
     - dependencies on other tasks or prior milestones

3. **Read governing ADRs.**
   - For each referenced ADR, read the full ADR and its appendix gap table.
   - Note which gaps are already closed vs. still open.

4. **Check current codebase state.**
   - For each task, identify the relevant source modules and their current
     implementation state.
   - Note any tasks already partially or fully completed.

5. **Draft the plan.**
   - Create `development/current-work/vX.Y.Z_plan.md` following the structure of
     `references/version_plan_reference.md` and existing plans
     (v0.11.1_plan.md, v0.11.5_plan.md).
   - Each task section must include:
     - goal statement
     - relevant references (ADRs, standards, source files)
     - current status assessment
     - implementation steps (concrete, actionable)
     - verification checklist
   - Include a release gate summary at the end.
   - Make the final `Release preparation` task reproduce the complete
     `release.md` handoff contract:
     - `make release-preflight` performs steps 1-10, including release-file and
       changelog updates;
     - `make release-finalize` guards the handoff;
     - the maintainer performs only steps 11-13 (commit/tag/push, RTD, PyPI);
     - `make release-postcommit` performs steps 14-17 (PyPI page/metadata,
       published-install smoke, plan handoff/archive, development-version bump).
   - If postcommit created a placeholder plan, replace it with the complete
     version plan required by this skill; a scaffold is not closure evidence.

6. **Cross-check completeness.**
   - Verify every task from the master plan milestone has a section.
   - Verify every open gap from the appendix for referenced ADRs is addressed.
   - List minimal new tests required.
   - Verify that no command, path, expected version, or next-version rule is
     hard-coded to the release that happened to precede the target milestone.

## Output contract

Produce `development/current-work/vX.Y.Z_plan.md` with:
- header identifying version, milestone type, and authoritative task source
- explicit exact release version and development version declarations
- source references reviewed
- global rules section (if applicable)
- numbered task sections matching the master plan
- release gate summary
- minimal new tests section

## Constraints

- Do not invent tasks not in `RELEASE_PLAN_v1.md` without explicit user approval.
- Do not modify `RELEASE_PLAN_v1.md` itself (use `ce-adr-author` for that).
- Mark tasks as completed only when verification evidence exists in the codebase.
- Respect the current plan format conventions from the reference and maintained
  finished-work examples.
- Do not duplicate the old manual release-file checklist in a version plan;
  reference the executable steps 1-10 / 11-13 / 14-17 contract from
  `release.md` and keep only milestone-specific closure evidence.
