---
name: ce-release-check
description: >
  Read release-plan state and select the next ADR-compliant actionable development
  step.
model: haiku
---

# CE Release Check

You are determining the next development step from the release plan.
This skill implements the "proceed according to plan" workflow defined in
`execution-plan.instructions.md`.

**Mandatory sequence:**
1. Read `development/current-work/RELEASE_PLAN_v1.md`.
2. Identify the current released version and the target next milestone.
3. List outstanding gates and work items for that milestone.
4. Verify that the proposed next step is allowed by all relevant ADRs.
5. If an ADR constraint and a plan step conflict, the ADR wins.

---

## Files to read

```
development/current-work/RELEASE_PLAN_v1.md   ← primary source: current version, milestones, gates
development/adrs/                ← governance constraints (ADR takes precedence)
CHANGELOG.md                          ← completed items; do not duplicate
```

---

## Step 1 — Identify current state

```markdown
Current released version: v<X.Y.Z>   (from RELEASE_PLAN_v1.md top section)
Target next milestone:    v<X.Y.Z+1>
```

---

## Step 2 — Scan open gates for the target milestone

Look for sections like:
```
### v0.11.0 — <milestone name>
#### Gates
- [ ] ADR-NNN gap <description>
- [x] (already closed)
```

Gates marked `[ ]` are outstanding. List them with their ADR reference.

---

## Step 3 — Cross-reference ADRs

For each outstanding gate, identify the governing ADR(s) using `ce-adr-consult`.
If the proposed work from the plan conflicts with an ADR decision, **stop** and
flag the conflict rather than proceeding.

---

## Step 4 — Select the next actionable item

Priority order:
1. **Blocking gates** — items explicitly labeled as release blockers.
2. **Open ADR implementation gaps** — items in the ADR roadmap summary with open status.
3. **Non-gate improvements** — feature additions scheduled for the milestone.

---

## Step 5 — Verify against CHANGELOG

```bash
# Check what was delivered recently
head -100 CHANGELOG.md
```

Do not propose work that is already present in `CHANGELOG.md`.

When recommending closure evidence for a planned task, prefer:
- `make local-checks-task TASK=<n>` for task completion
- `make local-checks-pr` for PR preflight
- `make local-checks-release` for release-boundary validation

---

## Output format

```
Release Check: <date>
======================
Current released version:  v<X.Y.Z>
Target next milestone:      v<X.Y.Z+1>

Outstanding gates:
  1. [ADR-NNN] <brief description>
  2. [ADR-NNN] <brief description>
  ...

Next actionable step:
  Work item: <title>
  ADR(s):    ADR-NNN (Decision section: <binding rule>)
  Plan ref:  RELEASE_PLAN_v1.md § <section>
  Rationale: <one sentence>

ADR conflicts detected: NONE | <list if any>
```

---

## Completing a work item (CHANGELOG update)

When an item is completed satisfactorily, add it to `CHANGELOG.md` under
the appropriate section header:

```markdown
## [Unreleased]

### Added
- Implemented `to_primitive` / `from_primitive` calibrator serialization (ADR-031).

### Fixed
- ...
```


---

## Evaluation Checklist

- [ ] `RELEASE_PLAN_v1.md` read before proposing any step.
- [ ] Current version and target milestone clearly identified.
- [ ] Outstanding gates listed with ADR references.
- [ ] No ADR constraint violated by the proposed next step.
- [ ] `CHANGELOG.md` checked to avoid duplicate work.
- [ ] Completed items added to `CHANGELOG.md` under `[Unreleased]`.
