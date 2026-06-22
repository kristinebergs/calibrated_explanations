# Capability Requirements

This directory contains CE requirement files derived from capability claims in
`development/capabilities/claims/`.

Each requirement file translates one or more capability claims into specific,
testable obligations on the CE public API, following the verification chain
defined in `development/README.md`:

```text
ADR / Standard
    -> constrains
Capability claim         -> development/capabilities/claims/
    -> decomposes into
Requirement              (this directory)
    -> is exercised through
TIF verification interface -> development/capabilities/verification/tif/
    -> is executed by
Test / verification gate -> tests/capabilities/
    -> produces
Evidence record          -> reports/verification/ (raw)
                         -> development/capabilities/evidence/ (curated)
```

## Location authority

This directory is the canonical location for CE requirements.
See `development/README.md` for the full location map.

## File naming

Requirement files use the prefix `CE-REQ-` and are stored as Markdown:

```
CE-REQ-<AREA>-<FACET>-<NNN>.md
```

Examples: `CE-REQ-EXPL-API-001.md`, `CE-REQ-PRED-API-001.md`

## Requirement structure

Each requirement file must state:

- **requirement_id**: unique CE-REQ-... identifier
- **obligation_type**: one of `api_contract`, `payload_schema`, `numerical_behavior`,
  `statistical_method_alignment`, `documentation_boundary`, `visualization_behavior`,
  `plugin_behavior`, `runtime_behavior`, `serialization_contract`, `static_policy`,
  `quality_gate`, `empirical_smoke`
- **claim_refs**: which CE-CAP-... claims this requirement serves
- **adr_refs** or **standard_refs**: which active ADRs / Standards govern the obligation
- **status**: whether the requirement is active, superseded, or retired
- **verification_status**: `verified` only when executable evidence exists, or
  `adr_gap_open` / `not_implemented` when implementation or executable evidence is missing
- **scope**: public API surface, task type, and workflow applicable
- **observable_behavior**: what must be true when the requirement is satisfied
- **acceptance_criterion**: the measurable or checkable condition
- **tif_refs** or **tif_exemption**: which CE-TIF-... interface exercises this requirement,
  or an explicit justification for why no TIF interface is needed
- **verification_method**: how the criterion is checked
- **verification_targets**: executable pytest targets or real quality gates
- **evidence_required**: metadata that a passing evidence record must include
- **assumption_boundary**: what the requirement and its evidence do not prove

## TIF requirement

Every implemented behavioral requirement must reference at least one TIF interface
in `tif_refs` unless it includes a `tif_exemption`.

### Valid TIF exemptions

TIF exemptions are rare. They are permitted only for:

- Static documentation-boundary checks (documentation structure, not behavior)
- Schema-only validation (YAML/JSON file structure, not runtime behavior)
- Repository policy checks (file existence, naming conventions)
- Pure metadata/linkage checks (ADR/requirement cross-reference integrity)

A TIF exemption must not be used for behavioral CE public-API requirements.
If a requirement specifies observable API behavior, it must have a TIF reference.

To declare a TIF exemption:

```markdown
## TIF exemption

tif_exemption: <one of: documentation_boundary | schema_validation | repository_policy | metadata_linkage>
tif_exemption_rationale: >
  <Specific reason why no TIF interface is needed. Cite why the behavior is not
  observable through WrapCalibratedExplainer. Must not be used for behavioral
  requirements.>
```

## Requirements-as-code evidence policy

Requirements-as-code means executable evidence, not traceability prose. For every
implemented behavioral requirement, the verification chain must terminate in at
least one executable target that directly verifies observable behavior. Prefer
`pytest: tests/.../test_*.py::test_should_...` references; quality scripts or CI
gates are acceptable only when they execute a concrete check for the stated
behavior.

Metadata/linkage tests are drift guards only. They can prove that ADR, claim, and
requirement references remain navigable, but they do not prove API behavior,
runtime behavior, serialization behavior, payload schema behavior, plugin
behavior, visualization behavior, numerical behavior, statistical method
alignment, static policy compliance, or empirical smoke behavior.

Human/manual review is not valid evidence for implemented behavioral
requirements. Terms such as `manual_review`, `manual_review_required`, or
`human verification` may appear only for requirements whose `verification_status`
is `adr_gap_open` or `not_implemented`, and those requirements must include a
`gap_ref` or `adr_gap_ref` that is present in
`development/current-work/RELEASE_PLAN_status_appendix.md`.

If an ADR-governed requirement is not implemented or lacks executable evidence,
do not mark it verified. Register it as an ADR gap with the requirement ID, ADR
ID, missing behavior or evidence, why it is not verified, and intended closure
path. Every behavioral requirement must therefore have exactly one of these
outcomes: executable evidence that exists and runs, or an explicit ADR gap.

## Verification strength model

Avoid generic `verified: true` without specifying what was proven. Use
`verification_strength` to distinguish what dimension was tested:

| Strength | Meaning |
|---|---|
| `api_contract` | The method exists and is callable from the public workflow |
| `behavioral_contract` | The method produces the documented observable behavior |
| `numerical_behavior` | Numeric output values satisfy documented invariants |
| `statistical_method_alignment` | Statistical properties satisfy documented assumptions |
| `empirical_smoke` | Basic end-to-end execution without detailed output validation |
| `documentation_boundary` | Documentation states what is and is not proven |
| `visualization_structure` | Visualization output matches expected structural contract |
| `policy_check` | Repository or CI policy compliance |

A requirement can be verified as `api_contract` while not verified as
`statistical_method_alignment` or semantic explanation quality.

Use `evidence_level` to record what form of evidence exists:

| Level | Meaning |
|---|---|
| `raw_evidence` | Machine-readable output in `reports/verification/` |
| `curated_summary` | Human-reviewed summary in `development/capabilities/evidence/` |
| `ci_gate` | CI pass/fail recorded in pipeline artifacts |
| `metadata_only` | Traceability links only — cannot prove behavioral requirements |

## Rules

1. Requirements are not tests. Do not embed test code in requirement files.
2. Every requirement must have at least one `claim_ref`.
3. Every requirement must have a stated `verification_method`, `verification_targets`, and `acceptance_criterion`.
4. Acceptance criteria must be visible here, not hidden inside test code only.
5. Statistical obligations must state their assumptions explicitly.
6. Implemented behavioral requirements must cite executable pytest targets or real quality gates.
7. Metadata-only tests cannot be the sole evidence for implemented behavioral requirements.
8. Unimplemented or unverified ADR obligations must be registered as ADR gaps, not treated as verified.
9. Every implemented behavioral requirement must have `tif_refs` pointing to a CE-TIF-... interface,
   or a `tif_exemption` with a valid exemption type and rationale.

---

## Structuring guide: when one operation spans multiple object types

### Rule R-1 — One requirement per OPERATION, not per class

Requirements decompose by **operation** (what is being called), not by the concrete
class it is called on. A single requirement covers the same operation on all object
types listed in its scope.

```
WRONG: CE-REQ-EXPL-FILTER-SUPER-COL-001  (super on collection only)
       CE-REQ-EXPL-FILTER-SUPER-IND-001  (super on individual only)

RIGHT: CE-REQ-EXPL-FILTER-SUPER-001      (super on both collection and individual)
       — acceptance criterion has separate entries for each object type
```

Exception: if the operation has **materially different contracts** on different object
types (different return types, different preconditions, or different failure modes that
users must handle separately), then separate requirements are appropriate.

### Rule R-2 — Always separate requirements for SEPARATE OPERATIONS

Operations that are semantically distinct (e.g., `super_explanations`, `semi_explanations`,
`counter_explanations`, `ensured_explanations`, `pareto_explanations`) must be separate
requirements even when they share parameters and return types.

```
WRONG: CE-REQ-EXPL-FILTER-001  (lumps super + semi + counter + ensured + pareto)

RIGHT: CE-REQ-EXPL-FILTER-SUPER-001
       CE-REQ-EXPL-FILTER-SEMI-001
       CE-REQ-EXPL-FILTER-COUNTER-001
       CE-REQ-EXPL-FILTER-ENSURED-001
       CE-REQ-EXPL-FILTER-PARETO-001
```

### Rule R-3 — State applicable object level; do not split requirements on it

When a method exists on both a collection type and an individual explanation type,
use ONE requirement that covers both. State which object levels apply in the
`applicable_on` field of the Metadata table:

```markdown
| applicable_on | collection (CalibratedExplanations, AlternativeExplanations) and individual (FactualExplanation, AlternativeExplanation) |
```

The acceptance criterion must contain separate sub-entries for collection and
individual to ensure each is independently verifiable. Tests can be separate
functions within the same test file — one per object level.

Do NOT create separate requirements just because the same operation is callable
on both a collection and an individual.

```
WRONG: CE-REQ-EXPL-CONJ-COL-001  (add_conjunctions on collection only)
       CE-REQ-EXPL-CONJ-IND-001  (add_conjunctions on individual only)

RIGHT: CE-REQ-EXPL-CONJ-API-001  (add_conjunctions; applicable_on: collection and individual)
```

Exception: if the operation has **materially different contracts** on different
object types (different return types, different preconditions, or different failure
modes that users must handle separately), then separate requirements are appropriate.

### Rule R-4 — Aliases do not need separate requirements

When a short-form alias delegates directly to the canonical method (e.g., `.super()`
delegates to `.super_explanations()`), one requirement covers both. State the alias
explicitly in scope and note "alias delegator — verified by the canonical test."

### Rule R-5 — Parameter variants: assertions within the same requirement

A parameter that selects a **different code path** (e.g., `max_rule_size=1` disables
conjunction generation; `max_rule_size=2` enables pairs) requires coverage in the
requirement's acceptance criterion and tests. Use `pytest.mark.parametrize` to cover
meaningful parameter values **within the SAME requirement**. Do NOT create a separate
requirement just because a test uses a different parameter value.

A parameter that only changes the **count or size of output** (e.g., `n_top_features`
controlling how many features are considered) does not require a separate requirement.

### Rule R-6 — Each behavioral requirement must have executable evidence

Every implemented behavioral requirement file must reference at least one named
pytest test or real quality gate in its `Verification targets` section. Tests for
a family of related requirements can live in one test file, but each behavioral
requirement must have its own executable target. Governance-only requirements may
use metadata/linkage drift guards, but those drift guards must not be reused as
sole evidence for behavioral requirements.

## Related locations

| Material | Location |
|---|---|
| Capability claims that generate requirements | `development/capabilities/claims/` |
| TIF verification interfaces | `development/capabilities/verification/tif/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated (raw) verification run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
