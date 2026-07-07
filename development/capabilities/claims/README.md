# Capability Claims

This directory contains CE capability claim files for the `calibrated_explanations` library.

Each claim file documents a user-visible statement about what CE provides, following
the verification chain defined in `development/README.md`:

```text
ADR / Standard
    -> constrains
Capability claim         (this directory)
    -> decomposes into
Requirement              -> development/capabilities/requirements/
    -> is exercised through
TIF verification interface -> development/capabilities/verification/tif/
    -> is executed by
Test / verification gate -> tests/capabilities/
    -> produces
Evidence record          -> reports/verification/ (raw)
                         -> development/capabilities/evidence/ (curated)
```

## Location authority

This directory is the canonical location for CE capability claims. Canonical
locations are ratified by
`development/adrs/ADR-040-capability-verification-framework.md` (D2);
`development/README.md` mirrors the full location map. On conflict, ADR-040 is
authoritative.

## File naming

Claim files use the prefix `CE-CAP-` and are stored as YAML:

```
CE-CAP-<AREA>-<NNN>.yaml
```

Examples: `CE-CAP-EXPL-001.yaml`, `CE-CAP-PRED-001.yaml`

## Claim schema (illustrative)

```yaml
claim_id: CE-CAP-EXPL-001
claim_type: capability
owner: calibrated_explanations
status: current
claim_text: >
  One-sentence statement of the user-visible capability.
public_api:
  - WrapCalibratedExplainer.explain_factual
requirements:
  - CE-REQ-EXPL-API-001
  - CE-REQ-EXPL-RETURN-001
  - CE-REQ-EXPL-DOC-001
verification:
  proves:
    - api_contract
evidence_required:
  - commit_sha
  - package_version
  - test_id
  - result
```

## Rules

1. **Claims describe what CE provides.** Requirements describe how the claim is proven.
2. Every claim must have an `owner` and at least one requirement ID.
3. Do not duplicate definitions that already exist in another claim.
4. Statistical claims must state their assumptions (calibration data,
   exchangeability, task-type scope, empirical vs theoretical boundary).
5. Do not mark roadmap or unsupported behavior as `status: current`.
6. Claims describe existing CE behavior only — they do not introduce new functionality.

### What makes a claim too detailed

A claim is too detailed if it contains:
- acceptance criteria
- exact test scenarios
- parameter-specific behavior
- return-type obligations
- implementation mechanics
- fixture details
- evidence fields

If a claim contains any of these, move that content to a requirement.

### Claim decomposition

A claim should normally fan out into multiple requirements.

A claim with exactly one requirement must include an explicit `atomic_rationale`
field explaining why the capability cannot usefully be decomposed — specifically
why no separate API, schema, semantic, error, documentation, or evidence
requirement is needed.

```yaml
atomic_rationale: >
  This claim maps to one requirement because <specific reason>. No separate
  API, schema, semantic, error, documentation, or evidence requirement is
  warranted because <specific reason>.
```

### Claim stability

Claims should remain stable under public API refactoring. Method names may
appear in the `public_api` list, but the `claim_text` itself should remain
at capability level — describing what CE provides to users, not which specific
method implements it.

---

## Structuring guide: when one operation spans multiple object types

### Rule C-1 — Claims describe capabilities, not implementations

A claim covers the full capability regardless of which concrete class exposes it.
Write ONE claim per conceptual capability group, even if the method exists on multiple
classes (e.g., collection and individual, factual and alternative).

```
WRONG: CE-CAP-EXPL-CONJ-FAC-001  (conjunctions — factual collection only)
       CE-CAP-EXPL-CONJ-ALT-001  (conjunctions — alternative collection only)

RIGHT: CE-CAP-EXPL-CONJ-001      (conjunctions — all applicable types)
```

### Rule C-2 — public_api lists all first-class entry points

List every public_api entry point users are expected to call directly.
Do not list implementation helpers that are only called transitively.

### Rule C-3 — requirements list must be exhaustive

The `requirements` list in a claim must reference every requirement that
derives from it. Requirements are separated by **operation** (see R-2), not by
object level (collection vs individual) — a single requirement covers all
applicable object levels and declares them in its `applicable_on` field.

### Rule C-4 — operation families share one claim

When several operations share the same conceptual purpose and similar parameter
signatures (e.g., super/semi/counter/ensured/pareto all filter alternative
explanations by different criteria), use ONE claim for the family. The
requirements list then contains one entry per distinct operation.

### Rule C-5 — use probabilistic_regression for threshold-based regression queries

`probabilistic_regression` is a distinct task type: a regression model is queried
with `threshold=` to return P(Y > threshold | X) rather than a point estimate or
interval. Valid task type values and when to use them:

| Task type | When to use |
|---|---|
| `binary_classification` | Two-class classification model |
| `multiclass_classification` | More than two classes |
| `regression` | Continuous output model: point estimates, UQ intervals |
| `probabilistic_regression` | Regression model queried with `threshold=` for P(Y > threshold) |

List `probabilistic_regression` explicitly when a capability applies to regression
models in threshold mode. A capability that applies to all regression modes must list
BOTH `regression` AND `probabilistic_regression`. Do NOT list `probabilistic_regression`
for capabilities that are specific to non-threshold regression (e.g., UQ intervals via
`predict(X, uq_interval=True)`) or to classification.

## Related locations

| Material | Location |
|---|---|
| Requirements derived from claims | `development/capabilities/requirements/` |
| TIF verification interfaces | `development/capabilities/verification/tif/` |
| Verification scenarios and helpers | `development/capabilities/verification/` |
| Pytest capability tests | `tests/capabilities/` |
| Generated (raw) verification run outputs | `reports/verification/` |
| Curated capability evidence summaries | `development/capabilities/evidence/` |
