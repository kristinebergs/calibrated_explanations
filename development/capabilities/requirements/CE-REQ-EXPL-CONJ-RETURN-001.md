# CE-REQ-EXPL-CONJ-RETURN-001 — Conjunction Return Type and Cardinality Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-CONJ-RETURN-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-EXPL-CONJ-001 |
| adr_refs | ADR-008 |
| status | active |
| verification_status | verified |
| applicable_on | collection (CalibratedExplanations, AlternativeExplanations) |
| supersedes | CE-REQ-EXPL-CONJ-001 (partial) |
| tif_refs | CE-TIF-EXPL-CONJ-001 |
| verification_strength | api_contract |
| evidence_level | raw_evidence |

## Scope

Public API:
- `CalibratedExplanations.add_conjunctions(...)` — factual collection
- `AlternativeExplanations.add_conjunctions(...)` — alternative collection

Applicable task types: binary classification, multiclass classification, regression, probabilistic_regression.

Applicable workflow: `WrapCalibratedExplainer` → `fit` → `calibrate` → `explain_factual` /
`explore_alternatives` → `add_conjunctions`. Individual explanation objects (`FactualExplanation`,
`AlternativeExplanation`) are out of scope for the cardinality part of this requirement; their
non-None return is covered by CE-REQ-EXPL-CONJ-API-001.

## Observable behavior

When `add_conjunctions()` is called on a collection returned by `explain_factual(X)` or
`explore_alternatives(X)`:

1. The return value is not `None`.
2. The return value supports `len()`.
3. `len(result) == len(X)` — collection cardinality is preserved.

## Acceptance criterion

For both `CalibratedExplanations` (factual) and `AlternativeExplanations` (alternative):

1. `observation.result_is_none` is `False`.
2. `observation.result_len` equals `len(X_test)` (the number of test instances).

This criterion is verified by CE-TIF-EXPL-CONJ-001 with default parameters.

## TIF reference

**TIF ID:** CE-TIF-EXPL-CONJ-001

TIF scenario function: `run_conjunction_tif_scenario()`
Observation fields verified:
- `result_is_none` (must be `False`)
- `result_len` (must equal `len(X_test)`)

## Verification method

Automated pytest tests calling CE-TIF-EXPL-CONJ-001, asserting return value and length.

## Verification targets

- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_preserve_cardinality_when_factual_collection_add_conjunctions`
- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_preserve_cardinality_when_alternative_collection_add_conjunctions`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| tif_ids | yes |
| dataset_id | yes |
| random_seed | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies the return type contract and collection cardinality only.
It does not verify:
- That the returned object contains semantically valid rules.
- That conjunction rules combine multiple features (see CE-REQ-EXPL-CONJ-RULE-001).
- Cardinality of individual explanation objects (those are not collections with len()).
- Any statistical or calibration properties.
