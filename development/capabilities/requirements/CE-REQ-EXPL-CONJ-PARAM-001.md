# CE-REQ-EXPL-CONJ-PARAM-001 — max_rule_size=1 Suppresses Multi-Feature Conjunctions

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-CONJ-PARAM-001 |
| obligation_type | behavioral_contract |
| claim_refs | CE-CAP-EXPL-CONJ-001 |
| adr_refs | ADR-008 |
| status | active |
| verification_status | verified |
| applicable_on | collection (CalibratedExplanations, AlternativeExplanations) |
| supersedes | CE-REQ-EXPL-CONJ-001 (partial) |
| tif_refs | CE-TIF-EXPL-CONJ-001 |
| verification_strength | behavioral_contract |
| evidence_level | raw_evidence |

## Scope

Public API:
- `CalibratedExplanations.add_conjunctions(n_top_features, max_rule_size)` — factual collection
- `AlternativeExplanations.add_conjunctions(n_top_features, max_rule_size)` — alternative collection

Applicable condition: `max_rule_size == 1`.

Applicable workflow: `WrapCalibratedExplainer` → `fit` → `calibrate` → `explain_factual` /
`explore_alternatives` → `add_conjunctions(max_rule_size=1)`.

## Observable behavior

When `add_conjunctions(max_rule_size=1)` is called on a collection, no item in the
collection should have `has_conjunctive_rules == True`. Single-feature rules remain
(the API still completes successfully), but no multi-feature conjunction rules are
generated.

## Acceptance criterion

For `factual = explainer.explain_factual(X_test)` after
`factual.add_conjunctions(max_rule_size=1)`:

1. No item in `[factual[i] for i in range(len(factual))]` has
   `item.has_conjunctive_rules == True`.
2. `observation.any_has_conjunctive_rules` is `False`.
3. `observation.exception_raised` is `False` (API still completes without error).

This is verified by CE-TIF-EXPL-CONJ-001 with `max_rule_size=1`.

## TIF reference

**TIF ID:** CE-TIF-EXPL-CONJ-001

TIF scenario function: `run_conjunction_tif_scenario()`
Observation fields verified:
- `any_has_conjunctive_rules` (must be `False` when `max_rule_size == 1`)
- `exception_raised` (must be `False`)

## Verification method

Automated pytest test calling CE-TIF-EXPL-CONJ-001 with `max_rule_size=1`,
asserting `any_has_conjunctive_rules == False`.

## Verification targets

- `pytest: tests/capabilities/test_conjunction_contracts.py::test_should_not_produce_conjunctive_rules_when_max_rule_size_one`

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

This requirement verifies that `max_rule_size=1` suppresses multi-feature conjunction
rule generation. It does not verify:
- The exact number or content of single-feature rules that remain.
- That the returned collection is empty or unchanged.
- Behavior under `max_rule_size=0` (undefined/invalid input).
- Any statistical or calibration properties.
