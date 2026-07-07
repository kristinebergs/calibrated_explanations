# CE-REQ-EXPL-ALT-RETURN-001 — Alternative Explanation Return Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-ALT-RETURN-001 |
| obligation_type | output_contract |
| claim_refs | CE-CAP-EXPL-002 |
| adr_refs | ADR-008, ADR-026 |
| status | active |
| verification_status | verified |
| applicable_on | collection (AlternativeExplanations) and individual (AlternativeExplanation) |
| tif_refs | CE-TIF-EXPL-001 |
| verification_strength | behavioral_contract |
| evidence_level | raw_evidence |

## Scope

Public API: `WrapCalibratedExplainer.explore_alternatives(X)` return contract.

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explore_alternatives.

## Observable behavior

The collection returned by `explore_alternatives(X)` must:

1. Have length equal to the number of rows in `X` (`len(result) == len(X)`).
2. Support indexing: `result[i]` is not `None` for all valid `i`.
3. Each `result[i]` is an `AlternativeExplanation` instance.
4. `type(result).__name__` is `"AlternativeExplanations"`.

## Acceptance criterion

For a fitted and calibrated `WrapCalibratedExplainer` and test set `X_test`:

- `len(result) == len(X_test)`.
- `result[0]` is not `None`.
- `type(result).__name__` is `"AlternativeExplanations"`.

## TIF reference

**TIF ID:** CE-TIF-EXPL-001

TIF scenario function: `run_alternative_tif_scenario()`
Observation fields: `result_len`, `result_is_none`, `first_item_is_none`, `result_type_name`

## Verification method

Automated pytest tests calling CE-TIF-EXPL-001.

## Verification targets

- `pytest: tests/capabilities/test_explanation_contracts.py::test_should_preserve_cardinality_when_alternative_explain`
- `pytest: tests/capabilities/test_explanation_contracts.py::test_should_return_alternative_explanations_type_when_explore_alternatives`

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

This requirement verifies the return type and structural contract only. It does not verify:

- That alternative scenarios are achievable in practice.
- That changed feature values remain within the natural data distribution.
- Statistical validity under distribution shift.

See `CE-CAP-EXPL-002` for the full assumption statement.
