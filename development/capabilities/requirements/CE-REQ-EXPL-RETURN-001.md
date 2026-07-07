# CE-REQ-EXPL-RETURN-001 — Factual Explanation Return Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-EXPL-RETURN-001 |
| obligation_type | output_contract |
| claim_refs | CE-CAP-EXPL-001 |
| adr_refs | ADR-008, ADR-026 |
| status | active |
| verification_status | verified |
| applicable_on | collection (CalibratedExplanations) and individual (FactualExplanation) |
| tif_refs | CE-TIF-EXPL-001 |
| verification_strength | behavioral_contract |
| evidence_level | raw_evidence |

## Scope

Public API: `WrapCalibratedExplainer.explain_factual(X)` return contract.

Applicable task types: binary classification, multiclass classification, regression.

Applicable workflow: fit-calibrate-explain_factual.

## Observable behavior

The collection returned by `explain_factual(X)` must:

1. Have length equal to the number of rows in `X` (`len(result) == len(X)`).
2. Support indexing: `result[i]` is not `None` for all valid `i`.
3. Each `result[i]` is a `FactualExplanation` instance (not `None`, not a raw dict).
4. `result[0].feature_weights` is accessible as a non-None public attribute containing
   per-feature contribution data.

## Acceptance criterion

For a fitted and calibrated `WrapCalibratedExplainer` and test set `X_test`:

- `len(result) == len(X_test)`.
- `result[0]` is not `None`.
- `result[0].feature_weights` is not `None`.
- `type(result).__name__` is `"CalibratedExplanations"`.

## TIF reference

**TIF ID:** CE-TIF-EXPL-001

TIF scenario function: `run_factual_tif_scenario()`
Observation fields: `result_len`, `result_is_none`, `first_item_is_none`, `feature_weights_accessible`

## Verification method

Automated pytest tests calling CE-TIF-EXPL-001.

## Verification targets

- `pytest: tests/capabilities/test_explanation_contracts.py::test_should_preserve_cardinality_when_factual_explain`
- `pytest: tests/capabilities/test_explanation_contracts.py::test_should_return_accessible_feature_weights_when_factual_explain`

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

- That feature weights are numerically correct or scientifically meaningful.
- That calibrated probabilities match any theoretical coverage bound.
- Statistical validity under distribution shift.

See `CE-CAP-EXPL-001` for the full assumption statement.
