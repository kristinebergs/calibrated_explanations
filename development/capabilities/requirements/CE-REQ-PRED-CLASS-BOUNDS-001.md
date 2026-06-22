# CE-REQ-PRED-CLASS-BOUNDS-001 — Classification Probability Bounds Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PRED-CLASS-BOUNDS-001 |
| obligation_type | output_contract |
| claim_refs | CE-CAP-PRED-CLASS-001 |
| adr_refs | ADR-021 |
| status | active |
| verification_status | verified |
| applicable_on | WrapCalibratedExplainer.predict_proba for classification |
| tif_refs | CE-TIF-PRED-CLASS-001 |
| verification_strength | numerical_behavior |
| evidence_level | raw_evidence |

## Scope

Public API: `WrapCalibratedExplainer.predict_proba(X)` for classification tasks.

Applicable task types: binary classification, multiclass classification.

Applicable workflow: standard offline fit-calibrate-predict.

## Observable behavior

For a classification task, `predict_proba(X)` must:

1. Return values strictly bounded in `[0, 1]` — no value below 0 or above 1.
2. Return an array-like with shape compatible with `(len(X), n_classes)`.
3. The bounds hold for all instances in the test set.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted and calibrated for binary classification:

- `np.all(predict_proba(X_test) >= 0.0)` is True.
- `np.all(predict_proba(X_test) <= 1.0)` is True.
- `len(predict_proba(X_test)) == len(X_test)`.

## TIF reference

**TIF ID:** CE-TIF-PRED-CLASS-001

TIF scenario function: `run_classification_tif_scenario()`
Observation fields: `proba_min`, `proba_max`, `proba_len`

## Verification method

Automated pytest tests calling CE-TIF-PRED-CLASS-001.

## Verification targets

- `pytest: tests/capabilities/test_classification_contracts.py::test_should_return_bounded_probabilities_when_classification_fitted_and_calibrated`

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

This requirement verifies that probability values are in [0, 1] numerically. It does not verify:

- That probabilities sum to 1 across classes (Venn-Abers intervals do not guarantee this).
- Calibration validity in a finite-sample or distribution-shift sense.
- That values represent true posterior probabilities.

See `CE-CAP-PRED-CLASS-001` for the full assumption statement.
