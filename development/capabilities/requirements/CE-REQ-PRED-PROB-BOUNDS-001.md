# CE-REQ-PRED-PROB-BOUNDS-001 — Probabilistic Regression Output Bounds Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PRED-PROB-BOUNDS-001 |
| obligation_type | output_contract |
| claim_refs | CE-CAP-PRED-PROB-001 |
| adr_refs | ADR-021 |
| status | active |
| verification_status | verified |
| applicable_on | WrapCalibratedExplainer.predict_proba with threshold for regression |
| tif_refs | CE-TIF-PRED-PROB-001 |
| verification_strength | numerical_behavior |
| evidence_level | raw_evidence |

## Scope

Public API: `WrapCalibratedExplainer.predict_proba(X, threshold=y_threshold)` for
probabilistic regression (regression model queried with a scalar threshold).

Applicable task types: probabilistic_regression.

Applicable workflow: standard offline fit-calibrate-predict with threshold.

## Observable behavior

For a probabilistic regression task, `predict_proba(X, threshold=y_threshold)` must:

1. Return values strictly bounded in `[0, 1]` — no value below 0 or above 1.
2. Return an array-like with length `len(X)`.
3. The bounds hold for all instances in the test set.

## Acceptance criterion

For a `WrapCalibratedExplainer` fitted and calibrated for regression and a scalar threshold:

- `np.all(predict_proba(X_test, threshold=y_threshold) >= 0.0)` is True.
- `np.all(predict_proba(X_test, threshold=y_threshold) <= 1.0)` is True.
- `len(predict_proba(X_test, threshold=y_threshold)) == len(X_test)`.

## TIF reference

**TIF ID:** CE-TIF-PRED-PROB-001

TIF scenario function: `run_prob_regression_tif_scenario()`
Observation fields: `proba_min`, `proba_max`, `proba_len`

## Verification method

Automated pytest tests calling CE-TIF-PRED-PROB-001.

## Verification targets

- `pytest: tests/capabilities/test_probabilistic_regression_contracts.py::test_should_return_bounded_probabilities_when_regression_threshold_query`

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

This requirement verifies that exceedance probability values are in [0, 1]. It does not verify:

- Frequency-calibration of P(Y > threshold | X).
- CPS coverage guarantees at any significance level.
- Distribution shift robustness.

See `CE-CAP-PRED-PROB-001` for the full assumption statement.
