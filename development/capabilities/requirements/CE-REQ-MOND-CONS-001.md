# CE-REQ-MOND-CONS-001 - Mondrian Inference Consistency Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-MOND-CONS-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-MOND-001 |
| adr_refs | ADR-039 |
| status | active |
| verification_status | verified |
| tif_refs | CE-TIF-MOND-001 |

## Scope

Public API: `WrapCalibratedExplainer.predict`, `predict_proba`,
`explain_factual`, and `explore_alternatives` after conditional or global
calibration.

Applicable task types: classification, regression, probabilistic regression.

## Observable behavior

1. A wrapper calibrated with inline `bins=` raises `ValidationError` when
   inference omits `bins=`.
2. A globally calibrated wrapper raises `ConfigurationError` when inference
   receives `bins=`.
3. A wrapper calibrated with `mc=` raises `ConfigurationError` when inference
   also receives explicit `bins=`.

## Acceptance criterion

The public wrapper APIs fail before calibrator internals run, and the raised
exception is a CE exception with structured `details`.

## Verification method

Automated pytest tests in `tests/capabilities/`.

## Verification targets

- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_validation_error_when_bins_calibrated_inference_omits_bins`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_validation_error_when_regression_bins_calibrated_predict_omits_bins`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_configuration_error_when_global_inference_receives_bins`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_configuration_error_when_mc_inference_receives_explicit_bins`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| dataset_id | yes |
| random_seed | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies API consistency and error behavior only. It does not
verify conditional validity, per-category calibration quality, or the scientific
meaning of the chosen categories.
