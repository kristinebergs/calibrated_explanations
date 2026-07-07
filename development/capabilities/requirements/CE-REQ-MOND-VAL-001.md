# CE-REQ-MOND-VAL-001 - Mondrian Structural Validation Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-MOND-VAL-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-MOND-001 |
| adr_refs | ADR-039 |
| status | active |
| verification_status | verified |
| tif_refs | CE-TIF-MOND-001 |

## Scope

Public API: `WrapCalibratedExplainer.calibrate`, wrapper inference methods, and
the public `CalibratedExplainer.explain_factual` boundary.

Applicable task types: classification, regression, probabilistic regression.

## Observable behavior

1. Calibration and inference reject Mondrian `bins` whose length differs from the
   input sample count with `DataShapeError`.
2. Inference rejects labels outside the calibration-time category vocabulary with
   `ValidationError`.

## Acceptance criterion

The public boundary raises CE exceptions with `details` containing the mismatched
lengths or unknown/known labels.

## Verification method

Automated pytest tests in `tests/capabilities/`.

## Verification targets

- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_validation_error_when_inference_bins_include_unknown_label`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_data_shape_error_when_inference_bins_length_mismatches_samples`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_data_shape_error_when_calibration_bins_length_mismatches_samples`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_data_shape_error_when_core_explainer_bins_length_mismatches_samples`

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

This requirement verifies structural input validation only. It does not prove
that category definitions are statistically appropriate or sufficiently powered.
