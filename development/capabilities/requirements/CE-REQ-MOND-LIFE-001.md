# CE-REQ-MOND-LIFE-001 - Mondrian Calibration Lifecycle Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-MOND-LIFE-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-MOND-001 |
| adr_refs | ADR-039, ADR-038 |
| status | active |
| verification_status | verified |
| tif_refs | CE-TIF-MOND-001 |

## Scope

Public API: `WrapCalibratedExplainer.calibrate(..., reuse_conditional=True)`.

Applicable task types: classification, regression, probabilistic regression.

## Observable behavior

1. Recalibrating without `bins=`, `mc=`, or `reuse_conditional=True` resets the
   wrapper to global calibration.
2. `reuse_conditional=True` reapplies a stored `mc` to the new calibration data.
3. `calibrate` rejects multiple conditional channels in one call.
4. `reuse_conditional=True` without a stored `mc` raises `ValidationError`.

## Acceptance criterion

The wrapper's active conditional state reflects only the current calibration call,
unless `reuse_conditional=True` explicitly reuses a stored categorizer.

## Verification method

Automated pytest tests in `tests/capabilities/`.

## Verification targets

- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_reset_conditional_state_when_recalibrated_without_channel`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_reuse_conditional_categorizer_when_requested`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_validation_error_when_calibrate_receives_multiple_conditional_channels`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_raise_validation_error_when_reuse_conditional_has_no_stored_mc`

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

This requirement verifies lifecycle state transitions only. It does not prove
that the reused categorizer defines valid categories for a changed data
distribution.
