# CE-REQ-PREPROC-GOV-001 - Preprocessor Mapping Runtime Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PREPROC-GOV-001 |
| obligation_type | runtime_behavior |
| claim_refs | CE-CAP-PREPROC-001 |
| adr_refs | ADR-009 |
| status | active |
| verification_status | verified |
| tif_exemption | repository_policy |
| tif_exemption_rationale | Preprocessor wiring is verified by integration tests that assert on telemetry metadata not exposed through WrapCalibratedExplainer; the preprocessor is a callsite-controlled pipeline stage below the public wrapper boundary. |

## Scope

ADR-009 wrapper preprocessing behavior: deterministic mapping learned during calibration, reused at inference, and surfaced through telemetry/provenance boundaries.

## Observable behavior

- Preprocessing is applied during calibration and reused for inference.
- Preprocessor metadata is exposed through telemetry without moving mapping policy into the core numeric explainer.
- Mapping behavior remains deterministic for the wrapped public workflow.

## Acceptance criterion

- The preprocessor wiring integration test confirms calibration and inference use the learned preprocessor path.
- The telemetry test confirms preprocessor metadata is exposed.
- The test workflow uses the public wrapper path rather than private shortcuts.

## Verification method

Automated integration pytest tests for wrapper preprocessor wiring and metadata exposure.

## Verification targets

- pytest: tests/integration/core/test_preprocessor_wiring.py::test_preprocessor_applied_on_calibrate_and_inference
- pytest: tests/integration/core/test_preprocessor_wiring.py::test_preprocessor_metadata_exposed_in_telemetry

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies wrapper-level preprocessing behavior. It does not verify every encoder implementation or post-v1.0 mapping-persistence open item.
