# CE-REQ-DEPREC-GOV-001 - Deprecation Closure Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-DEPREC-GOV-001 |
| obligation_type | quality_gate |
| claim_refs | CE-CAP-DEPREC-001 |
| adr_refs | ADR-011 |
| status | active |
| verification_status | verified |

## Scope

ADR-011 deprecation lifecycle enforcement: ordered local closure checks, active-deprecation blocking behavior, and report generation.

## Observable behavior

- The deprecation closure lane defines the expected steps in order.
- The lane writes reports when closure checks pass.
- The lane fails before running later commands when active deprecations remain.

## Acceptance criterion

- The deprecation closure step-order test passes.
- The report-writing test confirms closure artifacts are produced on pass.
- The active-deprecations test confirms the lane fails closed when the migration ledger still has active rows.

## Verification method

Automated pytest tests for the local deprecation closure gate.

## Verification targets

- pytest: tests/scripts/test_local_checks_deprecation_closure.py::test_should_define_deprecation_closure_steps_in_expected_order
- pytest: tests/scripts/test_local_checks_deprecation_closure.py::test_should_write_reports_when_deprecation_closure_passes
- pytest: tests/scripts/test_local_checks_deprecation_closure.py::test_should_report_active_deprecation_rows_as_blocking_when_eta_targets_v1_0_0

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies the deprecation gate and ledger failure behavior. It does not itself prove every deprecated public symbol has been removed.
