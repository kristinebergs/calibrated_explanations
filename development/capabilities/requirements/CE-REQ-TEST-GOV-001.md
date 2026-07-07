# CE-REQ-TEST-GOV-001 - Test Quality Gate Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-TEST-GOV-001 |
| obligation_type | quality_gate |
| claim_refs | CE-CAP-TEST-001 |
| adr_refs | ADR-023, ADR-030 |
| status | active |
| verification_status | verified |

## Scope

ADR-030 test-quality enforcement for naming, assertion strength, determinism checks, marker hygiene, coverage ratification lane structure, and no production test-helper exports.

## Observable behavior

- The anti-pattern detector flags assertion and determinism problems.
- The anti-pattern detector check mode fails when new violations appear.
- ADR-030 ratification local-check steps are defined and fail closed when steps fail or reports are missing.
- Production packages do not export test-helper scaffolding.

## Acceptance criterion

- Anti-pattern detector tests pass for assertion, determinism, and excessive mocking cases.
- ADR-030 ratification lane tests pass for step order, report writing, failing step behavior, and missing report behavior.
- No-test-helper-export checker tests pass for banned and clean export cases.
- Marker hygiene remains covered by the local-check ratification lane.

## Verification method

Automated pytest tests for ADR-030 quality scanners and local-check ratification behavior.

## Verification targets

- pytest: tests/scripts/test_detect_test_anti_patterns.py::test_detector_flags_new_assertion_and_determinism_patterns
- pytest: tests/scripts/test_detect_test_anti_patterns.py::test_detector_check_mode_enforces_no_new_violations
- pytest: tests/scripts/test_local_checks_adr030_ratification.py::test_should_define_adr030_ratification_steps_in_expected_order
- pytest: tests/scripts/test_local_checks_adr030_ratification.py::test_should_stop_and_return_failure_when_adr030_step_fails
- pytest: tests/scripts/test_check_no_test_helper_exports.py::test_checker_blocks_banned_registry_exports

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies executable quality gates and scanner behavior. It does not claim every individual test is high-value without the over-testing and coverage reports run by the broader ADR-030 process.
