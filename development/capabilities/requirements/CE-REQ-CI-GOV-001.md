# CE-REQ-CI-GOV-001 - CI Workflow Policy Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-CI-GOV-001 |
| obligation_type | quality_gate |
| claim_refs | CE-CAP-CI-001 |
| adr_refs | ADR-035 |
| status | active |
| verification_status | verified |

## Scope

CI workflow policy checks governed by ADR-035, including constrained installs, pinned external actions, reusable workflow policy, local reproduction updates, and CODEOWNERS coverage.

## Observable behavior

- Workflow validation rejects unconstrained pip installs where constraints are required.
- Workflow validation rejects externally hosted actions that are not pinned to full SHAs.
- New or strict workflow changes are checked against reusable-workflow policy unless explicitly allowlisted with dated rationale.
- Local reproduction and ownership paths stay covered for CI policy changes.

## Acceptance criterion

- Synthetic workflow changes without required constraints fail the CI policy validator.
- Synthetic workflow changes with major-tag external actions fail the CI policy validator.
- Non-allowlisted new workflows are flagged by the reusable-workflow gate.
- The validator test suite confirms `scripts/local_checks.py` policy ownership coverage.

## Verification method

Automated pytest tests for the CI workflow policy validator.

## Verification targets

- pytest: tests/scripts/test_validate_ci_policy.py::test_should_fail_when_pip_install_missing_constraints
- pytest: tests/scripts/test_validate_ci_policy.py::test_should_fail_when_external_action_is_major_tag
- pytest: tests/scripts/test_validate_ci_policy.py::test_should_flag_reusable_check_for_non_allowlisted_new_workflow
- pytest: tests/scripts/test_validate_ci_policy.py::test_should_cover_scripts_local_checks_path_in_codeowners

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies in-repository CI policy enforcement. GitHub branch-protection settings remain platform-governed operational constraints tracked by ADR-035, not requirements proven by this test.
