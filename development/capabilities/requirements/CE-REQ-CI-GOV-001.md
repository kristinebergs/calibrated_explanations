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

CI workflow policy checks governed by ADR-035 (v1 full-inventory model, v0.11.6 Task 60), including the approved workflow inventory, constrained installs, pinned external actions, least-privilege permissions, and CODEOWNERS coverage.

## Observable behavior

- Workflow validation rejects any workflow file outside the approved inventory until the approved set is deliberately updated.
- Workflow validation rejects unconstrained pip installs where constraints are required.
- Workflow validation rejects externally hosted actions that are not pinned to full SHAs.
- Local reproduction and ownership paths stay covered for CI policy changes.

## Acceptance criterion

- A synthetic unapproved workflow file fails the full-inventory CI policy validator.
- Synthetic workflow changes without required constraints fail the CI policy validator.
- Synthetic workflow changes with major-tag external actions fail the CI policy validator.
- The validator test suite confirms `scripts/local_checks.py` policy ownership coverage.

## Verification method

Automated pytest tests for the CI workflow policy validator.

## Verification targets

- pytest: tests/scripts/test_validate_ci_policy.py::test_should_reject_unapproved_workflow_file
- pytest: tests/scripts/test_validate_ci_policy.py::test_should_reject_pip_install_without_constraints
- pytest: tests/scripts/test_validate_ci_policy.py::test_should_reject_unpinned_external_action
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
