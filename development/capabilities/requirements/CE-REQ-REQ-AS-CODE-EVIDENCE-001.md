# CE-REQ-REQ-AS-CODE-EVIDENCE-001 — Behavioral requirements require executable evidence

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-REQ-AS-CODE-EVIDENCE-001 |
| obligation_type | quality_gate |
| claim_refs | CE-CAP-REQ-AS-CODE-001 |
| adr_refs | ADR-030, ADR-035 |
| status | active |
| verification_status | verified |

## Scope

All active ADR-governed capability requirements under
`development/capabilities/requirements/`, all capability claims under
`development/capabilities/claims/`, and the executable tests they cite.

## Observable behavior

The ADR-to-claim-to-requirement chain MUST terminate in executable evidence for every
active behavioral requirement:

1. Behavioral requirements MUST name at least one executable pytest target.
2. Documentation, source files, schemas, standards, and metadata-linkage tests MAY support
   interpretation, but MUST NOT be the only evidence for a behavioral requirement.
3. A cited pytest target MUST resolve to an existing test function.
4. Human or manual verification MUST NOT be used as a test method for implemented
   requirements.
5. If the requirement is not implemented, the requirement MUST use `verification_status`
   `adr_gap_open` or `not_implemented`, include `gap_ref` or `adr_gap_ref`, and the gap
   MUST be registered in `development/current-work/RELEASE_PLAN_status_appendix.md`.

## Acceptance criterion

The requirements-as-code governance integration tests pass without accepting metadata-only
or human verification as behavioral proof.

## Verification method

Automated pytest tests in `tests/capabilities/`.

## Verification targets

- pytest: `tests/capabilities/test_adr_capability_links.py::test_should_require_behavioral_requirements_to_name_concrete_evidence`
- pytest: `tests/capabilities/test_adr_capability_links.py::test_should_reject_human_verification_without_registered_adr_gap`
- pytest: `tests/capabilities/test_adr_capability_links.py::test_should_require_claimed_pytest_targets_to_be_real_tests`
- pytest: `tests/capabilities/test_adr_capability_links.py::test_should_validate_curated_semantic_claim_presence`

Test IDs:
- `test_should_require_behavioral_requirements_to_name_concrete_evidence`
- `test_should_reject_human_verification_without_registered_adr_gap`
- `test_should_require_claimed_pytest_targets_to_be_real_tests`
- `test_should_validate_curated_semantic_claim_presence`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |
| failing_requirement_ids | yes when failing |
| gap_ref | yes when requirement is not implemented |

## Assumption boundary

This requirement verifies the requirements-as-code evidence contract. It does not by
itself prove every feature behavior. Feature behavior remains proven by the concrete
pytest or quality-gate tests cited by each feature requirement.
