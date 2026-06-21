# CE-REQ-REJECT-RESULT-ENVELOPE-001 — Reject-aware outputs use a structured RejectResult envelope when reject integration is active.

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-REJECT-RESULT-ENVELOPE-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-REJECT-RESULT-ENVELOPE-001 |
| adr_refs | ADR-029 |
| status | active |
| verification_status | manual_review_required |

## Scope

ADR-governed implementation, documentation, and verification artifacts for `ADR-029`.

## Observable behavior

When reject integration is active, reject-aware outputs MUST expose structured RejectResult information rather than unstructured ad-hoc status fields.

## Acceptance criterion

Behavioral acceptance for this requirement is the ADR-specific obligation itself:

- When reject integration is active, reject-aware outputs MUST expose structured RejectResult information rather than unstructured ad-hoc status fields.
- Automated metadata tests MAY prove that the requirement is linked, but MUST NOT be
  treated as proof that the behavioral obligation is satisfied.
- Until a named runtime, static-policy, serialization, schema, or quality-gate check below
  directly verifies the obligation, the verification status remains `manual_review_required`.

## Verification method

Verification method: `manual_review` plus the concrete source, documentation, schema,
quality-script, or pytest targets listed below. Metadata tests are linkage guards only.

## Verification targets

- source: `development/adrs/ADR-029*.md`
- docs: `docs/practitioner/advanced/reject-policy.md`
- docs: `docs/contributor/reject_policy_usage.md`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_adr_claim_requirement_link_metadata`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_curated_semantic_claim_presence`
- manual_review: `development/current-work/RELEASE_PLAN_status_appendix.md` (ADR-029 section)

Test IDs:
- `test_should_validate_adr_claim_requirement_link_metadata`
- `test_should_validate_curated_semantic_claim_presence`
- `test_should_require_behavioral_requirements_to_name_concrete_evidence`
- `test_should_prevent_runtime_obligations_from_using_documentation_boundary_type`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail/partial) |

## Assumption boundary

This requirement captures the normative ADR obligation and its evidence chain. It does not
claim that metadata checks alone prove runtime correctness; behavior remains verified by the
specific source, test, documentation, or quality-script evidence maintained for `ADR-029`.
