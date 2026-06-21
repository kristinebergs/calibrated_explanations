# CE-REQ-SCHEMA-EXTENSION-SURFACE-001 — Provenance and metadata are the governed optional extension surfaces; top-level envelope validation remains deferred.

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-SCHEMA-EXTENSION-SURFACE-001 |
| obligation_type | payload_schema |
| claim_refs | CE-CAP-SCHEMA-EXTENSION-SURFACE-001 |
| adr_refs | ADR-005 |
| status | active |
| verification_status | manual_review_required |

## Scope

ADR-governed implementation, documentation, and verification artifacts for `ADR-005`.

## Observable behavior

Payload extensions MUST use governed provenance and metadata surfaces, while top-level envelope validation remains deferred unless a later ADR makes it mandatory.

## Acceptance criterion

Behavioral acceptance for this requirement is the ADR-specific obligation itself:

- Payload extensions MUST use governed provenance and metadata surfaces, while top-level envelope validation remains deferred unless a later ADR makes it mandatory.
- Automated metadata tests MAY prove that the requirement is linked, but MUST NOT be
  treated as proof that the behavioral obligation is satisfied.
- Until a named runtime, static-policy, serialization, schema, or quality-gate check below
  directly verifies the obligation, the verification status remains `manual_review_required`.

## Verification method

Verification method: `manual_review` plus the concrete source, documentation, schema,
quality-script, or pytest targets listed below. Metadata tests are linkage guards only.

## Verification targets

- source: `development/adrs/ADR-005*.md`
- schema: `development/schemas/primitives_schema.json`
- docs: `docs/schema_v1.md`
- docs: `docs/foundations/concepts/explanation_structures.md`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_adr_claim_requirement_link_metadata`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_curated_semantic_claim_presence`
- manual_review: `development/current-work/RELEASE_PLAN_status_appendix.md` (ADR-005 section)

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
specific source, test, documentation, or quality-script evidence maintained for `ADR-005`.
