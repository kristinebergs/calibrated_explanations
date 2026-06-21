# CE-REQ-CALIBRATOR-PRIMITIVE-SCHEMA-001 — Built-in calibrators implement JSON-safe to_primitive() and from_primitive() schema contracts.

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-CALIBRATOR-PRIMITIVE-SCHEMA-001 |
| obligation_type | serialization_contract |
| claim_refs | CE-CAP-CALIBRATOR-PRIMITIVE-SCHEMA-001 |
| adr_refs | ADR-031 |
| status | active |
| verification_status | manual_review_required |

## Scope

ADR-governed implementation, documentation, and verification artifacts for `ADR-031`.

## Observable behavior

Built-in calibrators MUST serialize to JSON-safe primitives and restore via from_primitive() with explicit schema_version metadata.

## Acceptance criterion

Behavioral acceptance for this requirement is the ADR-specific obligation itself:

- Built-in calibrators MUST serialize to JSON-safe primitives and restore via from_primitive() with explicit schema_version metadata.
- Automated metadata tests MAY prove that the requirement is linked, but MUST NOT be
  treated as proof that the behavioral obligation is satisfied.
- Until a named runtime, static-policy, serialization, schema, or quality-gate check below
  directly verifies the obligation, the verification status remains `manual_review_required`.

## Verification method

Verification method: `manual_review` plus the concrete source, documentation, schema,
quality-script, or pytest targets listed below. Metadata tests are linkage guards only.

## Verification targets

- source: `development/adrs/ADR-031*.md`
- docs: `docs/api/serialization.md`
- schema: `development/schemas/primitives_schema.json`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_adr_claim_requirement_link_metadata`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_curated_semantic_claim_presence`
- manual_review: `development/current-work/RELEASE_PLAN_status_appendix.md` (ADR-031 section)

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
specific source, test, documentation, or quality-script evidence maintained for `ADR-031`.
