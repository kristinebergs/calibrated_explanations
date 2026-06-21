# CE-REQ-LOG-STRUCTURED-COMPATIBILITY-001 — Logging remains compatible with structured logging consumers while preserving standard library logging behavior.

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-LOG-STRUCTURED-COMPATIBILITY-001 |
| obligation_type | static_policy |
| claim_refs | CE-CAP-LOG-STRUCTURED-COMPATIBILITY-001 |
| adr_refs | ADR-028 |
| status | active |
| verification_status | manual_review_required |

## Scope

ADR-governed implementation, documentation, and verification artifacts for `ADR-028`.

## Observable behavior

Logging helpers MUST preserve standard library logging compatibility and support structured-context consumers without imposing a non-standard logging dependency.

## Acceptance criterion

Behavioral acceptance for this requirement is the ADR-specific obligation itself:

- Logging helpers MUST preserve standard library logging compatibility and support structured-context consumers without imposing a non-standard logging dependency.
- Automated metadata tests MAY prove that the requirement is linked, but MUST NOT be
  treated as proof that the behavioral obligation is satisfied.
- Until a named runtime, static-policy, serialization, schema, or quality-gate check below
  directly verifies the obligation, the verification status remains `manual_review_required`.

## Verification method

Verification method: `manual_review` plus the concrete source, documentation, schema,
quality-script, or pytest targets listed below. Metadata tests are linkage guards only.

## Verification targets

- source: `development/adrs/ADR-028*.md`
- quality_script: `scripts/quality/check_logging_domains.py`
- standard: `development/standards/STD-005-logging-and-observability-standard.md`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_adr_claim_requirement_link_metadata`
- metadata_test: `tests/capabilities/test_adr_capability_links.py::test_should_validate_curated_semantic_claim_presence`
- manual_review: `development/current-work/RELEASE_PLAN_status_appendix.md` (ADR-028 section)

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
specific source, test, documentation, or quality-script evidence maintained for `ADR-028`.
