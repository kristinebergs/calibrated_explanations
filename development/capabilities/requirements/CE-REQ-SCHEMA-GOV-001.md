# CE-REQ-SCHEMA-GOV-001 - Explanation Payload Schema Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-SCHEMA-GOV-001 |
| obligation_type | payload_schema |
| claim_refs | CE-CAP-SCHEMA-001 |
| adr_refs | ADR-005, ADR-008 |
| status | active |
| verification_status | verified |

## Scope

ADR-005 explanation payload schema behavior: committed golden fixture validation, minimal payload validation, fallback validation when jsonschema is unavailable, and provenance preservation through adapters.

## Observable behavior

- The committed golden explanation fixture validates through the schema validator.
- Minimal payload checks reject missing or malformed required fields.
- Schema validation handles the missing-jsonschema fallback path.
- Domain/legacy adapters preserve provenance and metadata through conversion.

## Acceptance criterion

- Golden fixture validation passes and asserts interval invariants.
- Minimal schema validation tests pass.
- Missing-jsonschema fallback validation tests pass.
- Adapter provenance tests pass for legacy-to-domain and domain-to-legacy conversion.

## Verification method

Automated pytest tests for schema validation and adapter provenance.

## Verification targets

- pytest: tests/unit/test_schema_validation_minimal.py::test_should_validate_golden_explanation_fixture
- pytest: tests/unit/test_schema_validation_minimal.py::test_validate_payload_minimal_checks
- pytest: tests/unit/test_schema_validation_minimal.py::test_schema_validation_handles_missing_jsonschema_import
- pytest: tests/unit/explanations/test_adapters_provenance.py::test_should_preserve_provenance_and_metadata_when_legacy_to_domain
- pytest: tests/unit/explanations/test_adapters_provenance.py::test_should_preserve_provenance_and_metadata_when_domain_to_legacy

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies schema and adapter payload contracts. It does not certify every future schema extension unless covered by additional requirements.
