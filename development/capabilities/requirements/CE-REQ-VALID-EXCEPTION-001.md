# CE-REQ-VALID-EXCEPTION-001 - Validation Exception Taxonomy Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-VALID-EXCEPTION-001 |
| obligation_type | static_policy |
| claim_refs | CE-CAP-VALID-001 |
| adr_refs | ADR-002 |
| status | active |
| verification_status | verified |

## Scope

ADR-002 validation and exception taxonomy for public validation helpers, plugin metadata validation, and ADR-002 compliance scanning.

## Observable behavior

- Validation helpers raise CE `ValidationError` with structured details for invalid public inputs.
- Plugin metadata validators reject malformed plugin metadata with CE exceptions.
- The ADR-002 compliance quality gate scans source files for forbidden generic exception patterns.

## Acceptance criterion

- Validation unit tests pass for matrix shape, finite checks, model/fit-state validation, ADR-002 signature details, and details payloads.
- Plugin base validation tests pass for malformed metadata and config schemas.
- The ADR-002 compliance quality gate passes on the repository source tree.

## Verification method

Automated pytest tests and executable ADR-002 compliance quality gate.

## Verification targets

- pytest: tests/unit/core/test_validation_unit.py::test_validate_inputs_matrix_shape_checks
- pytest: tests/unit/core/test_validation_unit.py::test_validate_inputs_adr002_signature_details_payload
- pytest: tests/plugins/test_base_validation.py::test_validate_plugin_meta_rejects_non_dict
- pytest: tests/plugins/test_base_validation.py::test_validate_plugin_config_applies_defaults_and_rejects_unknown_keys
- quality-gate: python scripts/quality/check_adr002_compliance.py

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies public validation and scanner behavior in ADR-002 scope. It does not prohibit documented domain-specific exception subclasses or every internal implementation detail.
