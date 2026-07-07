# CE-REQ-CONFIG-GOV-001 - Configuration Governance Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-CONFIG-GOV-001 |
| obligation_type | static_policy |
| claim_refs | CE-CAP-CONFIG-001 |
| adr_refs | ADR-034, ADR-038 |
| status | active |
| verification_status | verified |

## Scope

Runtime and call-time configuration governance under ADR-034 and ADR-038: centralized config reads, lifecycle allowlist control, config governance events, and naming-policy guardrails.

## Observable behavior

- Migrated runtime modules do not read environment variables or pyproject configuration directly outside the ConfigManager boundary.
- The lifecycle allowlist for direct configuration reads remains empty.
- Configuration governance events are schema-valid and reject unsupported shapes.
- Public call-time parameter naming is checked by the naming guardrail tests.

## Acceptance criterion

- The real package scan reports zero targeted ConfigManager usage violations.
- The lifecycle allowlist test reports no remaining direct-read exceptions.
- Config governance event tests validate supported event types and reject invalid details.
- Parameter naming tests reject banned public signature names and confirm required references exist.

## Verification method

Automated pytest tests for config-manager usage, config governance events, and naming guardrails.

## Verification targets

- pytest: tests/scripts/test_check_config_manager_usage.py::test_should_report_zero_targeted_violations_against_real_package
- pytest: tests/scripts/test_check_config_manager_usage.py::test_should_have_empty_lifecycle_allowlist
- pytest: tests/observability/test_config_governance_events.py::test_should_validate_all_config_event_types_and_reject_plugin_validator_path
- pytest: tests/scripts/test_check_parameter_naming.py::test_no_banned_names_in_public_signatures

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies enforced configuration governance checks. ADR-034 post-v1.0 open items, such as wider redaction guarantees, remain outside this verified scope.
