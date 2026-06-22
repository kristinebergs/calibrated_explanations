# CE-REQ-MODALITY-GOV-001 - Modality Plugin Packaging Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-MODALITY-GOV-001 |
| obligation_type | plugin_behavior |
| claim_refs | CE-CAP-MODALITY-001 |
| adr_refs | ADR-033 |
| status | active |
| verification_status | verified |

## Scope

ADR-033 modality-extension plugin packaging and metadata behavior: external plugin discovery, `data_modalities` preservation, modality alias normalization, API-version checks, and no nonstandard loader fallbacks.

## Observable behavior

- Valid discovered plugins register through the governed entry-point path.
- Plugin metadata preserves `data_modalities`.
- Image modality aliases normalize to canonical modality names.
- Plugins with incompatible major API versions are rejected.
- Nonstandard entry-point loader fallbacks are not used.

## Acceptance criterion

- ADR-033 packaging smoke tests pass for valid plugin registration and metadata preservation.
- Alias normalization tests pass for image/vision modality declarations.
- Major-version mismatch tests reject incompatible plugins.
- Loader fallback tests confirm the governed entry-point path is used.

## Verification method

Automated pytest tests for ADR-033 plugin packaging and modality metadata.

## Verification targets

- pytest: tests/unit/plugins/test_adr033_packaging_smoke.py::test_should_register_valid_plugin_when_entry_point_is_discovered
- pytest: tests/unit/plugins/test_adr033_packaging_smoke.py::test_should_preserve_data_modalities_when_entry_point_is_discovered
- pytest: tests/unit/plugins/test_adr033_packaging_smoke.py::test_should_normalise_modality_alias_when_plugin_declares_image
- pytest: tests/unit/plugins/test_adr033_packaging_smoke.py::test_should_reject_plugin_when_plugin_api_version_major_mismatches
- pytest: tests/unit/plugins/test_adr033_packaging_smoke.py::test_should_not_use_nonstandard_entrypoint_loader_fallbacks

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies plugin packaging and metadata contracts. It does not implement or validate modality-specific model semantics for vision, audio, or text plugins.
