# CE-REQ-SERIAL-GOV-001 - Serialization Persistence Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-SERIAL-GOV-001 |
| obligation_type | serialization_contract |
| claim_refs | CE-CAP-SERIAL-001 |
| adr_refs | ADR-031 |
| status | active |
| verification_status | verified |
| tif_exemption | repository_policy |

## Scope

ADR-031 calibrator and wrapper persistence behavior: schema-versioned state, JSON-safe primitive serialization, round-trip prediction preservation, and fail-fast unsupported-version/checksum handling.

## Observable behavior

- Wrapper save/load round-trips preserve classification and regression behavior.
- Saved wrapper state writes schema version 2 manifests.
- Unsupported schema versions and checksum mismatches are rejected.
- VennAbers and IntervalRegressor primitives serialize to JSON-safe v2 state and round-trip predictions.

## Acceptance criterion

- Wrapper persistence tests pass for classification and regression round-trips.
- Manifest schema-version tests pass for v2 writes and v1 migration acceptance.
- Checksum and unsupported-version rejection tests pass.
- Primitive serialization tests pass for Venn-Abers and IntervalRegressor JSON-safe v2 round-trips.

## Verification method

Automated pytest tests for wrapper persistence and calibrator primitive serialization.

## Verification targets

- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_save_and_load_state_roundtrip_classification
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_save_state_writes_schema_version_2_manifest
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_load_state_rejects_checksum_mismatch
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_load_state_rejects_unsupported_schema_version
- pytest: tests/unit/calibration/test_calibrator_primitive_roundtrip.py::test_venn_abers_to_primitive_v2_is_json_serializable
- pytest: tests/unit/calibration/test_calibrator_primitive_roundtrip.py::test_interval_regressor_to_primitive_v2_is_json_serializable

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies supported persistence contracts. It does not promise compatibility with arbitrary pre-schema artifacts beyond explicitly tested migration paths.
