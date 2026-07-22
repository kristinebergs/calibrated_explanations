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
| tif_exemption_rationale | Serialization persistence is verified by unit tests that save/load state objects directly; WrapCalibratedExplainer scenarios cannot verify schema version manifests, checksum rejection, or primitive JSON-safe round-trips at the required granularity. |

## Scope

ADR-031 calibrator and wrapper persistence behavior: schema-versioned, non-executable state, JSON-safe primitive serialization, round-trip prediction preservation, fail-fast unsupported-version/checksum handling, and unconditional rejection of legacy pickle-based artifacts (schema v1/v2).

## Observable behavior

- Wrapper save/load round-trips preserve classification, multiclass, regression, and conditional/Mondrian-bins predictions, given a caller-supplied fitted learner.
- Saved wrapper state writes schema version 3 manifests containing only JSON-safe declarative data; no `wrapper.pkl` or other pickle payload is ever written.
- `load_state()` never calls `pickle.loads`/`pickle.load`/`joblib.load`/`cloudpickle.loads` on artifact-provided bytes, even when every checksum in the artifact is internally consistent (checksums detect corruption; they do not authenticate an artifact's origin).
- Legacy schema v1/v2 artifacts (which persisted the whole wrapper, and unsupported calibrators, via pickle) are rejected unconditionally by `load_state()`, before any pickle byte is read.
- A `calibrator_type="python_pickle"` primitive, nested directly or inside a `fast_collection`, is rejected without being base64-decoded or unpickled, regardless of checksum validity.
- Unsupported schema versions, malformed manifests, missing files, checksum mismatches, path traversal/absolute paths, disallowed files (e.g. `wrapper.pkl` in a v3 artifact), and unknown calibrator types are all rejected before component payloads are interpreted.
- VennAbers and IntervalRegressor primitives serialize to JSON-safe v2 state and round-trip predictions.

## Acceptance criterion

- Wrapper persistence tests pass for classification, multiclass, and regression round-trips using an explicitly supplied learner.
- Manifest schema-version tests pass for v3 writes and legacy v1/v2 rejection (not acceptance).
- Checksum, unsupported-version, and artifact-hardening (path/symlink/unexpected-file) rejection tests pass.
- The exploit-regression and nested-calibrator-rejection tests pass, proving pickle is never reached on the safe `load_state()` path.
- A static AST-based guard proves `wrap_explainer.py` never calls an arbitrary Python object deserializer.
- Primitive serialization tests pass for Venn-Abers and IntervalRegressor JSON-safe v2 round-trips.

## Verification method

Automated pytest tests for wrapper persistence, calibrator primitive serialization, and the state-persistence security trust boundary.

## Verification targets

- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_save_and_load_state_roundtrip_classification
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_save_and_load_state_roundtrip_multiclass
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_save_and_load_state_roundtrip_regression
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_save_state_writes_schema_version_3_manifest
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_load_state_rejects_legacy_schema_versions
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_load_state_rejects_checksum_mismatch
- pytest: tests/unit/core/test_wrap_explainer_persistence.py::test_load_state_rejects_unsupported_schema_version
- pytest: tests/unit/core/test_wrap_explainer_persistence_security.py::test_load_state_rejects_legacy_artifact_before_unpickling_malicious_wrapper
- pytest: tests/unit/core/test_wrap_explainer_persistence_security.py::test_load_state_rejects_python_pickle_calibrator_primitive
- pytest: tests/unit/core/test_wrap_explainer_persistence_security.py::test_load_state_rejects_python_pickle_nested_inside_fast_collection
- pytest: tests/unit/core/test_wrap_explainer_persistence_security.py::test_wrap_explainer_module_never_calls_unsafe_deserializers
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

This requirement verifies supported persistence contracts. State artifacts are treated as untrusted input: a valid SHA-256 checksum proves internal consistency only, never provenance or authenticity. This requirement does not promise compatibility with legacy pre-v3 (pickle-based) artifacts; migrating those requires a trusted, older calibrated-explanations environment the operator already controls, followed by re-saving with the current safe schema.
