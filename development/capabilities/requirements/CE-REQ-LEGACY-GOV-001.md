# CE-REQ-LEGACY-GOV-001 - Legacy User API Stability Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-LEGACY-GOV-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-LEGACY-001 |
| adr_refs | ADR-020 |
| status | active |
| verification_status | verified |

## Scope

ADR-020 legacy public API compatibility for wrapper lifecycle methods, prediction signatures, explanation collection APIs, and forwarding parity.

## Observable behavior

- Legacy wrapper lifecycle methods remain present.
- Prediction and explanation method signatures retain documented compatibility parameters.
- Explanation collection APIs retain stable public methods.
- Wrapper explanation forwarding preserves explicit kwargs and config default injection.

## Acceptance criterion

- Legacy user API contract tests pass for wrapper lifecycle and prediction signatures.
- Collection API stability tests pass.
- Wrapper factual and alternative explanation parity tests pass for explicit and default-injected kwargs.
- Var-keyword compatibility remains intentionally present until the documented removal boundary.

## Verification method

Automated pytest tests for legacy API compatibility.

## Verification targets

- pytest: tests/unit/api/test_legacy_user_api_contract.py::test_wrap_calibrated_explainer_lifecycle_methods_exist
- pytest: tests/unit/api/test_legacy_user_api_contract.py::test_wrap_predict_signature_includes_legacy_parameters
- pytest: tests/unit/api/test_legacy_user_api_contract.py::test_explanation_collection_api_is_stable
- pytest: tests/unit/api/test_legacy_user_api_contract.py::test_wrap_explain_methods_accept_var_keyword_for_ce_parity

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies the public compatibility surface named by ADR-020. It does not require preserving private helpers or removed/deprecated APIs beyond the documented compatibility window.
