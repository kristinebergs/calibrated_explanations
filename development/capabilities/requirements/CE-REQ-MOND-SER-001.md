# CE-REQ-MOND-SER-001 - Mondrian Serialization Visibility Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-MOND-SER-001 |
| obligation_type | api_contract |
| claim_refs | CE-CAP-MOND-001 |
| adr_refs | ADR-039, ADR-031 |
| status | active |
| verification_status | verified |
| tif_refs | CE-TIF-MOND-001 |

## Scope

Public API: pickle round-trip and `WrapCalibratedExplainer.save_state` /
`load_state` for wrappers calibrated with `mc=`.

Applicable task types: classification, regression, probabilistic regression.

## Observable behavior

1. Persistence emits `UserWarning` when a configured `mc` is dropped.
2. A loaded wrapper remains bins-calibrated and raises `ValidationError` when
   inference omits explicit `bins=`.

## Acceptance criterion

Persistence does not silently change conditional behavior. The user sees the
drop at persistence time and receives a clear CE exception after load if
inference bins are omitted.

## Verification method

Automated pytest tests in `tests/capabilities/`.

## Verification targets

- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_warn_and_require_explicit_bins_when_pickled_mc_wrapper_is_loaded`
- `pytest: tests/capabilities/test_mondrian_contracts.py::test_should_warn_and_require_explicit_bins_when_saved_mc_wrapper_is_loaded`

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| dataset_id | yes |
| random_seed | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies visible persistence behavior only. It does not require
portable serialization of arbitrary categorizer objects.
