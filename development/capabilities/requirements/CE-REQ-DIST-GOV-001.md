# CE-REQ-DIST-GOV-001 - Core Extras Separation Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-DIST-GOV-001 |
| obligation_type | quality_gate |
| claim_refs | CE-CAP-DIST-001 |
| adr_refs | ADR-010 |
| status | active |
| verification_status | verified |

## Scope

ADR-010 distribution boundary between core package behavior, optional core
components, and separately governed companion studies.

## Observable behavior

- The core-extras parity quality gate can be run as a standalone policy check.
- Core import smoke tests do not require optional visualization or
  study-specific dependencies.
- Mass import tests keep optional dependency exposure explicit.

## Acceptance criterion

- The core-extras parity quality gate reports no parity violations.
- Core import smoke tests pass without importing optional plotting backends at package-root import time.
- Mass import tests preserve the optional dependency split across supported modules.

## Verification method

Executable quality gate and automated pytest import-smoke tests.

## Verification targets

- quality-gate: python scripts/quality/check_core_extras_parity.py
- pytest: tests/test_core_import_no_matplotlib.py::test_import_package_does_not_eagerly_import_matplotlib
- pytest: tests/integration/test_import_smoke.py::test_import_smoke_modules
- pytest: tests/integration/test_mass_imports.py::test_mass_imports

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies import-time and parity boundaries. It does not certify packaging metadata for every downstream installer environment.
