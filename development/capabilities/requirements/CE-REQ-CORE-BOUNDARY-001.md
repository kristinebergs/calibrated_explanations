# CE-REQ-CORE-BOUNDARY-001 - Core Boundary Import Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-CORE-BOUNDARY-001 |
| obligation_type | static_policy |
| claim_refs | CE-CAP-CORE-001 |
| adr_refs | ADR-001 |
| status | active |
| verification_status | verified |

## Scope

Package import boundaries governed by ADR-001 across core, calibration, explanations, schema, plugins, cache, and parallel services.

## Observable behavior

- The import graph does not contain prohibited cross-sibling imports.
- Core packages import independently without forcing optional implementation packages at module load.
- ADR-001 package classifications and boundary rules remain documented and executable in CI.

## Acceptance criterion

- Import-graph tests find no cross-sibling import violations in `calibrated_explainer`.
- Top-level package circular-import checks pass.
- The ADR-001 CI enforcement test passes against the current package graph.
- Core package import tests preserve the sanctioned root export and lazy import behavior.

## Verification method

Automated pytest tests for import-graph and ADR-001 boundary enforcement.

## Verification targets

- pytest: tests/unit/test_import_graph_enforcement.py::TestImportGraphStaticAnalysis::test_should_not_have_cross_sibling_imports_in_calibrated_explainer
- pytest: tests/unit/test_import_graph_enforcement.py::TestImportGraphStaticAnalysis::test_should_find_no_circular_imports_in_top_level_packages
- pytest: tests/unit/test_import_graph_enforcement.py::TestImportGraphIntegration::test_should_enforce_adr001_boundaries_in_ci
- pytest: tests/unit/test_import_graph_enforcement.py::TestImportGraphRuntime::test_should_import_core_packages_independently

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies import and package-boundary policy. It does not validate every runtime behavior implemented inside each package.
