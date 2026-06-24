# CE-REQ-PLOTSPEC-GOV-001 - PlotSpec Validation Boundary Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PLOTSPEC-GOV-001 |
| obligation_type | visualization_behavior |
| claim_refs | CE-CAP-PLOTSPEC-001 |
| adr_refs | ADR-036, ADR-037 |
| status | active |
| verification_status | verified |
| tif_exemption | repository_policy |
| tif_exemption_rationale | PlotSpec IR round-trip and schema validation operate at the IR layer below the public API; they are verified by unit tests that construct and validate PlotSpec objects directly, not through WrapCalibratedExplainer. |

## Scope

ADR-036 and ADR-037 PlotSpec governance: canonical PlotSpec round-trip behavior, schema/primitives validation, and renderer-boundary validation before rendering.

## Observable behavior

- PlotSpec artifacts serialize and round-trip through the canonical model.
- PlotSpec schema and primitive tests validate supported artifact shapes.
- Invalid PlotSpec-shaped artifacts raise before renderer invocation.
- Non-PlotSpec artifacts pass through the validation boundary unchanged.

## Acceptance criterion

- PlotSpec round-trip and headless tests pass.
- PlotSpec schema/primitives tests pass.
- Validation-boundary tests raise before render for invalid PlotSpec-shaped artifacts.
- Non-PlotSpec boundary tests preserve pass-through behavior.

## Verification method

Automated pytest tests for PlotSpec schema, round-trip, and renderer-boundary validation.

## Verification targets

- pytest: tests/unit/viz/test_plotspec_roundtrip_and_headless.py::test_global_roundtrip_preserves_entries
- pytest: tests/unit/viz/test_plotspec_schema_and_primitives.py::test_example_plotspec_validates
- pytest: tests/unit/viz/test_plot_plugin_validation_boundary.py::test_plotspec_shaped_invalid_artifact_raises_before_render
- pytest: tests/unit/viz/test_plot_plugin_validation_boundary.py::test_non_plotspec_artifact_passes_through

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies PlotSpec validation and boundary behavior. It does not judge visual design quality or pixel-level rendering fidelity.
