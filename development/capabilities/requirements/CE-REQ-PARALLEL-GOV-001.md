# CE-REQ-PARALLEL-GOV-001 - Parallel Execution Runtime Contract

## Metadata

| Field | Value |
|---|---|
| requirement_id | CE-REQ-PARALLEL-GOV-001 |
| obligation_type | runtime_behavior |
| claim_refs | CE-CAP-PARALLEL-001 |
| adr_refs | ADR-004 |
| status | active |
| verification_status | verified |
| tif_exemption | repository_policy |
| tif_exemption_rationale | Parallel strategy resolution is driven by environment-variable configuration and is not observable through the WrapCalibratedExplainer public API; it is verified by unit tests that control the environment and assert on internal strategy state. |

## Scope

ADR-004 parallel execution behavior: explicit opt-in configuration, serial-by-default behavior, deterministic end-to-end parity, and deprecated auto-strategy visibility.

## Observable behavior

- Parallel execution is enabled only through explicit configuration.
- Absent parallel configuration defaults to disabled.
- Initializer parallel and sequential end-to-end paths produce matching results.
- Deprecated `strategy="auto"` emits a visible deprecation warning when enabled, while explicit strategies do not.

## Acceptance criterion

- Environment configuration tests pass for explicit opt-in and post-snapshot isolation.
- End-to-end initializer parallel tests pass against sequential output.
- Parallel strategy tests pass for explicit strategy resolution and deprecation warning behavior.
- Disabled parallel configuration does not emit the auto-strategy deprecation warning.

## Verification method

Automated pytest tests for parallel configuration, strategy resolution, and end-to-end parity.

## Verification targets

- pytest: tests/unit/core/test_calibrated_explainer_parallel_env.py::TestCalibratedExplainerParallelEnv::test_should_enable_parallel_executor_when_env_var_is_set
- pytest: tests/unit/core/test_calibrated_explainer_parallel_env.py::TestCalibratedExplainerParallelEnv::test_should_isolate_ce_parallel_from_env_changes_after_snapshot
- pytest: tests/integration/test_initializer_parallel_end_to_end.py::test_sequential_vs_initializer_parallel_end_to_end
- pytest: tests/unit/perf/test_parallel.py::test_should_emit_deprecation_when_strategy_auto_and_enabled
- pytest: tests/unit/perf/test_parallel.py::test_should_not_emit_deprecation_when_explicit_strategy_and_enabled

## Evidence required

| Field | Required |
|---|---|
| commit_sha | yes |
| package_version | yes |
| test_id | yes |
| result | yes (pass/fail) |

## Assumption boundary

This requirement verifies explicit parallel configuration and parity behavior. It does not prove speedup or performance characteristics on every platform.
