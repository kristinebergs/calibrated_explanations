---
title: Guards Plugin Migration (preview)
---

# Guards Plugin Migration

This guide prepares teams for the guard plugin architecture. Guards are now resolved through the plugin manager and explained via the developer documentation. Use this doc when migrating legacy guard configuration to the new plugin surfaces.

## Before (legacy)

- `CalibratedExplainer` accepted `guard_params` and exposed setter helpers (`set_guard`, `get_guard`).
- Guard logic lived on the explainer and there was a single hard-coded `ConformalRegionOracle` instance.
- Guard filtering occurred via name-mangled hooks inside `CalibratedExplainer` and sequential helpers.

## After (plugin-first)

- Guard state lives inside `GuardOrchestrator` under `core/explain/guards` and is referenced through `ExplanationContext.guard_orchestrator`.
- Configuration flows through `PluginManager.resolve_guard_plugin` with fallback identifiers (`core.guard.conformal_regions`).
- Guard plugins implement the `GuardPlugin` protocol and register via `register_guard_plugin(identifier, plugin)`.
- Execution plugins call `context.guard_orchestrator.filter_perturbations` (and `explain_predict_step` filters candidate pools) before computing weights.

## Migration Checklist

1. Remove direct `guard_params` usage from `CalibratedExplainer` callers; prefer `guard_params` kwargs on the constructor or `CE_GUARD_PARAMS`.
2. Register a custom guard plugin if the old workflow injected a bespoke `ConformalRegionOracle`. Implement `supports_mode`, `initialize`, and the filter hooks as needed.
3. Trust the new plugin if it lives outside the core tree (`trust_plugin("identifier")` or `CE_TRUST_PLUGIN`).
4. Update tests/documentation to interact with `context.guard_orchestrator` instead of `CalibratedExplainer` guard setters.

## Validation

- Run explanation suites in factual/alternative/fast modes with and without the guard enabled.
- Inspect `PluginManager.guard_orchestrator` to ensure the expected plugin identifier was selected.
- Verify `GuardContext.metadata` contains the expected `guard_params` so the plugin can fit correctly.

For deep dives, see {doc}`developer/guards-plugin-architecture` and {doc}`guides/guard-configuration`.
