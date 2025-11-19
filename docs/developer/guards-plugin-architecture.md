---
title: Guard Plugin Architecture
---

# Guard Plugin Architecture

This guide explains how **guards** are now first-class plugins in the explain subsystem.

## Why Guard Plugins

Guards filter perturbations/candidates to keep explanations within the training distribution. The new design moves all guard state out of `CalibratedExplainer`, routes configuration through `PluginManager`, and lets `ExplanationOrchestrator` and execution plugins consume a shared `GuardOrchestrator` built on the plugin registry.

## Architecture Overview

```
CalibratedExplainer
  └── PluginManager (single source of truth)
        ├── ExplanationOrchestrator
        │     └── GuardOrchestrator (initialized per explainer)
        │           └── GuardPlugin (e.g., ConformalRegionsGuardPlugin)
        └── Registry (resolves GuardPlugin identifiers)
```

Guard plugins are resolved by precedence (kwargs > env vars > pyproject > fallback chain), initialized with a read-only `GuardContext`, and used by execution plugins via `context.guard_orchestrator`.

## GuardPlugin Protocol

```python
class GuardPlugin(Protocol):
    plugin_meta: PluginMeta

    def supports_mode(self, mode: str, *, task: str) -> bool: ...
    def initialize(self, context: GuardContext) -> None: ...
    def filter_perturbations(...): ...
    def filter_candidates(...): ...
    def accept_batch(...): ...
```

`GuardContext` carries calibration data, the fitted learner, interval learner, feature metadata, and guard_params metadata. Plugins must treat it as frozen and thread-safe.

## Writing a Custom Guard Plugin

```python
class DomainGuardPlugin(GuardPlugin):
    plugin_meta = PluginMeta(...)

    def supports_mode(self, mode: str, *, task: str) -> bool:
        return True

    def initialize(self, context: GuardContext) -> None:
        self._bounds = context.metadata.get("bounds", {})

    def filter_perturbations(...):
        # drop rows outside bounds
        return filtered_x, filtered_feature
```

Register your guard with `register_guard_plugin("custom.guard.domain", DomainGuardPlugin())` and mark it trusted when needed.

## Integration Points

1. **Registration:** builtin guards register during `plugins.builtins`; custom guards call `register_guard_plugin` and can be trusted via env var controls.
2. **Configuration:** `PluginManager.resolve_guard_plugin` handles overrides (`guard_plugin`, `CE_GUARD_PLUGIN`, `pyproject` settings) and caches the chosen plugin.
3. **Initialization:** `GuardOrchestrator.initialize` receives `GuardContext` (new metadata includes guard_params and guard_enabled). Guards fit on calibration data here.
4. **Execution:** `ExplanationContext` exposes `guard_orchestrator`; sequential/parallel executors call `filter_perturbations` (and `explain_predict_step` now filters candidates before generating perturbations).

## Registration Flow

1. Import or define the plugin.
2. Call `register_guard_plugin("identifier", plugin_instance)`.
3. Ensure the plugin metadata declares supported `modes`, `tasks`, and `capabilities` (e.g., `"guard:conformal"`).
4. Trust it via `trust_plugin("identifier")` or `CE_TRUST_PLUGIN` if needed.

## Architecture Diagram

```
+-----------------+           +--------------------+
| Calibrated      |  delegates| PluginManager      |
| Explainer       |---------->| (guards, intervals, |
+-----------------+           |  explanations)     |
                              +--------------------+
                                       |
                  +--------------------+--------------------+
                  |                                         |
         ExplanationOrchestrator                   GuardOrchestrator
                  |                                         |
        ExplanationContext                        GuardPlugin (Conformal)
                  │                                         │
        Execution plugins (sequential, parallel) <────────────┘
```

## Key Takeaways

- Guards are now discoverable plugins; `CalibratedExplainer` itself no longer stores guard state.
- `GuardContext` ensures each GuardPlugin gets read-only visibility into calibration data.
- Execution plugins invoke `context.guard_orchestrator` before computing weights.
- Configuration occurs in a single place (PluginManager) via kwargs, env vars, and pyproject entries.
