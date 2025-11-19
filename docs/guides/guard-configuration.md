---
title: Guard Configuration Guide
---

# Guard Configuration Guide

Guards are now registered plugins. This guide explains the surfaces for configuring the guard plugin, enabling/disabling the feature, and passing `guard_params` to the plugin.

## Environment Variables

| Name | Purpose | Example |
| --- | --- | --- |
| `CE_GUARD_ENABLED` | Enable/disable guard evaluation globally (`true`/`false`). | `export CE_GUARD_ENABLED=true` |
| `CE_GUARD_PLUGIN` | Override the guard plugin identifier or module path (must be registered/trusted). | `export CE_GUARD_PLUGIN="core.guard.conformal_regions"` |
| `CE_GUARD_PARAMS` | JSON blob forwarded to `GuardContext.metadata["guard_params"]`. | `export CE_GUARD_PARAMS='{"alpha":0.1,"enforcement":false}'` |

Environment overrides are respected after keyword arguments but before the default fallback chain.

## pyproject.toml

Add a `[tool.calibrated_explanations.guards]` block to configure defaults:

```toml
[tool.calibrated_explanations.guards]
enabled = true
plugin = "core.guard.conformal_regions"
fallbacks = ["custom.guard.domain_constraints"]

[tool.calibrated_explanations.guards.params]
alpha = 0.15
n_clusters = 10
enforcement = true
```

Pyproject settings merge with env vars and kwargs; more specific overrides win.

## CalibratedExplainer Keyword Arguments

When creating an explainer you can pass:

```python
explainer = CalibratedExplainer(
    learner,
    x_cal,
    y_cal,
    guard_enabled=True,
    guard_plugin="core.guard.conformal_regions",
    guard_params={"alpha": 0.05, "n_clusters": 6},
)
```

`guard_plugin` accepts a string identifier, plugin instance, or callable returning a plugin. `guard_enabled=False` disables guarding even if other surfaces request it.

## Examples

1. **Default conformal guard** (no extra config): rely on built-in `core.guard.conformal_regions` and pass parameters through `guard_params`.
2. **Custom domain guard**: register `custom.guard.domain_constraints` and set `CE_GUARD_PLUGIN` to that identifier along with bounds via `guard_params` or `CE_GUARD_PARAMS`.
3. **Safely disable temporarily**: `guard_enabled=False` or `CE_GUARD_ENABLED=false` bypass guard filtering entirely.

Guards inherit the same trust controls as other plugins, so untrusted identifiers are ignored unless explicitly trusted via the registry helpers or `CE_TRUST_PLUGIN`.
