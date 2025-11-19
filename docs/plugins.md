# Plugins

Calibrated Explanations supports an optional, extensible plugin system. By default,
you don’t need plugins to run calibrated factual and alternative explanations.
When you want speed-ups (e.g., FAST) or custom visualizations, install a curated
external bundle and wire it in. If you’re extending the framework, follow the
plugin contract to preserve calibration semantics.

Choose your path:

- For practitioners: Use external plugins to enable optional speed-ups and plots → {doc}`practitioner/advanced/use_plugins`
- For contributors: Develop plugins that honor the CE contract → {doc}`contributor/plugin-contract`

Community listings and the curated install extra live here: {doc}`appendices/external_plugins`.

Notes

- Plugins are optional and externally distributed. Core workflows work without them (ADR-027).
- Wiring methods (priority order):
  1) Explainer parameters; 2) Environment variables; 3) pyproject.toml; 4) Plugin-declared dependencies.
- Trust/deny controls and discovery are available via the registry and CLI; see contributor docs for details.
- For detailed wiring (env vars, pyproject, dependency seeding), see {doc}`contributor/extending/plugin-advanced-contract`.

## Guard Plugins

Guard plugins filter perturbations, candidates, and entire batches before explanations reach the user. They implement the `GuardPlugin` protocol defined in `calibrated_explanations.plugins.guards`, are initialized with a `GuardContext`, and are resolved through the same registry used for explanation/interval plugins.

- **GuardPlugin** – Declares `plugin_meta` (identifier, modes, tasks, capabilities) plus methods for `supports_mode`, `initialize`, `filter_perturbations`, `filter_candidates`, and `accept_batch`. Execution builds guard-aware perturbations by invoking `context.guard_orchestrator`.
- **GuardContext** – A frozen dataclass that includes the calibrated learner, `x_cal/y_cal`, interval learner, feature metadata, and `metadata` (e.g., guard_params and guard_enabled). Guard plugins use it to fit conformal region or domain constraint logic.

Configure guards through the same three surfaces as other plugins:

1. **CalibratedExplainer kwargs**: pass `guard_plugin` (`str` identifier, callable, or instance), `guard_enabled` (`bool`), and `guard_params` (mapping) to override the defaults.
2. **Environment variables**: `CE_GUARD_PLUGIN`, `CE_GUARD_ENABLED`, and `CE_GUARD_PARAMS` (JSON blob) mirror kwarg precedence.
3. **pyproject.toml**: `[tool.calibrated_explanations.guards]` can set `plugin`, `enabled`, and `params`, and can extend fallbacks via `fallbacks`.

Guards no longer live on `CalibratedExplainer` directly (no `guard_params` parameter or setter), so refer to the plugin manager documentation for how guard state flows through `ExplanationOrchestrator` and `GuardOrchestrator`.
