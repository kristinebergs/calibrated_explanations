# External plugins

The external plugin index tracks optional extensions that remain outside the
core package. Each entry must:

- Reuse the calibrated prediction bridge and respect the ADR-006 (plugin
  registry trust model) and ADR-037 (visualization extension and rendering
  governance) guardrails.
- Highlight binary & multiclass classification plus probabilistic and interval
  regression coverage.
- Document optional telemetry or compliance hooks inside the `Optional extras`
  section of the hosting page.

Listing here does not mean CE core owns or maintains the package; ownership
and maintenance status are noted per entry.

## Vetted plugins

| Identifier | Summary | Install | Compatibility | Activation | Notes |
| --- | --- | --- | --- | --- | --- |
| `core.explanation.fast` / `core.interval.fast` | FAST explanations and interval calibrators packaged as an opt-in bundle. | `pip install "calibrated-explanations[external-plugins]"` | Ships with CE; tracks the installed CE version. | Auto-registers on import (in-tree bundle, not a third-party package). | Wheel and sdist installs auto-register the shipped FAST identifiers. The optional helper is `from external_plugins.fast_explanations import register; register()`. |
| `calibrated-explanations-visualization-plotly` (package `ce_visualization_plotly`) | Interactive Plotly layouts for factual, alternative, global, and dashboard explanation views (hover inspection, standalone HTML export, searchable feature controls). | `pip install calibrated-explanations-visualization-plotly` (add `[live]` for the optional Dash dashboard); also curated into the `calibrated-explanations-visualization` family metapackage (`pip install calibrated-explanations-visualization`). | Requires `calibrated-explanations>=1.0.0rc2,<2` and Python >=3.11; requires `plotly>=5.18` (mandatory) and `dash>=3.1` (only for `[live]`). | **Explicit, not automatic.** Importing the package has no side effects. Call `ce_visualization_plotly.register_plotly_visualization_components()` once per process to register styles (e.g. `style="plotly.local.factual_bars"`, `"plotly.local.alternative_bars"`, `"plotly.global.instance_explorer"`) with CE's public plugin registry (ADR-006). | Third-party package, not maintained by CE core; status `mature` as of 2026-07-21. Preserves CE calibrated semantics verbatim (no rescaling/re-derivation) — only presentation (hover cards, HTML output) differs from CE's built-in matplotlib renderer. Limitation: requires the exact CE floor above; older CE 1.0.0rc1 installs are not supported. |

### Optional: aggregated install extra for the in-tree FAST bundle

`pip install "calibrated-explanations[external-plugins]"` installs the pinned
versions of `numpy`, `pandas`, and `scikit-learn` required by FAST mode (see
`pyproject.toml`). Installed distributions auto-register the shipped FAST
identifiers on import. If you still want the helper, import
`external_plugins.fast_explanations` and call `register()` directly. The
helper is not exposed as a `python -m` command. This extra does not install
the third-party Plotly visualization plugin above; install that separately.

## Community submissions

To request a listing, open an issue with the plugin's distribution name,
installation command, plugin/style identifiers, the CE version range it
targets, and a short note on which ADR-006/ADR-037 guardrails it follows.
Maintainers add a row above once the listing is verified.

### Optional: telemetry disclosure

External plugins should clearly mark telemetry emission as opt-in and link back
to {doc}`../foundations/governance/optional_telemetry` whenever instrumentation is enabled.
