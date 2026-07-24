# Plugin catalog

This catalog provides stable discovery information. The
[generated package index](https://github.com/kristinebergs/calibrated-explanations-plugins/blob/main/docs/package-index.md)
is authoritative for the complete official inventory, versions, maturity, and
curation state.

## Official companion plugins

Official plugins are maintained and independently released from the public
[Calibrated Explanations Plugins](https://github.com/kristinebergs/calibrated-explanations-plugins)
repository.

### Plotly visualization

- **Distribution:** `calibrated-explanations-visualization-plotly`
- **Purpose:** interactive Plotly views for CE explanations.
- **Recommended install:** `pip install calibrated-explanations-visualization`
- **Activation:** explicit registration with
  `ce_visualization_plotly.register_plotly_visualization_components()`;
  importing or installing alone does not activate it.
- **Source and documentation:**
  [plugin package](https://github.com/kristinebergs/calibrated-explanations-plugins/tree/main/packages/visualization/calibrated-explanations-visualization-plotly)

## Core-provided optional components

The in-tree FAST explanation and interval components are shipped by the core
project and are not third-party packages. Install their optional dependency
bundle with:

```bash
pip install "calibrated-explanations[external-plugins]"
```

See {doc}`../practitioner/advanced/use_plugins` for the current activation and
usage guidance.

## Community plugins

No community plugins are listed currently. A future listing provides
discoverability only; it does not imply ownership, endorsement, or maintenance
by CE core. Submit listing requests through the
[authoritative plugin-intake form](https://github.com/Moffran/calibrated_explanations/issues/new?template=plugin_publication_request.yml).

Entry-point tier: Tier 3.
