# Plugins

Plugins add optional capabilities without changing the core Calibrated
Explanations installation. They are independently versioned packages and run in
the host Python process.

## Recommended visualization family

Install the currently recommended family:

```bash
pip install calibrated-explanations-visualization
```

This family currently installs the mature Plotly visualization plugin. It does
not replace CE's built-in matplotlib plotting, and installing or importing it
does not automatically trust or activate it. Register its components
explicitly:

```python
import ce_visualization_plotly as cevp

cevp.register_plotly_visualization_components()
```

Registration is explicit and idempotent.

## Navigation

- [Official plugin repository](https://github.com/kristinebergs/calibrated-explanations-plugins)
- [Generated package index](https://github.com/kristinebergs/calibrated-explanations-plugins/blob/main/docs/package-index.md)
- {doc}`Practitioner plugin-use guide <practitioner/advanced/use_plugins>`
- {doc}`Contributor plugin contract <contributor/plugin-contract>`
- [Plugin trust model](https://github.com/Moffran/calibrated_explanations/blob/main/development/adrs/ADR-006-plugin-registry-trust-model.md)
- {doc}`Plugin catalog <appendices/external_plugins>`

## Trust

Plugins execute in process with the permissions of the host application.
Install and register a plugin only when you trust its source and maintainer.

## Categories

1. **Official companion plugins** are project-maintained, independently
   released packages in the official plugin repository.
2. **Core-provided optional components** ship with the core project but remain
   opt-in, such as the in-tree FAST bundle.
3. **Community plugins** are independently owned and maintained packages.

Entry-point tier: Tier 1.

```{toctree}
:hidden:

appendices/external_plugins
```
