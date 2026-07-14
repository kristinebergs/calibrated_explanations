# External plugins

The external plugin index tracks optional extensions that remain outside the
core package. Each entry must:

- Reuse the calibrated prediction bridge and respect ADR-024/025/026 guardrails.
- Highlight binary & multiclass classification plus probabilistic and interval
  regression coverage.
- Document optional telemetry or compliance hooks inside the `Optional extras`
  section of the hosting page.

## Vetted plugins

| Identifier | Summary | Install | Notes |
| --- | --- | --- | --- |
| `core.explanation.fast` / `core.interval.fast` | FAST explanations and interval calibrators packaged as an opt-in bundle. | `pip install "calibrated-explanations[external-plugins]"` | Wheel and sdist installs auto-register the shipped FAST identifiers. The optional helper is `from external_plugins.fast_explanations import register; register()`. |

## Community submissions

This table is reserved for community-maintained plugins. Open an issue with the
plugin metadata, ADR alignment notes, and calibration guarantees to request
listing.

| Identifier | Contact | Status |
| --- | --- | --- |
| _TBD_ | _community_ | _Pending_ |

### Optional: aggregated install extra

`pip install "calibrated-explanations[external-plugins]"` installs every curated
external plugin dependency bundle plus the pinned versions of ``numpy``,
``pandas``, and ``scikit-learn`` required by FAST mode. Installed distributions
auto-register the shipped FAST identifiers on import. If you still want the
helper, import `external_plugins.fast_explanations` and call `register()`
directly. The helper is not exposed as a `python -m` command.

### Optional: telemetry disclosure

External plugins should clearly mark telemetry emission as opt-in and link back
to {doc}`../foundations/governance/optional_telemetry` whenever instrumentation is enabled.
