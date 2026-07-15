# Calibrated Explanations documentation

Use this site by audience and by task.

The public source, issue tracker, tags, and releases are maintained in
[`Moffran/calibrated_explanations`](https://github.com/Moffran/calibrated_explanations).
Install published packages from
[PyPI](https://pypi.org/project/calibrated-explanations/), report problems in the
[public issue tracker](https://github.com/Moffran/calibrated_explanations/issues),
and read the
[release notes](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md).

```{admonition} 1.0.0rc1 release candidate
:class: important

The public API is frozen as of `1.0.0rc1`; only release-blocking defect fixes
will be accepted before GA. Install the exact candidate with
`pip install calibrated-explanations==1.0.0rc1` (or use
`pip install --pre calibrated-explanations` to select the newest pre-release).
The package remains classified as Beta during the RC. Before adoption, review
the {doc}`v1.0.0 upgrade checklist <upgrade/v1.0.0-upgrade-checklist>`, the
{doc}`Explanation Schema v1 freeze <schema_v1>`, and the
[RC release notes](https://github.com/Moffran/calibrated_explanations/blob/main/CHANGELOG.md).
```

- New users start in {doc}`get-started/index`.
- Practitioners use {doc}`practitioner/index`.
- Researchers use {doc}`researcher/index`.
- Contributors use {doc}`contributor/index`.

Semantics are mode-specific. Classification, interval regression, and
probabilistic regression do not share one guarantee statement. For semantics,
assumptions, and non-guarantees, use
{doc}`foundations/concepts/calibrated_interval_semantics`.

```{toctree}
:maxdepth: 1
:caption: Start here

get-started/index
practitioner/index
researcher/index
contributor/index
```

```{toctree}
:maxdepth: 1
:caption: Core references

foundations/index
tasks/index
api/index
```

```{toctree}
:maxdepth: 1
:caption: Extensions and project docs

plugins
migration/index
upgrade/index
appendices/external_plugins
appendices/changelog_links
appendices/rtd_tier_map
get-started/ce_first_agent_guide
get-started/copilot-setup
maintenance/legacy-plotting-reference
compare
ROADMAP
citing
```

Entry-point tier: Tier 1.
