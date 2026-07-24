# Installation

Calibrated Explanations is published on PyPI and conda-forge. Install the base
package first, then add extras that unlock plotting, notebook examples, or
contributor tooling.

## PyPI

```bash
pip install calibrated-explanations
```

Extras are opt-in so you only pull the dependencies you need:

| Extra | Purpose | Install command |
| ----- | ------- | --------------- |
| `viz` | Matplotlib-based plotting and PlotSpec adapters. | `pip install "calibrated-explanations[viz]"` |
| `notebooks` | Jupyter notebook tutorials with pinned dependencies. | `pip install "calibrated-explanations[notebooks]"` |
| `dev` | Full development toolchain (linters, docs, tests). | `pip install "calibrated-explanations[dev]"` |
| `external-plugins` | In-tree optional FAST explanation and interval dependencies. | `pip install "calibrated-explanations[external-plugins]"` |

Published-study environments and reproduction dependencies are maintained in
the
[Calibrated Explanations Studies](https://github.com/kristinebergs/calibrated-explanations-studies)
repository, not as a core package extra.

## conda-forge

```bash
conda install -c conda-forge calibrated-explanations
```

If you rely on extras from PyPI inside a conda environment, install the base
package via conda and then add the relevant extras with `pip`.

## Verifying your environment

```bash
python -c "import calibrated_explanations; print(calibrated_explanations.__version__)"
```

This prints the version selected by your package manager (`1.0.0` or later).

## Optional contributor fast path

For repository development, `pip` remains the canonical fallback:

```bash
python -m pip install -e .[dev] -c constraints.txt
```

If you already use `uv`, you can install the editable development environment
with the same CI constraints:

```bash
uv pip install -e .[dev] -c constraints.txt
```

This `uv` command is optional and constraint-based. The project does not treat
`uv.lock` as an authoritative dependency lockfile.

Entry-point tier: Tier 2.
