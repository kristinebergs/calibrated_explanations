# Contributing — extras and test workflows

This project separates a lean core install from optional extras for visualization, notebooks and evaluation. The guidelines below explain how to work with the extras and run tests with or without them.

Repository authority
--------------------

> `Moffran/calibrated_explanations` is the authoritative upstream repository.
> Issues, discussions, milestones, active and archived release plans,
> pull-request review, plugin-intake requests, releases, tags, changelog
> entries, security advisories, and published documentation are managed
> there.
>
> Contributors may develop changes on branches in personal forks and submit
> pull requests targeting `Moffran/calibrated_explanations`.
>
> `kristinebergs/calibrated_explanations` is retired as an official
> development workspace, planning repository, and plugin-intake target. No
> active instruction, template, script, or skill may direct new official
> project activity there.

- Canonical development install (required dev extras + pinned constraints, then
  pre-commit hooks):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev] -c constraints.txt
pre-commit install
```

- Core-only install (core-feature development without the dev toolchain):

```powershell
pip install -e .
```

- Install visualization extras (required to run viz tests and examples):

```powershell
pip install -e .[viz]
```

- Install notebook extras (for notebook development):

```powershell
pip install -e .[notebooks]
```

- Install evaluation extras (for reproducing experiments):

```powershell
pip install -r evaluation/requirements.txt
# or using the project extras:
pip install -e .[eval]
```


Running tests
-------------

- Fast core-only test run (recommended for local development and CI):

```powershell
make test-core
```

- Run only visualization tests (install the `viz` extras first):

```powershell
pip install -e .[viz]
make test-viz
```

- Run the full test suite (includes viz tests; may be slower):

```powershell
pip install -e .[viz]
pytest
```

Local validation gates
-----------------------

- Inner-loop checks while editing:

```powershell
make quick
```

- Focused release-task verification before marking a release-plan task complete:

```powershell
make local-checks-task TASK=<n>
```

- Blocking PR-scope preflight before opening/updating a PR:

```powershell
make local-checks-pr
```

- Heavy local gate for merge-readiness or maintainer-requested final validation:

```powershell
make local-checks-full
```

- Release-boundary local validation only:

```powershell
make local-checks-release
```

`make local-checks` is a compatibility alias for `make local-checks-full`
(the heavy gate), not the routine contributor workflow — use `make quick` and
`make local-checks-pr` for day-to-day work.

CLI and plugin configuration
----------------------------

- The plugin CLI entry point is available as `ce.plugins` (packaged via
  `pyproject.toml`). Use it to inspect registered plugins and trust state:

```powershell
ce.plugins list all
```

- Configure plugins via environment variables or explainer kwargs:
  - Env vars: `CE_EXPLANATION_PLUGIN`, `CE_INTERVAL_PLUGIN`,
    `CE_INTERVAL_PLUGIN_FAST`, `CE_PLOT_STYLE`, plus their `*_FALLBACKS`
    counterparts.
  - Explainer kwargs: `factual_plugin`, `alternative_plugin`, `fast_plugin`,
    `interval_plugin`, `fast_interval_plugin`, and `plot_style`.
    - Preferred FAST activation: pass `fast=True` to `CalibratedExplainer` to
      enable FAST-mode execution. `fast_plugin` remains available when you need
      to target a specific FAST implementation, but `fast=True` is the primary
      activation mechanism.

Style guardrails
----------------

- Naming and documentation conventions are enforced in CI (Ruff naming + pydocstyle).
- Review the quick-reference checklist in `.github/CONTRIBUTING.md` before
  submitting changes that touch public APIs or new modules.
- Run naming guardrails locally with `pre-commit run ruff-naming --all-files`
  (or `ruff check --select N`) before opening a PR.
- Legacy API contract updates are required for user-facing API changes:
  verify against `development/finished-work/legacy_user_api_contract.md` (historical surface inventory) and
  update `tests/unit/api/test_legacy_user_api_contract.py` in the same PR (ADR-020).
  Review `release.md` and the active plan's `## Release-specific gates` section before cutting a release.

Logging and Observability
-------------------------

We follow [Standard-005](development/standards/STD-005-logging-and-observability-standard.md) for logging. When adding new features:
1. Use the appropriate logger domain (e.g. `calibrated_explanations.core.*`, `calibrated_explanations.plugins.*`).
2. Use `calibrated_explanations.logging.logging_context` to propagate identifiers like `explainer_id` or `plugin_identifier`.
3. Consult [ADR-028](development/adrs/ADR-028-logging-and-governance-observability.md) for architecture details.

Notes
-----
- The test suite automatically skips tests marked with `@pytest.mark.viz` when
  `matplotlib` cannot be imported. This makes local development faster and
  avoids false failures on minimal installs.
- If you need to run only viz tests, install the `viz` extras and run
  `pytest -m viz`.

- Local CI parity: the repo provides a local stacked-checks runner `scripts/local_checks.py`, exposed via the `make quick` / `make local-checks-task` / `make local-checks-pr` / `make local-checks-full` / `make local-checks-release` profiles described above. When adding, removing, or changing CI workflows under `.github/workflows/`, update `scripts/local_checks.py` (and `Makefile` if needed) so contributors can reproduce CI behaviour locally. Mark heavy checks as optional in the local runner to avoid slowing developer loops.

If you add or remove optional dependencies, please update `pyproject.toml`,
`evaluation/requirements.txt`, and `evaluation/environment.yml` accordingly.

Contributing a plugin
---------------------

Plugins for `calibrated-explanations` are incubated and published from a
separate plugin repository. Wrote a plugin, or want an existing one promoted
or listed? Open a short
["Plugin intake request"](https://github.com/Moffran/calibrated_explanations/issues/new?template=plugin_publication_request.yml)
issue here — a description, maintainer contact, licence, and known
limitations are all that is needed to start. The maintainers triage requests
and transfer accepted work into the plugin repository, where new plugins
start as experimental. Submitting a request does not authorise publication;
maturity and publication decisions rest with the CE plugin maintainers.
