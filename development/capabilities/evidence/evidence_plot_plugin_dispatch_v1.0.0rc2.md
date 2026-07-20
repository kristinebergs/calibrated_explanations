# Evidence: third-party plot-plugin dispatch fix (v1.0.0rc2 candidate)

Date: 2026-07-20 (baseline work recorded 2026-07-19; independent review and
correction pass recorded 2026-07-20)
Author: automated agent session (Claude Code), evidence verified by execution
Plan: `development/current-work/v1.0.0-rc2_plan.md`

## Independent review and correction pass (2026-07-20)

A second-pass review of the initial implementation (commit `8459df1c`)
identified three real gaps, verified against actual code/execution before
any fix was applied:

1. **Configured-preference styles bypassed the raw dispatcher.** The
   explicit-`style=` interception did not consult
   `PluginManager.plot_style_override` / `CE_PLOT_STYLE` / pyproject / the
   plugin-dependency chain, so a third-party style selected through any of
   those mechanisms still lost `filter_top`/`uncertainty`/`rnk_metric`/
   `rnk_weight`/options to built-in consumption. Confirmed by executing a
   configured-style probe against the fixed-but-not-yet-corrected candidate:
   the plugin was never invoked.
2. **`style_override` alone (no `style=`) bypassed interception on the
   factual/alternative surfaces**, while `plot_global` already checked both.
   Confirmed by executing `factual.plot(style_override="vendor.x", ...)`.
3. **`resolve_plot_plugin_strict` computed the `trusted` flag before applying
   `renderer_override`**, so overriding to an untrusted, unrelated renderer
   inherited the original style's trust and received full runtime access
   (explainer, model, request data) — a real ADR-006 violation. Confirmed by
   registering a trusted style, overriding its renderer to a separately
   registered *untrusted* renderer, and observing a non-empty
   `context.runtime`.

All three were fixed (see commit `9e1fa1e5`): `resolve_plot_plugin_strict`
now recomputes trust as `trusted and renderer_descriptor.trusted` whenever a
renderer override is applied; the three public surfaces now check
`style_override` alongside `style` for explicit selection and, when neither
is given, resolve the same configured-preference precedence the built-in
path already used, dispatching the full raw request through the existing
strict resolver when it names a registered third-party plugin. A guard
(`_configured_dispatch_blocked`) preserves the exact pre-existing
reachability condition for `use_legacy`/`return_plot_spec`/explicit style, so
none of those combinations changed behavior. Unresolvable configured styles
fall through to the existing built-in/legacy path unchanged (no broadened
fallback). 10 new regression tests were added directly reproducing each
finding (positive and negative), and both the installed-wheel proof and the
six-style Plotly no-bridge proof were rebuilt and rerun against the
corrected candidate — all still pass. The reviewer also flagged three gaps
in the temporary `tmp/no-bridge-proof` plugins-repo branch (stale
CE floor, a test asserting bridge installation, an implicit trust step);
all three were corrected there (commit `3629ca9`) since they were concrete
and verified (one stale assertion was confirmed to fail before the fix).

## Baselines

| Item | Value |
|---|---|
| CE baseline SHA | `9573af24e62171040dd872067da2f6aeac884944` (post-RC1 `main`) |
| CE candidate SHA | `9e1fa1e5` on branch `feat/plot-plugin-dispatch-rc2` (implementation `8459df1c` + review-driven correction `9e1fa1e5`); docs/plans/evidence committed on the same branch |
| Plotly proof candidate | `3629ca9` on local branch `tmp/no-bridge-proof` (supersedes `b545bf4`/`27b6a6d`) |
| CE `pyproject.toml` version | `1.0.0-dev` (unchanged; preflight tooling owns the RC bump) |
| v1.0.0rc1 anchor | release commit `0464593a`, PyPI-verified 2026-07-15; tags are authoritative on `Moffran/calibrated_explanations` (none on this mirror) |
| v1.0.0rc2 | does not exist anywhere (verified 2026-07-19) |
| Plotly plugin baseline SHA | `f4f7cc84561b032bbc01034f4056543f4bd438b1` (`calibrated-explanations-plugins`, `main`) |
| Plotly package | `calibrated-explanations-visualization-plotly` 0.3.2, declares `calibrated-explanations>=1.0.0rc1,<2` |
| Bridge files | `src/ce_visualization_plotly/_ce_compat.py` (installed via `plugin.register_plotly_visualization_components(install_compat_bridges=True)` and re-exported from `__init__.py`) |
| Plotly no-bridge proof commits | `b545bf4`, `27b6a6d` on local branch `tmp/no-bridge-proof` (not pushed, not published) |

## Root cause (verified executably on baseline, 2026-07-19)

Probe: synthetic builder/renderer/style registered through the public
registry, called through public plot APIs on baseline `9573af24`:

1. **Factual**: builder invoked only from inside `plot_probabilistic` after
   `FactualExplanation.plot` consumed `filter_top`/`uncertainty`/
   `rnk_metric`/`rnk_weight`/`filename` and ranked features. Observed
   `context.options == {'style', 'vendor_opt', 'instance_index'}`.
2. **Alternative**: builder **never invoked** — `AlternativeExplanation.plot`
   calls `plot_alternative()` without `**kwargs`; the `style` kwarg is
   dropped.
3. **Global**: builder invoked with `options == {'payload'}` only;
   `aggregate_positions=True` discarded (with a forwarded-kwargs warning).
4. **Unknown explicit style** (`vendor.missing`): no error; silently rendered
   via `plot_spec.default` appended by `resolve_plot_style_chain`.
5. **`None` renderer result**: `plugin_result is not None` gating caused
   fall-through into built-in rendering (`_require_matplotlib()` on that
   path).
6. **No runtime access**: `PlotRenderContext` had no field carrying the
   originating explainer or request.

## Files changed (CE candidate)

- `src/calibrated_explanations/plotting.py` — dispatch machinery (+ ~280 lines)
- `src/calibrated_explanations/explanations/explanation.py` — factual/alternative interception
- `src/calibrated_explanations/plugins/manager.py` — `resolve_plot_plugin_strict`
- `src/calibrated_explanations/plugins/plots.py` — `PlotRenderContext.runtime`
- `tests/unit/test_plot_third_party_dispatch.py` — new (25 tests)
- `tests/plugins/test_protocols.py` — +5 context tests
- `tests/unit/test_plot_default_promotion.py`, `tests/unit/test_plotting_coverage.py` — shim adaptations (assertion strength preserved; see plan Task 2)
- `CHANGELOG.md`, `docs/contributor/plugin-contract.md`, release plans — documentation

## Public API change

- `PlotRenderContext.runtime: Mapping[str, Any] = field(default_factory=dict)`
  appended after `plugin_config`; positional/keyword backward compatibility
  and pickle behavior covered by tests. Runtime is trust-gated (ADR-006) and
  dropped on serialization.
- `PluginManager.resolve_plot_plugin_strict(identifier, *, renderer_override=None)`
  → `(plugin, identifier, trusted)`; raises `ConfigurationError` for
  denied/unregistered/incomplete styles and unresolvable renderer overrides.
- API snapshot review: **pending** (governed process; see plan checklist).

## Option-forwarding matrix (proven)

| Surface | Options proven to arrive verbatim | Proof |
|---|---|---|
| Factual | `filter_top` (incl. default `None`), `uncertainty`, `rnk_metric`, `rnk_weight`, nested `vendor_options` | CE tests + wheel proof `factual options complete` |
| Alternative | `filter_top`, `rnk_metric`, `rnk_weight`, `vendor_flag`; `style` survives unrewritten | CE tests + wheel proof |
| Global | `aggregate_positions`, nested `global_options`, `bins`; reserved `payload` added by CE | CE tests + wheel proof `global options preserved` |
| Dashboard | `factual_options`, `alternative_options`, `available_cards`, `precompute` | CE dashboard test + Plotly proof |

## Transport behavior (proven)

`filename` → `context.path` verbatim (no `plots/` prefix, no suffix rewrite);
`path` preserved exactly; equal `path`+`filename` accepted; conflicting values
raise `ValidationError`; output path + omitted `show` → `False`; no output
path → `True`; explicit `show` wins; `save_ext` list→tuple, absent→`None`
(no invented static formats). Five dedicated tests.

## Strict resolution and trust behavior (proven)

- Registered trusted style resolves exactly (context.style == requested id).
- Unregistered explicit style → `ConfigurationError` naming the identifier,
  the remedy, and the registered styles; message is surface-neutral.
- `CE_DENY_PLUGIN` denial → `ConfigurationError` at dispatch.
- Untrusted registered style renders (rc1 parity) but `context.runtime == {}`.
- Builder error and renderer error propagate; no built-in fallback (built-in
  plot functions monkeypatched to fail loudly in tests — never called).
- `style="vendor.x", use_legacy=True` → `ValidationError`; differing string
  `style_override` → `ValidationError`.
- Configured-preference fallback (manager/env/pyproject) retains rc1
  chain semantics with visible `UserWarning` + INFO (existing tests, incl.
  the adapted renderer-override fallback test).

## `None` renderer-result behavior (proven)

`_PlotDispatchOutcome.handled` is independent of the result; a `None` result
propagates to the caller and built-in rendering never runs (guarded by
failing stubs in tests; wheel proof repeats this without Matplotlib
installed, where any fall-through would raise).

## Test results

- Focused: `tests/unit/test_plot_third_party_dispatch.py` (31, incl. 10 added
  during the review-correction pass: configured manager-override dispatch
  for factual/alternative/global, `None`-result handling on the configured
  path, `use_legacy=True` still bypassing configured dispatch, unregistered
  configured styles falling through silently, `style_override`-only
  selection for factual/alternative, and renderer-override trust
  recomputation both directions) + `tests/plugins/test_protocols.py`
  (21 incl. 5 new) + `tests/unit/test_plotting.py`,
  `tests/unit/test_plot_default_promotion.py`,
  `tests/unit/plugins/test_manager.py`, `tests/plugins/` — all green
  (2026-07-19/20, Python 3.11 and 3.14 envs).
- Built-in regression: full `tests/` run excluding `tests/viz` — **exit 0**
  both before (2026-07-19) and after (2026-07-20) the correction pass, Python
  3.11, command `python -m pytest tests -q -x --ignore=tests/viz`. Includes
  factual
  regular/legacy/PlotSpec/uncertainty, regression, alternative
  regular/ensured/triangular, global built-in PlotSpec/legacy, one-sided
  interval rejection, renderer override, visible-fallback tests.
- Repository gates (`make local-checks-pr`, docs build, viz lane, CI matrix):
  **pending** — see plan checklist; do not treat this report as claiming
  them.

## Installed-wheel evidence (2026-07-19)

- Wheels: `calibrated_explanations-1.0.0.dev0-py3-none-any.whl` (+ sdist)
  built from candidate `8459df1c` via `uv build`; synthetic plugin
  `ce_synthetic_viz-0.0.1-py3-none-any.whl` (one factual, one alternative,
  one global, one dashboard style; public registration + ADR-006 keyed trust
  helpers; no import-time side effects).
- Environment: fresh `uv venv` (CPython 3.11.9) at a short path
  (`%TEMP%\cewp`), packages: calibrated-explanations 1.0.0.dev0,
  ce-synthetic-viz 0.0.1, numpy 2.4.6, scikit-learn 1.9.0, scipy 1.17.1,
  pandas 3.0.3, crepes 0.9.1, venn-abers 1.5.3. **Matplotlib absent.**
- Result: **20/20 checks passed** — wheel-only import (no source tree on
  `sys.path`), all four styles dispatched with complete options, dashboard
  drove `explain_factual`/`explore_alternatives` through `context.runtime`,
  `None` result handled, callable identities unchanged before/after
  registration and rendering. Raw log: session scratchpad
  `wheelproof/proof_out.txt` + `wheelproof/env_packages.txt`.

## Plotly no-bridge evidence (2026-07-20)

- Temporary patch (branch `tmp/no-bridge-proof`, commits `b545bf4`,
  `27b6a6d`): removed `_ce_compat` import from `__init__.py` and bridge
  installation from `plugin.py` (parameter kept as a no-op);
  `instance_workspace.py` consumes `context.options["payload"]` and
  `context.runtime` for dashboard precomputation and strips plot-only keys
  (`filter_top` etc.) from explain-time kwargs — a latent bridge bug: the old
  bridge forwarded `factual_options` straight into `explain_factual`, which
  real CE rejects (`explain_kwargs_schema`); it only passed with fake test
  explainers.
- Environment: fresh `uv venv` (CPython 3.11.9, `%TEMP%\cewp2`): CE candidate
  wheel + patched plotly-plugin wheel (`--no-deps`; the 0.3.2 requirement
  `>=1.0.0rc1` predates the dev-version wheel), plotly 6.9.0, scikit-learn
  1.9.0. **Matplotlib absent.**
- Callable identities of `FactualExplanation.plot`,
  `AlternativeExplanation.plot`, `plotting.plot_global` recorded before
  import, after import, after registration, and after rendering **all six**
  bridge-covered styles — identical throughout; `_ce_compat` never appeared
  in `sys.modules`; no bridge marker attributes present.
- Styles exercised through public CE APIs with non-default options
  (**19/19 checks passed**; raw log: scratchpad
  `wheelproof/plotly_proof_out.txt`):
  - `plotly.local.factual_bars` — `filter_top=3, uncertainty=True,
    rnk_metric="ensured", rnk_weight=0.25`; artifact `options_used` echoed
    all four; `num_items == 3`; plotly figure produced.
  - `plotly.local.factual_simple` — `uncertainty=True` honored (consumed
    directly via CE's preserved `uncertainty` key, no Plotly-specific alias
    in CE).
  - `plotly.local.alternative_bars` — `filter_top=4, rnk_metric="ensured",
    rnk_weight=0.5` echoed; `num_alternatives == 4`.
  - `plotly.local.alternative_feature_summary` — ran, correct artifact type.
  - `plotly.global.instance_explorer` — `include_instance_records=True`
    consumed (8/8 records), `aggregate_positions=True` forwarded; figure
    produced.
  - `plotly.dashboard.instance_workspace` — `available_cards="auto",
    precompute="all", factual_options={"filter_top": 2},
    alternative_options={"filter_top": 3}`; 8 instances precomputed with
    local cards generated through `context.runtime["explainer"]`.
- Static scan of the patched plugin tree (excluding the now-unreferenced
  `_ce_compat.py`): zero assignments to CE plotting methods, no
  `__wrapped__` manipulation, no bridge markers, no import-time patching.
- Trust: styles registered untrusted (`source="entrypoint"`); the proof
  marked them trusted via the public `mark_plot_builder_trusted`/
  `mark_plot_renderer_trusted` helpers (operator action), which is what
  grants the dashboard runtime access.

## Remaining risks

1. **Resolved during the review-correction pass, not a residual risk:**
   configured (non-explicit) third-party styles that successfully resolve
   now go through the *same* `_dispatch_third_party_plot`/
   `_PlotDispatchOutcome` machinery as the explicit path, so a configured
   plugin legitimately returning `None` is also now treated as handled
   (test: `test_should_treat_none_renderer_result_as_handled_when_configured_style_dispatched`).
   An *unresolvable* configured style (unregistered, denied, or no strict
   resolver reachable) still falls through to the pre-existing built-in/
   chain-based path unchanged — that fallback governance was deliberately
   not broadened or narrowed.
2. `make local-checks-pr`, the ADR-023 viz lane, and packaging validation
   have been run and are green on the corrected candidate (`9e1fa1e5`,
   2026-07-20). The full repository-supported CI matrix has **not** run,
   since the branch has not been pushed — this remains open and RC2 must not
   be cut before it is green on the exact pushed candidate commit.
3. Public API snapshot regeneration must go through the governed review, not
   blind acceptance.
4. The Plotly package's `main` branch still ships `_ce_compat.py` and
   declares `>=1.0.0rc1`; the temporary `tmp/no-bridge-proof` branch
   (candidate `3629ca9`) removes the bridge install/import, raises the floor
   to `>=1.0.0rc2`, and fixes the three stale bridge-installed test
   assertions a reviewer identified — but `_ce_compat.py` itself was not
   deleted (confirmed dead code: unreferenced by `__init__.py`, the
   registration path, or any runtime dispatch path) and installation docs
   were not rewritten to make the required operator trust step (marking the
   dashboard builder/renderer trusted) explicit. Both are scoped to the
   separately authorized adoption task, per this task's "temporary
   adaptation, not final integration branch" constraint.
5. `FastExplanation.plot` third-party styles remain undefined (documented
   out of scope).

## Release recommendation

Implementation, contract tests (including the review-driven correction for
configured-style dispatch, `style_override`-only selection, and
renderer-override trust recomputation), installed-wheel proof, and the
external Plotly no-bridge proof are complete and green on the corrected
candidate. Recommend proceeding to the remaining release-preparation gates
(push the branch, full CI matrix, API snapshot review) and then cutting
`v1.0.0rc2`. Do not tag or publish before those gates are green on the exact
candidate commit.
