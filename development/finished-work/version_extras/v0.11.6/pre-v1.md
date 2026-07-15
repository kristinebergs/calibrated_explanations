# Pre-v1 Skeptical Release Audit

Status: complete
Date: 2026-07-08
Scope: adversarial audit of `kristinebergs/calibrated_explanations` before v1.

This file is a live findings log. Findings below are based on repository
evidence and reproducible commands, not plan status claims.

## 1. Executive Summary

Assessment: **Do not release v1 before fixing blockers.**

The ordinary pytest suite is green in this environment:

- `pytest tests/unit -q --maxfail=1 --no-cov` passed.
- `pytest tests -q --maxfail=1 --no-cov` passed.
- The deprecation-closure pytest slice passed 999 selected tests.

That green test result is not enough for v1. Two release-blocking problems were
confirmed from actual command output and built artifacts:

- the ADR-030 release/PR quality gate fails because 9 newly added tests have no
  assertions;
- the built wheel and sdist omit the BSD license text.

I also found several high-risk public-contract issues that should be fixed
before freezing v1 docs/API behavior: stale guarded-parameter guidance, a
silently ignored `to_narrative(format=...)` doc pattern, docs that import
repository test helpers, version/provenance drift, and an empty-calibration
path that leaks `IndexError`.

## 2. Blockers

### B1. ADR-030 anti-pattern gate fails on 9 new assertion-less tests

- Severity: blocker.
- Evidence:
  - `python scripts/local_checks.py --adr030-ratification` exited 1.
  - The failing step is the ADR-030 detector with `--check`; output reports
    `New violations versus baseline: 9`.
  - Generated report:
    `reports/anti-pattern-analysis/test_quality_report.json` has
    `"new_violations": 9`, all with `"pattern": "test without assertion"`.
  - PR/main workflows enforce this same check:
    `.github/workflows/ci-pr.yml:241-247` and
    `.github/workflows/ci-main.yml:105-111` run
    `detect_test_anti_patterns.py --check`.
  - Affected tests:
    - `tests/unit/core/test_calibrated_explainer_more_paths.py:661`
      `test_explain_methods_still_forward_genuinely_unknown_kwargs_to_plugin`
    - `tests/unit/core/test_calibrated_explainer_more_paths.py:697`
      `test_ce_skip_reject_still_works_on_predict_and_predict_proba`
    - `tests/unit/core/test_calibrated_explainer_more_paths.py:705`
      `test_plot_still_works_after_5c_predict_kwarg_scoping`
    - `tests/unit/core/test_parameter_surface_contracts.py:302`
      `test_interval_summary_accepted_on_wrap_predict`
    - `tests/unit/core/test_parameter_surface_contracts.py:378`
      `test_classification_use_legacy`
    - `tests/unit/core/test_parameter_surface_contracts.py:382`
      `test_regression_use_legacy`
    - `tests/unit/core/test_parameter_surface_contracts.py:386`
      `test_regression_thresholded_use_legacy`
    - `tests/unit/core/test_parameter_surface_contracts.py:390`
      `test_regression_low_high_percentiles_use_legacy`
    - `tests/unit/core/test_parameter_surface_contracts.py:394`
      `test_plot_only_kwargs_do_not_reach_prediction_gates`
- Why this matters for v1: the current tree cannot pass the documented
  ADR-030 release/PR quality gate. Several of these tests cover public
  parameter-surface and plotting regressions; execution-only tests are weak
  evidence for a v1 API freeze.
- How to reproduce:
  - `python scripts/local_checks.py --adr030-ratification`
  - or
    `python scripts/anti-pattern-analysis/detect_test_anti_patterns.py --tests-dir tests --check --output reports/anti-pattern-analysis/test_anti_pattern_report.csv --report reports/anti-pattern-analysis/test_quality_report.json --baseline .github/test-quality-baseline.json`
- Recommended fix: add explicit behavioral assertions to each listed test.
  For no-raise checks, assert returned object type/length/state and verify that
  the relevant kwarg reached or did not reach the intended boundary. Do not
  rebaseline without reviewer sign-off because the current baseline is empty.
- Estimated scope: small.

### B2. Built release artifacts omit the BSD license text

- Severity: blocker.
- Evidence:
  - The repository has a root `LICENSE` file with the BSD 3-Clause text.
  - `pyproject.toml:14-15` declares `license = "BSD-3-Clause"` and
    `license-files = []`.
  - `python -m build` succeeded and produced
    `dist/calibrated_explanations-0.11.6.dev0-py3-none-any.whl` and
    `dist/calibrated_explanations-0.11.6.dev0.tar.gz`.
  - `python -m zipfile -l dist/calibrated_explanations-0.11.6.dev0-py3-none-any.whl`
    shows package schemas/templates and dist-info metadata, but no `LICENSE`,
    `COPYING`, or `NOTICE` file.
  - Inspecting the wheel `METADATA` shows
    `License-Expression: BSD-3-Clause` and `License-File: None`.
  - Inspecting the sdist tarball for names containing `LICENSE`, `COPYING`, or
    `NOTICE` printed no entries.
  - `calibrated_explanations.egg-info/SOURCES.txt` includes package schemas and
    `templates/explain_template.yaml`, but no root `LICENSE`.
- Why this matters for v1: a v1 source/binary distribution should carry the
  license notice and conditions. The BSD 3-Clause license text itself requires
  redistributions of source and binary forms to reproduce the copyright notice,
  conditions, and disclaimer. Publishing artifacts with only a license
  expression but no license text is a release-artifact/legal hygiene defect.
- How to reproduce:
  - `python -m build`
  - `python -m zipfile -l dist/calibrated_explanations-0.11.6.dev0-py3-none-any.whl`
  - Run a tarball member listing and search for `LICENSE`.
- Recommended fix: remove the explicit empty `license-files = []`, or replace it
  with `license-files = ["LICENSE"]` if the current setuptools version supports
  that declaration. Rebuild and verify both wheel and sdist contain the license
  file and wheel metadata reports `License-File: LICENSE`.
- Estimated scope: small.

## 3. High-Risk Findings

### H1. CE-first guide teaches `to_narrative(format=...)`, which is silently ignored

- Severity: high.
- Evidence:
  - `docs/get-started/ce_first_agent_guide.md` documents
    `.to_narrative(format=...)` in the checklist and examples.
  - `AGENTS.md` line 129 also uses
    `print(explanations[0].to_narrative(format="short"))`.
  - The actual single-explanation signature is
    `to_narrative(..., output_format="dataframe", ..., **kwargs)` in
    `src/calibrated_explanations/explanations/explanation.py:570-574`.
  - The collection signature is the same in
    `src/calibrated_explanations/explanations/explanations.py:1512-1519`.
- Reproduction:
  - Ran a breast-cancer quickstart with `WrapCalibratedExplainer`.
  - `e[0].to_narrative(output_format="text", expertise_level="advanced")`
    returned `str`.
  - `e[0].to_narrative(format="short")` returned a pandas `DataFrame`
    using the default `output_format="dataframe"`; the `format` kwarg was
    silently forwarded/ignored.
- Risk: v1 users and agents following the CE-first guide will get the wrong
  output type without an error. This is worse than a hard failure because it
  creates silent API confusion at the recommended agent entry point.
- Recommended fix: either support `format=` as an explicit deprecated/alias
  parameter with a warning and documented mapping, or update all guides to use
  `output_format=` plus supported `expertise_level=` values. Add a regression
  test that `format=` does not silently change nothing.
- Estimated scope: small.

### H2. Runtime `__version__` and installed package metadata disagree

- Severity: high.
- Evidence:
  - `pyproject.toml:10` declares `version = "0.11.6-dev"`, which the build
    backend normalizes to `0.11.6.dev0`.
  - `src/calibrated_explanations/__init__.py:10` declares
    `__version__ = "v0.11.6-dev"`.
  - Reproduction command:
    `python -c "import importlib.metadata as md; import calibrated_explanations as ce; print(ce.__version__); print(md.version('calibrated_explanations'))"`
    printed `v0.11.6-dev` and `0.11.6.dev0`.
  - `src/calibrated_explanations/plugins/registry.py:29` and
    `src/calibrated_explanations/plugins/builtins.py:25` import
    `__version__ as package_version`; plugin descriptors then emit that value
    as metadata, e.g. `src/calibrated_explanations/plugins/builtins.py:235`.
  - Docs instruct users to retrieve audit/version evidence through
    `calibrated_explanations.__version__`, e.g.
    `docs/practitioner/playbooks/eu-ai-act-compliance.md:235-236`.
- Risk: release artifacts, provenance payloads, plugin metadata, and user audit
  logs can disagree about the installed version. This can break exact-version
  comparisons and weakens v1 traceability.
- Recommended fix: choose one PEP 440-compatible runtime version source. Prefer
  deriving `__version__` from package metadata or make the checked-in value match
  the normalized metadata at release time. Add a test comparing
  `calibrated_explanations.__version__` to `importlib.metadata.version(...)`
  after normalizing intentionally allowed tag prefixes, if any.
- Estimated scope: small.

### H3. Guarded-parameter docs still describe removed kwargs as deprecated/current

- Severity: high.
- Evidence:
  - Runtime guardrails reject the old kwargs. In
    `src/calibrated_explanations/api/params.py:32-36`, the removed guarded
    kwargs map to `GuardedOptions` replacements; line 102 raises
    `ConfigurationError` with "Guarded keyword arguments were removed in
    v0.11.5".
  - Reproduction command: a fitted/calibrated `WrapCalibratedExplainer` was
    called with each of `guarded=True`, `significance=0.1`,
    `n_neighbors=3`, `normalize_guard=True`, and `merge_adjacent=True`.
    Each call raised `ConfigurationError` with the v0.11.5 removal message.
  - Source docs and agent-facing instructions still say deprecated/current:
    - `CONTRIBUTOR_INSTRUCTIONS.md:35-37` says `guarded=True` emits
      `DeprecationWarning` and is removed in v1.0.0.
    - `docs/get-started/quickstart_guarded.md:37` says
      `guarded=True` is deprecated and removed v1.0.0.
    - `docs/foundations/concepts/parameter-reference.md:155` says
      `significance` applies to `explain_factual(guarded=True)` and
      `explore_alternatives(guarded=True)`.
    - `.github/skills/ce-factual-explain/SKILL.md:79`,
      `.github/skills/ce-alternatives-explore/SKILL.md:77`,
      `.github/skills/ce-pipeline-builder/SKILL.md:49`, and
      `.github/skills/ce-adr-consult/SKILL.md:120` repeat the
      removed-in-v1.0.0 framing.
    - `development/adrs/ADR-026-explanation-plugin-semantics.md:203-223`,
      `development/adrs/ADR-032-guarded-explanation-semantics.md:35-36`,
      and `development/adrs/ADR-038-call-time-configuration-taxonomy.md:161`
      still describe `guarded=True` as deprecated/current rather than removed.
- Risk: public docs, contributor rules, and agent skills steer users and
  maintainers toward calls that now fail. This is especially risky before v1
  because it mixes three lifecycle states for the same API: current, deprecated,
  and removed.
- Recommended fix: update every source doc/skill/ADR reference to the
  post-v0.11.5 state. Treat `guarded=True`, `significance=`, `n_neighbors=`,
  `normalize_guard=`, and `merge_adjacent=` as removed, and point all usage to
  `guarded_options=GuardedOptions(...)`. Add a docs grep check for
  "guarded=True.*deprecated" and "removed v1.0.0" outside historical files.
- Estimated scope: medium.

### H4. Public docs import `tests.helpers`, so installed-user examples fail

- Severity: high.
- Evidence:
  - `docs/foundations/how-to/tune_runtime_performance.md:33` imports
    `from tests.helpers.doc_utils import run_quickstart_classification`.
  - `docs/contributor/plugin-contract.md:515` and `:557` import
    `from tests.helpers.model_utils import get_classification_model` in
    copy-pasteable plugin wiring examples.
  - `docs/contributor/extending/plugin-advanced-contract.md:144`, `:177`, and
    `:295` also import `tests.helpers.model_utils`.
  - In the repository checkout, those imports resolve because `tests/` exists.
  - In a wheel-only temp install (`pip install --target <tmp>
    dist/calibrated_explanations-0.11.6.dev0-py3-none-any.whl`, run from the
    temp target), `importlib.util.find_spec("tests.helpers.doc_utils")` failed
    with `ModuleNotFoundError: No module named 'tests'`.
- Risk: public examples depend on repository test helpers that are not part of
  the installed package. This undermines docs trust and hides a test-helper
  coupling that normal repo-based docs checks may miss.
- Recommended fix: replace `tests.helpers.*` imports in docs with inline
  sklearn model construction or a documented public helper. Add a docs grep
  check that fails on `from tests` / `import tests` in source docs, except in
  explicitly marked test-authoring documentation.
- Estimated scope: small/medium.

### H5. Empty calibration data leaks raw `IndexError` instead of a CE error

- Severity: high.
- Evidence:
  - `WrapCalibratedExplainer.calibrate(...)` validates calibration arrays via
    `validate_inputs_matrix(..., require_y=True, allow_nan=False)` at
    `src/calibrated_explanations/core/wrap_explainer.py:464`.
  - `validate_inputs_matrix` records `n_samples = x_arr.shape[0]` at
    `src/calibrated_explanations/core/validation.py:226` but does not reject
    `n_samples == 0`.
  - `CalibratedExplainer.num_features` then reads `self._X_cal[0]` /
    `self._X_cal[0, :]` at
    `src/calibrated_explanations/core/calibrated_explainer.py:1411-1416`.
  - Reproduction:
    `WrapCalibratedExplainer(model).fit(X_train, y_train).calibrate(X_empty, y_empty)`
    raised `IndexError: index 0 is out of bounds for axis 0 with size 0`.
  - Neighboring checks are more disciplined: `calibrate(..., X_with_nan, ...)`
    raises `ValidationError: Argument 'x' contains NaN or infinite values.`
- Risk: a public v1 API leaks an internal NumPy indexing error for invalid input
  instead of a stable CE exception (`DataShapeError`/`ValidationError`) with an
  actionable message. Empty calibration data is a likely user mistake and should
  fail at the boundary.
- Recommended fix: update `validate_inputs_matrix` or the calibrate boundary to
  reject zero calibration samples when `require_y=True`. Add tests for wrapper
  and direct `CalibratedExplainer` empty-calibration paths asserting the CE error
  type and message.
- Estimated scope: small.

## 4. Medium/Low Findings

### M1. Notebooks contain stale captured warnings from an older kwarg policy

- Severity: medium.
- Evidence:
  - Several notebooks display old warning output:
    - `notebooks/advanced/demo_under_the_hood.ipynb:207`
    - `notebooks/advanced/demo_narrative_explanations.ipynb:121`
    - `notebooks/miscellaneous/demo_speeddating.ipynb:458`
    - `notebooks/advanced/demo_plugin_wiring.ipynb:156`, `277`, `460`,
      `626`, `739`, `836`
    - `notebooks/quickstart.ipynb:767`, `812`
    - `notebooks/quickstart_guarded.ipynb:1091`, `1136`
    - `notebooks/core_demos/demo_binary_classification.ipynb:1360`
    - `notebooks/core_demos/demo_multiclass_glass.ipynb:1484`
  - The captured warning says
    `WrapCalibratedExplainer received unknown keyword arguments: [...] These
    will be forwarded for compatibility but may be ignored.`
  - Reproduction check against the current code: `calibrate(...,
    class_labels=...)`, `calibrate(..., categorical_labels=...)`,
    `calibrate(..., plot_style=...)`, and `calibrate(..., factual_plugin=...)`
    all executed without warnings.
- Risk: examples visibly contradict the current fail-fast/allow-list policy and
  make legitimate kwargs look suspicious. This weakens notebook credibility even
  though the cells appear to be executable now.
- Recommended fix: re-execute or clear outputs for public notebooks before v1,
  then add a notebook-output hygiene check that fails on stale "unknown keyword
  arguments ... forwarded for compatibility" warning text.
- Estimated scope: small/medium depending on notebook execution cost.

### M2. Contributor guidance points to a missing legacy API contract path

- Severity: medium.
- Evidence:
  - `.github/CONTRIBUTING.md:67` says changes must preserve the contract in
    `development/current-work/legacy_user_api_contract.md`.
  - `Test-Path development/current-work/legacy_user_api_contract.md` returned
    `False`.
  - `Test-Path development/finished-work/legacy_user_api_contract.md` returned
    `True`.
  - The same file already uses the existing path correctly at
    `.github/CONTRIBUTING.md:20`.
- Risk: contributors following the public GitHub contribution guide hit a dead
  reference during public API review. This is small but directly affects the v1
  compatibility workflow.
- Recommended fix: update `.github/CONTRIBUTING.md:67` to
  `development/finished-work/legacy_user_api_contract.md`, or restore a
  deliberate current-work pointer if that is the intended canonical path.
- Estimated scope: small.

### M3. Narrative template docs point to `.json`, but the packaged default is `.yaml`

- Severity: medium.
- Evidence:
  - The implementation default is
    `src/calibrated_explanations/templates/explain_template.yaml` in
    `src/calibrated_explanations/viz/narrative_plugin.py:72-85`.
  - `Get-ChildItem src/calibrated_explanations/templates` shows only
    `explain_template.yaml`.
  - The wheel contains `calibrated_explanations/templates/explain_template.yaml`.
  - `docs/practitioner/narrative_templates.md:138` instructs users to copy
    `src/calibrated_explanations/templates/explain_template.json`.
  - `docs/foundations/how-to/to_narrative.md:81` says missing `exp.yaml`
    falls back to `explain_template.json`; line 105 tells users to pass
    `template_path="explain_template.json"` explicitly.
  - The narrative plugin docstring says `FileNotFoundError` is raised when a
    template is missing, but `src/calibrated_explanations/viz/narrative_plugin.py:139-160`
    falls back to the default template and emits `UserWarning`.
- Risk: users following docs look for or pass a template filename that does not
  exist in the package. The warning/fallback behavior also disagrees with the
  API docstring.
- Recommended fix: change docs to `.yaml`, update or remove the stale
  `FileNotFoundError` contract, and add a docs/source grep test for
  `explain_template.json`.
- Estimated scope: small.

### M4. Two incompatible PlotSpec schemas exist; the one under `src/` is not packaged

- Severity: medium.
- Evidence:
  - `src/calibrated_explanations/schemas/v1/plotspec_schema.json` exists under
    the production source tree and declares required fields
    `["plotspec_version", "plot_spec"]`.
  - `development/schemas/plotspec_schema.json` declares a different required
    shape: `["kind", "mode", "header", "body", "style", "uncertainty",
    "feature_order"]`.
  - Tests validate `development/schemas/plotspec_schema.json`, e.g.
    `tests/unit/viz/test_plotspec_schema_and_primitives.py:16-23`.
  - `pyproject.toml:185-190` includes package data for `schemas/*.json` but not
    nested `schemas/v1/*.json`; wheel listing confirms the nested source schema
    is omitted.
- Risk: maintainers and downstream users see an apparently production-owned
  schema under `src/` that disagrees with the tested schema and is absent from
  artifacts. This looks like a stale contract copy or an accidentally unshipped
  public schema.
- Recommended fix: decide which schema is canonical. Either delete the stale
  `src/.../schemas/v1` copy, or package it intentionally and add tests that
  verify the packaged schema matches the runtime PlotSpec payloads.
- Estimated scope: small/medium.

## 5. Suspicious But Unconfirmed

- Skill registry drift: `.codex/skills/ce-test-audit/SKILL.md` references
  `references/adr-030-test-quality.md`, but that file is absent under
  `.codex/skills/ce-test-audit/`. The same reference exists under `.agents/`
  and `.claude/`. Needs confirmation whether `.codex` is expected to be
  complete or whether `.agents` is the canonical local skill mirror.
- Built wheel omits `calibrated_explanations/schemas/v1/plotspec_schema.json`
  and `calibrated_explanations/utils/configurations/plot_config.ini`. Current
  runtime searches did not find production use of the source PlotSpec schema,
  and `plotting.update_plot_config()` creates the missing config directory on
  demand, so this is not confirmed as a defect. Confirm whether
  `src/calibrated_explanations/schemas/v1/plotspec_schema.json` is intended to
  be shipped as public package data or is a stale duplicate of
  `development/schemas/plotspec_schema.json`.
- Single-class classification calibration silently succeeds. In a smoke test,
  calibrating a two-class `RandomForestClassifier` on 8 calibration rows from
  only class `1` produced two-column probabilities but
  `explainer.class_labels == {1: "1"}`. `explain_factual` and
  `explore_alternatives` still ran, so this is not confirmed as a defect.
  Confirm whether single-class calibration should be rejected, warned, or
  explicitly documented; add a test that class labels/probability columns remain
  coherent if it is supported.

## 6. Test-Suite Blind Spots

1. ADR-030 can fail while pytest is green.
   - Current evidence: full `pytest tests` passed, but
     `scripts/local_checks.py --adr030-ratification` failed on assertion-less
     tests.
   - Add/strengthen tests: convert every no-assertion public-contract test into
     a behavioral assertion; keep the zero-new-violation detector blocking.

2. Built artifacts are not tested for license/package-data correctness.
   - Current evidence: `python -m build` succeeds, but wheel/sdist omit
     `LICENSE`.
   - Add test: a packaging smoke script that builds wheel/sdist and asserts
     `LICENSE`, required schemas, `py.typed`, and default templates are present.

3. Installed-user docs snippets are not isolated from the repository checkout.
   - Current evidence: docs import `tests.helpers.*`; a wheel-only temp install
     fails with `ModuleNotFoundError: No module named 'tests'`.
   - Add test: docs grep/snippet smoke run from a temp cwd with only the wheel
     on `PYTHONPATH`; fail on `from tests` in source docs.

4. Public invalid-input errors are not consistently asserted at the API boundary.
   - Current evidence: empty calibration data raises raw `IndexError`.
   - Add test: `WrapCalibratedExplainer.calibrate(empty_x, empty_y)` and direct
     `CalibratedExplainer(...)` should raise `DataShapeError` or
     `ValidationError` with a stable message.

5. Documentation drift around removed/deprecated parameters is not gated.
   - Current evidence: source docs still say `guarded=True` is deprecated even
     though runtime rejects it as removed.
   - Add test: grep source docs/skills for removed call forms outside
     historical migration tables.

6. Narrative template behavior lacks source-doc consistency tests.
   - Current evidence: docs name `explain_template.json`, implementation and
     artifact ship `explain_template.yaml`; docstring says `FileNotFoundError`
     while implementation falls back with `UserWarning`.
   - Add test: docs grep for stale template filename and a unit test asserting
     documented missing-template behavior.

7. PlotSpec schema authority is not enforced.
   - Current evidence: incompatible schemas exist under `src/` and
     `development/`; tests validate only the development schema.
   - Add test: one canonical schema location, plus a packaging/runtime test
     proving generated PlotSpecs validate against it.

8. Single-class calibration semantics need a policy test.
   - Current evidence: single-class calibration succeeds but leaves
     `class_labels` missing one model class while probabilities have two
     columns.
   - Add test: either reject single-class calibration with a CE error, or assert
     all probability columns and class labels remain coherent.

## 7. Documentation/API Mismatch Table

| Documented behavior | Actual behavior | Evidence | Risk | Fix |
| ------------------- | --------------- | -------- | ---- | --- |
| `to_narrative(format="short")` is shown in CE-first guidance. | `format` is accepted via `**kwargs` but ignored; default `output_format="dataframe"` is used. | `AGENTS.md:129`; `docs/get-started/ce_first_agent_guide.md`; `explanation.py:570-574`; repro returned DataFrame. | Silent wrong output type for recommended API. | Replace docs with `output_format=` or implement an explicit alias/warning. |
| `guarded=True` is deprecated and removed in v1.0.0. | Runtime raises `ConfigurationError` because it was removed in v0.11.5. | `CONTRIBUTOR_INSTRUCTIONS.md:35-37`; `docs/get-started/quickstart_guarded.md:37`; `api/params.py:32-36,102`; repro with `guarded=True`. | Users follow docs into a hard failure; lifecycle state is contradictory. | Update docs/skills/ADRs to removed-state wording and `GuardedOptions`. |
| Docs examples can be copied by installed users. | Several docs import `tests.helpers.*`, which exists only in the repo checkout. | `docs/foundations/how-to/tune_runtime_performance.md:33`; `docs/contributor/plugin-contract.md:515`; wheel-only smoke failed. | Installed-user examples fail. | Inline public sklearn setup or provide a public helper. |
| Default narrative template is `explain_template.json`. | Only `explain_template.yaml` exists and is packaged. | `docs/practitioner/narrative_templates.md:138`; `docs/foundations/how-to/to_narrative.md:81,105`; `narrative_plugin.py:72-85`. | Users look for/pass a non-existent template file. | Rename docs to `.yaml`; remove stale JSON references. |
| Missing narrative templates raise `FileNotFoundError`. | Implementation falls back to the default template and emits `UserWarning`. | `narrative_plugin.py` docstring around `Raises`; implementation at `narrative_plugin.py:139-160`. | Error-handling contract is ambiguous. | Align docstring/docs with chosen fallback behavior. |
| `calibrated_explanations.__version__` is suitable for audit/provenance logs. | Runtime version is `v0.11.6-dev`; package metadata version is `0.11.6.dev0`. | `__init__.py:10`; `pyproject.toml:10`; reproduction with `importlib.metadata.version`. | Audit logs/plugin metadata can disagree with installed artifact. | Derive runtime version from metadata or normalize consistently. |
| Contributor guide points to the legacy API contract. | One contributor-guide path is missing. | `.github/CONTRIBUTING.md:67`; `Test-Path development/current-work/legacy_user_api_contract.md` false; finished-work path true. | Public API review workflow points at a dead file. | Update to `development/finished-work/legacy_user_api_contract.md`. |

## 8. Recommended Pre-v1 Action Plan

### Must fix before v1

1. Fix B1: make the ADR-030 anti-pattern gate green without rebaselining away
   the 9 assertion-less tests.
2. Fix B2: include `LICENSE` in both wheel and sdist, then rebuild and verify
   `License-File`.

### Should fix before v1

1. Fix H5 empty-calibration error taxonomy.
2. Fix H1/H3/H4 documentation/API mismatches that direct users to wrong or
   repo-only APIs.
3. Fix H2 version/provenance drift before publishing release artifacts.
4. Re-execute or clear notebooks with stale unknown-kwarg warnings.
5. Resolve narrative template `.json` vs `.yaml` docs and missing-template
   behavior.
6. Decide the canonical PlotSpec schema and remove/package/test the other copy.

### Can defer after v1 if explicitly recorded

1. Skill registry drift under `.codex/skills` if those files are not part of
   the release contract.
2. Single-class calibration policy, but only if the current behavior is
   documented as supported/unsupported and a post-v1 issue is opened.
3. Wheel omission of `plot_config.ini` if maintainers confirm it is created on
   demand and not intended as package data.

## Commands Run

- `Get-Content -Raw CONTRIBUTOR_INSTRUCTIONS.md`
- `Get-Content -Raw AGENTS.md`
- `Get-Content -Raw docs/get-started/ce_first_agent_guide.md`
- `Get-Content -Raw development/standards/test-quality-method/README.md`
- `Get-Content -Raw development/standards/test-quality-method/code_quality_auditor.md`
- `Get-Content -Raw development/standards/test-quality-method/anti_pattern_auditor.md`
- `Get-Content -Raw tests/README.md`
- `Get-Content -Raw development/README.md`
- `Get-Content -Raw development/current-work/RELEASE_PLAN_v1.md`
- `Get-Content -Raw .agents/skills/ce-test-audit/references/adr-030-test-quality.md`
- `Get-Content -Raw development/adrs/ADR-030-test-quality-priorities-and-enforcement.md`
- `git status --short`
- `python -m build`
- `python -m pip install -e .`
- `python -m pip check`
- `python -c "import importlib.metadata as md; import calibrated_explanations as ce; print(ce.__version__); print(md.version('calibrated_explanations'))"`
- `python -m zipfile -l dist/calibrated_explanations-0.11.6.dev0-py3-none-any.whl`
- wheel/sdist inspection scripts for license/package data
- wheel-only temp install smoke for package resources and docs-helper imports
- `python scripts/quality/check_adr002_compliance.py`
- `python scripts/quality/check_import_graph.py`
- `python scripts/quality/check_docstring_coverage.py`
- `python scripts/quality/check_no_test_helper_exports.py`
- `python scripts/anti-pattern-analysis/detect_test_anti_patterns.py`
- `python scripts/anti-pattern-analysis/scan_private_usage.py --check`
- `python scripts/quality/check_marker_hygiene.py --check`
- `python scripts/anti-pattern-analysis/analyze_private_methods.py src tests --output reports/anti-pattern-analysis/private_method_analysis.csv`
- `python scripts/local_checks.py --adr030-ratification`
- `python scripts/local_checks.py --deprecation-closure`
- `pytest tests/unit -q --maxfail=1 --no-cov`
- `pytest tests -q --maxfail=1 --no-cov`
- targeted public API reproductions for `to_narrative(format=...)`, removed
  guarded kwargs, notebook kwargs, empty calibration data, and single-class
  calibration data
- targeted `rg` searches for guarded docs, notebook stale warnings, test-helper
  imports in docs, narrative template names, version strings, package data,
  PlotSpec schema copies, environment reads, private/test-helper imports, and
  plotting dependency imports
