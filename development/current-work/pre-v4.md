# Pre-vX skeptical release audit

Audit target: `kristinebergs/calibrated_explanations`
Branch/commit: `main` / `60d1918a03730b97541290d73d12141a9f299274`
Started: `2026-07-10 20:14:48 +02:00`
Auditor: `Codex / GPT-5 (CE code-quality Option C + release-check guidance)`

## Current expected state from repo evidence

- Package version: `0.11.6-dev` (`pyproject.toml`; normalized artifact form `0.11.6.dev0`).
- Current released version according to release docs: `v0.11.5` (`RELEASE_PLAN_v1.md` and `CHANGELOG.md`).
- Active milestone: `v0.11.6` pre-RC hardening (`v0.11.6_plan.md`). The stale master-plan control snapshot still says `v0.11.5`; existing v0.11.6 Task 28 owns release-plan/bookkeeping reconciliation.
- Next milestone: `v1.0.0-rc`, a validation/freeze milestone; planned implementation is prohibited there.
- Release blockers: the v0.11.6 gate summary still leaves Tasks 13, 20, 28, 35, and 45 unfinished; its opening sentence also incorrectly lists completed Tasks 26 and 27 as pending. This bookkeeping drift is already covered by Task 28. In addition, this audit has verified `S4-B1`, which is not covered by the current plans.
- Zero-deprecation expectation: the Active deprecations ledger must be empty and `make deprecation-closure` must fail on any active row or runtime deprecation emitter before RC.
- Kwarg policy: stable closed public surfaces fail fast with `ConfigurationError` on removed/unknown names; direct-core explanation methods retain an ADR-038 experimental plugin-forwarding seam for genuinely plugin-defined kwargs and log forwarded names at INFO. `multi_labels_enabled` and `interval_summary` remain experimental `**kwargs` on explanation methods pending the RC graduation gate.
- Public API freeze posture: v0.11.6 may harden behavior; `v1.0.0-rc` is freeze/validation only, with code changes limited to emergency release-blocking patches.
- Required gates: task-specific mappings for task closure; `make local-checks-pr` for milestone closure; strict docs/linkcheck and release profile; `make deprecation-closure`; `make capability-chain-check`; release-boundary capability evidence refresh; artifact build/inspection; `make release-preflight` before tagging and `make release-finalize` before manual publication.
- Canonical development locations: `development/current-work/`, `development/adrs/`, `development/standards/`, and `development/capabilities/` (per `CONTRIBUTOR_INSTRUCTIONS.md`).
- Commands attempted: repository baseline (`git rev-parse HEAD`, `git branch --show-current`, `git status --short`) and required-file presence checks.

### Startup file-presence record — 2026-07-10 20:14 +02:00

All required startup files exist at commit `60d1918a03730b97541290d73d12141a9f299274`:

- `CONTRIBUTOR_INSTRUCTIONS.md`
- `development/README.md`
- `development/current-work/RELEASE_PLAN_v1.md`
- `development/current-work/v0.11.6_plan.md`
- `development/current-work/bug-list.md`
- `development/current-work/pre-v1.md`
- `development/current-work/pre-v2-gaps.md`
- `development/current-work/RELEASE_PLAN_status_appendix.md`
- `CHANGELOG.md`
- `pyproject.toml`
- `README.md`
- `docs/migration/deprecations.md`
- `tests/README.md`
- `Makefile`
- `src/calibrated_explanations/api/params.py`
- `src/calibrated_explanations/core/wrap_explainer.py`
- `src/calibrated_explanations/core/calibrated_explainer.py`
- `tests/unit/core/test_parameter_surface_contracts.py`

`development/current-work/pre-vX.md` did not exist before this audit and was created as required. The worktree was clean before creation.

Immediate inconsistency pending classification: `CONTRIBUTOR_INSTRUCTIONS.md` declares `.claude/skills/` canonical and requires shadow skills to be redirect-only shims, but `.claude/skills/ce-doc-navigator/SKILL.md` is absent while `.codex/skills/ce-doc-navigator/SKILL.md` contains a full independent skill definition. This will be verified against the tracked skill inventory before being promoted to a finding.

## Live findings index

| ID | Severity | Status | Area | Short title | Covered by existing plan? |
|---|---|---|---|---|---|
| S4-B1 | BLOCKER | verified | Preprocessing / silent correctness | Transform failure silently bypasses configured preprocessing and changes predictions | No |
| S4-B2 | BLOCKER | verified | Release gates / fallback policy | Warning-policy gate passes while mandatory fallback signals are absent | No; Task 25 verifies filter precedence, not fallback-site completeness |
| S4-B3 | BLOCKER | verified | Lifecycle / failed-call atomicity | Rejected recalibration disables a working calibrator and replaces stored Mondrian state | No |
| S4-B4 | BLOCKER | verified | Explanation semantics / payload | `interval_summary` changes prediction API output but not displayed explanation probability | No; contradicts the v0.11.0 changelog claim and current acceptance-only tests |
| S4-H1 | HIGH | needs maintainer decision | Public API / test quality | Test-only aliases became mutable production API | No; introduced by historical v0.10.2 test remediation |
| S4-H2 | HIGH | verified | Validation / parameter contracts | Valid parameter names accept invalid values, silently coerce behavior, or leak deep errors | No |
| S4-H3 | HIGH | needs maintainer decision | Plugin extension seam | Direct-core misspelled explanation kwargs produce plausible output and disappear | Partly; completed v0.11.6 Task 27 deliberately retained INFO-only forwarding |
| S4-H4 | HIGH | verified | Documentation / executable examples | Guarded and FAST migration examples call removed or unavailable surfaces | No; contradicts completed Tasks 14, 26, and 37 |
| S4-H5 | HIGH | verified | Snapshot semantics / shared state | Frozen explanation snapshots expose and share mutable runtime state with the live explainer | No |
| S4-H6 | HIGH | verified | Plot contract / path divergence | Legacy global plotting bypasses prediction validation and plot typos disappear | Partly; Task 44 closed prediction surfaces but did not cover plotting |
| S4-M1 | MEDIUM | verified | Release metadata / documentation | README and compliance playbook understate the Python floor | Partly; Task 26 changed package metadata but missed these pages |
| S4-M2 | MEDIUM | candidate | Semantic test coverage | FAST tuning parameters are source-consumed but lack public differential evidence | No |
| S4-I1 | INFO | needs maintainer decision | Import/global state | Root import installs a process-wide MappingProxyType pickle reducer | No |

## Findings

## Second-pass adversarial gap audit

Pass label: `S4-startup` — repository state and canonical-source baseline.

The second pass is being appended incrementally. Finding IDs use `S4-B#`, `S4-H#`, `S4-M#`, `S4-L#`, and `S4-I#`.

### Paired direct-core versus wrapper surface matrix — `S4-surface`

| Case | Wrapper | Direct `CalibratedExplainer` | Disposition |
|---|---|---|---|
| Valid `interval_summary='regularized_mean'` on factual explain | accepted, 2 explanations | accepted, 2 explanations | identical / cleared |
| Typed `guarded_options=GuardedOptions()` | accepted (guard warnings were data-dependent) | accepted with explicit Mondrian bins | identical / cleared |
| Invalid `reject_policy=False` | `ValidationError` with details | same | identical / cleared control |
| Conditional prediction after wrapper calibration with `mc` | wrapper derives bins | direct core requires the same explicit bins | intentionally different lifecycle responsibility; paired outputs accepted |
| Removed `guarded=True` | `ConfigurationError` with replacement/details | same | identical; stale docs are `S4-H4` |
| Closed-surface `fast=True` on factual explain | `ConfigurationError` | same | identical / cleared |
| Genuinely unknown explain kwarg | `ConfigurationError` | INFO-log then forward; built-in ignores it | ADR-038 exception, but unconsumed-key defect `S4-H3` |
| Plot-only `show=` sent to `predict` | rejected as unknown | stripped before core validation and accepted when required bins are present | undocumented internal escape hatch; contributes to `S4-H6` |
| Classification `threshold=` on `predict` / `predict_proba` | `ValidationError` | `ValidationError` | identical / cleared at prediction surfaces; legacy plot bypass is `S4-H6` |

The matrix was executed with deterministic classification data and `random_state=42`; conditional direct calls were supplied the exact categories produced by the wrapper's stored categorizer.

### Semantic-effect classification for prioritized accepted parameters — `S4-semantic`

| Parameter | Classification from current code/evidence | Second-pass disposition |
|---|---|---|
| `interval_summary` | consumed by prediction; accepted by explanation, but only a hidden full-probability field changes | verified correctness defect `S4-B4` |
| `multi_labels_enabled` | behavior-changing on multiclass explanation orchestration; intentionally inert-with-warning on binary | consumed; non-bool truthiness coercion is `S4-H2` |
| `reject_policy`, `reject_confidence` | behavior-changing when a policy resolves active; confidence intentionally inactive without a policy | consumed conditionally; invalid-value boundary covered by `S4-H2` |
| `features_to_ignore` | consumed by discretizer/feature task and changes emitted rules | behavior-changing; out-of-range/weak typing failure is `S4-H2` |
| `bins` | selects Mondrian calibration/inference channel and is mandatory after bins calibration | behavior-changing; paired direct/wrapper control passed |
| `threshold` | selects probabilistic-regression event semantics | behavior-changing; classification legacy-plot bypass is `S4-H6` |
| `low_high_percentiles` | selects regression interval endpoints | behavior-changing; malformed sequence leaks low-level error (`S4-H2`) |
| `normalization` | consumed in multiclass Venn-Abers probability normalization | behavior-changing; current coercer rejects invalid strings/bools |
| `perf_cache`, `perf_parallel` | forwarded to prediction/explanation execution primitives | forwarded/consumed; setup failure is silently disabled (`S4-B2`) |
| `plot_style` | stored in `PluginManager` and changes plot resolution chain | behavior-changing; renderer-path contract gap is `S4-H6` |
| `condition_source` | chooses observed vs predicted labels for discretizer fitting | behavior-changing; exact enum validation present |
| `suppress_crepes_errors` | selects raise versus degraded-zero prediction fallback | behavior-changing; string truthiness can accidentally enable it (`S4-H2`), visibility governed by `S4-B2` |
| `sample_percentiles` | consumed by explanation computation/FAST percentile sampling | source-consumed, but public tests do not prove semantic effect; `S4-M2` |
| `noise_type`, `scale_factor`, `severity` | consumed by FAST interval perturbation setup | source-consumed, but public tests prove acceptance/property storage rather than output effect; `S4-M2`; invalid values in `S4-H2` |

No prioritized parameter was classified as intentionally no-op or compatibility-only by the current docstrings. Conditional inactivity must therefore be explicit (for example reject confidence without a policy), not inferred from a successful non-`None` result.

### S4-B1 — Preprocessor transform failure silently bypasses preprocessing and returns wrong predictions

- Severity: `BLOCKER`
- Status: `verified`
- Files/symbols: `src/calibrated_explanations/core/wrap_explainer.py::_pre_fit_preprocess`, `::_pre_transform`, `::predict`; `tests/unit/core/test_wrap_explainer_helpers.py::test_preprocess_failures_are_swallowed`
- Why the first audit prompt might miss this: the broad audit asks whether preprocessing fails, but the existing test deliberately treats raw-input fallback as success. Shape/type and no-raise checks therefore stay green while the prediction semantics change.
- Evidence: `_pre_transform` catches every ordinary exception from a configured preprocessor, emits only a logger warning, and returns the untransformed input. `_pre_fit_preprocess` has the same fail-open policy. The unit test named `test_preprocess_failures_are_swallowed` asserts that a failing preprocessor returns the original array and leaves the wrapper usable. A public-API differential reproduction fitted and calibrated a `DecisionTreeClassifier(random_state=42)` on values shifted by `+100`; the third transform call failed during inference. `WrapCalibratedExplainer.predict([[1.5]])` returned `[0]`, identical to the learner's raw-input prediction, while the correctly transformed learner prediction was `[1]`. `warnings.catch_warnings(record=True)` captured zero Python warnings.
- Minimal reproduction or inspection method: configure `wrapper.preprocessor` with a deterministic object whose `fit_transform` and first two `transform` calls return `X + 100`, whose third `transform` raises `RuntimeError`, then run `fit -> calibrate -> predict`. Compare `wrapper.predict(query)` with `wrapper.learner.predict(query + 100)` and inspect captured warnings.
- Observed: the wrapper silently substitutes raw features after the configured inference transform fails, produces a plausible but incorrect class, and does not emit the mandatory `UserWarning` + INFO fallback signal.
- Expected: a configured preprocessor failure must fail fast with a CE-owned `ValidationError`/`ConfigurationError`, or an explicitly governed fallback must be visible and must not feed representation-incompatible raw data into a model trained/calibrated on transformed features. Failed calls must leave state intact.
- v1 risk: wrong prediction/probability/explanation results can be returned from the primary public wrapper while the call appears successful. This violates the CE fallback policy and is a direct silent-correctness defect at the v1 API boundary.
- Recommended action: remove the raw-data fail-open path for user preprocessors and the built-in encoder. Convert preprocessing failures to CE exceptions with `details=` (stage, preprocessor type, original error class), retain unseen-category policy handling explicitly, and add public wrapper tests proving failed fit/calibrate/predict/explain calls do not mutate lifecycle or calibration state. If any fallback is intentionally retained, require `UserWarning` + INFO logging and prove representation compatibility before prediction.
- Verification after fix: rerun the deterministic shifted-feature reproduction; assert a CE exception, unchanged wrapper/explainer state, and no learner prediction on raw input. Add paired fit, calibrate, predict, predict_proba, factual, alternative, and serialization-round-trip preprocessing failure tests.

### S4-B2 — Warning-policy gate cannot detect missing fallback warnings and passes non-compliant runtime paths

- Severity: `BLOCKER`
- Status: `verified`
- Files/symbols: `scripts/quality/check_warning_policy.py::_find_warn_sites`, `::classify`; `tests/helpers/fallback_control.py::disable_all_fallbacks`; `src/calibrated_explanations/core/wrap_explainer.py::from_config`; `src/calibrated_explanations/plugins/builtins.py::_ExecutionExplanationPluginBase.explain_batch`; `src/calibrated_explanations/core/calibrated_explainer.py::_enforce_feature_filter_plugin_preferences`, `::reset`; `development/capabilities/claims/CE-CAP-OBS-001.yaml`; `development/capabilities/requirements/CE-REQ-OBS-GOV-001.md`; `development/adrs/ADR-027-fast-feature-filtering.md:Failure and fallback behavior`; `CONTRIBUTOR_INSTRUCTIONS.md §5`
- Why the first audit prompt might miss this: the gate reports “all warning sites classified,” which sounds like fallback closure. It only inventories `warnings.warn` calls already present, however; a fallback with no warning is outside its search universe and is guaranteed not to be reported.
- Evidence: `python scripts/quality/check_warning_policy.py --check` passed with 106 sites, 101 whole-file allowlisted, and zero unclassified. The script never scans `except` handlers, fallback messages, logger-only fallbacks, or return-value substitutions, and its own module doc says degraded fallbacks “SHOULD be routed to WARNING logs, not UserWarning,” contradicting the current repository-wide mandate of `UserWarning` plus INFO. `tests/helpers/fallback_control.py` likewise intercepts only `warnings.warn`, so logger-only fallbacks evade the autouse fallback detector. Concrete evading paths include: (1) a requested perf factory raising in `WrapCalibratedExplainer.from_config`, which silently sets cache/parallel to `None` with one DEBUG log and zero warnings; (2) execution-plugin unsupported/error paths that switch to legacy or even an empty collection using logger WARNING only; (3) feature-filter enforcement failures using logger WARNING only; and (4) reset/deepcopy suppressions. ADR-027 explicitly authorizes debug-only fail-open behavior unless future strict observability is enabled, conflicting with `CONTRIBUTOR_INSTRUCTIONS.md §5` and the v0.11.6 global rules. The current capability claim `CE-CAP-OBS-001` nevertheless states that every fallback emits `UserWarning` and INFO and marks this as `automated_executable_evidence`; its sole requirement samples governance-event cases and exempts itself from TIF as `repository_policy`. `make capability-chain-check` warned about that runtime-behavior exemption but still passed, so the capability chain also overclaims this behavior.
- Minimal reproduction or inspection method: pass `from_config` a valid config whose `_perf_factory.make_cache()` raises; capture warnings and DEBUG logs. Result: `perf_cache=None`, `parallel_executor=None`, zero warnings, DEBUG `Failed to initialize perf primitives...`. Then run `python scripts/quality/check_warning_policy.py --check`; it still passes.
- Observed: release validation can be green while known fallback paths are silent or only log. The gate classifies presence, not completeness or paired warning+log semantics; broad path allowlisting additionally treats every warning in a file as reviewed.
- Expected: the release gate must fail when a runtime fallback lacks the governed user-visible signal and must validate the warning/log pair, exception scope, changelog entry, and fallback test opt-in. Governing documents must agree on whether user warnings are mandatory.
- v1 risk: this is a release-gate false-confidence defect. Silent degraded execution, configuration loss, and empty-result substitution can ship while the named fallback-visibility gate reports full compliance.
- Recommended action: replace the presence-only inventory with an AST/manifest-backed fallback-site registry (or explicit governed helper) that couples each fallback to warning, INFO log, test, and policy disposition. Remove whole-file allowlisting. Reconcile ADR-027, ADR-028/STD-005 prose, the master plan's “log-first” wording, and current contributor/v0.11.6 rules before declaring closure. Fail fast where correctness preservation cannot be proven.
- Verification after fix: inject failures into perf setup, plugin supports/execute, feature-filter enforcement, parallel execution, plotting, cache, serialization migration, reset, and preprocessing. The gate must fail for each missing signal; fallback tests must require `enable_fallbacks` and assert both `UserWarning` and INFO, while normal tests fail on any unrequested fallback.

### S4-B3 — A rejected recalibration partially commits state and downgrades later predictions

- Severity: `BLOCKER`
- Status: `verified`
- Files/symbols: `src/calibrated_explanations/core/wrap_explainer.py::calibrate`; `src/calibrated_explanations/core/calibrated_explainer.py::__init__`, `::set_mode`; wrapper `predict` / `predict_proba` uncalibrated branches
- Why the first audit prompt might miss this: the implementation comment says calibration is invalidated only after every validation gate has passed, and ordinary tests check only whether the bad call raises. Constructor-time validation and plugin setup still occur after `self.mc` is replaced and `self.calibrated` is set false.
- Evidence: start from a successfully conditionally calibrated classifier (`calibrated=True`, stored `old_mc`, working explainer). Calling `calibrate(..., mc=new_mc, seed='bad')` raises raw NumPy `TypeError: SeedSequence expects int...`; afterward `calibrated=False`, the old explainer object remains installed, and `mc` is `new_mc`. The same partial commit occurs with `mode='Classification'` (deep `ConfigurationError`) and regression `fast=True, scale_factor='bad'`. A subsequent default `predict_proba` takes the uncalibrated branch and returns raw learner probabilities: they matched `learner.predict_proba`, differed from the pre-failure calibrated values by up to `0.3333`, and emitted only the generic “must be calibrated” warning. A failure rejected before the mutation point (`low_high_percentiles` sent to `calibrate`) correctly preserved all state, confirming the boundary is ordering-dependent.
- Minimal reproduction or inspection method: fit and calibrate a seeded `DecisionTreeClassifier` with `mc=old_mc`; snapshot `calibrated`, `explainer`, `mc`, and calibrated probabilities; call `calibrate` with `mc=new_mc, seed='bad'`; compare state and the next `predict_proba` result with both the snapshot and raw learner output.
- Observed: a failed public call is not atomic. It discards the lifecycle flag and conditional categorizer while retaining an explainer calibrated under the old state; later calls silently choose a different prediction backend after only a generic warning.
- Expected: validation and construction must occur in temporary state. On any exception, `calibrated`, `explainer`, `mc`, preprocessing state, plugin manager, interval learner, caches, and metadata must remain exactly as before the call.
- v1 risk: a caller can catch an expected configuration error and continue with the same object, unaware that calibrated probabilities have been replaced by raw model probabilities. This breaks the public lifecycle and can change decisions/results after a rejected operation.
- Recommended action: make recalibration transactional: normalize and validate every value first, derive bins without mutating the wrapper, construct the candidate explainer completely, then commit `mc`, `explainer`, and `calibrated` together. Normalize all constructor failures to CE exceptions with `details=`. Audit preprocessor fitting and perf/plugin setup for the same rollback requirement.
- Verification after fix: snapshot-and-compare tests for every rejected calibration phase (surface validation, preprocessor fit/transform, bins/mc, target validation, RNG, interval plugin, feature filter, perf setup). After each failure, assert byte/value identity of relevant state and equality of calibrated predictions/probabilities/explanations to the pre-call baseline.

### S4-B4 — Explanation `interval_summary` produces internally contradictory probabilities and narratives

- Severity: `BLOCKER`
- Status: `verified`
- Files/symbols: `src/calibrated_explanations/core/calibrated_explainer.py::explain_factual`, `::explore_alternatives`, `::predict_proba`; `src/calibrated_explanations/core/explain/orchestrator.py`; `src/calibrated_explanations/plugins/builtins.py::_LegacyExplanationBase.explain_batch`; `src/calibrated_explanations/explanations/explanations.py::to_json`; narrative rendering; `tests/unit/core/test_parameter_surface_contracts.py`; `CHANGELOG.md` v0.11.0 Interval Summary Selection entry
- Why the first audit prompt might miss this: current surface tests prove `interval_summary='mean'` is accepted and returns a non-`None` object; prediction integration tests prove the option changes `predict_proba`. No differential test compares the selected prediction point with explanation payload, JSON, rules, plots, or narrative.
- Evidence: on deterministic binary classification data, `wrapper.predict_proba(query, interval_summary='lower')` returned positive probability `0.5`, while `'upper'` returned `0.5714285714`. The paired factual explanations encoded those selected values only inside `prediction['__full_probabilities__']`; in both calls, `prediction['predict']` remained the default regularized mean `0.5333333333`. `to_json()` therefore contained `predict: 0.5333333333` beside full probabilities `[[0.5, 0.5]]` for lower and `[[0.42857, 0.57143]]` for upper. Both beginner narratives said `Calibrated Probability: 0.533`, and rule/feature point predictions were identical. The same no-difference occurred with the explicit legacy factual plugin. The changelog claims `predict_proba`, factual/alternative explanations, and legacy paths all respect the selected strategy.
- Minimal reproduction or inspection method: fit/calibrate `RandomForestClassifier(n_estimators=25, max_depth=4, random_state=42)` on a seeded five-feature classification set; for one query, compare lower vs upper `predict_proba`, `explain_factual()[0].prediction`, `batch.to_json()`, and `to_narrative(output_format='text', expertise_level='beginner')`.
- Observed: the accepted option changes a hidden probability matrix but not the explanation's canonical/displayed point estimate or rule effects. One payload contains mutually inconsistent representations of “the” selected probability.
- Expected: the requested summary must be applied consistently to the canonical point prediction, class probabilities, feature perturbation predictions/weights, JSON/domain payload, reject envelope, plot payload, and narrative—or the parameter must be removed from explanation surfaces if it is intentionally prediction-only.
- v1 risk: users selecting conservative lower or optimistic upper probabilities receive the default value in the primary explanation and narrative. This is wrong probability reporting and can mislead decisions while appearing internally calibrated.
- Recommended action: establish one canonical summary-selected probability vector at the prediction boundary and pass it through explanation construction. Eliminate separate default-summary recomputation in plugin/legacy paths. Clarify whether feature effects must use the same summary and update the changelog/docs if scope is narrower.
- Verification after fix: differential lower/upper/mean/regularized tests across factual, alternative, direct, wrapper, sequential, explicit legacy, multiclass, regression-threshold, reject, JSON, narrative, and PlotSpec. Assert every user-visible point equals the paired public prediction API value and that lower/upper differ whenever interval bounds differ.

### S4-H1 — ADR-030 remediation created a large mutable public API solely for tests

- Severity: `HIGH`
- Status: `needs maintainer decision`
- Files/symbols: `src/calibrated_explanations/core/calibrated_explainer.py:1025-1285` and `:2756-2815`; `src/calibrated_explanations/core/wrap_explainer.py:1199-1270` and `:1726-1815`; `CHANGELOG.md` v0.10.2 Release Task 6 entry; representative tests under `tests/unit/core/`
- Why the first audit prompt might miss this: ordinary private-member and export gates are green precisely because internal state was renamed/exposed as public. A test-quality audit that checks only for underscore access interprets these aliases as compliant and misses that the production contract expanded to satisfy the checker.
- Evidence: source comments explicitly say `Public aliases to replace test access of private members`, `Public aliases for testing`, and `Public alias for testing purposes`. The v0.10.2 changelog confirms that private-member remediation added public pyproject properties, public method aliases, and even resolved dead-code violations through public aliases. At least the following test-facing surfaces now appear on production instances, most with setters: `feature_names_internal`, `get_sigma_test`, `initialize_interval_learner_for_fast_explainer`, `bridge_monitors`, `explanation_plugin_instances`, `pyproject_explanations`, `pyproject_intervals`, `pyproject_plots`, `lime_helper`, `shap_helper`, `fast`, `noise_type`, `scale_factor`, `severity`; wrapper aliases include `serialise_preprocessor_value`, `extract_preprocessor_snapshot`, `build_preprocessor_metadata`, `pre_fit_preprocess`, `pre_transform`, `maybe_preprocess_for_inference`, `finalize_fit`, `format_proba_output`, `normalize_auto_encode_flag`, and `normalize_public_kwargs`. Repository search found test use for each and no user-facing source-doc use for almost all. `is_initialized` is additionally still exposed in generated API docs with a `.. deprecated:: 0.10.1` marker, is absent from the active/removed ledger, and is used only by a unit test.
- Minimal reproduction or inspection method: inspect `inspect.getmembers(CalibratedExplainer, predicate=...)` / `dir(WrapCalibratedExplainer)` and cross-reference names with source comments, `CHANGELOG.md`, tests, source docs, and the deprecation ledger. Mutating properties such as `explainer.pyproject_*`, `bridge_monitors`, caches, helpers, and FAST tuning fields can be set directly by any caller.
- Observed: tests have converted implementation state and internal helpers into user-visible, mutable names. The class API now exposes plugin instances, bridge monitors, config snapshots, caches, helper objects, preprocessing internals, and interval-initialization machinery without a deliberate v1 user contract or documentation boundary.
- Expected: tests should exercise public behavior or use helpers under `tests/helpers/`; production names should be public only when explicitly designed, documented, typed, validated, and included in the v1 compatibility contract. Internal/test seams should remain underscored or move to test fixtures.
- v1 risk: every non-underscored method/property becomes a plausible compatibility obligation at API freeze. Direct setters can violate invariants and create state combinations no public lifecycle supports. The stale `is_initialized` deprecation also contradicts the zero-active-deprecation posture.
- Recommended action: perform a maintainer-owned symbol disposition before RC: genuine public API (document/test/freeze), legacy compatibility (ledger and post-v1 lifecycle), or testing-only leak (remove/rename before freeze and move tests to public outcomes/helpers). Do not merely add all names to API docs; minimize the frozen surface.
- Verification after fix: add a deliberate class-surface snapshot/allowlist that distinguishes stable, experimental, compatibility, and internal names; make the private-usage gate reject production aliases whose only rationale is testing; verify source docs and the deprecation ledger cover every retained public/legacy symbol.

### S4-H2 — Invalid values cross public boundaries and fail late or silently change semantics

- Severity: `HIGH`
- Status: `verified`
- Files/symbols: `src/calibrated_explanations/core/calibrated_explainer.py::__init__`, `::set_mode`, `::predict`, `::predict_proba`; `src/calibrated_explanations/core/wrap_explainer.py::calibrate`; FAST initialization in `plugins/builtins.py`; percentile/threshold validation downstream
- Why the first audit prompt might miss this: `tests/unit/core/test_parameter_surface_contracts.py` proves each allow-listed name is accepted using benign values and mostly asserts only `result is not None`. It does not prove type/range validation or boundary-owned exceptions.
- Evidence: a deterministic public-wrapper matrix produced all of the following at current HEAD: `multi_labels_enabled="yes"` was truthiness-coerced and activated multiclass mode; `suppress_crepes_errors="no"` became `True`; `fast="yes"` was stored as a string and activated FAST behavior; invalid/empty/out-of-range `sample_percentiles` were accepted at calibration and later failed as plugin `ConfigurationError` (or ran unsorted); `features_to_ignore=[99]` calibrated successfully then failed during explanation with an index error wrapped by a plugin error; `mode="Classification"` followed inconsistent case-sensitive branches and failed during interval-plugin creation; `low_high_percentiles=(95,)` and `threshold=[]` leaked raw `IndexError`; `seed="42"` leaked NumPy `TypeError`; invalid FAST `noise_type`, zero/negative/string `scale_factor`, and negative/string `severity` failed only during interval-plugin setup with low-level messages and no `details=`. `interval_summary="not-a-mode"`, spaced `condition_source`, reversed/out-of-range percentile pairs, and NaN reject confidence with an active policy were correctly rejected and are cleared controls.
- Minimal reproduction or inspection method: create seeded classification/regression wrappers; pass one invalid value at a time through `calibrate`, `predict`, `predict_proba`, and explain methods; record acceptance, exception class, `details`, and when state changes. Commands and outputs are recorded under `S4-value-matrix` in the Commands section below.
- Observed: some invalid values silently activate features or fallback policy; others survive the public boundary and fail later as raw `IndexError`/`TypeError` or generic plugin `ConfigurationError` without CE validation details.
- Expected: valid names with invalid values should fail atomically at the public boundary with `ValidationError` and a populated `details=` payload. Boolean fields must require booleans; enum/string modes must be normalized once or rejected consistently; sequences must validate shape, ordering, bounds, and element types before plugin execution.
- v1 risk: users can unintentionally enable degraded-zero fallback (`suppress_crepes_errors="no"`), FAST mode, or multiclass behavior; error contracts vary by call path and expose low-level internals. This is likely user-visible breakage and can become silent correctness drift.
- Recommended action: add centralized typed/range validators for the full allow-list, with conditional validation where a parameter is mode-specific. Validate before mutating `mc`, calibration flags, preprocessors, plugin managers, or interval learners. Prioritize boolean flags, mode normalization, percentile/threshold shape, feature/categorical indices, FAST tuning, and seed.
- Verification after fix: a paired wrapper/direct invalid-value matrix must assert identical CE exception types/details and unchanged state for every rejected call; retain explicit cleared controls for values intentionally accepted (for example `seed=None` only if nondeterministic operation is deliberately documented).

### S4-H3 — Direct-core explanation typos are logged at INFO but otherwise disappear

- Severity: `HIGH`
- Status: `needs maintainer decision`
- Files/symbols: `src/calibrated_explanations/core/calibrated_explainer.py::_log_forwarded_explain_kwargs`, `::explain_factual`, `::explore_alternatives`; `src/calibrated_explanations/core/explain/orchestrator.py::invoke_factual`, `::invoke_alternative`; `src/calibrated_explanations/plugins/builtins.py::_LegacyExplanationBase.explain_batch`; `tests/unit/core/test_calibrated_explainer_more_paths.py::test_explain_methods_still_forward_genuinely_unknown_kwargs_to_plugin`, `::test_explain_methods_log_forwarded_plugin_kwargs`
- Why the first audit prompt might miss this: ADR-038 makes plugin forwarding an intentional exception, so a surface-contract audit can mark arbitrary kwargs as supported without proving that any selected plugin consumed them. The existing forwarding test uses the built-in plugin with `some_plugin_specific_key=123` and asserts only that a non-empty explanation is returned.
- Evidence: in a clean public-core reproduction, `core.explain_factual(x, typo_plugin_kee=123)` returned a normal one-item `CalibratedExplanations`, emitted zero Python warnings, and logged only `INFO ... forwarding ... ['typo_plugin_kee']`. The built-in plugin does not consume `ExplanationRequest.extras`; the value disappears. The wrapper correctly raises `ConfigurationError` for the same typo. A deliberately registered plugin overriding `explain_batch` did receive `{'consumed_key': 'value'}`, proving the extension seam itself works. Known cross-surface `fast=True` is rejected by both entry paths, so the defect is specifically the indistinguishable unconsumed-vs-consumed case.
- Minimal reproduction or inspection method: calibrate a wrapper, call `wrapper.explainer.explain_factual(x, typo_plugin_kee=123)` under `warnings.catch_warnings` and an INFO log handler, then repeat through `wrapper.explain_factual`; register a trusted test plugin that records `request.extras` as the consumed control.
- Observed: a misspelled direct-core kwarg and a valid plugin kwarg have identical API-level behavior. With normal application logging, the typo produces plausible output with no user-visible signal; telemetry does not identify the consuming plugin or confirm consumption.
- Expected: extension kwargs should be namespaced/schema-declared or the plugin dispatch result should report which keys were consumed. At minimum, unconsumed keys must raise or emit the repository fallback-policy signal (`UserWarning` plus INFO log). Direct-vs-wrapper divergence must be explicit in the stable v1 contract.
- v1 risk: users of the public `CalibratedExplainer` can misspell an explanation option and receive credible but semantically unchanged output. The completed Task 27 mitigation demonstrates forwarding, not consumption, and therefore gives false confidence around the highest-risk branch of the extension exception.
- Recommended action: before freeze, choose and document one policy: require plugin-declared kwarg schemas and reject unmatched keys; namespace plugin options by plugin id; or return consumption acknowledgements from dispatch and warn/raise on leftovers. Strengthen the current test so the selected plugin must record the value and add an explicit unconsumed-key case.
- Verification after fix: paired direct/wrapper tests for (1) unconsumed typo, (2) deliberately consumed plugin key, and (3) known closed-surface key, asserting exception/warning, telemetry, and exact consumption behavior.

### S4-H4 — Published guarded and FAST examples invoke removed or unshipped behavior

- Severity: `HIGH`
- Status: `verified`
- Files/symbols: `docs/migration/deprecations.md:Breaking changes/Guarded entrypoints`; `docs/practitioner/advanced/use_plugins.md:Quick Start`, `:Install and register`; `pyproject.toml:[tool.setuptools.packages.find]`, `[project.optional-dependencies].external-plugins`; `src/external_plugins/`; `CHANGELOG.md` v0.11.6 Task 26 entry
- Why the first audit prompt might miss this: each page contains some newer wording, so keyword-based stale-doc checks can pass. The contradiction appears only when the example is executed against the now-closed call surface and when the source-tree plugin package is compared with wheel package discovery.
- Evidence: the migration page says `explain_factual(guarded=True)` and `explore_alternatives(guarded=True)` now raise only on calibration-feature divergence, but both direct and wrapper calls currently raise `ConfigurationError` immediately because `guarded` was removed in v0.11.5. The plugin guide labels `explainer.explain_factual(x_test, fast=True)` as the preferred FAST method; both direct and wrapper calls reject `fast` as a cross-surface kwarg. The same guide tells users to install `[external-plugins]` and run `python -m external_plugins.fast_explanations register`, while Task 26 deliberately limits released package discovery to `calibrated_explanations*`; the extra installs dependencies but not the repository-only `src/external_plugins` module. The editable checkout masks the packaging failure because `find_spec('external_plugins')` succeeds locally.
- Minimal reproduction or inspection method: execute each documented call on a seeded fitted/calibrated wrapper; inspect `pyproject.toml` package discovery and the built wheel contents; test the module command in an isolated install of the wheel rather than the editable source tree.
- Observed: two advertised calls fail at the public boundary, and the documented post-install module command depends on a top-level package intentionally excluded from release artifacts.
- Expected: migration text must teach only `GuardedOptions`; FAST documentation must use the supported calibration-time/plugin-selection or `explain_fast` surface; install instructions must point to an actually published plugin distribution or clearly label the source-tree examples as repository-only.
- v1 risk: core migration guidance and the advanced plugin quick start are executable product claims. Users following them encounter immediate failures during the exact v1 migration and opt-in workflows the pages are meant to support.
- Recommended action: replace the stale guarded section, remove `fast=True` from explanation calls, and split repository-development plugin examples from installable distributions. Add docs example tests and a wheel-isolated command smoke test so editable installs cannot mask missing modules.
- Verification after fix: execute all code blocks in the two sections against a clean wheel install; assert no removed kwargs and no import of top-level packages absent from the wheel.

### S4-H5 — Frozen explanation snapshots share mutable learner, plugin, interval, and RNG state

- Severity: `HIGH`
- Status: `verified`
- Files/symbols: `src/calibrated_explanations/core/calibrated_explainer.py::__deepcopy__`, `::reset`; `src/calibrated_explanations/explanations/explanations.py::FrozenCalibratedExplainer`; `src/calibrated_explanations/explanations/__init__.py::__all__`
- Why the first audit prompt might miss this: direct `copy.deepcopy(explainer)` returns a distinct object and identical predictions, which looks correct. The custom deepcopy silently shares the exact mutable components whose independence matters across calls, and `FrozenCalibratedExplainer.__setattr__` creates the appearance of immutability while public accessors return those shared objects.
- Evidence: a public factual explanation produced a `FrozenCalibratedExplainer` (in the plugin path, nested twice). Unwrapping through its public `.explainer` property reached a distinct core copy, but the copy's `learner`, `plugin_manager`, `interval_learner`, and NumPy `rng` were object-identical to the live core explainer. Calling `reset()` on the core copy reachable through the public frozen wrapper cleared the live explainer's factual plugin identifier (`{'factual': 'core.explanation.factual.sequential'} -> {}`). The frozen wrapper's public `.learner` is the live learner itself; its arrays/attributes can be mutated despite the class claim that modification is prevented. `CalibratedExplainer.__deepcopy__` also shallow-copies `latest_explanation`, perf/cache helpers, prediction bridge, and any attribute whose deepcopy raises, with exceptions suppressed.
- Minimal reproduction or inspection method: `batch = core.explain_factual(x)`; follow `batch.calibrated_explainer.explainer` until the core copy; compare identity of learner/plugin manager/interval learner/RNG with `core`; call the copy's `reset()` and inspect the live manager. Separately attempt direct frozen attribute assignment (correctly rejected) and mutation through `.learner`/`.explainer` (allowed).
- Observed: “frozen” protects only attribute assignment on the outer proxy. It exposes mutable shared objects and operations that change the live explainer; deepcopy failures can silently degrade all the way to the original instance.
- Expected: either the snapshot is genuinely read-only/isolated, or the class/docstring/API must explicitly define shallow shared-state semantics and expose no mutation path. A deepcopy operation should not silently claim an independent snapshot while retaining lifecycle/plugin/RNG coupling.
- v1 risk: retained explanation objects can influence subsequent live explanation/plugin behavior, copied explainers are not independent, and concurrent/multi-call use can couple RNG and runtime state. The public class name and docstring materially overpromise safety.
- Recommended action: replace the raw `.explainer`/`.learner` accessors with immutable descriptors or controlled prediction handles; construct an explicit snapshot DTO for explanation metadata; independently clone or deliberately omit mutable runtime helpers. If model sharing is required for cost, document it narrowly and prevent mutation through the frozen API. Make deepcopy fallback visible and typed.
- Verification after fix: mutate every object reachable from `FrozenCalibratedExplainer` and prove the live explainer is unchanged; reset/close/copy/explain concurrently across original and copy; assert independent plugin identifiers, RNG state, latest explanation, caches, and interval metadata.

### S4-H6 — Legacy plotting accepts invalid prediction kwargs that PlotSpec rejects

- Severity: `HIGH`
- Status: `verified`
- Files/symbols: `src/calibrated_explanations/core/wrap_explainer.py::plot`; `src/calibrated_explanations/core/calibrated_explainer.py::plot`, `::predict`, `::predict_proba`; `src/calibrated_explanations/plotting.py::plot_global`; `src/calibrated_explanations/explanations/explanation.py::_log_forwarded_plot_kwargs`; `src/calibrated_explanations/explanations/explanations.py::_log_forwarded_plot_kwargs`; `tests/unit/core/test_parameter_surface_contracts.py::TestPlotUseLegacy`; `tests/unit/test_explanations_collection.py::test_should_log_forwarded_plot_kwargs_for_collection_surface`
- Why the first audit prompt might miss this: Task 44 proves direct prediction calls reject classification `threshold=`, and plot tests prove both renderer branches return. No paired test asserts that plotting applies the same prediction contract before selecting the renderer, or that renderer kwargs are consumed.
- Evidence: on the same calibrated classifier, wrapper `predict(..., threshold=.5)`, `predict_proba(..., threshold=.5)`, and default `plot(..., threshold=.5, use_legacy=False)` all raised the intended `ValidationError`. `plot(..., threshold=.5, use_legacy=True)` returned normally. `plot_global` selects legacy before gathering outputs through validated `predict_proba`, so the legacy adapter receives and ignores the classification-only-invalid threshold. Separately, a misspelled `filter_topp=1` returned normal output with zero warnings through collection/item default and legacy plots; only INFO forwarding logs appeared. Wrapper default and legacy global plots also accepted the typo, with no warning in the sampled plot loggers. Existing typo tests assert only the INFO log.
- Minimal reproduction or inspection method: calibrate a seeded classifier and call the four threshold cases above under `MPLBACKEND=Agg`; then call wrapper, collection, and item plots with `filter_topp=1`, `show=False`, under both `use_legacy` values while capturing warnings/logs.
- Observed: prediction validation depends on the selected visual renderer. Plot-only unknown kwargs have the same unconsumed-forwarding black hole as explanation plugin kwargs, so misspellings produce plausible plots.
- Expected: renderer selection must not change prediction semantics. Validate/compute the prediction payload once through the governed public contract, then render it. Plot extension kwargs must be schema-declared/consumed or rejected/warned when left over.
- v1 risk: a user can request an invalid classification threshold and see an ordinary legacy plot, while the default plot and direct prediction reject it. This undermines Task 44's fail-fast contract and makes visual outputs path-dependent.
- Recommended action: move prediction-argument validation ahead of the `use_legacy` branch and pass a canonical payload to both renderers. Introduce a closed set for built-in plot kwargs plus explicit renderer schemas/consumption acknowledgements for extensions. Add paired legacy/PlotSpec differential tests for every prediction and plot-only kwarg.
- Verification after fix: threshold, percentile, bins, reject, and interval-summary matrices must have identical value/exception semantics in legacy and PlotSpec paths; misspelled plot kwargs must fail or produce the documented user-visible signal.

### S4-M1 — User-facing Python requirements contradict package metadata and CI

- Severity: `MEDIUM`
- Status: `verified`
- Files/symbols: `README.md:182`; `docs/practitioner/playbooks/eu-ai-act-compliance.md:94`; `pyproject.toml:requires-python` and classifiers; `.github/workflows/ci-pr.yml:python-compat`
- Why the first audit prompt might miss this: Task 26 correctly hardened packaging metadata and CI, which can make the compatibility lane appear closed even though prose requirements were not included in the acceptance check.
- Evidence: `pyproject.toml` requires Python `>=3.10` and declares/tests 3.10–3.13, while the README still says Python `>=3.8` and the EU AI Act playbook says `>=3.9`.
- Minimal reproduction or inspection method: `rg -n "Python .*3\\.[0-9]|requires-python|python-version" README.md docs pyproject.toml .github/workflows`.
- Observed: users are told unsupported interpreters satisfy prerequisites; package installation will then reject them.
- Expected: every installation/prerequisite claim should use the authoritative `>=3.10` floor or link to one generated compatibility statement.
- v1 risk: this does not corrupt results, but it creates avoidable installation failures and weakens the truthfulness of compliance-oriented guidance.
- Recommended action: update both prose claims and make the packaging metadata checker scan canonical requirement statements (or generate them from metadata).
- Verification after fix: repository-wide compatibility search plus the Python matrix/package-artifact gate.

### S4-M2 — FAST tuning parameters are acceptance-tested, not behavior-tested through the public lifecycle

- Severity: `MEDIUM`
- Status: `candidate`
- Files/symbols: `tests/unit/core/test_parameter_surface_contracts.py::test_fast_calibration_accepts_tuning_knobs`; `tests/unit/core/test_calibrated_explainer_delegation.py` FAST property tests; `tests/unit/test_utils_perturbation.py`; `src/calibrated_explanations/plugins/builtins.py` FAST interval setup; `src/calibrated_explanations/core/explain/_computation.py`
- Why the first audit prompt might miss this: the allow-list test is named as contract coverage and supplies all knobs, but asserts only that calibration succeeds. Helper tests prove perturbation functions respond to values in isolation; they do not prove the public wrapper routes those values into the active plugin or that explanation/prediction outputs change.
- Evidence: source tracing shows `noise_type`, `scale_factor`, and `severity` reach FAST interval perturbation, while `sample_percentiles` reaches explanation computation. Repository tests found for the public tuning surface either assert acceptance, property get/set, or helper-level perturbation arrays. No paired seeded `WrapCalibratedExplainer.fit -> calibrate(fast=True, knob=A/B) -> predict/explain` test was found that asserts a semantic output/telemetry difference and active plugin identity. Invalid values also survive until deep plugin setup (`S4-H2`).
- Minimal reproduction or inspection method: cross-reference `_CALIBRATE_KWARGS`, public parameter-surface tests, FAST plugin construction, perturbation helper tests, and repository-wide references for each knob.
- Observed: code appears to consume the knobs, so this is not yet a verified no-op; the executable evidence cannot distinguish correct routing from storage-only behavior or a plugin-path regression.
- Expected: each behavior-changing accepted parameter should have at least one deterministic public differential proving active routing and the intended observable change, plus a control showing irrelevant modes are explicitly rejected or documented inert.
- v1 risk: a plausible wiring regression could make expensive FAST configuration silently ineffective while all current surface tests remain green.
- Recommended action: add one compact public differential matrix covering both supported noise types, scale factor, severity, and sample percentiles; assert plugin identity, telemetry/effective config, deterministic calibration size/state, and a stable output property rather than exact fragile floats.
- Verification after fix: run the matrix through wrapper and direct core, save/load, recalibration, and both FAST explanation entry paths.

### S4-I1 — Importing the package changes process-wide pickle behavior for mapping proxies

- Severity: `INFO`
- Status: `needs maintainer decision`
- Files/symbols: `src/calibrated_explanations/__init__.py::_reduce_mappingproxy` and import-time `copyreg.pickle`; `tests/unit/test_lazy_init_coverage.py::test_mappingproxy_reducer_registration_failure_is_non_fatal`
- Why the first audit prompt might miss this: the root import is correctly lazy with respect to NumPy, pandas, sklearn, and matplotlib, so a conventional import-cost smoke test passes. `copyreg` mutation is small and process-global rather than visible in the imported module graph.
- Evidence: in a clean process, `copyreg.dispatch_table` had no `MappingProxyType` reducer before import. After `import calibrated_explanations`, it contained CE's reducer, and pickling/unpickling any process-wide `MappingProxyType` produced a mutable `dict`. Warning filters were unchanged, heavy optional modules were not loaded, and the only logging mutation was the documented `NullHandler`. The sole reducer test verifies that registration failure is non-fatal; no test or ADR sampled here justifies the global semantic change or restoration/compatibility behavior.
- Minimal reproduction or inspection method: snapshot `copyreg.dispatch_table.get(MappingProxyType)`, warning filters, root logger handlers, and `sys.modules`; import CE; compare and pickle a mapping proxy created by unrelated code.
- Observed: merely importing CE changes how every library in the interpreter serializes `MappingProxyType`, converting immutability to a mutable dictionary on load. The mutation has no public opt-out and is not reversible through CE.
- Expected: process-global serialization changes should be explicitly governed/documented and proven necessary, collision-safe, and compatible with host applications; ideally CE-owned state should normalize mapping proxies at its own serialization boundary instead.
- v1 risk: low-probability host-application interference and surprising type changes outside CE. This needs a deliberate maintainer decision before the behavior becomes an implicit long-term import contract.
- Recommended action: prefer CE-local `__getstate__`/schema conversion. If global registration is retained, document it, test pre-existing reducer collision behavior, avoid overwriting host reducers, and provide a clear compatibility rationale.
- Verification after fix: clean-process import test covering reducer presence/collision, warning filters, log handlers, random state, environment reads, and heavy-module loading.

## False positives / cleared suspicions

## Commands run

- `git rev-parse HEAD`
- `git branch --show-current`
- `git status --short`
- required startup file-presence and line-count inspection

## Commands not run and why

- Behavioral, quality, release, and evidence gates: not yet run at startup.

## Recommended pre-v1 action plan

1. Must fix before v0.11.6 tag: pending audit evidence.
2. Must fix before v1.0.0-rc: pending audit evidence.
3. Must fix before v1.0.0 GA: pending audit evidence.
4. Can defer post-v1: pending audit evidence.
5. Maintainer decisions needed: pending audit evidence.
