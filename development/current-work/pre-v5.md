# Pre-v5 skeptical release audit (sixth pass)

Audit target: `kristinebergs/calibrated_explanations`
Branch/commit: `main` @ `15daa0eb` (Task 47 recalibration-atomicity commit landed mid-audit; all findings verified against `15daa0eb`)
Started: 2026-07-11
Auditor: Claude Code (claude-fable-5); adversarial sixth pass following `pre-v1.md`, `pre-v2-gaps.md`, `pre-v3.md`, `pre-v4.md`

**Headline:** this pass found **no new BLOCKER or HIGH findings**. All previously fixed silent-correctness defect classes were re-verified through public-API repros and hold at current HEAD. Every gate that was run passed. The genuinely release-gating work that remains is what `v0.11.6_plan.md` already tracks (open Tasks 13, 20, 26–28 bookkeeping, 35, 48–56 — in particular non-deferrable Task 48 / S4-B4). The findings below are two MEDIUM governance/documentation items and four LOW items, none of which block the v0.11.6 tag on their own.

## Current expected state from repo evidence

- Package version: `0.11.6-dev` (`pyproject.toml:10`); runtime `__version__` reports `0.11.6.dev0` (PEP 440 normalization of the same string — not drift; `scripts/quality/check_version_alignment.py` passes under the Task 36 dev-window policy).
- Current released version according to release docs: v0.11.5 (2026-07-07), per `RELEASE_PLAN_v1.md` and `CHANGELOG.md`.
- Active milestone: v0.11.6 (pre-RC hardening; `development/current-work/v0.11.6_plan.md`, baselined 2026-07-08).
- Next milestone: v1.0.0-rc (validation and freeze only), then v1.0.0 GA.
- Release blockers per plan: gate-summary states Tasks 45–48 (S4-B1–S4-B4) cannot be deferred. Tasks 45–47 are closed in the gate table; Task 48 (`interval_summary` consistency, S4-B4) is open at audit time. Also open: Tasks 13, 20 (in progress), 28, 35, 49–56, 57.
- Zero-deprecation expectation: `docs/migration/deprecations.md` Active deprecations table is empty ("All active deprecations were removed in v0.11.5"); `make deprecation-closure` passes.
- Explicit/experimental/unknown kwarg policy: fail-fast `ConfigurationError` for unknown public kwargs on all gated wrapper/core surfaces; direct-core `explain_factual`/`explore_alternatives` forward genuinely unknown kwargs to explanation plugins with INFO-level visibility (ADR-038 §3 exception; disposition of that seam is open Task 51).
- Public API freeze posture: freeze at RC tag; no active deprecation may survive into v1.0.0.
- Required gates: `make local-checks-pr` (includes capability-chain-check), `make local-checks-release`, `make deprecation-closure`, strict Sphinx docs (Task 41 turned these green 2026-07-10).
- Canonical development locations: `development/current-work/`, `development/adrs/`, `development/standards/`, `development/capabilities/`. All startup files listed in the audit brief exist (`pre-v1.md`, `pre-v2-gaps.md`, `pre-v3.md`, `pre-v4.md`, `bug-list.md`, plans, appendix, `CHANGELOG.md`, `README.md`, `QUICK_API.md`, `docs/migration/deprecations.md`, `tests/README.md`, `Makefile`).
- Commands attempted: see "Commands run" and "Commands not run and why".

## Live findings index

| ID | Severity | Status | Area | Short title | Covered by existing plan? |
|---|---|---|---|---|---|
| M1 | MEDIUM | verified | Docstrings / API docs | Core `predict`/`predict_proba`/`explain_fast` docstrings document wrong exception types and malformed numpydoc | Yes — added as v0.11.6 Task 57 (2026-07-11) |
| M2 | MEDIUM | verified | Release gates / typing governance | pyproject declares strict mypy for four modules; local gate and CI enforce only three; the fourth fails its declared contract | Yes — added as v0.11.6 Task 58 (2026-07-11) |
| L1 | LOW | verified | Documentation truthfulness | README and narrative-templates page show a narrative text format the runtime no longer produces, under an "exact format produced at runtime" claim | Yes — added to v0.11.6 Task 59 (2026-07-11) |
| L2 | LOW | verified | Release bookkeeping | Plan/gate-summary text contradicts its own gate table; RELEASE_PLAN control snapshot still names v0.11.5 as active milestone | Partly (open Task 28 covers "RELEASE_PLAN sync" and evidence reconciliation) |
| L3 | LOW | needs maintainer decision | Exception taxonomy (ADR-002) | New `calibrate()` exception normalization re-wraps `ModelNotSupportedError` as `ConfigurationError` | Yes — decision item in v0.11.6 Task 59 (2026-07-11) |
| L4 | LOW | verified | Governance drift | Live skill/agent/docs surfaces still reference removed `docs/improvement/`–`docs/standards/` locations; no gate covers this class | Yes — added to v0.11.6 Task 59 (2026-07-11) |
| I1 | INFO | verified | Validation matrix (supplements open Task 50) | Wrong-length `threshold` arrays: raw `IndexError` on `predict_proba`, `AssertionError` on `predict`/`explain_factual` | Yes (S4-H2 / open Task 50; recorded here as two additional concrete matrix rows) |

## Findings

### M1 — Core `predict`/`predict_proba`/`explain_fast` docstrings document wrong exception types and malformed numpydoc

- Severity: MEDIUM
- Status: verified
- Area: docstrings rendered into the public API reference (numpydoc/RTD)
- Files/symbols: `src/calibrated_explanations/core/calibrated_explainer.py` — `predict` Raises (~2273–2280), `predict_proba` Raises (~2499–2510), `explain_fast` Raises (~2035–2042)
- Existing-plan coverage: Yes as of 2026-07-11 — added as v0.11.6 Task 57 (sequenced with/after Task 50). At audit time: No — open Task 50 (S4-H2) fixes the *behavioral* validation matrix only, and the milestone's docstring scope in `RELEASE_PLAN_v1.md` was "Docstring corrections for `normalize=`/kwarg seams only".
- Evidence:
  - `predict` Raises documents `RuntimeError` "If the learner has not been fitted"; the fit-state check raises CE `NotFittedError` (`core/validation.py:313-318`), and the only two runtime `raise RuntimeError` sites left in `src/` are in `explanations/reject.py` (both `adr002_allow`-annotated).
  - `predict_proba` Raises documents `RuntimeError` **twice** (duplicate entry) plus `ValueError` "If the `threshold` parameter's length does not match the number of instances" — the verified actual behavior for that exact input is a raw `IndexError` from `calibration/interval_regressor.py:238` (see I1), and CE-boundary validation elsewhere raises `ValidationError`.
  - `explain_fast` Raises uses malformed inline numpydoc (`ValueError: ...` on a single line), lists `Warning` as a raised type, and documents `RuntimeError` for the fast-explainer precondition.
  - Standard-002 is recorded as "Completed — wrapper/public numpydoc closure" in `RELEASE_PLAN_v1.md`, which these sections contradict. The wrapper methods delegate via "See Also" to these core docstrings, so the wrapper API reference inherits the wrong contracts.
- Reproduction: `python -c "import inspect; from calibrated_explanations import CalibratedExplainer; print(inspect.getdoc(CalibratedExplainer.predict_proba))"` and compare with the observed exceptions in I1.
- Observed: the API reference tells users to catch `RuntimeError`/`ValueError` on the flagship prediction methods.
- Expected: Raises sections list the CE exception types the code actually raises (`NotFittedError`, `ValidationError`, `ConfigurationError`), once each, in valid numpydoc format.
- Why it matters for v1: docstrings are part of the frozen API contract at RC; users writing `except ValueError` around threshold validation get dead error-handling code. This does not corrupt any results.
- Recommended action: correct the three Raises sections (small, mechanical edit) in the same change that closes Task 50, so the documented and enforced exception contracts land together.
- Verification after fix: numpydoc validation of the three docstrings; a small test asserting the documented exception types match the ones raised by the I1/Task 50 matrix.

### M2 — Declared strict-typing scope and enforced mypy scope have drifted; the unenforced module fails its declared contract

- Severity: MEDIUM
- Status: verified
- Area: release-gate blind spot (typing governance)
- Files/symbols: `pyproject.toml` `[[tool.mypy.overrides]]` (modules: `core.exceptions`, `core.validation`, `api.params`, **`core.prediction_helpers`**); `scripts/local_checks.py::_mypy_targets` (lines 99–105, three files); `.github/workflows/ci-pr.yml` "Run mypy (Phase 1B scope)" (lines 102–119, same three files)
- Existing-plan coverage: Yes as of 2026-07-11 — added as v0.11.6 Task 58. At audit time no open task covered mypy-scope alignment.
- Evidence: `pyproject.toml` declares strict typing (`disallow_untyped_defs`, `warn_return_any`, `strict_equality`, …) for four modules. Both the local gate and CI enumerate only three; `core/prediction_helpers.py` is absent from both. Running the declared contract today fails: `python -m mypy src/calibrated_explanations/core/prediction_helpers.py --config-file pyproject.toml` → **10 errors** (e.g. `no-any-return` at line 497). Neither `.pre-commit-config.yaml` nor any workflow runs mypy more broadly. (For scale, not as a defect: full-tree mypy under the same config reports 514 errors in 51 files — the lenient global config is clearly intentional, and the repo nowhere claims full-tree mypy cleanliness. The finding is only the declared-but-unenforced strict module.)
- Reproduction: commands above; `grep -rn prediction_helpers scripts/local_checks.py .github/workflows/ci-pr.yml` returns nothing.
- Observed: every green "mypy" gate silently excludes one of the four modules the packaging metadata declares strict; that module currently violates its declared contract.
- Expected: the strict-override list and the gated file list are the same list, or the override is removed if the module is not meant to be strict.
- Why it matters for v1: this is exactly the "gate is green for the wrong reason" class — a declared engineering contract with no executable enforcement. It is not user-visible runtime breakage.
- Recommended action: maintainer picks one: (a) add `core/prediction_helpers.py` to `_mypy_targets()` and the CI list and fix the 10 errors, or (b) drop it from the strict-override list with a rationale. Either way, derive the gate list from `pyproject.toml` instead of duplicating it in two places.
- Verification after fix: `make quick` (or the PR profile) fails when a strictly-declared module has mypy errors; a unit check that `_mypy_targets()` matches the pyproject override list.

### L1 — README narrative example claims "the exact format produced at runtime" but shows a format the runtime no longer emits

- Severity: LOW
- Status: verified
- Area: documentation truthfulness
- Files/symbols: `README.md:23-39` (factual narrative fence and the "exact format produced at runtime" sentence); `docs/practitioner/narrative_templates.md:96-104`
- Existing-plan coverage: Yes as of 2026-07-11 — added to v0.11.6 Task 59 (step 1). Background: Task 43 (closed 2026-07-10) changed the runtime to ASCII-safe markers; Task 42's smoke test executes only the *Python* fences of README/QUICK_API, so text-fence output claims were unguarded.
- Evidence: runtime output (public-API probe, `random_state=42`) is `annual_income (…) >= 45000 - weight ~ 0.312 [0.198, 0.421]` style — ASCII hyphen and `~`, unsigned weights, an `Instance 0` banner, and a `Factual Explanation (Advanced):` heading. README shows `— weight ≈ +0.312 [+0.198, +0.421]` (em-dash, `≈`, signed weights, no banner, no `(Advanced)` suffix). The alternatives fence matches the runtime closely; only the factual fence and the templates page carry the old format.
- Reproduction: fit/calibrate a seeded `RandomForestClassifier` via `WrapCalibratedExplainer`, call `explanation[0].to_narrative(output_format="text", expertise_level="advanced")`, diff against README lines 26–40.
- Observed: the "exact format" claim is false in four particulars (marker glyphs, weight signing, banner, heading suffix).
- Expected: either update the fences to the current ASCII format or soften the "exact format" claim.
- Why it matters for v1: README is the most-read product claim; anyone parsing or eyeballing narrative output against the README sees a mismatch on first contact. No numerical or semantic impact.
- Recommended action: regenerate the README factual fence and the templates-page examples from a real run; consider extending the Task 42 smoke test to assert the presence of the current marker (`- weight ~`) in a live narrative.
- Verification after fix: the fences match a fresh runtime capture; grep for `weight ≈` in `README.md` and `docs/` returns nothing.
- Cosmetic side-note (no separate finding): the runtime itself prints `Calibrated Probability:` (capital P) for factual and `Calibrated probability:` for alternatives; README faithfully mirrors this inconsistency.

### L2 — Release bookkeeping text contradicts the authoritative gate table

- Severity: LOW
- Status: verified
- Area: release bookkeeping / governance text
- Files/symbols: `development/current-work/v0.11.6_plan.md:3843-3854` ("Overall status" paragraph); `development/current-work/RELEASE_PLAN_v1.md:37-38, 76-77` (control snapshot)
- Existing-plan coverage: Partly — open Task 28 is "Evidence/bookkeeping reconciled: CHANGELOG, ledger framing, RELEASE_PLAN sync, upgrade checklist", and the RELEASE_PLAN boundary-update policy defers plan grooming to milestone close.
- Evidence: the summary paragraph lists Tasks "26–28 … pending" and omits 26, 27, 34, 43, 44 from its completed enumeration, while the gate table directly below marks 26, 27, 34, 43, 44 "Completed" (2026-07-09/10). It also still says Task 41 "turns the currently red strict-docs and `local-checks-release` pre-tag gates green", though Task 41 closed 2026-07-10. Separately, `RELEASE_PLAN_v1.md`'s control snapshot still says "Active detailed milestone: v0.11.5" and "Next action: Execute v0.11.5" while v0.11.6 is executing.
- Reproduction: read the cited lines.
- Observed/Expected: summary paragraph and control snapshot should match the gate table and the actual active milestone.
- Why it matters for v1: release-audit traceability; a reader trusting the paragraph (or the snapshot) mis-reads the milestone state. No runtime impact.
- Recommended action: fold into Task 28's reconciliation sweep; regenerate the completed/pending lists from the gate table rather than maintaining them by hand.
- Verification after fix: the paragraph's two task lists partition 1–57 consistently with the gate table; the control snapshot names v0.11.6 (or its closed successor state).

### L3 — `calibrate()` now re-wraps `ModelNotSupportedError` as `ConfigurationError`

- Severity: LOW
- Status: needs maintainer decision
- Area: exception taxonomy fidelity (ADR-002)
- Files/symbols: `src/calibrated_explanations/core/wrap_explainer.py::calibrate` — pass-through tuple `(ConfigurationError, DataShapeError, IncompatibleStateError, NotFittedError, ValidationError)` followed by a blanket `except Exception` → `ConfigurationError` (`adr002_allow`-annotated), introduced by commit `15daa0eb` (Task 47, 2026-07-11)
- Existing-plan coverage: Yes as of 2026-07-11 — decision item in v0.11.6 Task 59 (step 2). The commit closing Task 47 introduced it the day of this audit.
- Evidence: `validate_model` (`core/validation.py:306-310`) raises `ModelNotSupportedError` for a learner without `predict`, and it is called inside the new try-block. Verified repro: a fitted learner exposing `fit`/`predict_proba` but no `predict` now gets `ConfigurationError` ("Calibration failed during surface_validation: Model must implement a 'predict' method.", cause=`ModelNotSupportedError`) from `calibrate()`; before `15daa0eb` the `ModelNotSupportedError` propagated unchanged. `ConvergenceError` is also absent from the tuple but has no raise site in `src/`, so `ModelNotSupportedError` is the only affected class today. Mitigating factors: both classes subclass `CalibratedError`; `details=` carries the original type; the `calibrate` docstring never promised `ModelNotSupportedError`; `DataShapeError` in the tuple is redundant (subclass of `ValidationError`) but harmless.
- Reproduction: `WrapCalibratedExplainer(NoPredictStub()).fit(X, y)` then `calibrate(Xc, yc)` — stub with `fit` setting `classes_` and `predict_proba` only.
- Observed: `ConfigurationError` wrapping `ModelNotSupportedError`.
- Expected (probable): taxonomy-specific classes defined by ADR-002 should survive the public boundary; a model-capability failure is arguably not a "configuration" error.
- Why it matters for v1: narrow — only callers catching `ModelNotSupportedError` from `calibrate()` on a mis-shaped learner are affected. Worth deciding *before* the RC freeze locks the behavior in.
- Recommended action: maintainer decision — add `ModelNotSupportedError` (and optionally `ConvergenceError`) to the pass-through tuple, or accept the normalization and note it in the `calibrate` docstring/CHANGELOG.
- Verification after fix: a unit test asserting the chosen exception class for the no-`predict` learner through `calibrate()`.

### L4 — Live agent/skill/docs surfaces still point at removed `docs/improvement/` / `docs/standards/` locations, and no gate covers the class

- Severity: LOW
- Status: verified
- Area: governance drift / stale references
- Files/symbols: `.claude/skills/ce-release-planner/references/version_plan_reference.md:5,191,193,213` (instructs creating `docs/improvement/vX.Y.Z_plan.md` and reviewing/archiving `docs/improvement/` — contradicting the same skill's `SKILL.md`, which correctly names `development/current-work/`); `.github/agents/architecture-gatekeeper.agent.md:28,30` ("Active docs/improvement/vX.Y.Z_plan.md", "Relevant STDs in docs/standards/"); user-facing `docs/foundations/concepts/architecture.md:3` ("See `docs/improvement/component_diagram.md`") and `docs/practitioner/advanced/parallel_execution_playbook.md:100`; cosmetic: `pyproject.toml:375` codespell skip lists the removed `docs/improvement/anti_pattern_gap_analysis.ipynb`
- Existing-plan coverage: Yes as of 2026-07-11 — added to v0.11.6 Task 59 (step 3). At audit time nothing covered it: Task 21 synced skill mirrors structurally; the docs migration closed in v0.11.4; `scripts/quality/check_forbidden_doc_patterns.py` has eight named checks, none for removed-location references. (CHANGELOG and `development/finished-work/` mentions are historical records and are correctly excluded here.)
- Reproduction: `grep -rn "docs/improvement\|docs/standards" .claude/skills .github/agents docs/` and filter out migration-history wording.
- Observed: an agent following the release-planner reference file, or the architecture-gatekeeper definition, is instructed to use directories whose recreation `CONTRIBUTOR_INSTRUCTIONS.md` explicitly forbids; two user-facing docs sentences point at paths that no longer exist.
- Expected: live guidance names only `development/` locations; user docs don't reference removed internal paths.
- Why it matters for v1: primarily an internal-tooling footgun (an agent could recreate forbidden directories during v1.0.0-rc planning); the user-facing instances are dead references only.
- Recommended action: fix the four files; optionally add a `removed-legacy-locations` check to `check_forbidden_doc_patterns.py` scoped to live surfaces (`.claude/skills/`, `.github/agents/`, `docs/`), excluding history files.
- Verification after fix: the grep above returns only historical-record hits; the new named check passes.

### I1 — Two additional concrete rows for the open Task 50 invalid-value matrix

- Severity: INFO (instances of the already-verified S4-H2 / open Task 50; recorded so the Task 50 closure covers them)
- Status: verified
- Area: value-sensitive invalid input
- Files/symbols: `calibration/interval_regressor.py:238`; `utils/helper.py:441` (`assert_threshold`)
- Existing-plan coverage: Yes — S4-H2 already lists "`threshold=[]` leaked raw `IndexError`" and its recommended action covers threshold shape validation; these are sibling instances.
- Evidence (regression wrapper, `random_state=42`, 6 test rows, `threshold=np.array([1.0, 2.0])`):
  - `predict_proba` → raw `IndexError: index 2 is out of bounds for axis 0 with size 2` at `interval_regressor.py:238` — the docstring-promised length validation never runs on this path.
  - `predict` and `explain_factual` → `AssertionError: list thresholds must have the same length as the number of samples` at `helper.py:441` — validation by `assert`, which is not a CE exception, carries no `details=`, and is stripped under `python -O`.
  - Same wrong input, three surfaces, three different failure modes.
- Recommended action: when Task 50 lands its centralized validators, include array-length thresholds explicitly and replace the `assert_threshold` assertion path with `ValidationError`; assert cross-surface consistency (predict/predict_proba/explain) in its verification matrix.

## False positives / cleared suspicions

- **Version drift `0.11.6-dev` vs `0.11.6.dev0`** — PEP 440 normalization of one source string; `check_version_alignment.py` passes (docs `0.11.6.dev0`/`0.11`, CITATION `v0.11.6`, METADATA `0.11.6` under the documented dev-window policy). Not drift.
- **Mondrian categorizer lost on persistence** — `save_state` on an mc-configured wrapper emits a `UserWarning` ("Pickle/state persistence drops the configured Mondrian categorizer (mc)…") plus INFO log; post-load bin-less inference fails fast with actionable guidance; explicit `bins=` post-load reproduces pre-save predictions exactly; `reuse_conditional=True` post-load raises `ValidationError` as designed; fresh `mc=` recalibration fully recovers. Visible-by-design (ADR-039 D5), not a defect.
- **Suspected leaked file handle on Windows state files** — `save_state` produces a *directory* artifact (ADR-031 manifest + per-file checksums); `os.remove` on a directory raises WinError 5. Probe error, not a product defect.
- **Thresholded-regression label/probability mismatch suspicion** — labels (`y_hat > t` / `y_hat <= t`) are consistent with `predict_proba(threshold=t)` columns (column 1 = P(y ≤ t)) across all probed rows. Cleared.
- **Multiclass narrative truthfulness (Task 39 regression check)** — with string labels {low, mid, high}: narrative explained class, runner-up class, and margin all match the stored calibrated probability payload to 3 decimals on every probed instance; `predict` equals argmax of calibrated `predict_proba`. Cleared.
- **Reject metadata contract (Task 40 regression check)** — `predict(reject_policy="only_accepted")` returns `RejectResult` with populated `matched_count`, `source_indices`, `novelty_mask`, schema_version; removed `confidence=` raises `ConfigurationError` with `reject_confidence=` guidance; `reject_confidence` −0.1/1.5/NaN all raise `ValidationError`. Cleared.
- **Import side effects beyond the known S4-I1** — importing the package mutates no warnings filters, no root logging handlers, imports no heavy modules (`matplotlib`/`pandas`/`joblib` absent from `sys.modules` after root import); the only CE-added global is the known `mappingproxy` copyreg reducer (S4-I1, open Task 56). Cleared.
- **ADR-039 conditional matrix** — bins-calibrated wrapper rejects bin-less predict/explain (`ValidationError` with recovery guidance); bins on a globally calibrated wrapper raise `ConfigurationError`; unseen bin labels raise `ValidationError` naming known/unknown labels; `calibrate(mc=…)` stores the categorizer and inference derives bins automatically. Cleared.
- **Task 47 atomicity (fresh commit)** — a rejected recalibrate (wrong-length bins → `DataShapeError`) leaves `calibrated=True` and the prior explainer fully functional. Behavioral goal confirmed (see L3 for the one taxonomy nuance it introduced).
- **Multi-call/lifecycle stability** — `explain_factual → explore_alternatives → explain_factual` yields identical feature weights; `deepcopy` of a calibrated wrapper predicts identically; collection `plot(show=False)` (Agg) succeeds and leaves subsequent predictions unchanged. Cleared.
- **Previously fixed defect classes re-verified at HEAD** — unknown kwargs raise `ConfigurationError` naming the rejecting method (wrapper calibrate/explain/predict); removed guarded kwargs raise with `GuardedOptions` migration text; classification `threshold=` fails fast; `{0,1}` int and `{1,2}` label spaces round-trip through `predict` with original dtype and no phantom labels; reversed tuple thresholds and reversed percentiles raise `ValidationError`. All hold.
- **S4-H4 stale guarded/FAST doc examples** — still present exactly as pre-v4 describes (`docs/practitioner/advanced/use_plugins.md:47-48,106` `fast=True`; `:69` `python -m external_plugins.fast_explanations register`; `docs/migration/deprecations.md:268-274` stale `guarded=True` wording). Not re-reported: open Task 52 covers precisely these.

## Commands run

- `python -m compileall src scripts` — exit 0.
- Full test suite `pytest -q --no-cov` (repo venv, Python 3.14.4) — 3315 tests, exit 0.
- `make deprecation-closure` — exit 0 (1094 focused tests passed; private-member scan, anti-pattern detector, test-helper export guard, marker hygiene, local-path guard, ADR-030 ratification lane all green; Active-deprecations ledger empty).
- `make capability-chain-check` — exit 0 (42 evidence files validated).
- `python scripts/quality/check_version_alignment.py` — PASS.
- `python scripts/quality/check_fallback_filter_live.py` — PASS ("fallback warnings fail the suite; benign warnings do not").
- `ruff check src tests` — "All checks passed!".
- `mypy` — gate scope (3 files) passes; declared-scope check of `core/prediction_helpers.py` fails with 10 errors (M2); informational full-tree run: 514 errors in 51 files under the intentionally lenient global config.
- Deterministic public-API probe batteries (all `random_state=42`, no network, no clock): kwarg/removed-alias/threshold/label-dtype battery; ADR-039 conditional battery; save/load round-trip battery (plain + mc); multiclass string-label + narrative-consistency battery; reject-policy battery; interval/threshold-consistency + CLI-import battery; import-side-effect probe; multi-call lifecycle battery.

## Commands not run and why

- `make local-checks-release` / `python -m build` — heavy (full docs build + artifact build + full coverage suite); the release profile last ran green 2026-07-10 under Task 41, and nothing in this pass touched packaging. The packaging-truthfulness claims were not independently re-verified here.
- `make capability-evidence-refresh` — mutating; audit-only pass, and `capability-chain-check` (non-mutating) is green.
- `make local-checks-pr` — its distinctive components (ruff, mypy gate scope, capability-chain-check, full pytest) were run individually instead, to attribute failures precisely.
- Notebook execution and docs linkcheck — heavy and outside this pass's focus; strict docs build last verified green under Task 41 (2026-07-10).

## Recommended pre-v1 action plan

1. **Must fix before v0.11.6 tag:** nothing new from this pass. The already-planned non-deferrable Task 48 (S4-B4 `interval_summary` consistency) remains the real gate. Fold L2 into the already-open Task 28 sweep. L3 should be decided (one line either way) before the tag, since the Task 47 commit shipping it is part of this milestone.
2. **Must fix before v1.0.0-rc:** M1 (docstring Raises corrections — cheap, and docstrings freeze with the API at RC); M2 decision (enforce or un-declare `core/prediction_helpers.py` strict typing); L1 (README "exact format" fence); L4 skill/agent-file corrections (before any agent runs v1.0.0-rc planning from the stale reference).
3. **Must fix before v1.0.0 GA:** none beyond the above; I1 rides along with the planned Task 50 closure.
4. **Can defer post-v1:** the optional `removed-legacy-locations` named gate (L4 hardening); deriving the mypy gate list from `pyproject.toml` (M2 hardening); the `Calibrated Probability:`/`Calibrated probability:` capitalization inconsistency.
5. **Maintainer decisions needed:** L3 (taxonomy pass-through vs. normalization in `calibrate()`); M2 option (a) vs (b).

## Sixth-pass conclusion

### Additional gaps found that earlier passes were likely to miss

- Docstring-level exception contracts on the flagship prediction methods (M1) — earlier passes checked docs pages and behavior, and pre-v2 flagged one `ValueError`→`ValidationError` drift, but the rendered Raises sections of `predict`/`predict_proba`/`explain_fast` were never diffed against actual raise sites.
- Declared-vs-enforced strict-typing drift (M2) — visible only by comparing the pyproject mypy override list against the two hand-maintained gate lists and then actually running the declared contract.
- A same-day regression nuance in freshly landed work (L3) — only findable by auditing the commit that closed Task 47 rather than trusting its closure note.
- Text-fence documentation claims that execution-based doc smoke tests cannot see (L1).

### Cleared high-risk suspicions

Mondrian persistence, thresholded-regression probability semantics, multiclass narrative truthfulness, reject metadata, import side effects, lifecycle/multi-call stability, and every previously fixed silent-correctness class re-probed this pass — all clean at `15daa0eb`. Details in "False positives / cleared suspicions".

### Highest-risk remaining v1 concern

Nothing new from this pass rises above MEDIUM. The highest-risk open item remains the plan's own non-deferrable Task 48 (S4-B4: `interval_summary` affecting prediction output but not displayed explanation probabilities), followed by the open S4-H* dispositions (Tasks 49–56). Judged from this pass's evidence, the codebase's runtime behavior is in materially better shape than its remaining paperwork: all executed gates are green and all re-probed behavioral contracts hold.

### Recommended additions to v0.11.6 or v1.0.0-rc plan

**Adopted 2026-07-11:** these findings were added to `v0.11.6_plan.md` as three
new tasks — Task 57 (M1 docstring exception contracts), Task 58 (M2 mypy-scope
alignment), and Task 59 (L1/L3/L4 LOW remainder cleanup, with L2 cross-referenced
to Task 28 and I1 to Task 50). Release preparation was renumbered from Task 57
to Task 60 and remains final. (2026-07-13 update: the v1 CI consolidation was
added as Task 60, so release preparation is now Task 61 and still final.)

Original recommendations:

- Add M1's docstring corrections to the Task 50 closure criteria (or as a one-line follow-up task). → Task 57 (sequenced with/after Task 50).
- Add an explicit mypy-scope alignment decision item (M2) to the v1.0.0-rc checklist. → Task 58.
- Record the L3 decision in the Task 47 closure note. → Task 59 step 2.
- Add L1/L4 to the Task 28-adjacent documentation sweep before RC. → Task 59 steps 1 and 3.
