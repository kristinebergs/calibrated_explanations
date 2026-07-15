> **Status (updated 2026-07-08, second pass):** Findings 1-5, 7, 8 were first incorporated
> into `v1.0.0-rc_plan.md` Task 0/0b; later the same day that scope was re-baselined out of
> the RC milestone into the new **`development/current-work/v0.11.6_plan.md`**, which is now
> the authoritative task tracker for closing these items (Finding 1 → Task 2, Finding 2 →
> Task 3, Findings 3-4 → Task 4, Finding 5 → Task 6, Findings 7-8 → Task 7, `ncf`/ruff
> low-severity items → Task 8, gate fix → Task 1, ADR-038 graduation → Task 5). Finding 6
> (version string) is deliberately *not* in v0.11.6 — it remains the RC plan Task 6's
> responsibility (decision D1). This file remains the source evidence/repro record — check
> the v0.11.6 plan for current status before re-deriving.

# Pre-RC bug and inconsistency hunt (2026-07-08)

Scope: functionality vs. `CHANGELOG.md` "Removed" claims, ADR-039 conditional-calibration
fail-fast behavior, deprecation-closure gate, capability-chain-check, full test suite, ruff/mypy,
and cross-check of docs/skills/instruction files against current source.

## Summary

Gates are green: full `pytest` suite passes (952 passed via `make deprecation-closure`'s focused
lane; full suite also green), `make deprecation-closure` passes, `make capability-chain-check`
validates all 31 evidence files, and all seven ADR-039 fail-fast behaviors were verified live
through the public API (see [Verified clean](#verified-clean)).

However, there is a cluster of **silent-correctness defects around the v0.11.5 kwarg removals**:
every "removed" keyword argument documented in `docs/migration/deprecations.md` as removed is in
practice silently swallowed instead of raising, and in the worst case `guarded=True` silently
returns an *unguarded* explanation. This is exactly the defect class v0.11.5 was shipped to
eliminate (see CHANGELOG "Fixed" / ADR-039 section), so findings 1-3 should block tagging RC.

---

## High severity — silent correctness

### 1. Removed guarded kwargs are silently ignored; wrapper allowlist actively suppresses the only warning

The deprecation ledger (`docs/migration/deprecations.md:247-251`) states `guarded=True`,
`significance=`, `n_neighbors=`, `normalize_guard=`, `merge_adjacent=` were removed in v0.11.5.

Verified: `explainer.explain_factual(X, guarded=True)` succeeds and returns a plain
(non-guarded) `CalibratedExplanations` — the in-distribution guard is silently dropped, no error,
no warning.

Root cause is two-layered:

1. The ADR-038 §3 `**kwargs` experimental sink in
   `src/calibrated_explanations/core/calibrated_explainer.py:1453` (`explain_factual`) accepts
   and discards unrecognized kwargs. Closing this is the one planned RC implementation item
   (v0.11.5 CHANGELOG "Added" entry on ADR-038 graduation), so this half is already tracked.
2. `_KNOWN_PUBLIC_KWARGS` in `src/calibrated_explanations/core/wrap_explainer.py:54-94` still
   lists `guarded`, `n_neighbors`, `merge_adjacent`, `normalize`, and `confidence` as known
   public kwargs. This means the wrapper's "unknown keyword arguments" `UserWarning` — which
   *does* fire correctly for `significance=` — is actively suppressed for the other four removed
   names. This half is a straight bug independent of the ADR-038 kwargs-gate closure and should
   be fixed by removing the stale entries from the allowlist.

**Repro:**
```python
w.explain_factual(X, guarded=True)         # SILENT-PASS, unguarded result
w.explain_factual(X, n_neighbors=5)        # SILENT-PASS, no warning (allowlisted)
w.explain_factual(X, merge_adjacent=True)  # SILENT-PASS, no warning (allowlisted)
w.explain_factual(X, significance=0.1)     # SILENT-PASS, but DOES warn (not allowlisted)
```

**Fix:** remove `guarded`, `n_neighbors`, `merge_adjacent`, `normalize` (guarded-path sense) from
`_KNOWN_PUBLIC_KWARGS`; close the `**kwargs` sink in `explain_factual`/`explore_alternatives` per
D3 in `v1.0.0-rc_plan.md`.

### 2. `predict_reject(x, confidence=0.5)` silently falls back to the 0.95 default

`confidence=` was removed in favor of `reject_confidence=` (ledger row, ETA v0.11.5). The old
name lands in the `**kwargs` parameter of
`src/calibrated_explanations/core/reject/orchestrator.py:2020` (`predict_reject`), which the
method body never reads.

**Verified:** reject rate 0.0 with `confidence=0.5` (identical to omitting the argument, default
0.95) vs. reject rate 0.05 with the correct `reject_confidence=0.5`. Zero warnings emitted either
way. The same `**kwargs` also swallows plain typos (e.g. `rejct_confidence=0.5`).

This is not covered by the ADR-038 experimental-forwarding rationale (`predict_reject` is not an
explanation-plugin call) and should either validate its kwargs or drop the catch-all.

### 3. `VennAbers.predict_proba(normalize=True/False)` silently produces SCALE, not the documented mapping

The docstring at `src/calibrated_explanations/calibration/venn_abers.py:236-238` still documents
`normalize=True → COHERENCE`, `normalize=False → NONE`. In code, both values are routed through
`coerce_normalization_strategy()` (`src/calibrated_explanations/calibration/normalization_strategy.py:64-87`),
which returns `NormalizationStrategy.SCALE` for any value that is not already a
`NormalizationStrategy` member or a matching string — i.e. **any bool**.

The deprecation ledger's language ("Bool passthrough removed") implies this now errors; it
doesn't — it's silently accepted with **changed semantics**. A caller who relied on
`normalize=True` for COHERENCE now silently gets SCALE results with no signal that behavior
changed.

**Fix:** either raise `ValidationError` for bool input to `normalize=`, or explicitly map
`True`/`False` to their documented `NormalizationStrategy` values and emit a `DeprecationWarning`
if the bool form is being kept as a compatibility shim (in which case it should also appear as an
*active* row in `docs/migration/deprecations.md`, not a "removed" row).

### 4. Silent-fallback coercers violate the fallback-visibility policy (CONTRIBUTOR_INSTRUCTIONS.md §5)

Two coercion helpers silently substitute a default on any invalid input, with zero warning and
zero log line:

- `coerce_normalization_strategy()` (`normalization_strategy.py:64-87`): unrecognized string
  (e.g. typo `"simplx"`) → silently returns `SCALE`. Verified: 0 warnings across bool and typo
  inputs.
- `coerce_interval_summary()` (`src/calibrated_explanations/core/prediction/interval_summary.py:24-29`):
  any invalid value → silently returns `REGULARIZED_MEAN` (bare `except Exception`, marked
  `# adr002_allow`).

CONTRIBUTOR_INSTRUCTIONS.md §5 (Fallback Visibility Policy) mandates a `UserWarning` plus an INFO
log entry for every fallback decision. Neither coercer does this. Given the RC posture is
fail-fast on invalid input (see ADR-039 remediation, D3), the more consistent fix is likely to
raise `ValidationError` on unrecognized input rather than to add a warning — but that is a
product decision, not just a bug fix, and should be made explicitly rather than left as an
undocumented silent default.

---

## Medium severity

### 5. Global `warnings.filterwarnings()` mutation in `venn_abers.py`

`src/calibrated_explanations/calibration/venn_abers.py:115,151,247,345` call
`warnings.filterwarnings("ignore"/"default", category=RuntimeWarning)` at module/call scope,
outside a `catch_warnings()` context manager. This permanently rewrites the *process-wide*
warnings filter state on every calibration/prediction call. The damage is already documented
in-repo at `src/calibrated_explanations/core/reject/orchestrator.py:34-36`: it increments
`_filters_version` and clears all warning registries, which is why that module needed a
module-level "warn once" flag instead of relying on Python's normal deduplication. This is a
latent source of warning-visibility bugs anywhere else in the codebase that relies on
`__warningregistry__`-based deduplication, and should use `warnings.catch_warnings()` scoping
instead of global mutation.

### 6. Version string mismatch across metadata sources

- `src/calibrated_explanations/__init__.py:10`: `__version__ = "v1.0.0-rc-dev"`
- `pyproject.toml:10`: `version = "1.0.0rc-dev"` — PEP 440-normalizes to `1.0.0rc0.dev0`
  (`packaging.version.Version` confirms this).
- `CITATION.cff:67`: `version: v0.11.5` (stale — expected, since CITATION.cff is presumably
  bumped only at release, but flagging since RC will need it updated too).

The `v`-prefixed `__version__` string flows into plugin metadata as `package_version` at
`src/calibrated_explanations/plugins/registry.py:2300` (`raw_meta.setdefault("version", ...,
package_version)`), so any comparison against installed-package metadata
(`importlib.metadata.version(...)`, used elsewhere in the same file for entry-point discovery)
will mismatch on the `v` prefix. `v1.0.0-rc_plan.md` Task 6 / decision D1 already tracks aligning
the version string to `1.0.0rc1` before tagging — this finding confirms the current dev state has
exactly the inconsistency that task is meant to fix, plus flags the `package_version` consumer
that should be checked once the string changes.

### 7. `CONTRIBUTOR_INSTRUCTIONS.md` is factually wrong about the guarded kwarg's current behavior

Lines 36-37: `guarded=True` boolean kwarg "(emits `DeprecationWarning`; removed in v1.0.0)".

Actual state: removed in **v0.11.5** (already shipped, per CHANGELOG and deprecation ledger), and
it emits **nothing** — see Finding 1. Any agent primed with this file will believe the kwarg still
works today and will emit a warning when misused; neither is true. Needs correction as part of
closing Finding 1, per the CONTRIBUTOR_INSTRUCTIONS.md update-cadence rule ("Update this file
whenever the public API changes").

### 8. `docs/foundations/concepts/parameter-reference.md` is written in pre-Task-17 future tense for changes that already shipped

Lines 19-22, 116, 145-164 describe the `significance` → `GuardedOptions.confidence` rename and
`confidence` → `reject_confidence` rename as "Planned rename (Task 17)", and line 155 still lists
`explain_factual(guarded=True)` as a current call form. Both renames are already complete and the
old names are removed (v0.11.5). This file needs to move from "planned" to "current" framing, and
the `guarded=True` example should be removed or explicitly marked historical.

---

## Low severity / needs a decision

- **Ruff:** one auto-fixable `C420` violation on `main` at
  `tests/unit/core/test_calibrated_explainer_runtime_helpers.py:26` (unnecessary dict
  comprehension; `ruff --fix` resolves it). Pre-commit's full ruff hook will trip on this.
- **`ncf="entropy"` silent alias:** documented at `docs/migration/deprecations.md:291` as
  "remains accepted and is silently normalized to `ncf=\"default\"`". This is a *documented*
  silent alias, which sits awkwardly next to both the fallback-visibility policy (§5) and the RC
  plan's D5/D6 "zero active deprecations" gate. Needs an explicit decision: warn, raise, or record
  a formal exemption before the deprecation-closure gate is trusted as RC evidence.
- **Mypy:** 516 errors across 52 files package-wide (`python -m mypy src/calibrated_explanations`).
  Not a gate failure — `make deprecation-closure` / CI mypy step only type-checks
  `core/exceptions.py`, `core/validation.py`, `api/params.py` (see `scripts/local_checks.py:43-49`)
  — but worth knowing the true package-wide mypy debt before any RC claim about type-checking
  coverage.

---

## Verified clean

- Full `pytest` test suite: all green, no failures, no unexpected skips.
- `make deprecation-closure`: passes (952 passed in the focused lane; full-suite run also clean).
- `make capability-chain-check`: all 31 evidence files validate OK.
- **ADR-039 conditional-calibration fail-fast behaviors**, verified live through
  `WrapCalibratedExplainer`/`predict`/`explain_factual`:
  1. Conditional calibration (`bins=` at `calibrate`) + inference without `bins=` → raises
     `ValidationError` ("calibrated with Mondrian bins; pass bins=...").
  2. Global calibration + `bins=` at inference → raises `ConfigurationError`.
  3. Conditional happy path (matching `bins=` at both stages) → succeeds.
  4. Unseen bin label at inference → raises `ValidationError`.
  5. Bin-length mismatch → raises `DataShapeError`.
  6. Re-calibrating without `bins=`/`mc=` resets to global; a subsequent `bins=` at inference then
     correctly raises `ConfigurationError`; omitting `bins=` after reset succeeds.
  7. `pickle.dumps()` on a wrapper calibrated with an `mc=` `MondrianCategorizer` emits exactly one
     `UserWarning` ("Pickle/state persistence drops the configured Mondrian categorizer...");
     the reloaded wrapper then correctly requires explicit `bins=` and raises `ValidationError`
     without it.
- Structural removals confirmed genuinely gone (not just undocumented-but-present): `core.reject`
  module shim, `core.explain.explain(...)`, `ExplainerHandle.learner` (no `.learner` on
  `WrapCalibratedExplainer` at the class level — only present as a runtime instance attribute,
  which is expected/correct), `legacy_payload` (now private `_legacy_payload`), VennAbers/
  IntervalRegressor schema-v1 pickle loaders (raise `ConfigurationError`, only `schema_version: 2`
  accepted), `ParallelConfig(strategy="auto", enabled=True)` (raises), legacy `plot_kinds`
  category vocabulary (only semantic names accepted in `plugins/base.py`).
- Notebooks, examples, and the runnable doc snippets I checked (including
  `docs/foundations/how-to/tune_runtime_performance.md`, whose `context.explainer.learner` is a
  valid instance attribute, not the removed `ExplainerHandle.learner` property) are consistent
  with current source.

---

## Suggested next steps

1. Treat Findings 1-3 as release-blocking for `v1.0.0-rc` per `v1.0.0-rc_plan.md` D3/D5 — they are
   silent-correctness bugs in the exact surface Task 0 already targets. Extend Task 0's scope
   statement to explicitly cover `predict_reject`'s `**kwargs` and the `coerce_*` helpers
   (Findings 2, 4), which are not currently named in the Task 0 description.
2. Fix `_KNOWN_PUBLIC_KWARGS` in `wrap_explainer.py` (Finding 1, half 2) as a quick, isolated
   correction independent of the larger ADR-038 kwargs-gate closure.
3. Decide the intended behavior for bool `normalize=` on `VennAbers.predict_proba` (Finding 3) and
   for the `coerce_*` helpers (Finding 4): fail-fast `ValidationError`, or warn-and-fallback per
   §5. Update `docs/migration/deprecations.md` and docstrings to match whichever is chosen.
4. Correct `CONTRIBUTOR_INSTRUCTIONS.md:36-37` (Finding 7) and de-future-tense
   `parameter-reference.md` (Finding 8) in the same PR as the code fix, per the
   CONTRIBUTOR_INSTRUCTIONS.md update-cadence rule.
5. Roll the `warnings.filterwarnings()` global-mutation cleanup (Finding 5) and the ruff C420 fix
   into the same pass if convenient; neither is release-blocking on its own.
6. Confirm version-string unification (Finding 6) is fully covered by `v1.0.0-rc_plan.md` Task 6 —
   it appears to be, this is a confirmation, not a new task.
