# Guard Implementation Plan — Confidence-Modulated Conformal Regions

Date: 2025-11-15
Status: Draft implementation plan (actionable)

This document summarizes the gap analysis between the documented guard design (see `improvement_docs/guards`) and the code currently present in `src/calibrated_explanations/guards` and related modules. It provides a concrete, low-risk step-by-step plan to (1) remove legacy label-conditional code, (2) wire a GuardOrchestrator consistent with the existing orchestrator pattern used for explanations and predictions, and (3) ensure the guard implementation is discoverable and actively used by the `CalibratedExplainer` thin delegators.

## Executive summary of findings (gap analysis)

- The design documents under `improvement_docs/guards` (notably `GUARD_DESIGN_CONFIDENCE_MODULATION.md`) specify a single conceptually-complete implementation: `ConformalRegionOracle` using confidence-modulated conformal regions.
- The codebase already contains a usable `ConformalRegionOracle` implementation in `src/calibrated_explanations/guards/regions.py` that closely matches the design documents. Good: the core algorithm exists.
- Activation gap: `CalibratedExplainer` contains guard wiring (`self.guard`, `guard_params`, `__fit_guard`, `set_guard`) and also private helpers `_label_ctx` and `_accept`. However these are not consolidated using a Guard orchestrator and parts of the code still use legacy label-context flows (e.g., `label_ctx`) and conditional logic. The guard is therefore partly present but not orchestrated or consistently invoked.
- Dead/legacy code: multiple call sites compute or pass `label_ctx` (see `src/calibrated_explanations/explanations/explanation.py` and `src/calibrated_explanations/core/calibrated_explainer.py`). The design forbids label-conditional APIs; these must be removed.
- Duplication: there are multiple `_filter_perturbations_by_guard` implementations and inline guard logic inside `CalibratedExplainer` and `Explanation` modules. This should be routed through a single orchestrator to match the explanation/prediction design.

Consequence: the algorithm is ready but un-orchestrated. To meet the stated design constraints we should (1) create a GuardOrchestrator that owns runtime guard lifecycle and plugin-like resolution, (2) replace direct guard logic in high-level classes with thin delegators to the orchestrator, and (3) remove label-based conditional code.

---

## Concrete goals

1. Make `ConformalRegionOracle` the canonical guard implementation (keep `src/calibrated_explanations/guards/regions.py`).
2. Add a `GuardOrchestrator` (pattern consistent with `ExplanationOrchestrator` and `PredictionOrchestrator`) under `src/calibrated_explanations/guards/orchestrator.py` that:
   - Is instantiated from `CalibratedExplainer` during initialization.
   - Exposes a minimal public API that `CalibratedExplainer` and `Explanation` modules can call:
     - initialize_chains() / initialize_runtime_state() (if needed)
     - fit_guard(guard_params) / build_default_guard(guard_params)
     - set_guard(guard)
     - get_guard() -> guard or None
     - accept(x_new, calibrated_prediction=None) -> bool
     - accept_batch(x_new_batch, calibrated_predictions=None) -> np.ndarray
     - intervals(x_orig, calibrated_prediction=None) -> list of per-feature intervals
     - filter_perturbations(perturbed_x, perturbed_feature, x, prediction) -> (filtered_x, filtered_feature)
     - filter_candidates(feature_index, candidates, x_orig, calibrated_pred) -> candidates filtered
   - Can hold fallback chains if plugins ever exist; for now it simply wraps `ConformalRegionOracle`.
3. Replace `CalibratedExplainer` private guard helpers with thin delegators. `calibrated_explainer.py` should not implement guard logic itself — only delegate to the `GuardOrchestrator` (mirroring the explanation/prediction orchestrator pattern).
4. Remove `label_ctx` variables and the `_label_ctx` function. All code paths that use `label_ctx` must be updated to call the orchestrator's `accept` with the canonical `calibrated_prediction` tuple format (or with None).
5. Remove duplicate guard helper implementations in `CalibratedExplainer` and move them into orchestrator methods (e.g., filtering perturbations and candidate intervals).
6. Add unit and integration tests to verify:
   - ConformalRegionOracle.fit/accept/intervals behavior (unit)
   - GuardOrchestrator API and its thin delegators (unit)
   - End-to-end flow: `CalibratedExplainer` with a fitted guard actually filters perturbations during explanation generation (integration test).

---

## Design — small contract for GuardOrchestrator

Purpose: centralize guard lifecycle, plugin resolution, and runtime calls so that `CalibratedExplainer` contains only thin delegators.

Public inputs/outputs (minimal):
- Input shapes: x (1D or 2D numpy arrays), calibrated_prediction either None or (predict_value, (low, high)).
- Exceptions: orchestrator methods should raise descriptive RuntimeError/NotFittedError if called before guard is fitted or set.
- Success: accept returns bool or boolean array; intervals returns list-of-lists-of-(low, high) tuples.

Edge cases to cover:
- guard not configured: accept(...) -> True (no filtering) — maintain backward compatibility with current behaviour
- guard.set_guard(None): explicitly disable
- malformed calibrated_prediction: orchestrator handles gracefully and falls back to no modulation
- batch acceptance: vectorized accept for performance

Behavior guarantees:
- All heavy code (distance computations, covariance inversion, etc.) remains in `ConformalRegionOracle`.
- Orchestrator delegates to `ConformalRegionOracle` methods and controls modulation parameters.

---

## Files to create / edit (concrete edits)

Planned file changes (smallest set of edits, single-purpose diffs for each file):

1. Add: src/calibrated_explanations/guards/orchestrator.py (new)
   - Contains `GuardOrchestrator` class that wraps `ConformalRegionOracle`.
   - API: initialize_chains(), fit_guard(params), set_guard(guard), get_guard(), accept(...), accept_batch(...), intervals(...), filter_perturbations(...), filter_candidates(...)
   - Minimal internal state: _explainer ref, _guard (None or ConformalRegionOracle instance), _guard_params

2. Edit: src/calibrated_explanations/core/calibrated_explainer.py
   - In __init__: instantiate `self._guard_orchestrator = GuardOrchestrator(self)` (phase 1 orchestration like others)
   - Add thin delegators (mirror existing orchestrator delegators):
     - _fit_guard -> self._guard_orchestrator.fit_guard(self.guard_params)
     - set_guard(self, guard) -> delegate to orchestrator.set_guard(guard)
     - get_guard or property -> self._guard_orchestrator.get_guard()
   - Remove private methods `_label_ctx` and `_accept` (they are legacy). Replace any internal call sites to these with delegators to orchestrator methods.
   - Replace duplicated `__fit_guard()` behaviour with a call into orchestrator
   - Remove duplicate implementations of perturbation filtering and replace with orchestrator.filter_perturbations (or delegate to orchestrator.accept_batch)

   Rationale: keep explainer file thin; orchestrator owns guard lifecycle.

3. Edit: src/calibrated_explanations/explanations/explanation.py
   - Delete `label_ctx` computation and related temporary variables.
   - Replace all uses of `expl._accept(perturbed_row[0], label_ctx)` / `expl._accept(x_prime, label_ctx)` with `expl._guard_orchestrator.accept(x_prime, calibrated_prediction)` or use `expl._get_explainer()._guard_orchestrator.accept(...)` if needed.
   - Update `_predict_conjunctive` and `add_conjunctions` code paths to not compute label_ctx. Instead, pass calibrated prediction tuples where available (the code already has access to prediction tuples in many places). For backward compatibility, if the orchestrator is not set, treat as accept=True.
   - Remove redundant double-computation of label_ctx seen in multiple places.

4. Keep: src/calibrated_explanations/guards/regions.py
   - No functional change to algorithmic code. It becomes the algorithm module that the orchestrator imports.

5. Edit: src/calibrated_explanations/guards/__init__.py
   - Export the orchestrator in `__all__` if desired, but ideally keep `ConformalRegionOracle` exported and make orchestrator internal under `guards.orchestrator`.

6. Tests to add:
   - tests/unit/guards/test_regions.py — unit tests for `ConformalRegionOracle` behavior (fit/accept/intervals, modulation function, accept_batch).
   - tests/unit/guards/test_orchestrator.py — test `GuardOrchestrator` thin wrappers, verify `CalibratedExplainer` delegates to orchestrator.
   - tests/integration/guards/test_explainer_integration.py — small integration test: create a simple learner, fit explainer and guard, call `.explain()` and assert that perturbations were filtered (e.g., simulated by swapping guard to one that rejects and verifying rules reduced).

7. Documentation & release artifacts:
   - Update `docs` or README guard index to reference the orchestrator API (short note how to use: explainer.set_guard(guard) or pass `guard_params` into CalibratedExplainer init to auto-fit).
   - Add a short example notebook in `notebooks/` demonstrating regression guard usage without threshold.

---

## Example API sketches (no code edits here — for reviewers)

GuardOrchestrator public API (proposed methods):

- __init__(explainer)
- initialize_chains()  # if plugin/fallback chains are ever needed
- fit_guard(guard_params: dict) -> None  # creates and fits ConformalRegionOracle from params
- set_guard(guard: ConformalRegionOracle | None) -> None
- get_guard() -> Optional[ConformalRegionOracle]
- accept(x_new: np.ndarray, calibrated_prediction: Optional[tuple] = None) -> bool
- accept_batch(x_batch: np.ndarray, calibrated_predictions: Optional[Sequence[tuple]] = None) -> np.ndarray
- intervals(x_orig: np.ndarray, calibrated_prediction: Optional[tuple] = None) -> list
- filter_perturbations(perturbed_x, perturbed_feature, x, prediction) -> tuple(filtered_x, filtered_feature)
- filter_candidates(feature_index, candidates, x_orig=None, calibrated_pred=None) -> np.ndarray

CalibratedExplainer thin delegators (examples):

- In __init__: self._guard_orchestrator = GuardOrchestrator(self)
- set_guard(guard) -> self._guard_orchestrator.set_guard(guard)
- _fit_guard -> self._guard_orchestrator.fit_guard(self.guard_params)
- any internal guard accept: replace expl._accept(x_prime, label_ctx) with self._guard_orchestrator.accept(x_prime, calibrated_pred)

---

## Migration steps (ordered, small commits)

1. Create `GuardOrchestrator` (new file). Implement all methods but have them delegate to `ConformalRegionOracle`. Add unit tests for orchestrator and a couple of smoke tests for ConformalRegionOracle. (Small commit)

2. Wire orchestrator into `CalibratedExplainer.__init__`. Add thin delegator methods in `CalibratedExplainer` that call orchestrator. Keep existing `_label_ctx` and `_accept` unchanged for now (so repo stays green). Add tests asserting dispatch to orchestrator. (Small commit)

3. Update a couple of internal call sites in `explanations/explanation.py` to use orchestrator.accept instead of `_accept(label_ctx)`. Keep the old `_label_ctx`/`_accept` in place but mark them as deprecated with comments and warnings. Run tests and iterate until green. (Small commit)

4. Remove `_label_ctx` and `_accept` implementations from `CalibratedExplainer` and remove all `label_ctx` variables and computations from `explanation.py`. Replace any calls to the old API with orchestrator calls. Run tests; fix any fallout. (Small commit — possibly larger)

5. Remove duplicate guard helper functions in `CalibratedExplainer` (two copies of `__filter_perturbations_by_guard` exist). Move that logic into orchestrator.filter_perturbations and update callers. (Small commit)

6. Add the remaining unit and integration tests described above. Run full test suite. Fix any failing tests and lints. (Small commit)

7. Documentation updates, changelog and PR. Reference `GUARD_DESIGN_CONFIDENCE_MODULATION.md` and ADRs where appropriate. Finalize PR. (Final commit)

Notes on commit granularity: keep each step as a small, reviewable commit. Aim for < 5 file changes per commit, and follow the repo test-generation policy (append tests to existing nearest files where possible).

---

## Tests & quality gates

Quality gates to run before merging:
- Unit tests (new and existing): PASS
- Selected integration tests that exercise explanation generation with perturbations: PASS
- Lint: PASS (pylint/flake as used in project)

Testing matrix to add:
- Unit: `test_regions.py` — small synthetic dataset checks: fitted guard accepts in-distribution points and rejects out-of-distribution.
- Unit: `test_orchestrator.py` — orchestrator delegates correctly; `set_guard` and `fit_guard` behave as expected.
- Integration: `test_explainer_integration.py` — run explain_factual with and without guard, compare number of candidate perturbations considered or number of rules produced to assert filtering.

Benchmark note: the conformal radius & Mahalanobis computations are O(d^2) per cluster inversion cost; acceptable for typical feature counts. Add an optional benchmark test under `benchmarks/` if desired.

---

## Risk, rollback & backward compatibility

- Risk: removing `label_ctx` may break external code if anyone relied on the private `_label_ctx` API. But the user said no part of guard codebase has been released — we can remove old code.
- Rollback: changes are local to a few files. Revert commit(s) if unexpected breakage arises.
- Backwards compatibility policy: not required here — we remove legacy code and prefer clarity.

---

## Timeline & estimated effort

Estimated hours (approx, single developer):
- Create orchestrator & unit tests: 2–4 hours
- Wire into `CalibratedExplainer` and add delegators + tests: 1–2 hours
- Migrate `explanation.py` call sites and remove label_ctx: 1–2 hours
- Add integration tests & fix fallout: 1–2 hours
- Linting, docs, PR packaging: 1–2 hours

Total: ~6–12 hours (iterative, can be split into 3–5 PRs).

---

## Next steps (what I'll do next if you want me to implement this plan)

- Create `src/calibrated_explanations/guards/orchestrator.py` with the `GuardOrchestrator` skeleton and unit tests for it (step 1). Run unit tests locally.
- Wire the orchestrator into `CalibratedExplainer` as thin delegators (step 2).
- Migrate call sites and remove `label_ctx` in small, tested commits (steps 3–5).

If you approve, I will implement step 1 now (create orchestrator skeleton + unit tests) and run the unit tests. If you prefer, I can instead open a PR plan with the file diffs listed here.

---

## Appendix: places in the codebase that reference `label_ctx` and guard calls (search results)

- `src/calibrated_explanations/core/calibrated_explainer.py` — `_label_ctx`, `_accept`, `__fit_guard` (guard wiring)
- `src/calibrated_explanations/explanations/explanation.py` — many occurrences where `label_ctx` is set and passed into `_predict_conjunctive` and other guard checks. These call sites need to be updated to use `GuardOrchestrator` accept/accept_batch/intervals.

(Use global search `label_ctx` across repository to find all remaining occurrences before deletion.)

---

Completion summary

This plan converts the current partially-present guard code into a fully orchestrated, consistent solution matching the repository's orchestrator pattern. It preserves the existing algorithm (`ConformalRegionOracle`) and centralizes runtime logic, cleaning up old label-conditional code and making the guard active and testable.

If you want me to proceed with implementation, say "Proceed: implement orchestrator" and I'll create the orchestrator skeleton and the first set of unit tests and wire the orchestrator into `CalibratedExplainer` (small commits, tested).
