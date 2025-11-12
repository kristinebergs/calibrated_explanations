# Guards Workplan — Small, Manageable Tasks

Date: 2025-11-11
Owner: perturbation_guard branch
Status: Draft (actionable)

This workplan breaks the guards effort into small, self-contained tasks sized for quick PRs (typically 30–120 minutes). Each task lists why, steps, acceptance criteria, and code locations. Tackle them in order within each phase. Keep diffs atomic (≤ 3 files) as per repo guidance.

Legend:
- Effort: S (≤ 1h), M (1–2h)
- Files use backticks; functions shown where relevant

## Phase 1 — Make Guards Functional (Integration + Correctness + Perf)

### 1. Wiring guards into the perturbation loop

G-01 Compute label context in explain paths (S)
- Why: interval filtering and acceptance checks require `label_ctx`.
- Steps:
  - In `CalibratedExplainer.explain` path, compute `label_ctx = self._label_ctx(x[i])` once per instance or batch (choose per existing batching pattern) and carry alongside.
  - If batching, consider per-row `label_ctx` array.
- Acceptance:
  - Unit print/log or debugger shows non-None `label_ctx` when `guard` is set.
  - No behavioural change when `guard=None`.
- Files: `src/calibrated_explanations/core/calibrated_explainer.py`

G-02 Pass `x` and `label_ctx` to value samplers (S)
- Why: `__get_greater_values/__get_lesser_values/__get_covered_values` contain guard filtering only if `x` and `label_ctx` are provided.
- Steps:
  - Update all call sites in `calibrated_explainer.py` to pass the current row `x[i]` (or slice) and `label_ctx`.
- Acceptance:
  - Calls compile and run; guard branches inside samplers execute when `guard` is set.
- Files: `core/calibrated_explainer.py` (callers around lines ~2570–2760, ~2657/2698/2739)

G-03 Activate interval filtering in samplers (no-op change) (S)
- Why: Confirm the existing filters engage.
- Steps:
  - No code change inside the three samplers; they already filter when args are provided. Add a small comment and keep behaviour.
- Acceptance:
  - With a CRO configured to be strict, candidate lists shrink vs baseline.
- Files: `core/calibrated_explainer.py` (`__get_*` methods ~3388–3432)

G-04 Conjunction acceptance check (M)
- Why: Multi-feature combinations can leave the valid region even if each 1D move was admissible.
- Steps:
  - After constructing a combined point (conjunction) but before including the rule, call `self.guard.accept(x_conj, label_ctx)` and skip if False.
  - If conjunctions are built outside core, add a small adapter in the core where the combined point is materialized or exposed.
- Acceptance:
  - Guarded runs may drop some conjunctions; unguarded behaviour unchanged when `guard=None`.
- Files: likely `core/calibrated_explainer.py` (conjunction builder), optionally `guards/conjunctions.py`

### 2. Interval quality and performance

G-05 Intersect intervals with feature domain (S)
- Why: Prevents proposing values outside valid feature ranges.
- Steps:
  - In `guards/regions.py:intervals`, after computing [low, high], intersect with known bounds (e.g., global min/max from training `X` stored at fit-time). Store per-feature bounds at `fit`.
- Acceptance:
  - Intervals clipped to data-supported ranges; empty intervals possible and handled by samplers (skip feature).
- Files: `guards/regions.py`

G-06 Merge overlapping intervals (S)
- Why: Fewer tiny overlapping ranges → faster, more stable sampling.
- Steps:
  - Use `guards/intervals.py:union_intervals` before returning per-feature intervals.
- Acceptance:
  - Overlapping intervals are merged; unit test with crafted overlaps.
- Files: `guards/regions.py`, `guards/intervals.py`

G-07 Cache S_total across features (M)
- Why: `intervals()` recomputes S per feature/cluster; caching cuts work from O(d^2) to O(d).
- Steps:
  - For each cluster, compute `S_total = Σ_i ((x_i-μ_i)^2/σ_i^2)` once, then for feature `j` use `S = S_total - ((x_j-μ_j)^2/σ_j^2)`.
- Acceptance:
  - Same intervals as before; micro-benchmark shows fewer ops or time drop.
- Files: `guards/regions.py`

### 3. Martingale e-test correctness (keep optional)

G-08 Fix e-value and threshold semantics (S)
- Why: Current `reject = exp(-d_k) > gamma` never rejects with default `gamma=10`.
- Steps:
  - Choose a monotone mapping where larger distances increase evidence for OOD and compare against a reasonable threshold (e.g., `reject if e_value >= gamma`, with e_value growing with distance). Alternatively compare distance to a fitted baseline quantile.
  - Update defaults so the test can actually trigger (documented).
- Acceptance:
  - Unit test where a far-away point is rejected while an inlier is not.
- Files: `guards/martingale.py`, `guards/regions.py (accept)`

G-09 Wire `use_martingale` through accept (S)
- Why: Ensure accept checks the e-test only when enabled.
- Steps:
  - Keep current pattern; validate config path; add docstring.
- Acceptance:
  - Toggle works; off-by-default preserves behaviour.
- Files: `guards/regions.py`

## Phase 2 — Guarantees, Evaluation, and Docs

### 4. Tests and evaluation basics

G-10 Unit: CRO accept/intervals on synthetic diagonal Gaussians (S)
- Steps: Generate small 2D clusters; assert accept True near centers, False far; intervals non-empty near x.
- Files: `tests/unit/guards/test_regions.py`

G-11 Unit: Sampler filtering (S)
- Steps: With a strict CRO, ensure `__get_*` drop out-of-interval candidates when `x,label_ctx` passed; untouched when `guard=None`.
- Files: `tests/unit/core/test_guard_sampling.py`

G-12 Integration: Guarded vs unguarded perturbation counts (S)
- Steps: Run explain on a tiny dataset; assert fewer candidates and identical API; time overhead within budget.
- Files: `tests/integration/guards/test_pipeline.py`

### 5. Documentation and ADR

G-13 ADR-028 skeleton (S)
- Steps: Create ADR file defining guard semantics, constraints, guarantees, and caveats about structured perturbations.
- Files: `docs/appendices/adr/ADR-028-guard-semantics.md`

G-14 User guide `docs/practitioner/guards.md` (M)
- Steps: Usage examples; configuration; limitations; metrics to monitor.

### 6. Diagnostics and metrics

G-15 Add `GuardDiagnostics` helper (S)
- Steps: Provide `compute_oob_rate`, `feature_admissibility`, simple 2D plot util for regions.
- Files: `src/calibrated_explanations/guards/diagnostics.py`

G-16 Minimal evaluation script (S)
- Steps: Script for synthetic 2D mixture reporting oob rate, stability proxy and timing.
- Files: `evaluation/guards/synthetic_basics.py`

## Phase 3 — Enhancements (Optional, Post-v1.0)

G-17 Full covariance experimental flag (M)
- Steps: Allow covariance="full" with regularization; document cost.

G-18 Adaptive alpha (M)
- Steps: Basic heuristic (calibration curve match) to tune alpha; optional.

G-19 Feature-domain policies (S)
- Steps: Configurable per-feature bounds/whitelist for categoricals; intersect in `intervals()`.

---

## Working notes
- Keep `guard=None` as no-op. All changes must be inert when guard isn’t provided.
- Prefer small PRs: 1–3 files each, with a focused unit test.
- Benchmarks: target ≤ 20% overhead vs unguarded for typical tabular shapes.
- Known limitation: Conformal membership provides marginal coverage; structured perturbations are not i.i.d. — document clearly in ADR and docs.
