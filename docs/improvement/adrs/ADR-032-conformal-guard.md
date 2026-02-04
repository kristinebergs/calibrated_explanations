# ADR-032 — Conformal Guarding for Explanations

## Context

Explanations can over-sample unrealistic feature values when perturbing an instance. To limit explanations to values supported by the calibration distribution, we introduce a conformal guard that computes per-instance conforming ranges and constrains rule generation.

## Decision

- `CalibratedExplainer` accepts an optional `conformal_guard` parameter (boolean or configuration dictionary) that is passed through the explanation plugin system.
- Conformal guard computations are performed within the explanation plugin lifecycle and do not require model predictions for the tree-based guard.
- The tree guard provides per-feature marginal plausibility rather than joint/conditional coverage.
- When enabled, the explanation plugins compute conformal outer limits per instance and feature, and store them alongside rule metadata.
- Conformal guard uses a tree-based plausibility filter to generate per-instance candidate points for numerical features. Candidates are deterministic per seed and cached per instance/feature.
- Factual explanations skip features whose observed values are non-conforming.
- Perturbation generation can be limited to conforming ranges when `use_for_perturbation=True`.
- Rule conditions are expressed as conjunctions of discretizer conditions and conformal outer limits.
- High-cardinality categorical features are capped via `max_category_values` to bound compute cost.

## Consequences

- Explanation rules become more conservative and may exclude features that fall outside conformal ranges.
- Rule metadata carries conformal intervals or conforming categorical values for downstream serialization and display.
- When the tree-based guard is enabled, rule metadata includes candidate points and `tree_used`/`fallback` flags to explain whether the guard produced the candidates.
- Additional conformal computations add overhead, but results are cached for reuse.

## Tree Guard Contract

The tree-based conformal guard generates candidate grids per numeric feature:

- **Input:** `(instance, feature_idx)`
- **Output:** `candidate_points` (1D numeric array, deterministic for the same seed)
- **Constraints:** Bounded by `candidate_grid`, includes the observed value, and is cached per instance + feature.
- **Fallbacks:** If a feature tree or leaf samples are unavailable, the guard falls back to global percentiles and marks `fallback=True`.
- **Serialization:** Metadata must be JSON-serializable (`candidate_points` lists, booleans, strings) to support downstream storage and telemetry.

## Determinism & Thread Safety

- Candidate generation uses deterministic per-instance seeds derived from the explainer seed, feature index, and instance payload.
- Tree fits and candidate caches are guarded by locks and are safe to access from parallel explanation runs.
- PredictBridge is retained for interface compatibility but is not used by the tree guard today.

## Telemetry / Logging

When the tree guard is enabled, the guard logs:

- Tree-based guard usage (tree or fallback)
- Feature index
- Candidate count
- `tree_used`/`fallback` status

These signals are surfaced in explanation metadata to help downstream consumers reason about rule coverage and failure modes.
