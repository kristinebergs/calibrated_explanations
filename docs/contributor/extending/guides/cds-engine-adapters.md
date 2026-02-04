# Tree-Conformal Guard Adapters

This guide documents how to extend the Tree-Conformal Plausibility Filter used by
`ConformalGuard`. The guard fits per-feature trees and computes conformal
p-values over leaf-level empirical distributions.

## Contract

An adapter or extension must:

- Accept `(instance, feature_idx)` and return JSON-serializable metadata:
  `intervals`, `candidate_points`, and `values` for categoricals.
- Be deterministic for the same seed and inputs.
- Limit `candidate_points` to `candidate_grid` and include the observed value.
- Be safe to call in parallel and cached per instance + feature.
- Ensure adapter functions and model handles are picklable when used in
  process-based workers.

## Integration points

1. **ConformalGuardConfig**
   - Add new configuration parameters (e.g., leaf statistics or candidate
     synthesis settings).
2. **TreeConformalGuard._fit_feature**
   - Fit per-feature models and compute calibration nonconformity scores.
3. **TreeConformalGuard._compute_feature_meta_for_instance**
   - Generate candidate points and intervals, then compute p-values against the
     calibration score distribution.
4. **Plugin metadata**
   - Ensure metadata fields are JSON-serializable lists/tuples and booleans.

## Example adapter sketch

```python
class MyLeafSampler:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample_candidates(self, leaf_samples: np.ndarray, observed: float, grid: int) -> np.ndarray:
        candidates = np.quantile(leaf_samples, np.linspace(0.0, 1.0, grid))
        if not np.any(np.isclose(candidates, observed)):
            candidates = np.concatenate(([observed], candidates))
        return np.unique(candidates)
```

## Validation checklist

- ✅ Candidates are deterministic for the same seed.
- ✅ Metadata is JSON-serializable (`json.dumps` succeeds).
- ✅ Candidates are clipped to calibration min/max bounds.
- ✅ Tests cover fallback behavior when leaf samples are empty.
