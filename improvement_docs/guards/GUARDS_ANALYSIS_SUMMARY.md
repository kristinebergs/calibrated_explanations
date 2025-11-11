# Guards Expansion Analysis – Executive Summary

**Date:** November 11, 2025
**Status:** Analysis Complete

---

## Question 1: Does the Guards Expansion Need Improvements?

**Answer: YES – Critical improvements are needed.**

### Current State

The guards expansion has **solid architecture but incomplete execution**:

- ✅ Core classes exist: `BaseGuard` protocol, `ConformalRegionOracle` implementation
- ✅ Integration points defined: `_label_ctx()`, `_accept()`, `set_guard()` methods
- ✅ Pipeline hooks identified in implementation instructions
- ❌ **Integration is non-functional** – guards are instantiated but never called during explanation
- ❌ **Martingale e-test not implemented** (stubbed as TODO)
- ❌ **Performance optimizations missing** (S_total caching not done)
- ❌ **Formal guarantees undocumented** – no clarity on what "in-distribution" means for perturbations

### Key Finding

Searching the codebase reveals that `_accept()` and `_label_ctx()` methods are **defined but never invoked**:

```bash
$ grep -c "_accept(" src/calibrated_explanations/**/*.py
2  # Only in definition and a TODO comment

$ grep -c "_label_ctx(" src/calibrated_explanations/**/*.py
2  # Only in definition
```

**Impact:** Guards do not actually filter perturbations. The feature is **cosmetic only**.

---

## Question 2: In-Distribution Guarantees for Perturbed Instances

### Formal Problem

When generating synthetic instances `x'` for explanation, we want **high confidence** that `x'` is not OOD.

### Current Approach

The `ConformalRegionOracle` uses **split-conformal calibration**:

1. **Fit phase:** Cluster calibration data by label, compute Mahalanobis radius per cluster
2. **Check phase:** Accept `x'` if `||x' - nearest_cluster_center||²_Σ ≤ radius`

### What This Guarantees

- ✅ **Population coverage:** For random draws from training distribution, ~(1-α) fall in region
- ✅ **Conformal validity:** Guarantees are distribution-free (hold for any distribution)

### What This Does NOT Guarantee

- ❌ **Perturbation fidelity:** No guarantee that additive perturbations stay in-distribution
- ❌ **Method compatibility:** Uniform/Gaussian noise may have different properties than training data
- ❌ **Single-instance coverage:** Guarantees are marginal (average over population), not per-instance

### Critical Gap: Perturbations ≠ Random Samples

**Example:**

```
Calibration data: Gaussian cluster N(μ=0, σ=1)
Conformal radius: r = 2 (covers 95%)

Original point: x = 0 (well-centered)
Perturbation: x' = x + U(-0.5, 0.5) ∈ [-0.5, 0.5]

Question: Does perturbation stay in region?
- Mathematically: Yes, since |x'| ≤ 0.5 < 2
- But this ignores: Perturbations are NOT uniform random draws
  They are structured (relative to original) and may accumulate at boundaries

For many features, perturbations can exit the region despite being "close" to original.
```

### Recommended Guarantee Framework

**Adopt a two-tier approach:**

**Tier 1: Conformal Membership** (default, simpler)
- Method: Check `accept(x', label_ctx)`
- Guarantee: Marginal coverage ≥ (1-α)
- Overhead: O(n_clusters)

**Tier 2: Martingale E-Test** (advanced, stronger)
- Method: Likelihood ratio test on local k-NN density
- Guarantee: Anytime-valid test for OOD with controlled Type-I error
- Overhead: O(1) with precomputed stats
- Status: **TODO – not yet implemented**

**Perturbation Compatibility Audit:**
- Numeric features: Uniform/Gaussian noise ✓ compatible (bounded by data range)
- Categorical: Permutation ✓ compatible (within observed values)
- Transforms: ⚠️ Validate manually (may distort geometry)

---

## Question 3: Evaluation Framework – When Does Guarded CE Succeed vs Unguarded?

### Failure Modes

| Mode | Scenario | Detection |
|------|----------|-----------|
| **Sparse regions** | Perturbations land in low-density areas → spurious rules | Measure rule stability across repeats |
| **Boundary effects** | Near cluster boundaries, perturbations exit region | Track per-feature acceptance rate |
| **High dimensions** | Calibration set sparse → region too large | Compute coverage vs empirical density |
| **Label imbalance** | Minority class → loose constraints → more OOD | Stratify metrics by class |

### Proposed Metrics

1. **OOD Detection Rate**
   ```
   oob_rate = fraction of perturbations rejected by guard
   Expected: ~α (should equal target miscoverage)
   ```

2. **Explanation Stability** (key measure of success)
   ```
   Repeat explanation k times with different seeds
   overlap_unguarded = agreement between rules (unguarded)
   overlap_guarded = agreement between rules (guarded)
   stability_ratio = overlap_guarded / overlap_unguarded

   Expected: stability_ratio > 1.0 (guarded is more stable)
   ```

3. **Fairness Parity Across Classes**
   ```
   acceptance_rate_per_class should be similar
   Metric: max - min acceptance rate
   Expected: < 5 percentage points
   ```

4. **Computational Overhead**
   ```
   overhead = (time_guarded - time_unguarded) / time_unguarded
   Target: ≤ 20%
   ```

5. **Coverage Calibration**
   ```
   For different α values (0.01 to 0.2):
     estimated_coverage = 1 - α
     actual_coverage = empirical OOD rate on test set

   Plot (estimated vs actual)
   Expected: good calibration (near diagonal)
   ```

### Test Case Examples

**Test 1: Synthetic Mixture (Factual Mode)**
```python
# Generate 3 Gaussian clusters, ensure guard rejects inter-cluster perturbations
X, y = make_blobs(n_centers=3, n_samples=500)
guard = ConformalRegionOracle(alpha=0.1, n_clusters=3)

# Metric: unguarded rules should be less stable between cluster boundaries
stability_improvement = (stability_guarded - stability_unguarded) / stability_unguarded
assert stability_improvement > 0.2  # 20% improvement
```

**Test 2: COMPAS Fairness (Classification)**
```python
# Ensure guard doesn't discriminate across demographic groups
X, y, groups = load_compas()
acceptance_by_group = [guard.compute_oob_rate(X[groups==g]) for g in groups.unique()]

# Metric: acceptance rate disparity
assert max(acceptance_by_group) - min(acceptance_by_group) < 0.05
```

**Test 3: Housing Regression (Counterfactual Mode)**
```python
# For regression, identify which features are admissible (have non-empty intervals)
for feature_j in range(n_features):
    intervals = guard.intervals(x_test, label_ctx)[feature_j]
    if not intervals:
        print(f"Feature {j} is inadmissible (isolated point)")
    else:
        print(f"Feature {j} can vary in {intervals}")
```

### Diagnostic Tools (Proposed)

```python
class GuardDiagnostics:
    def compute_oob_rate(self, X_perturbed, y_ctx) -> float:
        """Fraction of perturbations rejected by guard."""

    def feature_admissibility(self, x, label_ctx) -> List[bool]:
        """For each feature, is interval non-empty?"""

    def compute_redundancy_ratio(self, X_train, y_train) -> float:
        """Coverage of conformal region vs empirical distribution."""

    def plot_region_2d(self, X_train, y_train, feature_pair=(0,1)):
        """Visualize conformal region boundaries."""
```

---

## Recommended Action Plan

### Phase 1: Complete Integration (v0.10.0)

1. **Wire guard into perturbation loop**
   - Find where single-feature perturbations are generated
   - Filter by `guard.intervals()` before adding to rule set
   - Add conjunction validation with `guard.accept()`

2. **Implement martingale e-test**
   - Implement missing `MartingaleETest` class
   - Wire into `ConformalRegionOracle.accept()`

3. **Optimize performance**
   - Cache S_total across features
   - Profile KDTree queries
   - Target: ≤ 20% overhead

4. **Add tests**
   - Unit tests for guard filters (synthetic data)
   - Integration tests with explanation pipeline
   - Benchmark tests (time + overhead)

### Phase 2: Formalize Guarantees (v0.11.0)

1. **Write ADR-028: Perturbation Guard Semantics**
   - Define "in-distribution" formally
   - State coverage guarantees with caveats
   - Document perturbation compatibility

2. **Implement evaluation framework**
   - Run synthetic benchmarks (mixture of Gaussians)
   - Real datasets (COMPAS, housing, high-dimensional)
   - Publish metrics and findings

3. **Add user documentation**
   - `docs/guards.md` – usage guide
   - When to use guards, interpretation, limitations

### Phase 3: Future Enhancements (Post-v1.0.0)

1. Make guards default (breaking change + deprecation path)
2. Full covariance support (not just diagonal)
3. Adaptive alpha selection
4. Feature domain enforcement (e.g., [0,1] bounds)

---

## Key Deliverables

1. **Analysis Document** ✅ (this file + detailed companion)
2. **Implementation Plan** – Wire guards into perturbation loop
3. **Test Suite** – Synthetic + real datasets with metrics
4. **Formal ADR** – ADR-028 on guard semantics
5. **User Guide** – `docs/guards.md` with examples

---

## Questions for Stakeholders

1. Should guards be **opt-in (default: off)** or **opt-out (default: on)**?
2. Is **90% coverage (α=0.1)** acceptable as default, or should users configure?
3. **Martingale e-test priority:** Critical for v0.10.0, or can defer to v0.11.0?
4. **Evaluation timeline:** Before or after pipeline integration?
5. **Real-world feedback:** Have production users reported issues with OOD perturbations?

---

## References

- **Code:** `src/calibrated_explanations/guards/` (195 LOC)
- **Integration:** `src/calibrated_explanations/core/calibrated_explainer.py` (lines 715–939, 1772–1803)
- **Instructions:** `src/calibrated_explanations/guards/implementation_instructions.md` (267 LOC)
- **Related ADRs:** ADR-008 (Domain Model), ADR-021 (Calibrated Intervals)
- **Release Plan:** `improvement_docs/RELEASE_PLAN_V1.md` (guards not yet listed)

---

**Analysis completed. Ready for next steps: design review, stakeholder feedback, or implementation planning.**
