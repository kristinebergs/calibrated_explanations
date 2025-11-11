# In-Depth Analysis: Guards Expansion for Calibrated Explanations

**Date:** November 11, 2025
**Status:** Analysis Phase
**Related ADRs:** ADR-008 (Domain Model), ADR-009 (Preprocessing), ADR-021 (Calibrated Interval Semantics)

---

## Executive Summary

The guards expansion introduces **perturbation guards**—optional in-distribution filters for calibrated explanations. The goal is to ensure that synthetic instances used during explanation generation remain faithful to the data distribution, thereby avoiding spurious rules that exploit OOD artifacts.

### Current State
- ✅ Core infrastructure (protocol, `ConformalRegionOracle`, integration slots) exists
- ✅ Basic split-conformal calibration logic is in place
- ⚠️  **Integration incomplete**: guards are instantiated but not actually used during explanation
- ⚠️  **Theoretical gaps**: no formal in-distribution guarantees for perturbed instances
- ⚠️  **Evaluation absent**: no systematic framework to detect failure modes

### Critical Gaps
1. **Pipeline integration** – Guards exist but don't filter perturbations in the explanation loop
2. **Formal guarantees** – Unclear what "in-distribution" means for perturbations under conformal logic
3. **Evaluation design** – No methodology to show when unguarded CE fails vs guarded CE succeeds
4. **Martingale e-test** – Stubbed but not implemented

---

## Part 1: Current Guards Architecture

### 1.1 Design Overview

The guards expansion follows a **label-conditional clustering + conformal calibration** approach:

```
┌─────────────────────────────────────────────────────┐
│ Training Phase: ConformalRegionOracle.fit(X, y)    │
├─────────────────────────────────────────────────────┤
│ 1. Compute label context: y_ctx = f(y)             │
│    - Classification: y_ctx = y                      │
│    - Regression: y_ctx = 1{y >= threshold}         │
│                                                     │
│ 2. For each label k:                               │
│    a. Subset X_k = X[y_ctx == k]                   │
│    b. Fit K KMeans clusters, get centers μ_{k,j}   │
│    c. Compute covariances Σ_{k,j} (diagonal)       │
│    d. Score each point: d_i = ||x_i - μ_j||²_Σ    │
│    e. Split-conformal calibration:                 │
│       - Calibration set: first 50% of scores       │
│       - Radius: r_k = quantile(cal_scores, 1-α)   │
│                                                     │
│ 3. Build KDTree on cluster centers for fast lookup │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Explanation Time: Filtering Perturbations          │
├─────────────────────────────────────────────────────┤
│ 1. label_ctx ← guard.label_context(x)              │
│ 2. For each feature j:                             │
│    a. intervals ← guard.intervals(x, label_ctx)[j] │
│    b. Filter candidates to fall within intervals   │
│ 3. For combined rules (conjunctions):              │
│    accepted ← guard.accept(x_conj, label_ctx)      │
│    if not accepted: skip this conjunction          │
└─────────────────────────────────────────────────────┘
```

### 1.2 Core Components

#### ConformalRegionOracle

**Location:** `src/calibrated_explanations/guards/regions.py` (195 lines)

```python
class ConformalRegionOracle:
    def __init__(self, alpha=0.1, mode="clf", threshold=None,
                 n_clusters=5, covariance="diag", random_state=None,
                 use_martingale=False, e_gamma=10.0, e_knn=30, e_neigh=500)
```

**Attributes:**
- `alpha` – Conformal miscoverage rate (default 0.1 = 90% coverage)
- `mode` – "clf" (classification) or "reg" (regression)
- `threshold` – For regression: threshold to convert to binary classification
- `n_clusters` – Number of KMeans clusters per label context
- `covariance` – "diag" (diagonal) or full (not yet implemented)
- `use_martingale` – Enable optional martingale e-test (default False)

**Methods:**

1. **`fit(X, y)`** – Build label-conditional regions
   - Stratified by `y` (or binarized for regression)
   - Per-label KMeans clustering
   - Split-conformal calibration of Mahalanobis radii

2. **`label_context(x, *, clf_predict_proba=None, reg_predict=None)`**
   - Returns predicted label (argmax for classification, thresholded for regression)
   - Used to select which region to check

3. **`intervals(x, label_ctx)`**
   - Returns per-feature allowed 1D intervals under the Mahalanobis ball
   - Solves quadratic constraint for each cluster:
     ```
     Let S = Σ_{i≠j} ((x_i - μ_i)² / σ_i²)
     If S > r²: no interval from this cluster
     Else: x_j ∈ [μ_j - √(D), μ_j + √(D)] where D = (r² - S)·σ_j²
     ```
   - Returns union of intervals across all clusters

4. **`accept(x_prime, label_ctx)`**
   - Fast membership check: Is `||x_prime - nearest_μ||²_Σ ≤ r²`?
   - If `use_martingale=True`: also applies e-test (currently a TODO)

#### Integration in CalibratedExplainer

**Location:** `src/calibrated_explanations/core/calibrated_explainer.py` (lines 715–939, 945–953, 1772–1803)

- **Init parameters:** `guard=None, guard_params=None`
- **Helper methods:**
  - `_label_ctx(x)` – Compute label context
  - `_accept(x_prime, label_ctx)` – Check acceptance
  - `set_guard(guard, guard_params)` – Instantiate and fit guard
  - `_update_guard()` – Re-fit guard (used during preprocessing)

**Current integration flow:**
```python
# In __init__:
self.set_guard(guard, guard_params)  # Instantiates and fits if data available

# Helpers exist:
def _label_ctx(self, x):
    if self.guard is None: return None
    return self.guard.label_context(x, ...)

def _accept(self, x_prime, label_ctx):
    return True if self.guard is None else self.guard.accept(x_prime, label_ctx)
```

---

## Part 2: Does It Need Improvements?

### **Answer: Yes, critical improvements are needed.**

#### 2.1 Integration Gaps

**Finding:** The helpers `_label_ctx()` and `_accept()` are **defined but never called** in the codebase.

```bash
$ grep -n "_accept\|_label_ctx" src/calibrated_explanations/core/calibrated_explainer.py
945:    def _label_ctx(self, x):
952:    def _accept(self, x_prime, label_ctx):

# No calls to these methods found in explain pipeline
```

**Impact:** Guards are instantiated and fit but never actually filter perturbations. The feature is non-functional.

**Improvement required:** Integrate guard filtering into the perturbation generation pipeline.

---

#### 2.2 Theoretical Gaps: What "In-Distribution" Means for Perturbations

**Problem:** The current design assumes that if a perturbed point `x'` is within the Mahalanobis ball of a cluster center (under the calibration-set-estimated covariance), it is "in-distribution." However, there are several subtle issues:

##### Issue 2.2.1: Conformal Calibration Guarantees Don't Directly Apply to Perturbations

**Current Logic:**
```
1. Fit regions on calibration set X_cal
2. Compute radius r_k = quantile_{1-α}(scores on X_cal)
3. Accept x' if ||x' - μ_k||²_Σ ≤ r_k
```

**What this guarantees:**
- For a random point drawn from the same distribution as X_cal, the ball covers ≥ (1-α) of the population in expectation
- This is a **population-level guarantee**

**What it does NOT guarantee:**
- That a **perturbation of an in-distribution point** stays in-distribution
- That the perturbation method (uniform/gaussian noise) samples uniformly or densely within the ball
- That the ball captures the actual conditional distribution P(X | Y = y_ctx)

##### Issue 2.2.2: Perturbation Methods Have Implicit Distributions

When generating perturbed instances, calibrated_explanations uses:
- **Categorical:** permutation (samples uniformly from observed categories)
- **Numeric:** uniform or gaussian noise added to the calibration set (replicates × scale_factor)

**Problem:** These perturbations are **not constrained to stay within the conformal region**. A uniform perturbation can move far outside the learned radius.

**Example:**
```
Original: x = [0.5] (1D)
Cluster center: μ = 0.5, σ = 0.1, r = 0.3
Allowed interval: [0.2, 0.8]

Uniform perturbation with severity=0.5:
  x' = 0.5 + U(-0.25, 0.25) ∈ [0.25, 0.75]  ✓ Mostly in-distribution

But the perturbation generates m points per original, and some may land outside [0.2, 0.8]
```

##### Issue 2.2.3: Split Conformal Assumes IID Data

**Problem:** If calibration data has systematic biases or label imbalance, the conformal radius may:
- Overestimate safe regions (if calibration set has outliers)
- Underestimate safe regions (if calibration set is too clean)
- Be misaligned with the actual conditional distribution

---

### 2.3 Missing Martingale E-Test Implementation

**Current state in `regions.py:189-194`:**
```python
# TODO: implement martingale if use_martingale
if self._martingale is not None and self._martingale.reject(x_prime):
    return False
return True
```

**Purpose:** The martingale e-test provides an anytime-valid test for OOD detection based on local density in a k-NN neighborhood. This would complement the conformal region check.

**Issue:** The test is never initialized or used.

---

### 2.4 Missing Performance Optimizations

From `implementation_instructions.md`:
> Cache per-instance `S` constants across features: reuse `S_total = Σ_i ((x_i-μ_i)^2/σ_i^2)`. For feature `j`, `S = S_total − ((x_j-μ_j)^2/σ_j^2)`.

**Current state:** Each call to `intervals(x, label_ctx)` recomputes S for each feature and each cluster.

**Cost:** O(n_clusters × n_features²) instead of O(n_clusters × n_features)

---

### 2.5 No Feature Domain Enforcement

**Issue:** `intervals()` returns mathematical bounds but doesn't intersect with actual feature ranges (e.g., [0, 1] for normalized features).

**Example:**
```
Feature j ∈ [0, 1] (normalized)
Allowed interval from guard: [-0.3, 1.5]
Should be intersected to [0, 1]
```

---

### 2.6 Incomplete Categorical Handling

**Issue:** For one-hot encoded features, the current implementation doesn't:
- Enforce that exactly one category is active per group
- Handle unseen categories gracefully
- Provide guidance on which categories are admissible

---

## Part 3: In-Distribution Guarantees for Perturbed Instances

### 3.1 Formal Problem Statement

**Goal:** Define and enforce that perturbed instances used in explanations remain faithful to the training data distribution.

**Definition (informal):** An instance `x'` is **admissible** if there is high confidence that `x'` is not an outlier or OOD point under the conditional distribution `P(X | Y = label_context(x))`.

---

### 3.2 Current Guarantee (Split Conformal)

**Assumption:** Training data is IID from some distribution P(X, Y).

**Conformal region construction:**
1. Compute nonconformity score for each calibration point: `α_i = d(x_i, nearest_cluster_under_y_i)`
2. Set radius: `r_k(α) = quantile_{⌈(n+1)(1-α)⌉ / n}(α_i)`
3. The region is: `R_k = {x : d(x, nearest_cluster_under_k) ≤ r_k}`

**Guarantee:** For any new point `x_new` sampled from the same distribution P(X, Y):
$$\Pr_{x_new \sim P}[\text{x_new} \in R_{y_new} \mid y_new] \geq 1 - \alpha$$

**Caveat:** This is a **marginal coverage guarantee** – it applies to the population, not a single instance.

---

### 3.3 Extension to Perturbations: What We Need

When generating perturbed instances `x_1', x_2', \ldots, x_m'` from a calibration point `x`, we want:

**Proposition:** If `x ~ P(X | Y = k)` and we perturb it additively, the perturbed instances should remain in `R_k` with high probability.

**Formal statement (candidate):**
Let `x ~ P(X | Y = k)` be an in-distribution point under label k.
Let `x' = x + \delta` where `δ ~ Perturbation_method(x)` (e.g., uniform or gaussian).
Then: `\Pr[x' \in R_k] ≥ f(α, \delta_max)` where `f` is some function of `α` and the maximum perturbation magnitude.

**Challenge:** This requires understanding the **joint distribution of X and the perturbation magnitude**, which is not provided by split conformal alone.

---

### 3.4 Proposed Guarantee Hierarchy

#### Level 1: Conformal Membership Check (Weakest)
- Check: Is `x' ∈ R_k`?
- Guarantee: Population-level coverage (applies to average over many perturbations)
- Overhead: O(n_clusters)

#### Level 2: Density-Based Check (Moderate)
- Check: Is `x'` in a high-density region under the empirical distribution of X_cal[Y=k]?
- Method: k-NN distance or kernel density estimate
- Guarantee: x' is similar to observed training points
- Overhead: O(n_cal log n_cal) with tree index

#### Level 3: Martingale E-Test (Strong)
- Check: Is the likelihood ratio under the conformal model acceptably low (e < e_gamma)?
- Guarantee: Anytime-valid sequential test with controlled Type-I error
- Method: Likelihood ratio between null (ID) and alternative (OOD) hypothesis
- Overhead: O(1) with precomputed statistics
- Current state: **TODO**

#### Level 4: Perturbation-Specific Bounds (Ideal but complex)
- Compute: max deviation `δ_max` under the perturbation method
- Check: Can the radius "absorb" the perturbation?
- Guarantee: Deterministic bounds on where perturbations can land
- Challenge: Requires analysis of perturbation method + data distribution

---

### 3.5 Recommended Formal Approach

**Adopt a two-tier guarantee:**

**Tier A (Default):** Conformal region membership
- Require: `accept(x', label_ctx)` returns True
- Guarantee: Marginal coverage ≥ (1-α)
- Rationale: Simple, interpretable, aligns with calibrated_explanations philosophy

**Tier B (Advanced):** Martingale e-test
- Require: `use_martingale=True` and e-value < e_gamma
- Guarantee: Anytime-valid test for OOD; controls Type-I error
- Rationale: Provides defense against adversarial or highly skewed data

**Perturbation-specific handling:**
1. Document that guards assume perturbations are **compatible** with the conformal model
   - Uniform/gaussian noise on numeric features: ✓ reasonable
   - Permutations on categorical: ✓ within observed space
   - Feature transformations: ⚠️  validate with domain expert

2. Add a diagnostic method: `guard.compute_oob_rate(perturbed_instances, label_ctx)` → fraction of perturbations outside conformal region → user can adjust α or perturbation magnitude

---

## Part 4: Evaluation Design – Detecting Failure Modes

### 4.1 Research Question

**When does unguarded CE fail (produce spurious rules) while guarded CE succeeds?**

### 4.2 Failure Mode Taxonomy

#### Failure Mode A: OOD Perturbations in Sparse Regions

**Scenario:**
- Calibration set has low density in feature space
- CE generates perturbations that land in sparse regions
- Rules learned from sparse regions are unstable and spurious

**Detection:**
- Generate multiple repetitions of same explanation
- Measure rule overlap (Jaccard similarity)
- Metric: `stability_improvement = (overlap_guarded - overlap_unguarded) / overlap_unguarded`

#### Failure Mode B: Cluster Boundary Effects

**Scenario:**
- A test point is near the boundary of a cluster
- Perturbations land outside the cluster
- CE learns rules that depend on cluster boundary artifacts

**Detection:**
- Measure fraction of perturbations accepted by guard
- Low acceptance rate ⟹ feature is inadmissible
- Check if excluded features correlate with spurious rules

#### Failure Mode C: Dimensionality Concentration

**Scenario:**
- High-dimensional data: calibration set is sparse
- Conformal region is huge relative to actual density
- Many perturbations are far from observed data

**Detection:**
- Compute coverage of conformal region relative to empirical distribution
- Compare to random baseline: what fraction of random draws from training set are in region?
- Metric: `redundancy_ratio = |region| / |empirical_support|`

#### Failure Mode D: Label Imbalance

**Scenario:**
- Minority class has few calibration samples
- Conformal radius is large for minority class
- Perturbations of minority examples are less constrained

**Detection:**
- Stratify metrics by label/class
- Compare acceptance rates across classes
- Flag classes with low acceptance rate as problematic

---

### 4.3 Proposed Evaluation Framework

**Setup:**

```python
# Synthetic benchmark: known ground truth
# e.g., mixture of Gaussians with known clusters

synthetic_data = generate_mixture_of_gaussians(
    n_components=3,
    n_samples=1000,
    separation=2.0,  # how well-separated are clusters?
    contamination=0.1  # inject 10% OOD points
)

# Real benchmark: challenging datasets
# e.g., high-dimensional, imbalanced, or sparse
real_data = [
    ("adult_income", adult_df, "income"),
    ("compas", compas_df, "recidivism"),
    ("moons", make_moons(n_samples=500)),
    ("blobs_sparse", make_blobs(n_samples=500, n_features=20, centers=5)),
]
```

**Metrics:**

1. **OOD Detection Rate**
   ```
   oob_rate = fraction of perturbations rejected by guard
   Expected: ~α (e.g., 0.1 for α=0.1)
   If oob_rate >> α: guard is too strict (too few features admissible)
   If oob_rate << α: guard is too loose (not actually filtering OOD)
   ```

2. **Explanation Stability**
   ```
   For each test instance x:
     rules_unguarded = explain(x, guard=None)
     rules_guarded = explain(x, guard=guard)

     # Repeat k times with different random seeds
     overlap_unguarded_k = agreement between rules_unguarded[1:k]
     overlap_guarded_k = agreement between rules_guarded[1:k]

   stability_ratio = overlap_guarded_k / overlap_unguarded_k
   Expected: > 1.0 (guarded is more stable)
   ```

3. **Fairness Parity Across Classes**
   ```
   For each class c:
     acceptance_rate_c = fraction of perturbations accepted for class c

   acceptance_disparity = max_c acceptance_rate_c - min_c acceptance_rate_c
   Expected: small (guards should treat all classes similarly)
   ```

4. **Computational Overhead**
   ```
   time_unguarded = runtime of explain(x, guard=None) [ms]
   time_guarded = runtime of explain(x, guard=guard) [ms]

   overhead = (time_guarded - time_unguarded) / time_unguarded [%]
   Expected: ≤ 20% (goal from implementation_instructions)
   ```

5. **Coverage vs Accuracy Trade-off**
   ```
   For various α values (0.01, 0.05, 0.1, 0.2):
     coverage_estimate_α = compute from guard
     actual_coverage_α = empirical OOD rate on test set

   Plot: (coverage_estimate, actual_coverage) across α
   Expected: good calibration (points near diagonal)
   ```

---

### 4.4 Concrete Test Cases

#### Test Case 1: Synthetic Mixture – Factual Explanations

```python
def test_guards_synthetic_factual():
    """
    Generate 3 Gaussian clusters, train classifier,
    explain on test points.

    Expected: Guarded CE should reject perturbations
    that land between clusters.
    """
    X, y = make_blobs(n_samples=500, n_features=5, centers=3)

    clf = RandomForestClassifier().fit(X, y)

    # Unguarded explainer
    ce_unguarded = CalibratedExplainer(clf, X_cal, y_cal, guard=None)
    fx_unguarded = ce_unguarded.explain(X_test)

    # Guarded explainer
    guard = ConformalRegionOracle(alpha=0.1, mode="clf", n_clusters=3)
    ce_guarded = CalibratedExplainer(clf, X_cal, y_cal, guard=guard)
    fx_guarded = ce_guarded.explain(X_test)

    # Metrics
    assert_more_stable(fx_guarded, fx_unguarded)
    assert_lower_oob_rate(ce_guarded)
```

#### Test Case 2: COMPAS – Fairness Across Groups

```python
def test_guards_compas_fairness():
    """
    On COMPAS dataset, ensure guard doesn't discriminate
    by demographic group.
    """
    X, y, groups = load_compas_with_groups()

    guard = ConformalRegionOracle(alpha=0.1, mode="clf")
    ce = CalibratedExplainer(model, X_cal, y_cal, guard=guard)

    # Acceptance rate should be similar across race/gender
    for group in groups.unique():
        mask = groups == group
        acceptance_rate_group = ce.guard.compute_oob_rate(
            perturbed_instances[mask], label_ctx[mask]
        )
        acceptance_rates.append(acceptance_rate_group)

    assert std(acceptance_rates) < 0.05  # small disparity
```

#### Test Case 3: Regression – Counterfactual Explanations

```python
def test_guards_regression_counterfactual():
    """
    Regression task with guarded CE.
    For counterfactual mode: should identify which features
    can be perturbed while staying in-distribution.
    """
    X, y = load_housing_dataset()
    reg = RandomForestRegressor().fit(X, y)

    guard = ConformalRegionOracle(alpha=0.1, mode="reg", threshold=y_median)
    ce = CalibratedExplainer(reg, X_cal, y_cal, guard=guard)

    x_test = X_test[0]
    label_ctx = guard.label_context(x_test, reg_predict=reg.predict)

    # For each feature
    for j in range(X.shape[1]):
        intervals = guard.intervals(x_test, label_ctx)
        admissible_interval = intervals[j]

        if not admissible_interval:
            # Feature cannot be perturbed – why? (isolated point, etc.)
            print(f"Feature {j} is inadmissible near {x_test[j]}")
        else:
            # Feature can be perturbed within interval
            print(f"Feature {j} can vary in {admissible_interval}")
```

---

### 4.5 Diagnostic Tools

**Proposed addition to guards module:**

```python
class GuardDiagnostics:
    """Utilities for debugging and evaluating guard behavior."""

    def __init__(self, guard: ConformalRegionOracle):
        self.guard = guard

    def compute_oob_rate(self, X_perturbed, y_ctx):
        """Fraction of perturbed instances rejected by guard."""
        return np.mean([
            not self.guard.accept(x, y_ctx[i])
            for i, x in enumerate(X_perturbed)
        ])

    def feature_admissibility(self, x, label_ctx):
        """For each feature, is it admissible (has non-empty interval)?"""
        intervals = self.guard.intervals(x, label_ctx)
        return [len(iv) > 0 for iv in intervals]

    def compute_redundancy_ratio(self, X_train, y_train):
        """Coverage of conformal region vs empirical distribution."""
        # sample random points from training set
        # check what fraction are accepted by guard
        # ratio = accepted / total
        pass

    def plot_region_2d(self, X_train, y_train, feature_pair=(0, 1)):
        """Visualize conformal region for two features."""
        # create 2D grid, evaluate guard on each point
        # overlay with training data
        pass
```

---

## Part 5: Recommended Improvements

### 5.1 Short-term (v0.10.0)

**Priority 1: Complete Pipeline Integration**

1. **Identify perturbation generation code**
   - Find where single-feature perturbations are created
   - Add guard filtering before adding candidates to rule set

2. **Wire guard filters into explain pipeline**
   ```python
   # In calibrated_explainer.py or helper module

   label_ctx = self._label_ctx(x)

   for feature_j in features:
       candidates = generate_perturbation_candidates(x, feature_j)

       if self.guard is not None:
           admissible_intervals = self.guard.intervals(x, label_ctx)[feature_j]
           candidates = filter_by_intervals(candidates, admissible_intervals)

       # Continue with filtered candidates
   ```

3. **Add conjunction validation**
   ```python
   # After computing x_conj
   if self.guard is not None and not self._accept(x_conj, label_ctx):
       continue  # Skip OOD conjunction
   ```

4. **Add tests** for guard integration (see Part 4.4 test cases)

**Priority 2: Implement Martingale E-Test**

1. Implement `MartingaleETest` class (currently stubbed)
2. Wire into `ConformalRegionOracle.accept()` when `use_martingale=True`
3. Add tests for martingale correctness

**Priority 3: Performance Optimization**

1. Cache `S_total` in `ConformalRegionOracle.intervals()`
2. Profile and optimize KDTree queries
3. Benchmark overhead on real datasets (target: ≤ 20%)

---

### 5.2 Medium-term (v0.11.0)

**Priority 1: Formal Guarantees Documentation**

1. Write ADR-028 (or extend ADR-021): "Perturbation Guard Semantics and Guarantees"
   - Define what "in-distribution" means formally
   - State coverage guarantees with caveats
   - Document perturbation compatibility assumptions

2. Add user-facing documentation in `docs/guards.md`:
   - When to use guards
   - Interpretation of acceptance/rejection
   - Limitations and assumptions

**Priority 2: Expanded Evaluation**

1. Implement test cases from Part 4.4
2. Run on synthetic benchmarks + real datasets
3. Generate evaluation report with metrics
4. Publish results (paper, documentation, or ADR)

**Priority 3: Feature Enhancements**

1. Full covariance support (not just diagonal)
2. Adaptive alpha selection based on calibration set size
3. Feature domain enforcement (e.g., [0, 1] for normalized)
4. Better categorical feature handling (one-hot constraints)

---

### 5.3 Long-term (Post-v1.0.0)

1. **Default guard**: Make `guard=ConformalRegionOracle(alpha=0.1)` the default (breaking change, needs deprecation path)
2. **Adaptive perturbation**: Constrain perturbation magnitude to guarantee staying in region
3. **Confidence intervals**: Return confidence scores for accepted perturbations
4. **Multi-level guards**: Combine conformal + martingale + density checks

---

## Part 6: Summary Table

| Aspect | Current State | Gap | Improvement |
|--------|---------------|-----|-------------|
| **Architecture** | ✅ Defined | — | — |
| **Core implementation** | ✅ CRO exists | — | — |
| **Integration** | ❌ Defined but unused | High | Wire into explain pipeline |
| **Martingale e-test** | 🟡 Stubbed | High | Implement fully |
| **Formal guarantees** | 🟡 Implied | High | Write ADR-028 |
| **Evaluation** | ❌ Absent | Critical | Design framework + run tests |
| **Performance** | 🟡 Unoptimized | Medium | Cache S_total, profile |
| **Documentation** | 🟡 Internal only | Medium | User guide + examples |
| **Feature scope** | 🟡 Diagonal covariance | Low | Add full covariance option |

---

## Part 7: Questions for Stakeholders

1. **Guarantee threshold**: Is 90% coverage (α=0.1) acceptable, or should it be configurable by default?
2. **Guard default**: Should guards be on by default or opt-in?
3. **Martingale priority**: Is the e-test critical for v0.10.0, or can it wait for v0.11.0?
4. **Evaluation timeline**: When should the evaluation framework be ready (before or after pipeline integration)?
5. **Real-world feedback**: Have users reported issues with OOD perturbations in production?

---

## References

- `src/calibrated_explanations/guards/`
- `src/calibrated_explanations/core/calibrated_explainer.py` (lines 715–939, 1772–1803)
- `improvement_docs/RELEASE_PLAN_V1.md` (guards not yet mentioned)
- `improvement_docs/adrs/ADR-021-calibrated-interval-semantics.md` (related to interval guarantees)

---

**End of Analysis**
