# Guards Expansion Analysis – Visual Summary

**Status:** Analysis Complete | Date: 2025-11-11

---

## Question 1: Does It Need Improvements?

### YES – But with nuance

```
┌──────────────────────────────────────┐
│ GUARDS EXPANSION STATUS             │
├──────────────────────────────────────┤
│ ✅ Architecture             [SOLID] │
│ ✅ Core Implementation      [SOLID] │
│ ⚠️  Integration             [90%]   │
│ ❌ Pipeline Usage           [0%]    │
│ ❌ Martingale E-Test        [0%]    │
│ ⚠️  Performance Optimization [0%]   │
│ ❌ Evaluation Framework      [0%]   │
│ ⚠️  User Documentation       [0%]   │
└──────────────────────────────────────┘
```

### Key Finding: Non-Functional Integration

The helper methods `_label_ctx()` and `_accept()` **exist but are never called**:

```python
# File: calibrated_explainer.py

# ✅ Defined
def _label_ctx(self, x):
    if self.guard is None: return None
    return self.guard.label_context(...)

def _accept(self, x_prime, label_ctx):
    return True if self.guard is None else self.guard.accept(...)

# ❌ But where are they used?
# Grep shows: nowhere in the explanation pipeline
```

**Result:** Guards do NOT filter perturbations. Feature is cosmetic.

---

## Question 2: In-Distribution Guarantees

### Guarantee Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│ LEVEL 1: CONFORMAL MEMBERSHIP CHECK (Currently Implemented)   │
├────────────────────────────────────────────────────────────────┤
│ Method:   ||x' - cluster_center||²_Σ ≤ radius²               │
│ Guarantee: Population coverage ≥ (1-α)                       │
│ Overhead:  O(n_clusters)                                     │
│ Strength:  ✅ Distribution-free, theoretically sound         │
│ Weakness:  ❌ No guarantee for PERTURBATIONS specifically    │
│ Weakness:  ❌ Applies to population average, not single x'   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ LEVEL 2: DENSITY-BASED CHECK (Not Implemented)                │
├────────────────────────────────────────────────────────────────┤
│ Method:   Is x' in high-density region of training data?     │
│ Guarantee: x' is similar to observed points                  │
│ Overhead:  O(log n) with k-NN index                          │
│ Strength:  ✅ Empirically grounded                           │
│ Status:    ❌ TODO                                            │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ LEVEL 3: MARTINGALE E-TEST (Stubbed Only)                    │
├────────────────────────────────────────────────────────────────┤
│ Method:   Likelihood ratio test on local k-NN neighborhood   │
│ Guarantee: Anytime-valid OOD test (Type-I error controlled) │
│ Overhead:  O(1) with precomputed stats                      │
│ Strength:  ✅ Strong formal guarantees                       │
│ Status:    ❌ TODO (code marked with TODO comment)           │
└────────────────────────────────────────────────────────────────┘
```

### Critical Gap: Perturbations ≠ Random Draws

**Problem:**
- Conformal calibration gives guarantees for random samples from the population
- But perturbations are **structured** (additive noise relative to a base point)
- No guarantee that structured perturbations stay within the calibrated region

**Example:**

```
1D Gaussian calibration set: X ~ N(0, 1)
Conformal region: [-2, +2] (covers 95%)

Original point: x = 0 (center)

Perturbation method 1: Uniform noise
  x' = 0 + U(-0.5, 0.5) ∈ [-0.5, 0.5]  ✓ Stays in region

Perturbation method 2: Many small perturbations
  x'₁ = 0 + U(-0.5, 0.5)  ∈ [-0.5, 0.5]  ✓
  x'₂ = x'₁ + U(-0.5, 0.5) ∈ [-1, 1]     ✓
  ...
  x'₂₀ = ... ∈ [-5, +5]               ❌ Outside region!

Key insight: Perturbation direction/magnitude matters, but conformal model is **isotropic**.
```

### Recommended Formalization

**Adopt two-tier guarantee system:**

```
TIER A: Conformal Membership (Default)
├─ Method: Check ||x' - μ||²_Σ ≤ r²
├─ Guarantee: Marginal coverage ≥ (1-α)
├─ Computational cost: O(n_clusters)
├─ When to use: Always (baseline)
└─ Caveat: Population-level, not per-instance

TIER B: Martingale E-Test (Optional)
├─ Method: Anytime-valid hypothesis test
├─ Guarantee: Type-I error ≤ e_gamma⁻¹
├─ Computational cost: O(1)
├─ When to use: High-stakes + small calibration sets
├─ Status: TODO (not yet implemented)
└─ Caveat: Requires local density estimation
```

**Perturbation Compatibility:**
```
Numeric (uniform noise):  ✓ Compatible (bounded by data range)
Numeric (gaussian noise): ✓ Compatible (concentrated near original)
Categorical (permutation):✓ Compatible (within observed values)
Transformed features:     ⚠ Requires manual validation
```

---

## Question 3: Evaluation – When Does Guarded CE Succeed?

### Failure Mode Map

```
┌─────────────────────────────────────────────────────────────┐
│ FAILURE MODE                 DETECTION METRIC              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ A) OOD in Sparse Regions                                  │
│    └─ Perturbations land in low-density areas             │
│       → Rule stability drops (inconsistent rules)          │
│       🔍 Metric: Rule overlap across 10 random seeds      │
│          stability_ratio = overlap_guarded / overlap_unguarded
│          Expected: > 1.2 (20% improvement)                │
│                                                             │
│ B) Cluster Boundary Effects                               │
│    └─ Near cluster edges, perturbations exit region       │
│       → Per-feature acceptance rate varies                │
│       🔍 Metric: Feature admissibility rate per class     │
│          acceptance_rate = (accepted / total) per feature │
│          Expected: ~(1-α) e.g., ~0.9 for α=0.1          │
│                                                             │
│ C) High-Dimensional Curse                                 │
│    └─ Calibration sparse → conformal region huge          │
│       → Many perturbations superficially "in-region"      │
│       🔍 Metric: Redundancy ratio                         │
│          ratio = |conformal_region| / |empirical_support| │
│          Expected: < 2 (not overly conservative)         │
│                                                             │
│ D) Label Imbalance                                        │
│    └─ Minority class → looser constraints → more OOD      │
│       → Different classes have different acceptance rates  │
│       🔍 Metric: Fairness parity                         │
│          disparity = max_class - min_class acceptance     │
│          Expected: < 0.05 (< 5pp difference)             │
│                                                             │
│ E) Computational Overhead                                 │
│    └─ Guard evaluation time matters in production         │
│       🔍 Metric: Overhead ratio                          │
│          ratio = (time_guarded - time_unguarded) / time_unguarded
│          Target: ≤ 0.20 (20% overhead)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Proposed Test Suite

#### Test 1: Synthetic Mixture – Factual Explanations

```
Setup:
  • Generate 3 well-separated Gaussian clusters
  • Train RandomForest classifier
  • Generate test instances

Unguarded:
  ├─ Explain test point
  ├─ Repeat 10 times with different seeds
  └─ Measure rule consistency

Guarded:
  ├─ Explain test point
  ├─ Repeat 10 times with different seeds
  └─ Measure rule consistency

Metric:
  stability_ratio = overlap_guarded / overlap_unguarded

Expected:
  stability_ratio > 1.2  (20% improvement)
  oob_rate ≈ 0.1        (matches α)
```

#### Test 2: COMPAS – Fairness Across Groups

```
Setup:
  • Classify recidivism (COMPAS dataset)
  • Stratify by race/gender

For each group:
  ├─ Measure acceptance rate
  ├─ Compute OOD rate
  └─ Store metrics

Fairness check:
  acceptance_disparity = max(group_rates) - min(group_rates)

Expected:
  acceptance_disparity < 0.05  (all groups similar)
  (Prevents guard from being biased)
```

#### Test 3: Housing – Counterfactual Explanations (Regression)

```
Setup:
  • Regression task (predict house price)
  • Generate counterfactual explanations

For each test instance:
  ├─ Compute label context (high/low price)
  ├─ For each feature:
  │  └─ Get admissible interval from guard
  ├─ Report which features can be perturbed
  └─ Report which features are "locked" (inadmissible)

Insight:
  • Features with empty intervals → isolated in feature space
  • Can use to detect data anomalies or feature importance
```

### Diagnostic Tools (Proposed Addition)

```python
class GuardDiagnostics:
    """Debug and evaluate guard behavior."""

    def compute_oob_rate(X_perturbed, y_ctx) -> float:
        """What fraction of perturbations are rejected?"""
        # Expected: ~α (e.g., 0.1)

    def feature_admissibility(x, label_ctx) -> List[bool]:
        """For each feature, is interval non-empty?"""
        # True = feature can be perturbed
        # False = feature is "locked" (isolated point)

    def redundancy_ratio(X_train) -> float:
        """How much larger is conformal region vs empirical?"""
        # Indicator of curse of dimensionality
        # Expected: < 2

    def acceptance_by_class(X_by_class) -> Dict[class, float]:
        """Per-class acceptance rate."""
        # Check fairness parity
        # Expected: similar across classes

    def plot_region_2d(X_train, y_train, features=(0, 1)):
        """Visualize conformal region boundaries."""
        # Helps understand region geometry
```

---

## Implementation Roadmap

### Phase 1: Complete Integration (v0.10.0)

```
Priority 1: Wire Guard Into Pipeline
  ├─ [ ] Find perturbation generation code
  ├─ [ ] Add guard.intervals() filtering
  ├─ [ ] Add guard.accept() for conjunctions
  ├─ [ ] Test integration end-to-end
  └─ [ ] Verify guards actually filter perturbations

Priority 2: Implement Martingale E-Test
  ├─ [ ] Implement MartingaleETest class
  ├─ [ ] Wire into ConformalRegionOracle.accept()
  ├─ [ ] Add unit tests
  └─ [ ] Benchmark overhead

Priority 3: Performance
  ├─ [ ] Cache S_total across features
  ├─ [ ] Profile KDTree queries
  ├─ [ ] Target: ≤ 20% overhead
  └─ [ ] Benchmark on real datasets

Timeline: 4-6 weeks
Deliverables: Functional guards + tests + benchmarks
```

### Phase 2: Formal Guarantees (v0.11.0)

```
Priority 1: ADR-028 (New Architecture Decision)
  ├─ [ ] Define "in-distribution" formally
  ├─ [ ] State coverage guarantees with caveats
  ├─ [ ] Document perturbation compatibility
  └─ [ ] Get stakeholder review

Priority 2: Evaluation Framework
  ├─ [ ] Implement test suite (synthetic + real)
  ├─ [ ] Run benchmarks on COMPAS, housing, etc.
  ├─ [ ] Collect metrics (stability, OOD rate, overhead)
  ├─ [ ] Generate report
  └─ [ ] Publish findings

Priority 3: User Documentation
  ├─ [ ] Write docs/guards.md
  ├─ [ ] Add usage examples
  ├─ [ ] Document limitations and assumptions
  └─ [ ] Create quick-start notebook

Timeline: 4-8 weeks
Deliverables: ADR + evaluation report + user guide
```

### Phase 3: Long-term (Post-v1.0.0)

```
- Make guards default (with deprecation path)
- Full covariance support (not just diagonal)
- Adaptive alpha selection
- Feature domain enforcement
- Multi-level guard combination
```

---

## Summary Table

| Aspect | Status | Gap | Next Step |
|--------|--------|-----|-----------|
| **Architecture** | ✅ Complete | — | — |
| **Core Implementation** | ✅ Complete | — | — |
| **Integration** | 🟡 Partial | High | Wire into pipeline |
| **Martingale E-Test** | 🔴 TODO | High | Implement |
| **Formal Guarantees** | 🟡 Implied | High | Write ADR-028 |
| **Evaluation** | 🔴 Missing | Critical | Design + run tests |
| **Performance** | 🟡 Unoptimized | Medium | Cache + profile |
| **Documentation** | 🔴 Absent | Medium | User guide |

---

## Key Questions for Stakeholders

1. **Default or opt-in?** Should guards be on by default (v1.0+) or remain opt-in?
2. **Coverage target?** Is 90% (α=0.1) acceptable, or should users configure?
3. **Martingale priority?** Critical for v0.10.0, or can defer to v0.11.0?
4. **Evaluation scope?** Just synthetic, or also real-world datasets?
5. **Production feedback?** Have users reported OOD perturbation issues?

---

## Conclusion

The guards expansion has **strong theoretical foundations** but **incomplete practical execution**. The core issue is that guards are instantiated but never actually used during explanation generation.

**Immediate action:** Complete the integration (wire into perturbation loop) and then formalize guarantees.

**Benefit:** Ensures explanations are based on in-distribution perturbations, improving reliability and trust.

---

**Analysis: Complete | Status: Ready for Implementation Planning**
