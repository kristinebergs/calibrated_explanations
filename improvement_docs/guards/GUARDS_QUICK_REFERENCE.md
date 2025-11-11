# Guards Expansion Analysis – Quick Reference

**Analysis Date:** November 11, 2025
**Status:** ✅ Complete

---

## Three Key Questions – Answered

### ❓ Question 1: Does it need improvements?

**✅ YES – Critical Gaps Identified**

| Component | Status | Issue | Priority |
|-----------|--------|-------|----------|
| Architecture | ✅ Complete | None | — |
| Implementation | ✅ Complete | None | — |
| **Pipeline Integration** | ❌ 0% | Helpers defined but never called | **CRITICAL** |
| **Martingale E-Test** | ❌ 0% | Marked TODO, not implemented | **HIGH** |
| **Performance Optimization** | ❌ 0% | S_total caching not done | **MEDIUM** |
| **Formal Guarantees** | 🟡 50% | Implied but not documented | **HIGH** |
| **Evaluation Framework** | ❌ 0% | No tests for guard effectiveness | **CRITICAL** |

**Key Finding:** Guards are **instantiated but never used**. The feature is non-functional.

---

### ❓ Question 2: What guarantees can be provided for in-distribution perturbations?

**Answer: Conformal Membership + Optional Martingale E-Test**

#### Tier 1: Conformal Membership (Default)

```
Guarantee:  For random samples from training distribution:
            P(point in region) ≥ (1-α)

Method:     ||x' - cluster_center||²_covariance ≤ radius²

Overhead:   O(n_clusters) ≈ fast

Caveat:     • Population-level (average), not per-instance
            • No guarantee for STRUCTURED perturbations
            • Assumes elliptical label-conditional distributions
```

#### Tier 2: Martingale E-Test (Optional)

```
Guarantee:  Anytime-valid test for OOD
            Type-I error ≤ e_gamma⁻¹

Method:     Likelihood ratio on k-NN local density

Overhead:   O(1) with precomputed stats

Status:     TODO (not yet implemented)

When:       High-stakes applications + adversarial concern
```

#### Critical Gap: Perturbations ≠ Random Samples

```
Problem:    Conformal calibration assumes RANDOM draws from population
            But CE generates STRUCTURED perturbations (additive noise)

            These have different properties!

Example:    • Random draw: X ~ N(0, 1)     ✓ Stays in [-2, 2] with 95% prob
            • Perturbation: X' = 0 + U(-0.5, 0.5)  ✓ Stays in [-0.5, 0.5]

            But:  Many sequential perturbations can drift outside region!

Recommendation: Document that guardrails apply to **marginal coverage**,
                not individual perturbation trajectories.
                For strong guarantees, use Tier 2 (martingale).
```

#### Perturbation Compatibility

| Type | Method | Compatibility | Notes |
|------|--------|---------------|-------|
| **Numeric** | Uniform/Gaussian noise | ✅ Good | Bounded by feature range |
| **Categorical** | Permutation | ✅ Good | Within observed values |
| **Transforms** | Feature engineering | ⚠️ Caution | May distort distance metric |

---

### ❓ Question 3: How to evaluate if guarded CE succeeds where unguarded fails?

**Answer: Five Key Metrics**

#### Metric 1: Explanation Stability (PRIMARY)

```
Procedure:
  1. Explain test point x with unguarded CE
  2. Repeat explanation 10 times with different random seeds
  3. Measure rule agreement (Jaccard similarity)
  4. Repeat with guarded CE
  5. Compute ratio

Result:
  stability_ratio = agreement_guarded / agreement_unguarded

Expected:
  > 1.2  (guarded should be ≥20% more stable)

Why this matters:
  • More stable rules → more trustworthy explanations
  • Shows guards prevent spurious rules from random OOD perturbations
```

#### Metric 2: OOD Detection Rate

```
Procedure:
  Generate perturbations from calibration set
  Check what fraction guard rejects

Result:
  oob_rate = (num_rejected / num_total)

Expected:
  oob_rate ≈ α  (e.g., 0.1 for α=0.1)

Interpretation:
  • If oob_rate >> α: Guard is too strict
  • If oob_rate << α: Guard is too loose
```

#### Metric 3: Fairness Parity (Across Classes)

```
Procedure:
  Compute acceptance_rate per class

Result:
  disparity = max(rates) - min(rates)

Expected:
  < 0.05  (less than 5 percentage points)

Why this matters:
  • Prevents guard from discriminating
  • Ensures minority classes aren't overly restricted
```

#### Metric 4: Computational Overhead

```
Procedure:
  time_unguarded = runtime without guard [ms]
  time_guarded = runtime with guard [ms]
  overhead = (time_guarded - time_unguarded) / time_unguarded

Expected:
  ≤ 20%  (from implementation_instructions.md)

Acceptable for production use
```

#### Metric 5: Coverage Calibration

```
Procedure:
  For α ∈ {0.01, 0.05, 0.1, 0.2}:
    estimated_coverage_α = 1 - α
    actual_oob_rate = empirical rate on test set

Plot:
  (estimated, actual) should be near diagonal

Expected:
  Good calibration means guard predictions are reliable
```

---

## Implementation Checklist

### Phase 1: Complete Integration (v0.10.0) – 4-6 weeks

- [ ] **CRITICAL** Find where perturbation candidates are generated
  - Search: `feature_values`, `perturb_dataset`, `uniform_perturbation`, `gaussian_perturbation`
  - Wire: Add `guard.intervals()` filtering before candidate addition

- [ ] **CRITICAL** Integrate guard filtering into explanation pipeline
  ```python
  # Pseudocode
  label_ctx = self._label_ctx(x)

  for feature_j in features:
      candidates = generate_candidates(x, feature_j)
      if self.guard is not None:
          allowed = self.guard.intervals(x, label_ctx)[feature_j]
          candidates = filter_by_intervals(candidates, allowed)
      # Continue with filtered candidates
  ```

- [ ] **CRITICAL** Add conjunction validation
  ```python
  if self.guard is not None and not self._accept(x_conj, label_ctx):
      continue  # Skip OOD conjunction
  ```

- [ ] **HIGH** Implement Martingale E-Test
  - Complete `MartingaleETest` class in `guards/martingale.py`
  - Wire into `ConformalRegionOracle.accept()`

- [ ] **HIGH** Performance optimization
  - [ ] Cache S_total across features in `intervals()`
  - [ ] Profile KDTree queries
  - [ ] Benchmark overhead (target ≤ 20%)

- [ ] **HIGH** Add unit tests
  - [ ] Guard filtering tests (synthetic)
  - [ ] Integration tests (with explanation pipeline)
  - [ ] Benchmark tests (time + overhead)

### Phase 2: Formal Guarantees (v0.11.0) – 4-8 weeks

- [ ] Write ADR-028: Perturbation Guard Semantics
  - Define "in-distribution" formally
  - State coverage guarantees with caveats
  - Document perturbation compatibility

- [ ] Implement evaluation suite
  - [ ] Synthetic benchmark (mixture of Gaussians)
  - [ ] Real datasets (COMPAS, housing)
  - [ ] Run all 5 metrics
  - [ ] Generate report

- [ ] Create user documentation
  - [ ] `docs/guards.md` (usage guide, examples)
  - [ ] Limitations and assumptions
  - [ ] Troubleshooting guide

### Phase 3: Future (Post-v1.0.0)

- [ ] Make guards default (with deprecation path)
- [ ] Full covariance support (not just diagonal)
- [ ] Adaptive alpha selection
- [ ] Feature domain enforcement
- [ ] Combine multiple guard levels

---

## Code Locations Reference

**Core Implementation:**
- `src/calibrated_explanations/guards/__init__.py` (28 LOC)
- `src/calibrated_explanations/guards/regions.py` (195 LOC)
- `src/calibrated_explanations/guards/implementation_instructions.md` (267 LOC)

**Integration Points:**
- `src/calibrated_explanations/core/calibrated_explainer.py`
  - Lines 715–939: `__init__` and guard initialization
  - Lines 945–953: `_label_ctx()` and `_accept()` helpers
  - Lines 1772–1803: `set_guard()` and `_update_guard()` methods

**Missing Implementation:**
- Perturbation filtering (needs to be added to explain pipeline)
- Martingale e-test (stubbed in `regions.py:189–194`)
- Evaluation framework (needs to be created)

---

## Stakeholder Questions

1. **Integration Priority:** Should pipeline integration be v0.10.0 or v0.9.2?
2. **Default Behavior:** Opt-in (current) or opt-out (default on)?
3. **Alpha Default:** Is 0.1 (90% coverage) acceptable?
4. **Martingale Urgency:** Critical for v0.10.0, or can defer?
5. **Real-World Data:** Any reported issues with OOD perturbations in production?

---

## Summary: The 60-Second Version

**What is the guards expansion?**
An optional in-distribution filter for perturbations used during explanation generation.

**Does it need improvements?**
Yes. It's architecturally sound but functionally incomplete:
- Integration: Defined but not called (0% functional)
- Martingale e-test: Stubbed only
- Evaluation: No tests
- Guarantees: Not documented

**Key guarantee:**
Population-level coverage ≥ (1-α) for conformal region membership. Works for random samples; weaker for structured perturbations.

**How to evaluate:**
- Primary metric: Explanation stability (10+ repeats, measure rule overlap)
- Secondary: OOD detection rate, fairness parity, overhead, calibration
- Real-world test: COMPAS (fairness), housing (regression), synthetic (controlled)

**Next steps:**
1. Wire guards into perturbation loop (v0.10.0)
2. Implement martingale e-test (v0.10.0)
3. Formalize guarantees in ADR-028 (v0.11.0)
4. Run evaluation suite (v0.11.0)

---

**Status: Analysis Complete ✅ | Ready for: Implementation Planning, Stakeholder Review**
