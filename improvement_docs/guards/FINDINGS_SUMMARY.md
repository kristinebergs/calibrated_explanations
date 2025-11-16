# Perturbation Guard: Analysis Summary & Key Findings

**Analysis Date:** November 15, 2025  
**Repository:** kristinebergs/calibrated_explanations  
**Branch:** perturbation_guard

---

## Quick Summary

The `ConformalRegionOracle` is **architecturally sound** but **broken in implementation**. The core design—augmenting perturbation features with calibrated predictions/probabilities, then using the resulting interval width to normalize conformal filtering—is correct but **cannot activate** due to critical unpacking bugs in the `fit()` method.

**Impact:** Confidence modulation is **always disabled** in practice, causing the guard to fall back to static conformal radii that don't adapt to prediction confidence.

---

## Key Findings

### What the Design Intends ✓

The guard is supposed to implement **Normalized Conformal Regression (NCR)**:

```
For each perturbation x_new:
  1. Form the augmented feature `[x_new || calibrated_prediction/probability]`
  2. Extract its nearest cluster center in this augmented space
  3. Compute Mahalanobis distance to that center
  4. Get the original instance's prediction interval (L, U)
  5. Compute interval width: w = U - L
  6. Apply confidence modulation: r_eff = q_norm * w
  7. Accept if: mahal_dist ≤ r_eff
```

**Why This Matters:**
- **Narrow intervals** (high confidence) → small acceptance radius → strict filtering
- **Wide intervals** (low confidence) → large acceptance radius → lenient filtering
- This aligns predictions with perturbation filtering constraints
- **Provides theoretical guarantees:** (1 - α) coverage on in-distribution perturbations

### What Actually Happens ✗

Due to Gap 1 and Gap 2, the code:

1. Tries to unpack `intervals_cal` and `(lower, upper)` separately
2. Attempts to iterate over `intervals_cal` as if it contains tuples
3. Hits exception (TypeError)
4. Falls back to `widths_cal = np.ones(len(cal_scores))`
5. Proceeds with uniform widths instead of actual intervals
6. Confidence modulation never activates

**Result:** Guard behaves like **static conformal prediction**, ignoring prediction confidence entirely.

---

## Root Cause Analysis

### The Core Bug: Unpacking Mismatch

**What the code expects:**
```python
result = some_structure_with_intervals_and_separate_bounds
intervals_cal, (lower, upper) = result
```

**What interval_learner.predict(uq_interval=True) actually returns:**
```python
result = (predictions_array, (lower_bounds_array, upper_bounds_array))
preds, (lower, upper) = result
```

**Why the confusion?**
- Variable naming suggests `intervals_cal` should be interval tuples
- But it's actually the prediction array
- Line 234 then tries to unpack `intervals_cal` item-by-item, which fails
- Exception is caught silently, falling back to uniform widths

### The Secondary Bug: Missing uq_interval Parameter

At line 299, the code calls:
```python
intervals = interval_learner.predict(x)  # No uq_interval=True!
```

Without `uq_interval=True`:
- `predict()` returns only predictions, not intervals
- Line 302 tries to unpack the return value as tuples
- Fails again, falls back to defaults

This is **not the intended behavior**—the docstring and comments suggest width statistics should be collected from the full training set.

---

## How Explain Uses Intervals (Reference)

The `LegacyPredictBridge` in `plugins/builtins.py` shows the correct pattern:

```python
prediction = self._explainer.predict(x, uq_interval=True, bins=bins)
if isinstance(prediction, tuple):
    preds, interval = prediction
    low, high = interval
```

This is the **canonical pattern** that the guard should mimic.

---

## Normalized Conformal Regression Mechanism

### The Mathematical Foundation

Given:
- Calibration set with nonconformity scores: $s_1, s_2, \ldots, s_n$
- Calibration set with interval widths: $w_1, w_2, \ldots, w_n$
- Miscalibration level: $\alpha$

**Normalized scores (difficulty-adjusted):**
$$s^{norm}_i = \frac{s_i}{w_i + \epsilon}$$

**Normalized quantile (global):**
$$q^{norm} = \text{quantile}_{1-\alpha}(s^{norm}_1, \ldots, s^{norm}_n)$$

**At test time, effective radius:**
$$r_{eff}(w_{test}) = q^{norm} \cdot w_{test}$$

**Acceptance rule:**
$$\text{accept} = \left( \text{mahal\_dist}(x_{new}) \leq r_{eff}(w_{test}) \right)$$

### Why This Works

1. **Normalizes by difficulty:** High-uncertainty predictions (wide intervals) have larger nonconformity budgets
2. **Provides coverage:** The (1 - α) quantile ensures $(1 - \alpha)$ of calibration perturbations are accepted
3. **Adaptive:** Test-time acceptance adapts to prediction confidence
4. **Theoretically grounded:** Extends conformal prediction to heteroscedastic settings

---

## Implementation Status Map

| Component | Status | Notes |
|-----------|--------|-------|
| ICP split (proper/calib) | ✓ Working | Lines 165-176 |
| KMeans clustering | ✓ Working | Lines 179-182 |
| Covariance computation | ✓ Working | Lines 185-200 |
| Nonconformity scores | ✓ Working | Lines 203-208 |
| Width extraction | ✗ **BROKEN** | Line 234 unpacking fails |
| Normalized score computation | ✓ Working (if widths were extracted) | Lines 242-247 |
| Normalized quantile caching | ✓ Working (if widths were extracted) | Lines 248-251 |
| Width statistics | ✗ **BROKEN** | Line 302 unpacking fails |
| Effective radius in accept() | ✓ Working (if widths existed) | Lines 447-465 |
| Per-cluster quantiles | ✓ Working (if widths existed) | set_alpha() method |

---

## Five Critical Questions

### Q1: Is the guard currently providing confidence modulation?

**A:** No. The logs would show:
```
Confidence modulation disabled: width_min=0.0, width_max=1.0
```

The fallback values (0.0, 1.0) indicate that width statistics were never computed.

### Q2: Why don't tests catch this?

**A:** Current tests likely:
1. Don't verify that confidence modulation is active
2. Don't provide a mock interval_learner that returns proper tuple structure
3. Don't assert on `_cluster_norm_quantiles` values
4. Only test the "happy path" (fitting without checking internals)

### Q3: Can the guard be used without fixing these bugs?

**A:** Yes, but it behaves like **static conformal prediction**, ignoring prediction confidence. It's still safe (respects α coverage) but loses the intended confidence-awareness.

### Q4: What happens if an explainer doesn't have an interval_learner?

**A:** Line 150 raises ValueError—this is correct. NCR fundamentally requires calibrated intervals.

### Q5: How does this affect the perturbation generation workflow?

**A:** In `explain()` → `explain_predict_step()` → `filter_perturbations()`:
1. Explainer extracts `(pred, (low, high))` with `uq_interval=True` ✓ Working
2. Guard receives these values in orchestrator.filter_perturbations() ✓ Working
3. Guard checks `accept_batch(x_perturbed, calibrated_predictions)` ✓ Working
4. Guard uses the `calibrated_prediction` tuple in accept() ✓ Working **IF** it had the quantiles

**The missing link:** The guard never computed `_cluster_norm_quantiles` due to bugs in fit().

---

## Connection to Explainer Architecture

### How Predictions Flow

```
CalibratedExplainer.predict(x, uq_interval=True)
  ↓ delegates to
PredictionOrchestrator.predict()
  ↓ delegates to
interval_learner.predict(x, uq_interval=True)
  ↓ returns
(predictions, (lower_bounds, upper_bounds))
  ↓ returns
(predictions, (lower_bounds, upper_bounds))
  ↓ used by
GuardOrchestrator.filter_perturbations()
  ↓ calls
ConformalRegionOracle.accept_batch(x_new, calibrated_predictions)
```

The guard receives the correct tuple structure but **cannot use it** because it never computed the normalized quantiles.

---

## Files & Locations

### Critical Code (Bugs Here)
- `src/calibrated_explanations/guards/regions.py:232-237` — Gap 1: Unpacking
- `src/calibrated_explanations/guards/regions.py:299-302` — Gap 2: Missing uq_interval

### Reference Implementations
- `src/calibrated_explanations/plugins/builtins.py:96-100` — Correct unpacking pattern
- `src/calibrated_explanations/core/calibrated_explainer.py:1849-1880` — predict() signature
- `src/calibrated_explanations/guards/orchestrator.py:122-145` — filter_perturbations usage

### Documentation & Tests (To Update)
- `tests/unit/guards/` — Add NCR-specific tests
- `improvement_docs/guards/IMPLEMENTATION_ANALYSIS.md` — New analysis doc
- `improvement_docs/guards/IMPLEMENTATION_PLAN.md` — New action plan

---

## Success Metrics (Post-Fix)

After implementing the plan:

1. **Logs show:**
   ```
   Guard fit summary: alpha=0.1, n_cal=25, ...
   Confidence modulation active: width_min=X, width_max=Y
   Normalized conformal regression (NCR) active. q_norm range: [Z1, Z2]
   ```

2. **Tests verify:**
   - `widths_cal` is computed correctly (direct subtraction)
   - `_cluster_norm_quantiles` is not None
   - `accept()` with narrow interval rejects more perturbations than with wide interval
   - `accept()` with wide interval accepts more perturbations

3. **Behavior shows:**
   - Guard adapts acceptance threshold based on prediction confidence
   - High-confidence predictions → strict filtering
   - Low-confidence predictions → lenient filtering
   - Coverage guarantee (1 - α) is maintained

---

## Recommendations for Immediate Action

1. **Fix Immediately (Blocking):**
   - Apply Tasks 1.1 and 1.2 from IMPLEMENTATION_PLAN
   - Add one integration test to verify confidence modulation activates
   - Update CHANGELOG

2. **Fix Soon (High Priority):**
   - Add validation (Task 2.1)
   - Update docstrings (Tasks 3.1-3.3)
   - Add comprehensive unit tests (Task 4.1, 4.2)

3. **Future (Nice-to-Have):**
   - Create ADR documenting normalized conformal regression
   - Add diagnostic visualization for confidence modulation
   - Extend to per-cluster normalized quantiles

---

## Conclusion

The guard implementation has **solid foundations** but is **blocked by implementation bugs**. The fixes are straightforward (correct unpacking, add parameter) and will unlock the intended confidence-modulated filtering behavior. Once fixed, the guard will provide theoretically-grounded, adaptive perturbation filtering that respects prediction confidence.

**Priority:** CRITICAL — Cannot fully leverage the explainability system without this fix.
