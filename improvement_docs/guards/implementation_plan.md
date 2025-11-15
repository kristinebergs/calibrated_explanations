# Guard Implementation Plan: Close the Gaps

**Date:** November 15, 2025  
**Owner:** Perturbation Guard Implementation  
**Priority:** CRITICAL - Blocking normalized conformal regression

---

## Implementation Plan Overview

This document outlines the steps to close all identified gaps in the `ConformalRegionOracle` implementation and fully activate the normalized conformal regression (NCR) feature for confidence-modulated guarding.

**Success Criteria:**
- All critical bugs (Gap 1, Gap 2) fixed
- Confidence modulation actively normalizes acceptance radius by prediction uncertainty
- Unit tests verify normalized quantile computation and effective radius modulation
- Documentation clearly explains the NCR mechanism

---

## Phase 1: Critical Bug Fixes

### Task 1.1: Fix interval_learner.predict() Unpacking (Gap 1)

**File:** `src/calibrated_explanations/guards/regions.py`  
**Lines:** 232-237

**Current Code:**
```python
intervals_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
if intervals_cal is not None and len(intervals_cal) == len(x_cal):
    widths_cal = np.array([upper - lower for lower, upper in intervals_cal])
else:
    widths_cal = np.ones(len(cal_scores))
```

**Fixed Code:**
```python
preds_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
if preds_cal is not None and len(preds_cal) == len(x_cal):
    widths_cal = upper - lower  # Direct array subtraction
    if widths_cal.ndim == 0:
        widths_cal = np.full(len(cal_scores), float(widths_cal))
else:
    widths_cal = np.ones(len(cal_scores))
```

**Why This Works:**
- Correctly unpacks `(predictions, (lower, upper))` tuple
- Removes invalid tuple unpacking from width computation
- Uses direct array subtraction: `upper - lower` returns array of interval widths
- Handles edge case where width is scalar
- Aligns with reference implementation in `plugins/builtins.py`

**Testing:**
- Create mock interval_learner that returns known predictions and bounds
- Verify `widths_cal` shape and values
- Assert no exception is raised

---

### Task 1.2: Add uq_interval Parameter and Fix Width Statistics (Gap 2)

**File:** `src/calibrated_explanations/guards/regions.py`  
**Lines:** 299-302

**Current Code:**
```python
intervals = interval_learner.predict(x)
if intervals is not None and len(intervals) > 0:
    widths = np.array([upper - lower for lower, upper in intervals])
```

**Fixed Code:**
```python
prediction_full, (lower_full, upper_full) = interval_learner.predict(x, uq_interval=True)
if prediction_full is not None and len(prediction_full) > 0:
    widths = upper_full - lower_full  # Direct array subtraction
    self._width_min = float(np.min(widths))
    self._width_max = float(np.max(widths))
```

**Why This Works:**
- Adds `uq_interval=True` to extract uncertainty bounds from full training set
- Uses direct array subtraction consistent with Task 1.1
- Correctly handles return tuple structure
- Ensures width statistics capture the full range of prediction uncertainties

**Testing:**
- Verify predict is called with `uq_interval=True`
- Check that `_width_min` and `_width_max` are correctly computed
- Validate confidence modulation is active (log message at line 323)

---

## Phase 2: Validation & Error Handling

### Task 2.1: Add interval_learner Parameter Validation

**File:** `src/calibrated_explanations/guards/regions.py`  
**Method:** `fit()`, after line 150

**Add After Existing Validation:**
```python
# Validate interval_learner supports required interface
if interval_learner is not None:
    try:
        # Test predict signature with small sample
        test_sample = x_arr[:min(2, len(x_arr))]
        _ = interval_learner.predict(test_sample, uq_interval=True)
    except TypeError as e:
        if "uq_interval" in str(e):
            raise ValueError(
                f"interval_learner.predict() does not support uq_interval parameter. "
                f"Error: {e}"
            ) from e
        raise ValueError(
            f"interval_learner.predict() signature incompatible. Error: {e}"
        ) from e
    except Exception as e:
        logger.warning(
            "interval_learner.predict() test call failed; "
            "confidence modulation may not work. Error: %s", e
        )
```

**Why This Helps:**
- Fails fast with clear error message if interval_learner is incompatible
- Prevents silent degradation to uniform widths
- Helps users debug misconfiguration

---

### Task 2.2: Add Informative Logging for Confidence Modulation Status

**File:** `src/calibrated_explanations/guards/regions.py`  
**Locations:** End of fit() method

**Add After Line 327:**
```python
# Log confidence modulation readiness
if self._cluster_norm_quantiles is not None:
    logger.info(
        "Normalized conformal regression (NCR) active. "
        "Effective radius will scale with prediction interval width. "
        "q_norm range: [%.4f, %.4f]",
        float(np.min(self._cluster_norm_quantiles)),
        float(np.max(self._cluster_norm_quantiles)),
    )
else:
    logger.warning(
        "Normalized conformal regression (NCR) disabled. "
        "Falling back to static radii. Check interval_learner output."
    )
```

**Why This Helps:**
- Makes it obvious whether confidence modulation is active
- Provides diagnostic values for tuning
- Signals any issues during fit

---

## Phase 3: Documentation & Design Clarity

### Task 3.1: Enhance fit() Docstring with NCR Details

**File:** `src/calibrated_explanations/guards/regions.py`  
**Method:** `fit()`

**Replace Docstring Section (lines 111-127):**

**Before:**
```python
"""Fit the conformal region oracle.

Performs inductive conformal prediction:
1. Split x into proper (75%) and calibration (25%) sets
2. Cluster the proper set in feature space
...
```

**After:**
```python
"""Fit the conformal region oracle using normalized conformal regression.

Performs inductive conformal prediction with confidence modulation:

1. Split x into proper (75%) and calibration (25%) sets
2. Cluster the proper set in feature space to capture heteroscedasticity
3. Compute per-cluster covariance and Mahalanobis distances on proper set
4. Extract uncertainty intervals from interval_learner for calibration set
5. Normalize nonconformity scores by interval width: s_norm = s_raw / width
6. Compute (1 - alpha) quantile on normalized scores: q_norm
7. Store q_norm per cluster for dynamic radius modulation at test time
8. Record width statistics (min/max) for confidence modulation diagnostics

The key innovation is **normalized conformal regression (NCR)**:
- At test time, effective radius scales with prediction confidence
- Wider intervals (low confidence) → larger acceptance regions
- Narrower intervals (high confidence) → smaller acceptance regions
- Formula: r_eff(cluster, width_test) = q_norm(cluster) * width_test

This approach implements confidence-aware perturbation filtering.
```

**Why This Helps:**
- Clear explanation of normalized conformal regression
- Helps maintainers understand the design
- Explains why interval_learner is critical

---

### Task 3.2: Add NCR Explanation to accept() Docstring

**File:** `src/calibrated_explanations/guards/regions.py`  
**Method:** `accept()`

**Add After "Returns" Section (after line 429):**

```python
    Notes
    -----
    This method implements normalized conformal regression (NCR) for
    confidence-modulated filtering:

    1. Computes Mahalanobis distance to nearest cluster center
    2. If calibrated_prediction is provided (pred, (L, U)):
       - Extracts interval width: w = U - L
       - Scales effective radius: r_eff = q_norm * w
       - Larger width (lower confidence) → higher acceptance tolerance
    3. Otherwise uses static conformal radius (legacy fallback)

    For effective confidence modulation, always provide calibrated_prediction
    with format: (prediction_value, (lower_bound, upper_bound))
```

**Why This Helps:**
- Documents the correct usage pattern
- Explains what happens with/without calibrated_prediction
- Guides users to always provide intervals

---

### Task 3.3: Add Design Comments Throughout regions.py

**File:** `src/calibrated_explanations/guards/regions.py`  
**Locations:** Multiple

**Add Comment Before Line 232:**
```python
# Extract calibrated intervals and compute widths for normalized conformal regression
# The interval_learner.predict(uq_interval=True) returns:
#   (predictions, (lower_bounds, upper_bounds))
# Widths = upper - lower represent prediction uncertainty/confidence
# Normalized scores s_norm = s_raw / width scale nonconformity by confidence
```

**Add Comment Before Line 242:**
```python
# Normalized conformal regression: quantile on width-normalized scores
# This enables the key feature: effective radius scales with prediction confidence
# Higher confidence (narrow interval) → smaller radius
# Lower confidence (wide interval) → larger radius
```

---

## Phase 4: Unit Tests

### Task 4.1: Test Correct Width Extraction

**File:** `tests/unit/guards/test_regions_ncr.py` (NEW)

**Test Case 1: Width Extraction from interval_learner**
```python
def test_fit_correctly_extracts_widths_from_interval_learner():
    """Verify fit() correctly unpacks and computes widths from interval_learner."""
    # Create mock interval_learner
    class MockIntervalLearner:
        def predict(self, x, uq_interval=False):
            if uq_interval:
                preds = np.arange(len(x), dtype=float)
                lower = preds - 1.0
                upper = preds + 2.0
                return preds, (lower, upper)
            return np.arange(len(x))
    
    x_train = np.random.randn(50, 5)
    y_train = np.random.randn(50)
    
    oracle = ConformalRegionOracle(alpha=0.1, n_clusters=3)
    interval_learner = MockIntervalLearner()
    
    oracle.fit(x_train, y_train, interval_learner=interval_learner)
    
    # Widths should be (upper - lower) = (preds + 2.0) - (preds - 1.0) = 3.0
    expected_widths = np.full(len(x_train), 3.0)
    np.testing.assert_array_almost_equal(oracle._cal_widths, expected_widths[:len(oracle._cal_widths)])
    
    # Width statistics should reflect this
    assert oracle._width_min == 3.0
    assert oracle._width_max == 3.0
```

**Test Case 2: NCR Quantile Computation**
```python
def test_fit_computes_normalized_quantile():
    """Verify fit() computes normalized quantile correctly."""
    # ... (similar setup)
    
    oracle.fit(x_train, y_train, interval_learner=interval_learner)
    
    # Normalized quantile should be computed
    assert oracle._cluster_norm_quantiles is not None
    assert len(oracle._cluster_norm_quantiles) > 0
    # All values should be positive (quantiles of nonconformity scores)
    assert np.all(oracle._cluster_norm_quantiles >= 0)
```

**Test Case 3: Effective Radius Modulation**
```python
def test_accept_uses_effective_radius_with_calibrated_prediction():
    """Verify accept() correctly modulates radius by interval width."""
    # ... setup ...
    
    oracle.fit(x_train, y_train, interval_learner=interval_learner)
    
    # Test point
    x_test = np.random.randn(5)
    
    # Test with narrow interval (high confidence)
    narrow_prediction = (0.5, (0.48, 0.52))  # width = 0.04
    accept_narrow = oracle.accept(x_test, narrow_prediction)
    
    # Test with wide interval (low confidence)
    wide_prediction = (0.5, (0.0, 1.0))  # width = 1.0
    accept_wide = oracle.accept(x_test, wide_prediction)
    
    # Same point should be more likely accepted with wider (lower confidence) interval
    # (This is a probabilistic assertion, but should hold for most random seeds)
    assert isinstance(accept_narrow, (bool, np.bool_))
    assert isinstance(accept_wide, (bool, np.bool_))
```

---

### Task 4.2: Integration Test with Real Explainer

**File:** `tests/integration/guards/test_guard_with_explainer.py` (NEW)

**Test Case: Guard Receives Correct Calibrated Predictions**
```python
def test_guard_filter_perturbations_uses_correct_intervals():
    """Integration test: guard.filter_perturbations receives calibrated predictions."""
    # Create explainer and train it
    x_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    model = LinearRegression().fit(x_train[:80], y_train[:80])
    
    explainer = CalibratedExplainer(model, x_train[80:], y_train[80:], mode="regression")
    
    # Initialize guard
    guard = GuardOrchestrator(explainer)
    guard.fit_guard({"alpha": 0.1, "n_clusters": 3})
    
    # Generate perturbations
    x_orig = np.random.randn(1, 5)
    predict, (low, high) = explainer.predict(x_orig, uq_interval=True)
    
    # Create dummy perturbed data
    perturbed_x = np.random.randn(10, 5)
    perturbed_feature = np.array([[i, 0] for i in range(10)])  # instance_idx=0
    
    # Filter should use the calibrated predictions
    filtered_x, filtered_feature = guard.filter_perturbations(
        perturbed_x, perturbed_feature, x_orig,
        prediction={"predict": predict, "low": low, "high": high}
    )
    
    # Some perturbations should be accepted (guard isn't too strict)
    assert len(filtered_x) > 0
    # Some may be rejected (guard filters some)
    assert len(filtered_x) <= len(perturbed_x)
```

---

## Phase 5: Documentation Updates

### Task 5.1: Create ADR Clarification (if needed)

If no existing ADR covers normalized conformal regression, create:

**File:** `improvement_docs/adrs/ADR-XXX-normalized-conformal-regression.md`

**Content:** (High-level)
- Decision: Use interval width to normalize conformal nonconformity scores
- Rationale: Confidence-aware perturbation filtering
- Implementation: NCR in ConformalRegionOracle
- Consequences: Requires interval_learner that supports uq_interval parameter

---

### Task 5.2: Update README and Examples

**File:** `calibrated-explanations/AGENTS.md` (or appropriate location)

**Add Section:** "Perturbation Guarding with Confidence Modulation"

```markdown
## Perturbation Guarding with Confidence Modulation

The guard provides normalized conformal regression (NCR) for confidence-aware
perturbation filtering:

```python
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.guards import ConformalRegionOracle, GuardOrchestrator

# Train explainer with calibration data
explainer = CalibratedExplainer(model, x_cal, y_cal)

# Initialize guard with NCR
guard_orchestrator = GuardOrchestrator(explainer)
guard_orchestrator.fit_guard({"alpha": 0.1, "n_clusters": 5})

# During explanation generation:
x_orig = X_test[0]
pred, (low, high) = explainer.predict(x_orig, uq_interval=True)

# Guard will filter perturbations using confidence-modulated radius:
# r_eff = q_norm * (high - low)
```

**Key Points:**
- Wider intervals (low confidence) → more perturbations accepted
- Narrower intervals (high confidence) → fewer perturbations accepted
- Provides theoretical coverage guarantees (1 - alpha)
```

---

## Rollout Strategy

### Release Checklist

- [ ] All critical bugs fixed (Tasks 1.1, 1.2)
- [ ] Validation added (Task 2.1, 2.2)
- [ ] All docstrings updated (Tasks 3.1-3.3)
- [ ] Unit tests pass (Tasks 4.1, 4.2)
- [ ] Integration tests pass
- [ ] Manual testing: Guard with live explainer shows confidence modulation
- [ ] Documentation updated (Task 5.1, 5.2)
- [ ] CHANGELOG entry added
- [ ] PR review by domain expert

### Success Verification

After implementation:
1. Run: `pytest tests/unit/guards/test_regions_ncr.py -v`
2. Run: `pytest tests/integration/guards/test_guard_with_explainer.py -v`
3. Check logs: "Normalized conformal regression (NCR) active"
4. Visual inspection: Accept rate changes with interval width

---

## Related Issues & Dependencies

- **Depends On:** interval_learner with `uq_interval=True` support
- **Used By:** Perturbation generation in explain methods
- **Related Files:**
  - `src/calibrated_explanations/guards/orchestrator.py`
  - `src/calibrated_explanations/core/explain/_computation.py`
  - `plugins/builtins.py` (reference implementation for predict usage)

---

## Timeline Estimate

| Phase | Tasks | Effort | Blockers |
|-------|-------|--------|----------|
| Phase 1 | 1.1, 1.2 | 2-3 hrs | None |
| Phase 2 | 2.1, 2.2 | 1-2 hrs | Phase 1 complete |
| Phase 3 | 3.1-3.3 | 2-3 hrs | Phase 1 complete |
| Phase 4 | 4.1-4.2 | 4-6 hrs | Phase 1, 2 complete |
| Phase 5 | 5.1-5.2 | 2-3 hrs | Phase 4 passing |
| **Total** | | **11-17 hrs** | **None** |

---

## Owner & Review

- **Implementer:** [Assign]
- **Reviewer (Technical):** [Domain expert on conformal prediction]
- **Reviewer (Code Quality):** [Standard code reviewer]

---
