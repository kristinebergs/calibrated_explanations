# Guard Implementation Analysis: Conformal Clustering with Interval Learning

**Date:** November 15, 2025  
**Status:** Gap Analysis Complete - Implementation Plan Generated

## Executive Summary

The `ConformalRegionOracle` is designed as a Conformal clustering-based solution for perturbation guarding. It enriches perturbed instances with calibrated prediction/probability intervals and uses the **width of the uncertainty interval** to normalize conformal out-of-distribution filtering.

**Current Status:** The implementation has **critical bugs** in how it calls and unpacks the `interval_learner.predict()` method, preventing proper uncertainty interval extraction and normalization.

---

## Intended Architecture

### Design Principles

1. **Conformal Prediction Foundation**
   - Uses inductive conformal prediction (ICP) split
   - Proper set (75%) for clustering
   - Calibration set (25%) for conformal radius computation

2. **Feature-Space Clustering**
   - KMeans clustering on proper set to capture heteroscedasticity
   - Per-cluster covariance matrices computed
   - Mahalanobis distance-based nonconformity scores

3. **Normalized Conformal Regression (NCR)**
   - **Core Feature:** Interval width is used to normalize nonconformity scores
   - Width normalization formula: `s_norm = s_raw / (width + eps)`
   - Enables adaptive acceptance radius based on prediction confidence
   - Wider intervals (low confidence) → larger acceptance regions
   - Narrower intervals (high confidence) → smaller acceptance regions

4. **Confidence Modulation**
   - Effective radius computed as: `r_eff = q_norm * width_test`
   - Where `q_norm` is the (1 - α) quantile of normalized scores
   - Dynamic radius scales with test-time prediction uncertainty

### How Explain Uses Intervals (Reference Implementation)

From `plugins/builtins.py` (LegacyPredictBridge):
```python
prediction = self._explainer.predict(x, uq_interval=True, bins=bins)
if isinstance(prediction, tuple):
    preds, interval = prediction
    low, high = interval  # Unpack (lower, upper) bounds
```

The return signature of `predict(uq_interval=True)` is:
```
(predictions, (lower_bounds, upper_bounds))
```

- `predictions`: array of prediction values (regression) or probabilities (classification)
- `lower_bounds`: array of lower uncertainty bounds
- `upper_bounds`: array of upper uncertainty bounds

---

## Gap Analysis

### Gap 1: Incorrect Unpacking of interval_learner.predict() Return Value

**Location:** `regions.py`, lines 232-237

**Current Code:**
```python
intervals_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
if intervals_cal is not None and len(intervals_cal) == len(x_cal):
    widths_cal = np.array([upper - lower for lower, upper in intervals_cal])
else:
    widths_cal = np.ones(len(cal_scores))
```

**Problem:**
- Line 232 unpacks as `(intervals_cal, (lower, upper))` 
- But `predict(uq_interval=True)` returns `(predictions, (lower, upper))`
- Variable name `intervals_cal` is misleading; it's actually predictions
- Line 234 tries to iterate over `intervals_cal` with tuple unpacking `for lower, upper in intervals_cal`
- This assumes `intervals_cal` contains interval tuples, but it contains scalar predictions

**Impact:**
- Runtime TypeError when trying to unpack scalars
- Exception is silently caught (line 236), falling back to uniform widths
- Confidence modulation never activates

**Correct Pattern (from explain code):**
```python
preds, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
widths_cal = upper - lower  # Direct array subtraction
```

---

### Gap 2: Incorrect Unpacking When Computing Width Statistics

**Location:** `regions.py`, lines 299-302

**Current Code:**
```python
intervals = interval_learner.predict(x)  # NO uq_interval parameter!
if intervals is not None and len(intervals) > 0:
    widths = np.array([upper - lower for lower, upper in intervals])
```

**Problems:**
1. Missing `uq_interval=True` parameter → returns only predictions, not intervals
2. Tries to unpack non-tuple return value as if it were intervals
3. Variable name `intervals` is confusing for a predictions array
4. Line 302 attempts tuple unpacking that will fail

**Impact:**
- Exception on line 302 since `intervals` is a 1D array, not a list of tuples
- Falls back to default `_width_min=0.0, _width_max=1.0`
- Width statistics never captured for confidence modulation

---

### Gap 3: Incomplete Documentation of Normalized Conformal Regression

**Location:** `regions.py`, method docstrings

**Current State:**
- `fit()` docstring mentions widths but lacks detail on normalized conformal regression
- `accept()` docstring explains effective radius but not the normalization formula
- Comments in code (lines 106, 242, 497) hint at the mechanism but aren't comprehensive

**Impact:**
- Maintenance burden; future developers won't understand the design intent
- No clear contract for what `_cluster_norm_quantiles` represents

---

### Gap 4: Missing Parameter Validation for interval_learner

**Location:** `regions.py`, line 150

**Current Code:**
```python
if interval_learner is None:
    raise ValueError("interval_learner must be provided; None is not allowed")
```

**Missing:**
- No check that `interval_learner.predict()` supports `uq_interval=True` parameter
- No validation that predict returns the expected tuple structure
- Silent fallback to uniform widths on any exception (line 236)

**Impact:**
- Silent failures are hard to debug
- Users don't know if their interval learner is compatible
- Warnings logged but not obvious to end users

---

### Gap 5: Inconsistency Between Fit and Accept

**Location:** `regions.py`, fit vs. accept methods

**Issue:**
- `fit()` attempts to normalize nonconformity scores by interval width (lines 242-247)
  ```python
  s_cal_norm = self._cal_scores / (widths_cal + self._eps_width)
  global_norm_q = float(np.quantile(s_cal_norm, 1.0 - self.alpha))
  self._cluster_norm_quantiles = np.full(n_clusters_actual, global_norm_q)
  ```

- But `accept()` never validates that the input `x_new` comes with its corresponding calibrated prediction (lines 447-465)
- Without `calibrated_prediction`, cannot compute effective radius
- Falls back to legacy base_radius without normalization

**Impact:**
- Design intent of normalized conformal regression not fully realized
- Confidence modulation only works if caller provides `calibrated_prediction`
- Guard behavior is inconsistent across different call patterns

---

## Summary of Gaps

| Gap | Severity | Category | Location |
|-----|----------|----------|----------|
| Incorrect predict() unpacking | **CRITICAL** | Implementation Bug | regions.py:232-237 |
| Missing uq_interval=True parameter | **CRITICAL** | Implementation Bug | regions.py:299 |
| Incomplete NCR documentation | **HIGH** | Documentation | regions.py docstrings |
| Missing interval_learner validation | **MEDIUM** | Error Handling | regions.py:150+ |
| Accept/Fit inconsistency | **MEDIUM** | Design Gap | regions.py methods |

---

## Verification Checklist

### What IS Implemented Correctly ✓

1. ✓ Conformal prediction framework (ICP split, clustering, covariance)
2. ✓ Mahalanobis distance-based nonconformity scores
3. ✓ Caching of calibration scores and widths for alpha adjustment
4. ✓ Normalized conformal regression quantile computation (lines 242-247)
5. ✓ Effective radius modulation in accept() (lines 447-465)
6. ✓ Per-feature interval computation in intervals() method
7. ✓ Batch accept processing

### What IS NOT Working ✓

1. ✗ Width extraction from interval_learner (line 234)
2. ✗ Width statistics computation (line 302)
3. ✗ Flow of normalized scores → normalized quantiles → effective radius
4. ✗ Confidence modulation in practice (never activates due to Gap 1)

---

## Related Code References

### Explain's Correct Usage Pattern
- File: `plugins/builtins.py` (line 96-100)
- File: `core/calibrated_explainer.py` (line 1849-1880)
- Pattern: `preds, (low, high) = explainer.predict(x, uq_interval=True)`

### Orchestrator's Prediction Handling
- File: `guards/orchestrator.py` (line 122-145)
- Shows filter_perturbations expects `(predict, low, high)` tuple structure
- Correctly constructs calibrated_predictions for guard acceptance

### Interval Learner Return Values
- File: `core/calibration/interval_regressor.py` (line 202-206, 244-260)
- Returns: `(probabilities, interval_lower, interval_upper, None)` for classification
- Returns: `(predictions, lower_bounds, upper_bounds, None)` for regression

---

## Recommendations

1. **Immediate Fix (Critical)**
   - Correct unpacking logic in lines 232-237
   - Add `uq_interval=True` parameter in line 299
   - Update width calculation to direct array subtraction

2. **Short-term (High Priority)**
   - Add comprehensive documentation for normalized conformal regression
   - Add parameter validation and informative error messages
   - Add unit tests verifying width extraction and normalization

3. **Medium-term (Architecture)**
   - Consider making `calibrated_prediction` parameter mandatory in accept()
   - Add diagnostic logging to verify confidence modulation is active
   - Document the contract between explainer and guard

4. **Long-term (Documentation)**
   - Create design document explaining normalized conformal regression
   - Document calibration diagnostics and how to interpret them
   - Add examples showing proper guard initialization and usage
