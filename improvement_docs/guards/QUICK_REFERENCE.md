# Quick Reference: Guard Implementation Bugs

## TL;DR

**Problem:** ConformalRegionOracle cannot extract calibrated intervals from interval_learner due to unpacking bugs.

**Effect:** Confidence modulation is disabled; guard behaves like static conformal prediction.

**Fix:** Two lines in `regions.py` — change unpacking logic.

---

## The Two Critical Bugs

### Bug #1: Line 232-237 (Calibration Set Widths)

**Current (Broken):**
```python
intervals_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
if intervals_cal is not None and len(intervals_cal) == len(x_cal):
    widths_cal = np.array([upper - lower for lower, upper in intervals_cal])
```

**Fixed:**
```python
preds_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
if preds_cal is not None and len(preds_cal) == len(x_cal):
    widths_cal = upper - lower  # Direct array subtraction
```

**Why:** 
- `predict(uq_interval=True)` returns `(predictions, (bounds))`
- Cannot iterate over predictions with tuple unpacking
- Use direct array subtraction instead

---

### Bug #2: Line 299-302 (Full Training Set Width Stats)

**Current (Broken):**
```python
intervals = interval_learner.predict(x)  # Missing uq_interval=True!
if intervals is not None and len(intervals) > 0:
    widths = np.array([upper - lower for lower, upper in intervals])
```

**Fixed:**
```python
prediction_full, (lower_full, upper_full) = interval_learner.predict(x, uq_interval=True)
if prediction_full is not None and len(prediction_full) > 0:
    widths = upper_full - lower_full
```

**Why:**
- Missing `uq_interval=True` parameter
- `predict(x)` returns only predictions, not intervals
- Need both lower and upper bounds to compute widths

---

## Verification Checklist

After applying fixes, verify:

- [ ] Import numpy: `np.array(...)` syntax works
- [ ] Width computation: `widths_cal = upper - lower` is array subtraction
- [ ] Width stats: `_width_min` and `_width_max` are computed (not defaulting to 0.0, 1.0)
- [ ] Normalized quantiles: `_cluster_norm_quantiles` is not None
- [ ] Log output includes: "Normalized conformal regression (NCR) active"
- [ ] Confidence modulation: `r_eff = q_norm * width_test` is used in accept()

---

## What Gets Fixed

| Aspect | Before | After |
|--------|--------|-------|
| Width extraction | Crashes silently | Works ✓ |
| Normalized scores | Never computed | Computed correctly ✓ |
| Confidence modulation | Disabled | Active ✓ |
| Effective radius | Static | Adapts to confidence ✓ |
| Test acceptance rate | Independent of interval width | Depends on width ✓ |

---

## How to Test

```python
from calibrated_explanations.guards import ConformalRegionOracle
from sklearn.linear_model import LinearRegression
from calibrated_explanations import CalibratedExplainer
import numpy as np

# Create data
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Train explainer with calibration
model = LinearRegression().fit(X[:80], y[:80])
explainer = CalibratedExplainer(model, X[80:], y[80:], mode="regression")

# Fit guard
oracle = ConformalRegionOracle(alpha=0.1, n_clusters=3)
oracle.fit(X, y, interval_learner=explainer.interval_learner)

# Check if confidence modulation is active
print(f"Width stats: min={oracle._width_min}, max={oracle._width_max}")
print(f"Has normalized quantiles: {oracle._cluster_norm_quantiles is not None}")

# Expected output after fix:
# Width stats: min=X (not 0.0), max=Y (not 1.0)
# Has normalized quantiles: True (not None)
```

---

## Impact on Broader System

```
explain()
  ├─ Get predictions with intervals: predict(x, uq_interval=True)
  ├─ Generate perturbations
  ├─ Filter with guard
  │  ├─ Guard should scale acceptance by interval width
  │  ├─ Currently: Always uses static radius (BUG)
  │  ├─ After fix: Uses r_eff = q_norm * width (CORRECT)
  └─ Return filtered explanations
```

---

## Files to Change

Only **one file** needs modification:

```
src/calibrated_explanations/guards/regions.py
```

Changes needed at:
- Line 232: Unpacking
- Line 234: Width calculation  
- Line 299: Add `uq_interval=True`
- Line 302: Width calculation (again)

---

## Rollback Plan

If problems arise:
1. Revert `regions.py` to previous version
2. Guard will revert to static conformal prediction
3. Still safe but no confidence modulation

---

## Questions & Answers

**Q: Will this break existing code?**  
A: No. It only fixes the confidence modulation feature; the guard still works without it (falls back to static behavior).

**Q: Do I need to update other files?**  
A: No immediate changes needed to other files, but added comprehensive documentation in `improvement_docs/guards/`.

**Q: How do I know if confidence modulation is working?**  
A: Check logs for "Normalized conformal regression (NCR) active" and verify `_width_min != _width_max`.

**Q: What if interval_learner returns bad data?**  
A: Exception handling (try/except) is already in place at lines 236 and 317.

---

## Related Documentation

See full details in:
- `improvement_docs/guards/IMPLEMENTATION_ANALYSIS.md` — Complete gap analysis
- `improvement_docs/guards/IMPLEMENTATION_PLAN.md` — Detailed implementation plan
- `improvement_docs/guards/FINDINGS_SUMMARY.md` — Summary and context
