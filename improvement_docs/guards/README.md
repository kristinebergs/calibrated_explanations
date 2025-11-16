# Perturbation Guard Analysis - Document Index

**Analysis Complete:** November 15, 2025

This folder contains comprehensive analysis of the ConformalRegionOracle implementation and an implementation plan to fix identified gaps.

---

## Documents Overview

### 1. **QUICK_REFERENCE.md** ⚡ Start Here

- 2-minute overview of the bugs
- Before/after code comparison
- Verification checklist
- Quick testing guide

**Use when:** You need to understand the problem quickly

---

### 2. **FINDINGS_SUMMARY.md** 📊 Executive Summary

- High-level findings and key insights
- Root cause analysis
- Connection to broader explainer architecture
- Five critical questions with answers
- Success metrics post-fix

**Use when:** You need context and justification for the fix

---

### 3. **IMPLEMENTATION_ANALYSIS.md** 🔬 Deep Dive Analysis

- Complete gap analysis with all 5 identified gaps
- Verification checklist (what works, what doesn't)
- Related code references
- Comprehensive recommendations

**Use when:** You're implementing the fix or need full technical details

---

### 4. **IMPLEMENTATION_PLAN.md** 📋 Action Plan

- Phased implementation roadmap (5 phases)
- Step-by-step tasks with code examples
- Validation and error handling requirements
- Unit and integration test specifications
- Documentation update requirements
- Timeline estimate and resource planning

**Use when:** You're assigned to implement the fixes

---

## At a Glance

### The Problem
The ConformalRegionOracle has **critical unpacking bugs** that prevent it from extracting calibrated intervals from the interval_learner, disabling the core feature: confidence modulation through normalized conformal regression. Additionally, the guard's clustering pipeline must run on the **concatenated vector of each perturbation's input features and its calibrated prediction/probability** so that cluster assignments respect both feature geometry and calibrated outputs before normalization.

### The Impact
- Confidence modulation is always disabled
- Guard falls back to static conformal prediction
- Loses intended per-prediction uncertainty adaptation

### The Fix
- **2 bug locations** in `src/calibrated_explanations/guards/regions.py`
- **~10 lines of code** need correction
- **Straightforward changes:** fix unpacking, add one parameter

### Timeline
- Phase 1 (Critical bugs): 2-3 hours
- Phase 2 (Validation): 1-2 hours
- Phase 3 (Documentation): 2-3 hours
- Phase 4 (Tests): 4-6 hours
- Phase 5 (Updates): 2-3 hours
- **Total: 11-17 hours**

---

## Key Concepts

### Normalized Conformal Regression (NCR)

The intended feature:

```
For each perturbation:
  1. Concatenate the perturbation input with its calibrated prediction/probability before clustering
  2. Compute Mahalanobis distance to nearest cluster in this augmented space
  3. Get original prediction's interval width: w = upper - lower
  4. Scale acceptance radius by width: r_eff = q_norm * w
  5. Accept if: mahal_dist ≤ r_eff

Result:
  - Narrow intervals (high confidence) → strict filtering
  - Wide intervals (low confidence) → lenient filtering
```

### The Two Critical Bugs

**Bug #1 (Line 232):** Unpacking error in calibration set width extraction
```python
# Wrong: tries to unpack predictions as interval tuples
intervals_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
widths_cal = np.array([upper - lower for lower, upper in intervals_cal])

# Right: unpacks correctly, uses direct array subtraction
preds_cal, (lower, upper) = interval_learner.predict(x_cal, uq_interval=True)
widths_cal = upper - lower
```

**Bug #2 (Line 299):** Missing parameter in full training set width computation
```python
# Wrong: missing uq_interval=True
intervals = interval_learner.predict(x)

# Right: includes required parameter
prediction_full, (lower_full, upper_full) = interval_learner.predict(x, uq_interval=True)
```

---

## How to Navigate

1. **Just want the facts?** → Read QUICK_REFERENCE.md
2. **Need to understand why?** → Read FINDINGS_SUMMARY.md
3. **Going to implement?** → Follow IMPLEMENTATION_PLAN.md
4. **Need all details?** → See IMPLEMENTATION_ANALYSIS.md

---

## Status Tracking

- [x] Root cause identified
- [x] Gap analysis completed
- [x] Reference implementations found
- [x] Documentation created
- [x] Implementation plan detailed
- [ ] Bugs fixed
- [ ] Tests written
- [ ] Code reviewed
- [ ] Changes merged

---

## Related Files in Repository

### Code Files
- `src/calibrated_explanations/guards/regions.py` — Contains the bugs
- `src/calibrated_explanations/guards/orchestrator.py` — Uses the guard
- `src/calibrated_explanations/plugins/builtins.py` — Reference implementation
- `src/calibrated_explanations/core/calibrated_explainer.py` — predict() method

### Test Files
- `tests/unit/guards/` — Existing guard tests
- `tests/integration/guards/` — Existing integration tests

### Documentation
- `docs/` — General documentation
- `improvement_docs/` — This analysis folder

---

## Contact & Questions

For questions about this analysis, see the specific document sections:
- Technical details → IMPLEMENTATION_ANALYSIS.md
- Implementation steps → IMPLEMENTATION_PLAN.md
- High-level overview → FINDINGS_SUMMARY.md

---

## Last Updated

November 15, 2025

**Status:** Complete - Ready for implementation phase
