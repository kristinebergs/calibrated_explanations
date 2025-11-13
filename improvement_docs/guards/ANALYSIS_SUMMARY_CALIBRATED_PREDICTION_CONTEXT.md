# Guard Threshold Elimination Analysis - Summary

**Date:** November 13, 2025  
**Status:** ✅ Analysis Complete & Final Design Locked  
**Related Documents:**
- 📐 `GUARD_MATHEMATICAL_FOUNDATIONS.md` — Formal conformal prediction theory
- 🏗️ `GUARD_DESIGN_CONFIDENCE_MODULATION.md` — Implementation architecture
- 📊 `GUARD_CALIBRATED_PREDICTION_CONTEXT_ANALYSIS.md` — Historical approach analysis (superseded)

---

## Final Design: Confidence-Modulated Conformal Regions

### Key Finding

**Eliminate both thresholds and categorical contexts.** Instead, implement **confidence-modulated conformal regions**, which:

1. ✅ Eliminates threshold requirement completely (for all tasks, not just classification)
2. ✅ Eliminates `context_mode` parameter (single unified approach)
3. ✅ Uses **continuous calibrated confidence** instead of categorical binning
4. ✅ Provides strongest possible guarantees (distribution-free conformal validity)
5. ✅ Works identically for classification and regression
6. ✅ Simpler API (fewer parameters, more interpretable)

---

## The Solution in One Sentence

Use **feature-space clustering** to find representative regions, compute **per-cluster conformal radii** (based on Mahalanobis distance), and **modulate acceptance by calibrated interval width** to adapt to model confidence:

$$r_{\text{eff}}(w) = r_{\text{base}} \cdot \left(1 + (1 - c(w)) \cdot \lambda\right)$$

where:
- $r_{\text{base}}$ = conformal radius for the cluster
- $w$ = interval width from calibrated predictor
- $c(w) = 1 - \frac{w - w_{\min}}{w_{\max} - w_{\min}}$ = normalized confidence [0, 1]
- $\lambda \geq 0$ = relaxation factor (controls flexibility)

---

## Why This Is Better

| Criterion | Threshold | Categorical Context | **Confidence Modulation** |
|-----------|-----------|-------------------|---------------------------|
| **Threshold req.** | ✗ Mandatory | ✗ Mandatory | ✅ None |
| **Categorical binning** | Binary | 4-way or n-way | ✅ None (continuous) |
| **For regression** | ✗ Weak | ✗ Loses structure | ✅ Works naturally |
| **For classification** | ✓ Works | ✓ Works | ✅ Works naturally |
| **Parameter count** | Many (threshold + others) | Many (quantile thresholds) | ✅ Few (relaxation_factor) |
| **Confidence awareness** | ✗ None | Partial (binned) | ✅ Continuous |
| **Theory** | Conformal (weak) | Conformal + calibration | ✅ **Conformal + modulation** |
| **Implementation complexity** | Simple | Moderate | ✅ Moderate (cleaner) |
| **API simplicity** | Simple | Simple | ✅ **Simplest** |

---

## Implementation Overview

### API Before (Old Threshold Approach)
```python
# ❌ Mandatory threshold, arbitrary for regression
guard = ConformalRegionOracle(alpha=0.1, mode="regression", threshold=50_000)
guard.fit(X_train, y_train)
```

### API After (Categorical Contexts - Rejected)
```python
# ❌ Still needs quantile thresholds, adds complexity
guard = ConformalRegionOracle(alpha=0.1, mode="regression", 
                               context_mode="calibrated", 
                               quantile_thresholds=(0.5, 0.5))
guard.fit(X_train, y_train, interval_learner=explainer.interval_learner)
```

### API Final (Confidence Modulation - Implemented)
```python
# ✅ No thresholds, no contexts, no mode parameter needed
explainer = CalibratedExplainer(model, X_cal, y_cal, mode="regression")
guard = ConformalRegionOracle(alpha=0.1, relaxation_factor=1.0, n_clusters=5)
guard.fit(X_train, y_train, model=model, interval_learner=explainer.interval_learner)

# Use the same guard for both classification and regression!
```

---

## What Changes in Implementation

### `ConformalRegionOracle.__init__`

**REMOVE parameters:**
- `threshold` ❌ (was only used for regression binarization)
- `context_mode` ❌ (no longer needed; always use modulation)
- `quantile_thresholds` ❌ (no binning)
- `mode` ❌ (classifier/regression distinction removed)

**KEEP parameters:**
- `alpha` ✓ — Conformal miscalibration level
- `n_clusters` ✓ — Number of feature-space clusters
- `covariance` ✓ — Covariance type (full, diag, shrunk)
- `random_state` ✓ — Reproducibility
- `ncm_method` ✓ — Nonconformity measure

**ADD parameters:**
- `relaxation_factor` (λ) — Controls confidence modulation flexibility (default: 1.0)

---

### `ConformalRegionOracle.fit`

**ADD parameters:**
- `model` — Required; the trained classifier/regressor (for getting predictions)
- `interval_learner` — Required; calibrated interval predictor (for getting confidence)

**REMOVE steps:**
- Label binarization for regression ❌
- Context-based region construction ❌

**ADD steps:**
- Single global K-means clustering on training features
- Per-cluster conformal radius computation (Mahalanobis distance based)
- Interval width statistics computation ($w_{\min}, w_{\max}$)

---

### `ConformalRegionOracle.accept`

**Replaces context-based check with confidence modulation:**

```python
def accept(x_prime, x_original):
    # 1. Find nearest cluster
    cluster_id = nearest_cluster(x_original, centers)
    base_radius = radii[cluster_id]
    
    # 2. Compute distance
    dist = mahalanobis_distance(x_prime, centers[cluster_id])
    
    # 3. Get confidence
    interval = interval_learner.predict(x_original)
    width = interval[1] - interval[0]
    confidence = 1 - (width - w_min) / (w_max - w_min)
    
    # 4. Modulate radius
    effective_radius = base_radius * (1 + (1 - confidence) * relaxation_factor)
    
    # 5. Accept/reject
    return dist <= effective_radius
```

---

## Theoretical Guarantee

**Old approach:**
> Perturbation is in-distribution if within radius of binary-labeled region.

**Categorical approach (rejected):**
> Perturbation is in-distribution if within radius of 4-way categorized region.

**New approach (implemented):**
> Perturbation is in-distribution if within **confidence-modulated** radius of feature-space cluster, where confidence modulation preserves conformal validity while adapting to model's stated confidence.

### Formal Statement

Let:
- $r_k$ = conformal radius for cluster $k$ (guaranteed by ICP)
- $r_{\text{eff}}(w) = r_k \cdot f(w)$ = modulated radius
- $f(w)$ = monotone non-decreasing modulation function, $f(w) \geq 1$

**Theorem:** The acceptance criterion:
$$A(x') \leq r_{\text{eff}}(w(x))$$

provides coverage guarantee:
$$\Pr[\text{accept} \mid x \text{ in-distribution}] \geq 1 - \alpha$$

**Proof:** By conformal validity + monotone modulation property (see `GUARD_MATHEMATICAL_FOUNDATIONS.md`)

---

## Design Evolution Summary

| Phase | Approach | Status | Why Rejected/Why Chosen |
|-------|----------|--------|---------------------------|
| 1 | **Threshold-based** | ✗ Rejected | Arbitrary, no semantic meaning, fails for regression |
| 2 | **Auto-threshold** (median, quantile, etc.) | ✗ Rejected | Still arbitrary, just hidden from user |
| 3 | **Categorical contexts** (4-way binning) | ✗ Rejected | User questioned necessity; loses continuous structure |
| 4 | **Confidence modulation** (continuous) | ✅ **Accepted** | Simplest, most elegant, theoretically rigorous, unified for all tasks |

---

## Key Advantages of Final Design

### 1. **Theoretically Sound**
- Built on conformal prediction (distribution-free validity)
- Modulation by monotone function preserves guarantees
- Formal proofs in `GUARD_MATHEMATICAL_FOUNDATIONS.md`

### 2. **Unified Across Tasks**
- Same API for classification and regression
- Same mechanism (confidence modulation)
- No special cases or mode parameters

### 3. **Parameter-Parsimonious**
- Eliminates all arbitrary thresholds
- Only semantic parameter: `relaxation_factor` (controls flexibility)
- Other parameters (`alpha`, `n_clusters`) are standard conformal hyperparameters

### 4. **Adaptive to Model Confidence**
- High-confidence predictions → strict perturbation bounds
- Low-confidence predictions → lenient perturbation bounds
- Modulation is continuous, not categorical

### 5. **Interpretable**
- Mahalanobis distance → "how far from cluster center"
- Interval width → "how confident is the model"
- Modulation → "adjust radius based on confidence"
- All quantities are understandable and debuggable

---

## Implementation Roadmap

1. **Modify `ConformalRegionOracle`** (src/calibrated_explanations/guards/regions.py)
   - Update `__init__` signature
   - Rewrite `fit()` method
   - Rewrite `accept()` method
   - Add `_normalize_confidence()` helper
   - Remove context-based code

2. **Update `CalibratedExplainer.set_guard()`** (src/calibrated_explanations/core/calibrated_explainer.py)
   - Pass `model` and `interval_learner` to `guard.fit()`

3. **Add comprehensive tests**
   - Modulation function behavior
   - Confidence normalization
   - Integration with calibrators
   - Comparison tests (vs. baseline)

4. **Update documentation & examples**
   - Update API docs
   - Update quickstart notebook
   - Add regression example

---

## Next Steps

Proceed to:
1. 📐 Review `GUARD_MATHEMATICAL_FOUNDATIONS.md` for formal proofs
2. 🏗️ Review `GUARD_DESIGN_CONFIDENCE_MODULATION.md` for implementation details
3. 🔨 Begin implementation (see checklist in design doc)

**Why it's stronger:**
- Threshold is arbitrary and independent of model behavior
- Calibrated context aligns with where model operates
- Confidence is respected (tight regions for high-confidence predictions)
- Formal guarantee via conformal prediction theory + calibration framework

---

## Implementation Effort

- **Complexity:** Medium (100–150 LOC)
- **Time:** 3–5 days (including tests & documentation)
- **Risk:** Low (feature not yet public, no backward compatibility needed)
- **Payoff:** High (eliminates user pain point, improves guarantees)

### Files to Modify
1. `src/calibrated_explanations/guards/regions.py` — Context computation logic
2. `src/calibrated_explanations/core/calibrated_explainer.py` — Integration hook
3. `notebooks/quickstart_guard.ipynb` — Example demonstrating regression workflow
4. Tests & documentation (standard)

---

## Testing Strategy

### Unit Tests
- Context computation: verify 4-way split, thresholds at quantiles
- Balance: ensure ~25% in each context
- Stability: same seed → same contexts

### Integration Tests
- End-to-end regression: explain_factual works without threshold
- OOB rate ≈ α (guard is well-calibrated)
- Explanations are stable and reasonable

### Validation
- Run on regression datasets (housing, energy, stock prices)
- Compare with old approach (if temporarily maintained)
- Check explanation quality metrics

---

## Edge Cases & Mitigations

| Edge Case | Scenario | Handling |
|-----------|----------|----------|
| **Poor model** | Noisy predictions | Warning; wide regions (acceptable) |
| **Imbalanced contexts** | Few samples in a context | Validation warning; min 5% per context |
| **High model confidence** | Narrow intervals everywhere | Contexts collapse to high-confidence; acceptable |
| **High model uncertainty** | Wide intervals everywhere | Contexts collapse to low-confidence; acceptable |
| **Distribution shift** | Test ≠ training | Guard rejects more; correct behavior |

---

## Success Criteria

✅ **Implementation:**
- Guard fits without explicit threshold
- Contexts computed from predictions + intervals
- All tests pass (no regression)
- Performance unaffected

✅ **Quality:**
- OOB rate ≈ α
- Explanations stable
- Rule quality comparable or better

✅ **User Experience:**
- No "what threshold should I use?" questions
- Clear documentation of context semantics
- Working example in notebook

---

## Next Steps

1. ✅ **Review this analysis** — Ensure recommendation aligns with design philosophy
2. **Plan sprint** — Allocate 3–5 days for implementation
3. **Implement Phase 1** — Core context computation logic
4. **Add tests** — Unit + integration
5. **Update documentation** — Docstrings + example notebook
6. **Validate** — Run empirical tests on regression datasets

---

## Why Not Just Auto-Threshold?

**Auto-threshold (median, quantile, etc.) is tempting but insufficient:**

❌ Still arbitrary (median is just as arbitrary as user-provided threshold)  
❌ Doesn't use model predictions or confidence  
❌ Loses regression structure (still binary)  
❌ Doesn't leverage calibration machinery already in place  
❌ Doesn't improve guarantees over manual threshold  

**Calibrated prediction context is the right solution:**

✅ Principled (model predictions + calibration)  
✅ Automatic (no parameter tuning)  
✅ Stronger guarantees (conformal + calibration aligned)  
✅ Richer structure (multi-way context)  
✅ Leverages existing calibration framework  

---

## Document Location

📄 **Full analysis:** `improvement_docs/guards/GUARD_CALIBRATED_PREDICTION_CONTEXT_ANALYSIS.md`

Sections:
1. Executive Summary
2. Current Architecture & Limitations (Problem analysis)
3. Why Calibrated Predictions Are the Solution (Theory)
4. Proposed Design (Detailed architecture)
5. Implementation Plan (Phase 1 roadmap)
6. Advantages & Guarantees Comparison (Benchmarking)
7. Edge Cases & Mitigations (Robustness)
8. Migration Path (User-facing changes)
9. Success Criteria (Definition of done)
10. Code Snippets (Ready to implement)

---

**Status:** ✅ Ready for technical review and implementation planning.
