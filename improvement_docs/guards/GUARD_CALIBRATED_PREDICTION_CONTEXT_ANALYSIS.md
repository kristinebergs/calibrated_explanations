# In-Depth Guard Analysis: Eliminating Threshold via Calibrated Prediction Context

**Date:** November 13, 2025
**Status:** Technical Analysis (Ready for Implementation)
**Scope:** Architecture redesign to eliminate regression threshold requirement
**Audience:** Technical team, Architecture review

---

## Executive Summary

The current `ConformalRegionOracle` requires an explicit `threshold` parameter for regression mode, binarizing continuous targets into artificial classes. This analysis proposes **a unified solution leveraging calibrated predictions with uncertainty intervals** as the basis for context definition, applicable to both classification and regression with **stronger theoretical guarantees** and **cleaner API semantics**.

### Key Insight

Instead of user-provided or auto-determined thresholds, use the **calibrated prediction intervals** from `CalibratedExplainer`'s built-in calibration machinery (Venn-Abers for classification, CPD for regression) to define contexts:

- **Classification:** Combine predicted class + confidence level → multi-way context
- **Regression:** Combine predicted value + interval width → multi-way context

This approach:
1. ✅ Eliminates threshold entirely
2. ✅ Aligns regions with model behavior + calibration
3. ✅ Provides strongest in-distribution guarantees
4. ✅ Unifies classification and regression under one semantics
5. ✅ No backward compatibility burden (not yet public)

---

## Part 1: Current Architecture & Limitations

### 1.1 The Threshold Requirement

**Current regression flow:**

```python
guard = ConformalRegionOracle(alpha=0.1, mode="reg", threshold=50_000)
guard.fit(X_train, y_train)
```

**What happens internally (lines 109–118, `regions.py`):**

```python
if self.mode == "reg":
    if self.threshold is None:
        raise ValueError("Threshold must be provided for regression mode")
    labels = np.array([0, 1])
    y_prop = (y_prop >= self.threshold).astype(int)  # Binarize!
    y_calib = (y_calib >= self.threshold).astype(int)
```

The threshold converts continuous regression into binary classification:
- Context `0`: $ y < \text{threshold} $
- Context `1`: $ y \geq \text{threshold} $

### 1.2 The Three Core Problems

#### Problem 1: No Principled Threshold Selection
- User must guess: median? mean? domain value? percentile?
- Example: house price regression — use $500k? $100k? Arbitrary.
- **Result:** Users avoid guards in regression mode.

#### Problem 2: Information Loss & Weak Context Semantics

**Example:**

```
y_train = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
threshold = 50

Binary contexts:
  0 (y < 50):  {10, 20, 30, 40}
  1 (y ≥ 50):  {60, 70, 80, 90, 100}

Problem:
- Values 10 and 40 share a region, but are far apart in y-space
- Values 60 and 100 share a region, but are far apart in y-space
- Regions don't reflect true data structure
- In-distribution guarantee is weak: "similar to training data ≥ $50k"
  doesn't mean "in model's natural decision regime"
```

#### Problem 3: Weak In-Distribution Guarantees

**Current guarantee (informal):**
> "A perturbation is in-distribution if within a Mahalanobis ball of points with arbitrary label value."

**Issues:**
1. Binarization is arbitrary (independent of model behavior)
2. Regions don't account for how the model actually predicts
3. No connection to prediction confidence
4. Binary split loses valuable regression structure

---

## Part 2: Why Calibrated Predictions Are the Solution

### 2.1 The Calibration Machinery Already Exists

`CalibratedExplainer` has two powerful calibration mechanisms:

**Classification (Venn-Abers, in `venn_abers.py`):**
```python
class VennAbers:
    def predict(self, x) -> Tuple[float, float]:
        """Return (low, high) confidence interval for class probability."""
```

Returns **calibrated prediction with uncertainty interval**.

**Regression (Conformal Predictive Distribution, in `interval_regressor.py`):**
```python
class IntervalRegressor:
    def predict(self, x) -> Tuple[float, float]:
        """Return (low, high) prediction interval for continuous target."""
```

Returns **prediction interval reflecting model's uncertainty**.

### 2.2 Why Calibrated Predictions Are Better Than Thresholds

| Aspect | Threshold | Calibrated Prediction |
|--------|-----------|------------------------|
| **Semantics** | Arbitrary split | Model behavior + reliability |
| **Adaptability** | Fixed across data | Per-instance uncertainty |
| **Guarantees** | Weak (label-based) | Strong (prediction + calibration aligned) |
| **User input** | Required | None (automatic) |
| **Regression structure** | Lost (binary) | Preserved via intervals |
| **Classification logic** | None (just labels) | Combined with confidence |

### 2.3 The Unified Context Formula

**For any sample $ i $ with prediction $ \hat{y}_i $ and interval width $ w_i $:**

Define a **multi-way context** by binning both:

$$\text{context}(x_i) = f(\hat{y}_i, w_i)$$

where $ f $ maps to discrete contexts, e.g., 4-way:

| Prediction | Confidence | Context Label |
|------------|-----------|----------------|
| High | High | 3 |
| High | Low | 2 |
| Low | High | 1 |
| Low | Low | 0 |

Or more granularly (e.g., quintile of each dimension) → 25-way context.

**Why this works:**

1. Regions are defined by **model's predicted behavior** (not arbitrary threshold)
2. **Confidence is respected** (tight regions for high-confidence, relaxed for low-confidence)
3. **Perturbations align with model's operating regime** (predictions similar → contexts often similar)
4. **Continuous structure is preserved** (not binary, but stratified by prediction + confidence)

---

## Part 3: Proposed Design

### 3.1 High-Level Architecture

**New parameters for `ConformalRegionOracle`:**

```python
class ConformalRegionOracle:
    def __init__(
        self,
        alpha=0.1,
        mode="clf",           # "clf" or "reg"
        context_mode="calibrated",  # NEW: "calibrated" (recommended for reg)
        quantile_thresholds=None,   # NEW: (pred_quantile, conf_quantile)
        n_clusters=5,
        # ... rest unchanged
    ):
```

**New fit signature:**

```python
def fit(
    self,
    xs,
    ys,
    x_cal=None,
    y_cal=None,
    prop_size=None,
    interval_learner=None,  # NEW: required if context_mode="calibrated"
):
```

### 3.2 Fit Logic: Computing Calibrated Contexts

**For regression (recommended path):**

```python
elif self.mode == "reg" and self.context_mode == "calibrated":
    if interval_learner is None:
        raise ValueError("interval_learner required for context_mode='calibrated'")

    # Get predictions and intervals from calibration
    y_pred_prop = model.predict(x_prop)  # shape (n,)
    intervals_prop = interval_learner.predict(x_prop)  # shape (n, 2)

    # Compute interval widths
    widths_prop = intervals_prop[:, 1] - intervals_prop[:, 0]

    # Bin by median prediction and median interval width
    pred_threshold = np.median(y_pred_prop)
    conf_threshold = np.median(widths_prop)

    # 4-way context: (high/low prediction) × (high/low confidence)
    # High confidence = narrow interval = width ≤ median
    context_prop = (
        2 * (y_pred_prop >= pred_threshold) +  # prediction: 0 or 2
        (widths_prop <= conf_threshold)         # confidence: 0 or 1
    )  # Result: 0, 1, 2, 3

    labels = np.array([0, 1, 2, 3])
    y_prop_ctx = context_prop
    y_calib_ctx = (similarly computed)
```

**For classification (for comparison):**

```python
elif self.mode == "clf" and self.context_mode == "calibrated":
    # Use predicted class + confidence
    y_proba_prop = model.predict_proba(x_prop)  # shape (n, n_classes)
    max_proba = np.max(y_proba_prop, axis=1)  # confidence per sample
    predicted_class = np.argmax(y_proba_prop, axis=1)

    # Bin confidence
    conf_threshold = np.median(max_proba)
    is_high_conf = (max_proba >= conf_threshold).astype(int)

    # 2×n_classes context
    context_prop = 2 * predicted_class + is_high_conf
    labels = np.unique(context_prop)
    y_prop_ctx = context_prop
```

### 3.3 Integration with CalibratedExplainer

**Ideal workflow:**

```python
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.guards import ConformalRegionOracle

# Train model
model = RandomForestRegressor().fit(X_train, y_train)

# Create explainer (auto-builds calibration)
explainer = CalibratedExplainer(
    model, X_cal, y_cal, mode="regression"
)

# Create guard with calibrated context
guard = ConformalRegionOracle(
    alpha=0.1,
    mode="reg",
    context_mode="calibrated"  # NEW
)

# Fit guard with explainer's interval learner
guard.fit(
    X_train, y_train,
    interval_learner=explainer.interval_learner  # Pass calibration
)

# Use guard
explainer.set_guard(guard)
explanations = explainer.explain_factual(X_test)
```

### 3.4 Context Semantics & Guarantees

**What does each context mean?**

For regression with 4-way context:

| Context | Meaning | Perturbation Strictness |
|---------|---------|------------------------|
| 0 | Low prediction, low confidence | Relaxed (model uncertain) |
| 1 | Low prediction, high confidence | Strict (model sure) |
| 2 | High prediction, low confidence | Relaxed (model uncertain) |
| 3 | High prediction, high confidence | Strict (model sure) |

**Why this is principled:**
- Regions align with **model predictions** (not arbitrary threshold)
- Regions reflect **model's own confidence** (tighter when model is confident)
- Perturbations stay **in regime where model made original prediction**

**Formal guarantee (sketch):**

Let $ (x_i, y_i) $ be a training pair where model predicts $ \hat{y}_i $ with interval $ [L_i, U_i] $.

If guard accepts perturbation $ x' $ for context $ c(x_i) $, then:
1. $ x' $ is within Mahalanobis ball of cluster $ k \in c(x_i) $
2. Cluster $ k $ was trained on samples with similar $ (\hat{y}, w) $ as $ x_i $
3. Therefore, $ \hat{y}' \approx \hat{y}_i $ (approximately same prediction)
4. And $ [L', U'] \approx [L_i, U_i] $ (approximately same uncertainty)

Thus: **$ x' $ is similar to data where model made similar predictions with similar confidence.** ✓

---

## Part 4: Implementation Plan

### 4.1 Phase 1: Core Implementation (Priority: HIGH)

**Files to modify:**

1. **`src/calibrated_explanations/guards/regions.py`**
   - Add `context_mode`, `quantile_thresholds` parameters to `__init__`
   - Add `interval_learner` parameter to `fit()`
   - Implement `_compute_calibrated_context()` method
   - Update `fit()` to compute calibrated contexts when `context_mode="calibrated"`
   - Update `label_context()` to handle new mode (if needed for prediction-time context)

2. **`src/calibrated_explanations/core/calibrated_explainer.py`**
   - Update `set_guard()` to auto-pass `interval_learner` if available
   - Update docstring to document new workflow

3. **`notebooks/quickstart_guard.ipynb`**
   - Add example: "Regression guard without threshold"
   - Show before/after (old way vs. new way)

### 4.2 Implementation Details

#### 4.2.1 New Parameters

```python
class ConformalRegionOracle:
    def __init__(
        self,
        alpha=0.1,
        mode="clf",
        context_mode="calibrated",  # NEW: "calibrated" (default for regression)
        quantile_thresholds=None,   # NEW: custom quantiles for binning
        # ... rest unchanged
    ):
        self.context_mode = context_mode
        self.quantile_thresholds = quantile_thresholds or (0.5, 0.5)  # (pred_q, conf_q)
```

#### 4.2.2 Fit Method Enhancement

```python
def fit(self, xs, ys, x_cal=None, y_cal=None, prop_size=None, interval_learner=None):
    x = check_array(xs)
    y = np.asarray(ys)
    # ... existing split logic ...

    if self.mode == "clf":
        labels = np.unique(y_prop)
        y_prop_ctx = y_prop
        y_calib_ctx = y_calib

    elif self.mode == "reg":
        if self.context_mode == "calibrated":
            if interval_learner is None:
                raise ValueError(
                    "interval_learner required for context_mode='calibrated' in regression mode"
                )
            labels, y_prop_ctx, y_calib_ctx = self._compute_calibrated_contexts_regression(
                x_prop, y_prop, x_calib, y_calib, interval_learner
            )
        else:
            # Fallback or error (if removing old threshold path)
            raise ValueError("For regression, use context_mode='calibrated'")

    # Rest of fit: cluster per label, etc.
    for label in labels:
        # ... existing clustering logic ...
```

#### 4.2.3 Context Computation Method

```python
def _compute_calibrated_contexts_regression(
    self, x_prop, y_prop, x_calib, y_calib, interval_learner
):
    """Compute 4-way contexts from predictions + interval widths."""

    # Predictions on proper set
    y_pred_prop = self.learner.predict(x_prop)  # or pass as param
    intervals_prop = interval_learner.predict(x_prop)  # (n, 2)
    widths_prop = intervals_prop[:, 1] - intervals_prop[:, 0]

    # Predictions on calibration set
    y_pred_calib = self.learner.predict(x_calib)
    intervals_calib = interval_learner.predict(x_calib)
    widths_calib = intervals_calib[:, 1] - intervals_calib[:, 0]

    # Thresholds (configurable quantiles)
    pred_q, conf_q = self.quantile_thresholds
    pred_thresh = np.quantile(y_pred_prop, pred_q)
    conf_thresh = np.quantile(widths_prop, conf_q)

    # Compute contexts (0–3)
    context_prop = (
        2 * (y_pred_prop >= pred_thresh).astype(int) +
        (widths_prop <= conf_thresh).astype(int)
    )
    context_calib = (
        2 * (y_pred_calib >= pred_thresh).astype(int) +
        (widths_calib <= conf_thresh).astype(int)
    )

    labels = np.array([0, 1, 2, 3])
    return labels, context_prop, context_calib
```

#### 4.2.4 Storing References

```python
def fit(self, ...):
    self._interval_learner = interval_learner
    self._model = model  # Store for predict-time context (if needed)
    # ...
```

### 4.3 Testing Strategy

**Unit tests** (in `tests/unit/guards/test_regions.py`):

1. **Context computation:**
   - Verify contexts are 4-way (0–3)
   - Verify thresholds are at specified quantiles
   - Verify context balance (roughly 25% each)

2. **Integration with CalibratedExplainer:**
   - Guard fits without explicit threshold
   - Contexts are stable (same seed → same contexts)
   - OOB rate ≈ α (guard is well-calibrated)

3. **Regression + Classification:**
   - Both modes produce consistent results
   - Classification still works (2n_classes contexts)
   - Regression no longer requires threshold

**Integration tests** (in `tests/integration/`):

1. End-to-end regression workflow:
   ```python
   explainer = CalibratedExplainer(model, X_cal, y_cal, mode="regression")
   guard = ConformalRegionOracle(alpha=0.1, mode="reg", context_mode="calibrated")
   guard.fit(X_train, y_train, interval_learner=explainer.interval_learner)
   explainer.set_guard(guard)
   explanations = explainer.explain_factual(X_test)
   assert len(explanations) == len(X_test)
   ```

2. Comparison with old approach (if kept):
   - Ensure new contexts are different from binary threshold
   - Verify explanations are still reasonable

### 4.4 Backward Compatibility

**Since guards are not yet public:**
- ✅ No backward compatibility required
- ✅ Can remove `threshold` parameter entirely
- ✅ Can make `context_mode="calibrated"` the default (only mode)

---

## Part 5: Advantages & Guarantees Comparison

### 5.1 Advantages Over Current Approach

| Aspect | Current (Threshold) | Proposed (Calibrated) |
|--------|------------|-----------|
| **Threshold required** | ❌ Yes (manual) | ✅ No |
| **User guidance** | ❌ None | ✅ Automatic |
| **Context semantics** | ❌ Arbitrary (label-based) | ✅ Model-aware (prediction + confidence) |
| **Confidence reflected** | ❌ No | ✅ Yes (narrow regions for high-confidence) |
| **Regression structure** | ❌ Lost (binary) | ✅ Preserved (4-way, 25-way, etc.) |
| **In-distribution guarantee** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Theory** | Conformal prediction on binary targets | Conformal prediction aligned with calibration |

### 5.2 Theoretical Guarantee Evolution

```
Old threshold approach:
  "Perturbation is in-distribution if within radius of points
   where model.predict(·) ≥ arbitrary_threshold"
  → Weak (no connection to model behavior)

Proposed calibrated approach:
  "Perturbation is in-distribution if within radius of points
   where model made similar predictions with similar confidence"
  → Strong (prediction + calibration aligned)
```

### 5.3 Empirical Expectations

**Expected improvements:**

1. **OOB rate:** Should be ≈ α (well-calibrated by design)
2. **Explanation stability:** Higher (regions are more meaningful)
3. **Rule length:** Similar or shorter (more constrained by confidence)
4. **Perturbation diversity:** Higher in low-confidence regions (intentional)

---

## Part 6: Edge Cases & Mitigations

### 6.1 Identified Edge Cases

| Edge Case | Scenario | Mitigation |
|-----------|----------|-----------|
| **Poor model** | Model predictions very noisy | Warning at fit-time; guard intervals become very wide |
| **Imbalanced contexts** | Some contexts have few samples | Validate context balance; warn if < 5% in any context |
| **Narrow intervals** | Model very confident everywhere | Contexts collapse (mostly high-confidence); acceptable (model is confident) |
| **Wide intervals** | Model very uncertain everywhere | Contexts collapse (mostly low-confidence); acceptable (model is uncertain) |
| **Distribution shift** | Test data far from training | Guard rejects more perturbations (correct behavior) |

### 6.2 Diagnostic Tool

Recommend adding a `GuardDiagnostics` class:

```python
class GuardDiagnostics:
    def __init__(self, guard):
        self.guard = guard

    def check_context_balance(self) -> Dict[int, float]:
        """Return percentage of samples per context."""
        # Check all contexts have ≥ 5% of samples

    def check_context_isolation(self) -> Dict[int, float]:
        """Return average intra-context distance vs inter-context."""
        # Ensure contexts are well-separated

    def check_model_confidence(self, interval_learner) -> float:
        """Return average interval width as % of y-range."""
        # Warn if model is too confident or too uncertain
```

---

## Part 7: Migration Path (No Breaking Changes)

### 7.1 User-Facing Changes

**Old usage (if any exists):**
```python
guard = ConformalRegionOracle(alpha=0.1, mode="reg", threshold=50_000)
guard.fit(X_train, y_train)
# ERROR: threshold no longer accepted
```

**New usage:**
```python
explainer = CalibratedExplainer(model, X_cal, y_cal, mode="regression")
guard = ConformalRegionOracle(alpha=0.1, mode="reg", context_mode="calibrated")
guard.fit(X_train, y_train, interval_learner=explainer.interval_learner)
# SUCCESS: no threshold needed
```

**If we want limited backward compatibility:**
```python
# Accept threshold but warn
if threshold is not None:
    warnings.warn(
        "threshold parameter is deprecated for regression. "
        "Use context_mode='calibrated' with interval_learner instead.",
        DeprecationWarning
    )
    # Fall back to old binary logic
```

### 7.2 Documentation Updates

1. **Update `ConformalRegionOracle` docstring:**
   - Remove threshold documentation
   - Document `context_mode="calibrated"`
   - Add example: "Regression without threshold"

2. **Update quickstart notebook:**
   - Add regression section with guard
   - Show end-to-end workflow
   - Explain context semantics

3. **Add architecture documentation:**
   - `docs/guides/guards_design.md`
   - Explain why calibrated contexts are better
   - Show theory + empirical validation

---

## Part 8: Success Criteria

### 8.1 Implementation Success

- ✅ `ConformalRegionOracle(mode="reg", context_mode="calibrated")` works without `threshold`
- ✅ `guard.fit(X, y, interval_learner=...)` accepts interval learner
- ✅ Contexts are computed from predictions + intervals
- ✅ All existing tests pass (no regression)
- ✅ New unit tests cover context computation
- ✅ Integration tests show end-to-end workflow

### 8.2 Quality Criteria

- ✅ OOB rate ≈ α (guard is well-calibrated)
- ✅ Explanations are stable (same seed → same rules)
- ✅ Rule length comparable or better than old approach
- ✅ No performance regression (guard filtering still fast)
- ✅ Documentation is clear (users understand context semantics)

### 8.3 User-Facing Success

- ✅ No more "what threshold should I use?" questions
- ✅ Users can get started with: `guard.fit(X, y, interval_learner=explainer.interval_learner)`
- ✅ Guarantees are clearly documented and understood

---

## Part 9: Code Snippets for Implementation

### 9.1 Complete `_compute_calibrated_contexts_regression` Method

```python
def _compute_calibrated_contexts_regression(
    self,
    x_prop: np.ndarray,
    y_prop: np.ndarray,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    interval_learner: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 4-way contexts from calibrated predictions and intervals.

    Returns:
        labels: [0, 1, 2, 3]
        context_prop: context assignment for proper set
        context_calib: context assignment for calibration set
    """
    import numpy as np

    # Get predictions
    try:
        y_pred_prop = self._model.predict(x_prop)
        y_pred_calib = self._model.predict(x_calib)
    except Exception as e:
        raise ValueError(f"Failed to get predictions: {e}")

    # Get intervals
    try:
        intervals_prop = interval_learner.predict(x_prop)
        intervals_calib = interval_learner.predict(x_calib)
    except Exception as e:
        raise ValueError(f"Failed to get intervals from interval_learner: {e}")

    # Ensure intervals are (n, 2)
    intervals_prop = np.asarray(intervals_prop)
    intervals_calib = np.asarray(intervals_calib)

    if intervals_prop.ndim == 1 or intervals_prop.shape[1] != 2:
        raise ValueError("interval_learner must return (n, 2) array")

    # Compute interval widths
    widths_prop = intervals_prop[:, 1] - intervals_prop[:, 0]
    widths_calib = intervals_calib[:, 1] - intervals_calib[:, 0]

    # Get thresholds
    pred_q, conf_q = self.quantile_thresholds
    pred_thresh = np.quantile(y_pred_prop, pred_q)
    conf_thresh = np.quantile(widths_prop, conf_q)

    _logger.info(
        f"Calibrated context: pred_threshold={pred_thresh:.4f} "
        f"(q={pred_q}), conf_threshold={conf_thresh:.4f} (q={conf_q})"
    )

    # Compute contexts
    context_prop = (
        2 * (y_pred_prop >= pred_thresh).astype(int) +
        (widths_prop <= conf_thresh).astype(int)
    )
    context_calib = (
        2 * (y_pred_calib >= pred_thresh).astype(int) +
        (widths_calib <= conf_thresh).astype(int)
    )

    labels = np.array([0, 1, 2, 3])

    return labels, context_prop, context_calib
```

### 9.2 Updated `fit` Method (Regression Branch)

```python
elif self.mode == "reg":
    if self.context_mode == "calibrated":
        if interval_learner is None:
            raise ValueError(
                "interval_learner required for context_mode='calibrated' in regression mode"
            )
        self._interval_learner = interval_learner

        labels, y_prop_ctx, y_calib_ctx = (
            self._compute_calibrated_contexts_regression(
                x_prop, y_prop, x_calib, y_calib, interval_learner
            )
        )
    else:
        raise ValueError(
            "regression mode requires context_mode='calibrated'; "
            "threshold-based approach is no longer supported"
        )
else:
    raise ValueError("Mode must be 'clf' or 'reg'")
```

### 9.3 Example Usage in Notebook

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.guards import ConformalRegionOracle

# Load data
dataset = fetch_openml(name="house_sales", version=3)
X = dataset.data.values.astype(float)
y = dataset.target.values / 1000
y_filter = y < 500
X, y = X[y_filter, :], y[y_filter]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_prop_train, y_prop_train)

# Create explainer (builds calibration)
explainer = CalibratedExplainer(
    model, X_cal, y_cal, mode="regression"
)

# Create guard with calibrated context (NO THRESHOLD!)
guard = ConformalRegionOracle(
    alpha=0.1,
    mode="reg",
    context_mode="calibrated"  # NEW: no threshold needed
)

# Fit guard
guard.fit(
    X_prop_train, y_prop_train,
    interval_learner=explainer.interval_learner  # Pass calibration
)

# Use guard
explainer.set_guard(guard)
explanations = explainer.explain_factual(X_test)

print(f"Guard fitted: {guard._fitted}")
print(f"Contexts: {np.unique(guard._clusters.keys())}")
```

---

## Conclusion

**This analysis proposes leveraging CalibratedExplainer's built-in calibration machinery to define guard contexts via calibrated predictions and uncertainty intervals.** This approach:

1. **Eliminates the threshold entirely** — No more user guessing
2. **Strengthens guarantees** — Regions align with model predictions + confidence
3. **Unifies classification and regression** — Both use (prediction, confidence) semantics
4. **Improves user experience** — Cleaner API, zero configuration
5. **Future-proofs the design** — Leverages calibration, the foundation of the entire framework

**Recommendation:** Implement immediately. No backward compatibility burden (feature not yet public), and the payoff in terms of API clarity and theoretical guarantees is substantial.

**Estimated effort:** 100–150 LOC, 3–5 engineering days (including tests & documentation).

---

**Document Version:** 1.0
**Status:** Ready for Architecture Review & Implementation Planning
**Next Step:** Engineering team to review and plan sprint allocation.
