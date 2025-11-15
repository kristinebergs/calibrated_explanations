# Guard Design: Confidence-Modulated Conformal Regions

**Date:** November 13, 2025
**Status:** Final Design (Ready for Implementation)
**Architecture:** Conformal prediction + calibrated confidence modulation
**Scope:** Both classification and regression

**Related Documents:**
- 📐 **[GUARD_MATHEMATICAL_FOUNDATIONS.md](./GUARD_MATHEMATICAL_FOUNDATIONS.md)** — Rigorous mathematical theory, conformal prediction proofs, formal guarantees
- 📊 **[GUARD_DESIGN_CONFIDENCE_MODULATION.md](./GUARD_DESIGN_CONFIDENCE_MODULATION.md)** — This document; implementation architecture and examples
- 🔍 **[ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md](./ANALYSIS_SUMMARY_CALIBRATED_PREDICTION_CONTEXT.md)** — Historical analysis and design evolution

---

## Executive Summary

The `ConformalRegionOracle` implements **confidence-modulated conformal regions** to filter out-of-distribution perturbations during explanation generation.

**Key insight:** Instead of categorical contexts based on arbitrary thresholds or binned predictions, use **continuous calibrated confidence** to modulate the conformal acceptance criterion.

**Design principle:** Conformal prediction provides validity guarantees; calibrated confidence determines *when* to be strict vs. lenient.

### No Threshold Required
- ✅ Regression guard works without threshold
- ✅ Uses calibrated predictions + intervals directly
- ✅ Single global clustering in feature space
- ✅ Confidence modulation at acceptance time
- ✅ Same mechanism for classification and regression

---

## Part 1: Mathematical Foundations (Conformal Prediction)

### 1.1 Conformal Prediction Theory

Conformal Prediction (CP) is a framework that wraps any predictor to provide **finite-sample coverage guarantees** without distributional assumptions.

**Fundamental theorem (informal):**

> For any miscalibration level $\alpha \in (0, 1)$, if we construct a conformal predictor using i.i.d. calibration data, the prediction set for any new point will contain the true target with probability at least $1 - \alpha$, with only the exchangeability assumption.

**Why this matters for guards:**
- We can construct "admissible regions" that are guaranteed to contain in-distribution data
- The guarantee holds **without assuming any particular data distribution**
- Only requirement: exchangeability (i.i.d. + any permutation-invariant data)

### 1.2 Nonconformity Measures

The core of conformal prediction is the **nonconformity measure** — a function that quantifies how "unusual" a point is.

$$A(z) = \text{measure of how different } (x, y) \text{ is from the training set}$$

Common choices:
- **Regression:** $A(x, y) = |y - \hat{y}(x)|$ (prediction error)
- **Classification (distance-based):** $A(x) = $ distance to nearest cluster center
- **Mahalanobis distance:** $A(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$

### 1.3 Inductive Conformal Prediction (ICP)

In practice, we don't use the full training set for calibration (avoids data waste). Instead:

1. **Proper set:** Train model and clustering on $\{\mathcal{Z}_{\text{prop}}\}$
2. **Calibration set:** Compute nonconformity scores on $\{\mathcal{Z}_{\text{cal}}\}$ (held-out)
3. **Radius computation:**
   $$r_\alpha = \text{quantile}_{1-\alpha}(A_{\text{cal}})$$

   This ensures: with probability $\geq 1 - \alpha$, a new point $x$ satisfies $A(x) \leq r_\alpha$

**Implementation in `ConformalRegionOracle`:**
```
Proper set (X_prop, y_prop) → Train clustering, compute centers
Calibration set (X_cal, y_cal) → Compute nonconformity scores
Radius: (1-α) quantile of scores → Conformal region boundary
```

### 1.4 Why Clustering Matters

Current implementation clusters data in feature space:

$$\text{Clusters: } C_k = \{x_i : \text{nearest center is } c_k\}$$

**Purpose:** Capture heteroscedasticity (different feature distributions in different regions)

**How it helps:**
- Points near cluster center are "typical" for that region
- Radius accounts for local variability
- More regions → captures more structure

### 1.5 Conformal Intervals (The Key for Guards)

For any point $x$, we can compute an **admissible interval** per feature.

**Given:**
- Cluster center $\mu$ with covariance $\Sigma$
- Conformal radius $r$

**For feature $j$, solve:**
$$\sum_{i=1}^d \frac{(x_i - \mu_i)^2}{\sigma_i^2} \leq r^2$$

For feature $j$, this yields an interval $[x_j - \Delta_j, x_j + \Delta_j]$ where:
$$\Delta_j = \sqrt{r^2 \sigma_j^2 - \sum_{i \neq j} (x_i - \mu_i)^2 / \sigma_i^2}$$

**Guarantee:** With probability $\geq 1 - \alpha$, perturbations within these intervals stay in-distribution.

---

## Part 2: Calibrated Confidence Modulation

### 2.1 The Role of Calibration

`CalibratedExplainer` computes calibrated predictions with uncertainty intervals:

**Classification (Venn-Abers):**
$$\text{predict}(x) \to (p_{\text{pred}}, [p_L, p_U])$$
- $p_{\text{pred}}$: predicted probability
- $[p_L, p_U]$: confidence interval (calibrated)

**Regression (Conformal Predictive Distribution):**
$$\text{predict}(x) \to (\hat{y}, [y_L, y_U])$$
- $\hat{y}$: predicted value
- $[y_L, y_U]$: prediction interval (calibrated)

### 2.2 Confidence as Information

The **interval width** is information about model confidence:

$$w = U - L$$

**Interpretation:**
- $w$ small (narrow interval) → model is confident
- $w$ large (wide interval) → model is uncertain

**Why use this?**
- High-confidence predictions should have strict perturbation bounds
- Low-confidence predictions should have lenient perturbation bounds
- This is *adaptive* per-instance

### 2.3 Modulation Function

We modulate the conformal radius by confidence:

$$r_{\text{eff}}(w) = r_{\text{base}} \cdot f(w)$$

where $f(w)$ is a modulation function.

**Default choice (linear):**
$$f(w) = 1 + (1 - \text{confidence}) \cdot \lambda$$

where $\text{confidence} \in [0, 1]$ is normalized:
$$\text{confidence} = 1 - \frac{w - w_{\min}}{w_{\max} - w_{\min}}$$

and $\lambda$ is a relaxation factor (e.g., $\lambda = 2.0$).

**Interpretation:**
- $\text{confidence} = 1$ (narrow interval): $f(w) = 1.0$ (use base radius)
- $\text{confidence} = 0$ (wide interval): $f(w) = 1 + \lambda$ (relax radius)

### 2.4 Why This Works Theoretically

**Claim:** Confidence-modulated acceptance preserves conformal guarantees.

**Reasoning:**

1. **Base radius $r$ has coverage guarantee:** By conformal prediction theory, $\Pr[A(x_{\text{new}}) \leq r] \geq 1 - \alpha$

2. **Modulation is data-driven:** The interval width $w$ is computed from calibrated interval learner, which has its own guarantees

3. **Modulation is monotonic:** Wider intervals (lower confidence) allow *larger* effective radius (more lenient). This is conservative — we don't reject things when the model says "I'm unsure"

4. **For perturbations within effective radius:** If $A(x') \leq r_{\text{eff}}(w)$, and if $(x', y')$ follows the same distribution as training data when the model is equally confident, then $y'$ should be similar to true target

**Key insight:** We're not weakening the conformal guarantee; we're making it *adaptive* based on model confidence. Higher confidence → stricter. Lower confidence → more lenient.

---

## Part 3: Implementation Details

### 3.1 Algorithm Overview

```
ALGORITHM: ConformalRegionOracle.fit(X_train, y_train, interval_learner, model)

Input:
  X_train: n × d feature matrix
  y_train: n × 1 target (for interface; not strictly used)
  interval_learner: fitted calibrator returning (L, U) intervals

Output:
  Fitted guard ready for accept() calls

Steps:
  1. Cluster X_train using KMeans into K clusters
  2. For each cluster k:
     - Center: μ_k = cluster center
     - Covariance: Σ_k = estimated covariance
     - Scores: A_k = {Mahalanobis(x_i, μ_k) : x_i in cluster k}
     - Radius: r_k = quantile(A_k, 1 - α)
  3. Compute width statistics:
     - w_min = min width across all training data
     - w_max = max width across all training data
  4. Store: μ_k, r_k, Σ_k, w_min, w_max
```

### 3.2 Acceptance Criterion

```
ALGORITHM: ConformalRegionOracle.accept(x', x_orig)

Input:
  x': candidate perturbation
  x_orig: original instance

Output:
  Boolean: accept or reject perturbation

Steps:
  1. Find nearest cluster to x_orig:
     k* = argmin_k || x_orig - μ_k ||_2

  2. Compute distance from x' to cluster center:
     dist = Mahalanobis(x', μ_k*)

  3. Get interval width for x_orig:
     (L, U) = interval_learner.predict(x_orig)
     w = U - L

  4. Normalize confidence:
     confidence = 1 - (w - w_min) / (w_max - w_min)

  5. Compute effective radius:
     r_eff = r_k* * (1 + (1 - confidence) * λ)

  6. Accept if within effective radius:
     return dist ≤ r_eff
```

### 3.3 Feature-Wise Intervals (For Perturbation Sampling)

For each feature $j$, compute allowed intervals given the other features are fixed at $x_{\text{orig}}$:

```
ALGORITHM: ConformalRegionOracle.intervals(x_orig)

For each feature j:
  1. Compute sum of squared Mahalanobis distances from all features except j:
     s_j = sum_{i ≠ j} ((x_orig[i] - μ[i])^2 / σ_i^2)

  2. Remaining "budget" for feature j:
     budget_j = r_eff^2 - s_j

  3. If budget_j < 0, no interval (feature is at boundary)

  4. Otherwise, interval for feature j:
     delta_j = sqrt(budget_j * σ_j^2)
     interval_j = [x_orig[j] - delta_j, x_orig[j] + delta_j]

Return: intervals = {interval_0, interval_1, ..., interval_{d-1}}
```

---

## Part 4: Formal Guarantees

### 4.1 Coverage Guarantee

**Theorem 1 (Conformal Coverage):**

Let $\{(x_i, y_i)\}_{i=1}^n$ be exchangeable samples. Let $A$ be a nonconformity measure and $r = \text{quantile}_{1-\alpha}(\{A(x_i, y_i)\})$.

Then for any new exchangeable point $(x_{\text{new}}, y_{\text{new}})$:
$$\Pr[A(x_{\text{new}}, y_{\text{new}}) \leq r] \geq 1 - \alpha$$

**In our context:**
- $A(x)$ = Mahalanobis distance to nearest cluster center
- $r = r_{\text{base}}$ (conformal radius)
- Coverage: $\Pr[\text{new point in conformal region}] \geq 1 - \alpha$

**Note:** No distributional assumptions. Only exchangeability required.

### 4.2 Modulation-Adjusted Guarantee

**Theorem 2 (Adaptive Coverage with Modulation):**

If we modulate the radius by a monotone-increasing function of interval width $f(w)$ (i.e., $f(w)$ is non-decreasing in $w$), the adjusted acceptance criterion still provides *conservative* coverage:

$$\Pr[A(x_{\text{new}}) \leq r_{\text{eff}}(w)] \geq 1 - \alpha$$

with equality approaching as modulation factor approaches 1.

**Proof sketch:**
- Since $f$ is monotone-increasing, $r_{\text{eff}} \geq r_{\text{base}}$ when $f \geq 1$
- By conformal theorem, $\Pr[A(x) \leq r_{\text{base}}] \geq 1 - \alpha$
- Therefore, $\Pr[A(x) \leq r_{\text{eff}}] \geq 1 - \alpha$ (weaker criterion)
- Coverage is **preserved** (possibly loose, but never violated)

### 4.3 In-Distribution Guarantee

**Theorem 3 (In-Distribution Perturbations):**

Let $x_{\text{orig}}$ be a test instance with calibrated interval $[L, U]$. If perturbation $x'$ satisfies:

1. $\text{Mahalanobis}(x', \mu_k) \leq r_{\text{eff}}(w)$, and
2. Model confidence on $x_{\text{orig}}$ is $w = U - L$

Then with probability $\geq 1 - \alpha$ (relative to training distribution):
- $x'$ is from the same region as points where the model made similarly-confident predictions
- The perturbation stays within the "natural domain" of features for that confidence level

**Interpretation:** Perturbations accepted by the guard are **in-distribution** in the sense that they come from regions where the model operates with similar confidence.

### 4.4 When Guarantees Apply

**Exchangeability requirement:**
- Training and test data must be exchangeable (i.i.d. + any permutation-invariant property)
- Covariate shift violates this (features change distribution, targets stay same)

**What's guaranteed:**
- Coverage level (at least $1 - \alpha$ of perturbations will be accepted as in-distribution)
- NOT: that explained rules are correct (explainer may still generate spurious rules)

**What's NOT guaranteed:**
- That rejected perturbations are truly out-of-distribution
- That the features we identify in rules are the "true" causes (that's up to the explainer)

---

## Part 5: Parameter Selection

### 5.1 Choosing $\alpha$ (Coverage Level)

**Meaning:** $\alpha = 1 - \text{coverage probability}$

| $\alpha$ | Coverage | Typical Use Case |
|----------|----------|------------------|
| 0.01 | 99% | Strict: allow only very similar perturbations |
| 0.05 | 95% | Moderate: balance between coverage and strictness |
| 0.10 | 90% | Relaxed: allow diverse perturbations |
| 0.20 | 80% | Very relaxed: almost never reject |

**Recommendation:** $\alpha = 0.1$ (default) balances validity and diversity

### 5.2 Choosing $n\_clusters$ (Feature Space Stratification)

**Meaning:** Number of regions to cluster feature space into

| $n\_clusters$ | Effect | Typical Use Case |
|---------------|--------|-----------------|
| 5 | Few regions, simple model | Small datasets (< 1K samples) |
| 10 | Moderate regions | Medium datasets (1K–10K samples) |
| 20 | Many regions, complex | Large datasets (> 10K samples) |

**Rule of thumb:** $n\_clusters \approx \sqrt{n\_\text{samples} / 10}$

### 5.3 Choosing $\lambda$ (Relaxation Factor)

**Meaning:** How much to relax radius for low-confidence predictions

$$r_{\text{eff}} = r_{\text{base}} \cdot (1 + (1 - \text{confidence}) \cdot \lambda)$$

| $\lambda$ | Effect | Typical Use Case |
|-----------|--------|-----------------|
| 0.5 | Minimal relaxation | Conservative: only slightly relax for uncertainty |
| 1.0 | Linear scaling | Balanced: proportional to uncertainty |
| 2.0 | Aggressive relaxation | Lenient: double radius for low-confidence |
| ∞ | Disable modulation | No effect: accept all perturbations at confidence=0 |

**Recommendation:** $\lambda = 1.0$ (default) provides balanced modulation

### 5.4 Choosing $\text{ncm\_method}$ (Nonconformity Measure)

**Options:**

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| `"mahalanobis"` | $\sqrt{(x-\mu)^T \Sigma^{-1}(x-\mu)}$ | Accounts for covariance | Sensitive to outliers in covariance |
| `"knn"` | Sum of distances to k-NN | Robust, local | Depends on k, slower |

**Recommendation:** `"mahalanobis"` (default) for efficiency

---

## Part 6: Usage Examples

### 6.1 Classification with Guard

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.guards import ConformalRegionOracle

# Load data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_prop, X_cal, y_prop, y_cal = train_test_split(X_train, y_train, test_size=0.25)

# Train classifier
clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_prop, y_prop)

# Create explainer (builds calibration)
explainer = CalibratedExplainer(clf, X_cal, y_cal, mode="classification")

# Create guard (confidence-modulated)
guard = ConformalRegionOracle(alpha=0.1, n_clusters=5, relaxation_factor=1.0)

# Fit guard (no threshold needed!)
guard.fit(X_prop, y_prop, interval_learner=explainer.interval_learner, model=clf)

# Use guard
explainer.set_guard(guard)
explanations = explainer.explain_factual(X_test[:10])

print(f"Guard fitted: {guard._fitted}")
print(f"OOB rate (should be ~{0.1}): {guard.empirical_oob_rate}")
```

### 6.2 Regression with Guard (No Threshold!)

```python
from sklearn.ensemble import RandomForestRegressor
from calibrated_explanations import CalibratedExplainer
from calibrated_explanations.guards import ConformalRegionOracle

# Load data
X, y = load_regression_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_prop, X_cal, y_prop, y_cal = train_test_split(X_train, y_train, test_size=0.25)

# Train regressor
reg = RandomForestRegressor(n_jobs=-1, random_state=42)
reg.fit(X_prop, y_prop)

# Create explainer
explainer = CalibratedExplainer(reg, X_cal, y_cal, mode="regression")

# Create guard (same API as classification!)
guard = ConformalRegionOracle(alpha=0.1, n_clusters=5, relaxation_factor=1.0)

# Fit guard (no threshold parameter at all!)
guard.fit(X_prop, y_prop, interval_learner=explainer.interval_learner, model=reg)

# Use guard
explainer.set_guard(guard)
explanations = explainer.explain_factual(X_test[:10])

print(f"Guard fitted: {guard._fitted}")
```

---

## Part 7: Comparison with Alternative Approaches

### 7.1 Why NOT Binary Contexts?

**Alternative:** Categorical contexts (4-way or n-way based on prediction/confidence bins)

| Aspect | Binary Contexts | Confidence Modulation |
|--------|-----------------|----------------------|
| **Binarization** | Loses information | Preserves continuous structure |
| **Per-context radius** | Different radius per context | Single radius, modulated by confidence |
| **Confidence handling** | Implicit (different context) | Explicit (modulation function) |
| **API complexity** | Extra parameter (`context_mode`) | Automatic, no extra parameters |
| **Theoretical justification** | Weaker (arbitrary bins) | Stronger (calibration + conformal) |

**Verdict:** Modulation is simpler and theoretically cleaner.

### 7.2 Why NOT Augmented Feature Space?

**Alternative:** Cluster in augmented space $[x, \hat{y}, w]$

| Aspect | Augmented Space | Modulated Regions |
|--------|-----------------|-------------------|
| **Clustering** | $[x, \hat{y}, w]$ mixed | $x$ only (features) |
| **Scaling issues** | Need careful normalization | Features in natural units |
| **Interpretability** | Distance in mixed space | Conformal + confidence |
| **Computational cost** | Higher-dimensional clustering | Lower-dimensional clustering |

**Verdict:** Modulation avoids scaling complexity.

---

## Part 8: Potential Issues & Mitigations

### 8.1 Poor Model Quality

**Issue:** If model predictions are noisy, confidence modulation may be ineffective

**Symptom:** Low OOB rate (guard rejects too much)

**Mitigation:**
- Check model performance: $R^2$ or accuracy
- If $R^2 < 0.7$, guard may be too strict
- Increase $\lambda$ to relax more for uncertain predictions
- Or increase $\alpha$ to be more lenient globally

### 8.2 Imbalanced Confidence

**Issue:** Most predictions have either high or low confidence

**Symptom:** Modulation factor is mostly extreme (1.0 or very large)

**Mitigation:**
- Check: `np.std(interval_widths)` should be substantial
- If interval learner is over-confident, that's good (model actually confident)
- If interval learner is uniformly wide, model is uncertain everywhere

### 8.3 Heteroscedasticity

**Issue:** Feature distribution varies dramatically by prediction value

**Symptom:** Explanations vary wildly between low and high predictions

**Mitigation:**
- Increase $n\_clusters$ to capture local structure
- Check conformal radius per cluster: should be similar if homoscedastic

### 8.4 Distribution Shift

**Issue:** Test data comes from different distribution than training

**Symptom:** OOB rate >> $\alpha$ (guard rejects too much)

**Mitigation:**
- This is correct behavior (guard rejects actual out-of-distribution)
- Increase $\alpha$ if you want to accept more
- Refit guard on representative calibration set

---

## Part 9: Mathematical Details: Conformal Intervals

### 9.1 Computing Feature-Wise Intervals

**Goal:** Given $x_{\text{orig}}$ and confidence $w$, compute interval $[L_j, U_j]$ for feature $j$

**Setup:**
- Cluster center: $\mu$
- Covariance: $\Sigma$ with diagonal elements $\sigma_i^2$
- Effective radius: $r_{\text{eff}}$

**Mahalanobis distance constraint:**
$$\sum_{i=1}^d \frac{(x_i - \mu_i)^2}{\sigma_i^2} \leq r_{\text{eff}}^2$$

**Solving for feature $j$:**

Let $x_{\text{orig}} = (x_1, \ldots, x_d)$ be fixed at all dimensions except $j$.

For dimension $i \neq j$: $(x_i - \mu_i)$ is fixed.

Compute sum of squared normalized distances from other dimensions:
$$S_j = \sum_{i \neq j} \frac{(x_i - \mu_i)^2}{\sigma_i^2}$$

Remaining "budget" for dimension $j$:
$$B_j = r_{\text{eff}}^2 - S_j$$

If $B_j < 0$: No valid interval (point is already outside radius for other features).

If $B_j \geq 0$:
$$\frac{(x_j - \mu_j)^2}{\sigma_j^2} \leq B_j$$

$$|x_j - \mu_j| \leq \sqrt{B_j \sigma_j^2}$$

$$\Delta_j = \sqrt{B_j \sigma_j^2}$$

$$\text{Interval}_j = [x_j - \Delta_j, x_j + \Delta_j]$$

### 9.2 Relative Intervals (For Perturbation Sampling)

Intervals are expressed relative to $x_{\text{orig}}[j]$:

$$\text{Relative interval}_j = [-\Delta_j, +\Delta_j]$$

This tells the perturbation sampler: "You can perturb feature $j$ by at most $\pm \Delta_j$"

---

## Part 10: Implementation Checklist

- [ ] Remove `threshold` parameter from `__init__`
- [ ] Add `model` parameter to `fit()`
- [ ] Remove `context_mode` and `quantile_thresholds` (no categorical contexts)
- [ ] Add `relaxation_factor` parameter to `__init__`
- [ ] Implement `fit()` with single global clustering
- [ ] Implement `accept()` with confidence modulation
- [ ] Implement `intervals()` using effective radius
- [ ] Compute width statistics: `_widths_min`, `_widths_max`
- [ ] Add method: `_normalize_confidence(width)`
- [ ] Add method: `_modulate_radius(base_radius, confidence)`
- [ ] Update docstrings with mathematical details
- [ ] Add examples to notebook (regression without threshold!)
- [ ] Add unit tests for modulation function
- [ ] Add integration tests with CalibratedExplainer
- [ ] Benchmark performance (should be same or faster)

---

## Conclusion

The confidence-modulated approach:

1. **Eliminates threshold requirement** for regression
2. **Preserves conformal guarantees** through monotone modulation
3. **Adapts to model confidence** — stricter for confident predictions
4. **Simpler API** — no `context_mode` parameter
5. **Universal mechanism** — works identically for classification and regression
6. **Theoretically grounded** — based on conformal prediction + calibration

This is the cleanest and most principled design.

---

**Document Version:** 2.0 (Revised to confidence modulation)
**Status:** Ready for implementation
**Next Step:** Implement core algorithm and tests
